"""GRAIL-CD：当前建筑实例提取 + 多源历史编码 + 空间约束对齐关联 + 建筑变化解码 + 二维视频记忆。

在线（逐关键帧）：检测（跟踪先验查询 + 新查询）-> 屋顶多边形与 η -> 地平面单应 + 径向位移律得基底
               -> GRAIL（跟踪模式，紧先验）关联到实例记忆 -> 记忆更新（二维多视角三角化）。
低频（1–2 Hz/航段结束）：记忆导出当前集合 -> GRAIL（跨时相，GNSS 先验）对齐 + 关联历史统一集合
               -> 变化解码（身份头 + 几何头）。
离线：历史多源 -> 统一建筑集合（GRAIL 源间融合）。
"""
import copy
import math
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

from .change_decoder import ChangeDecoder
from .config import GrailCDCfg
from .current_branch import RoofDetector, to_map
from .geometry import camera_from_meta, ground_to_img, poly_moments
from .grail_layer import GRAIL, extract_matches
from .history_branch import HistoryBranch
from .memory import TrackMemory


class GrailCD(nn.Module):
    def __init__(self, cfg: GrailCDCfg):
        super().__init__()
        self.cfg = cfg
        self.detector = RoofDetector(cfg)
        self.memory = TrackMemory(cfg)
        tcfg = copy.deepcopy(cfg.grail)
        tcfg.n_layers, tcfg.em_iters, tcfg.use_hough = 1, 2, False
        tcfg.lattice_expand, tcfg.use_tor, tcfg.hard_refine = False, False, 0       # 逐帧跟踪：紧先验，无需格网补全/精化
        self.grail_trk = GRAIL(tcfg)                                            # 跟踪模式（紧先验）
        self.grail = GRAIL(cfg.grail)                                           # 跨时相
        self.history = HistoryBranch(cfg)
        self.change = ChangeDecoder(cfg)

    # ------------------------------------------------------------ 离线
    def encode_history(self, hist: Dict, hist_label_fn=None) -> Dict:
        U, sets, raw, logs = self.history(hist, hist_label_fn)
        return {"U": U, "sets": sets, "raw": raw, "logs": logs}

    def init_state(self, b, device):
        return {"mem": self.memory.init(b, device), "metas": [], "last_kf": None}

    # ------------------------------------------------------------ 在线逐帧
    def is_keyframe(self, meta, state):
        if state["last_kf"] is None:
            return True
        mv = (meta[:, :2] - state["last_kf"][:, :2]).norm(dim=-1)
        dyaw = (meta[:, 3] - state["last_kf"][:, 3]).abs()
        return bool(((mv > self.cfg.mem.keyframe_move_m) | (dyaw > 2.0)).any())

    def step(self, img, meta, state, label_fn: Optional[Callable] = None, obj_thr=0.5):
        """label_fn(det) -> (B,Q) 每个查询匹配到的 GT 建筑 ID（仅训练时用于给新轨迹打 GT 标签）。"""
        cfg = self.cfg
        if not self.training and not self.is_keyframe(meta, state):
            return None, state                                                   # 关键帧门控：跳过整帧
        hw = cfg.backbone.img_hw
        mem = state["mem"]
        tq = self.memory.track_queries(mem, meta, hw, cfg.det.n_track)
        det = self.detector(img, meta, tq)
        geo = to_map(det, meta, hw)
        b, q = det["obj"].shape
        n_trk = det["n_trk"]
        live = (torch.sigmoid(det["obj"]) > obj_thr) & geo["valid"] & det["qmask"]
        gt_lab = label_fn(det) if label_fn is not None else None
        if gt_lab is not None:                                                  # 训练：教师强制（匹配到 GT 的检测视为有效）
            live = live | ((gt_lab >= 0) & geo["valid"] & det["qmask"])
        A = {"poly": geo["foot"], "emb": det["emb"].float(), "mask": live, "pos_cov": geo["cov"]}
        Bset = self.memory.export_set(mem)
        if Bset["mask"].any() and live.any():
            c0 = (poly_moments(A["poly"])[1] * A["mask"][..., None]).sum(1) / A["mask"].sum(1, keepdim=True).clamp(min=1)
            pm = torch.zeros(b, 4, device=img.device)
            pv = torch.tensor([1.5 ** 2, 1.5 ** 2, math.radians(0.5) ** 2, 0.01 ** 2], device=img.device).expand(b, 4)
            go = self.grail_trk(A, {**Bset, "vis": torch.ones_like(Bset["mask"], dtype=torch.float)}, c0, pm, pv)
            m_a, _ = extract_matches(go["logP"], A["mask"], Bset["mask"])
        else:
            go, m_a = None, torch.full((b, q), -1, dtype=torch.long, device=img.device)
        birth = live & (m_a < 0)
        # 跟踪查询解码出的实例若未被关联，不新建轨迹（其身份已由查询携带）：交给关联结果，避免重复
        if n_trk > 0:
            birth[:, :n_trk] = False
        obs = {**geo, "emb": det["emb"], "eta": det["eta"], "eta_logsig": det["eta_logsig"], "obj": det["obj"]}
        if gt_lab is not None:
            obs["gt_id"] = gt_lab
        mem = self.memory.update(mem, obs, m_a, birth, meta)
        state = {"mem": mem, "metas": state["metas"] + [meta.detach()], "last_kf": meta.detach()}
        return {"det": det, "geo": geo, "trk": go, "trk_A": A, "trk_B": Bset, "assign": m_a, "tq": tq}, state

    # ------------------------------------------------------------ 覆盖率（历史建筑是否被当前视频拍到）
    def coverage(self, state, poly_map, margin=0.02):
        hw = self.cfg.backbone.img_hw
        _, c, _ = poly_moments(poly_map)
        vis = torch.zeros(c.shape[:2], device=c.device)
        for meta in state["metas"]:
            K, R, C = camera_from_meta(meta, hw)
            uv, front = ground_to_img(c, K, R, C)
            uvn = uv / torch.tensor([hw[1], hw[0]], device=c.device)
            inside = ((uvn > margin) & (uvn < 1 - margin)).all(-1) & front
            vis = torch.maximum(vis, inside.float())
        return vis

    # ------------------------------------------------------------ 低频：跨时相对齐关联 + 变化解码
    def associate(self, state, H: Dict, gt_pairs=None, sigma_t=8.0, sigma_deg=4.0):
        A = self.memory.export_set(state["mem"])
        U = H["U"]
        b = A["mask"].shape[0]
        dev = A["mask"].device
        vis = self.coverage(state, U["poly"])
        B = {**U, "vis": vis}
        c0 = (poly_moments(A["poly"])[1] * A["mask"][..., None]).sum(1) / A["mask"].sum(1, keepdim=True).clamp(min=1)
        pm = torch.zeros(b, 4, device=dev)
        pv = torch.tensor([sigma_t ** 2, sigma_t ** 2, math.radians(sigma_deg) ** 2, 0.04 ** 2], device=dev).expand(b, 4)
        go = self.grail(A, B, c0, pm, pv)
        m_a, m_b = extract_matches(go["logP"], A["mask"], B["mask"])
        id_a, id_b = self.change.identity(go, A, B)
        if gt_pairs is None:                                                     # 推理：用预测匹配
            P = A["mask"].shape[1]
            pairs = torch.stack([torch.arange(P, device=dev).expand(b, P), m_a.clamp(min=0)], -1)
            pmask = m_a >= 0
        else:
            pairs, pmask = gt_pairs
        geo = self.change.geometry(go, A, B, c0, pairs, pmask)
        return {"go": go, "A": A, "B": B, "c0": c0, "m_a": m_a, "m_b": m_b, "id_a": id_a, "id_b": id_b,
                "geom": geo, "pairs": pairs}

    @torch.no_grad()
    def summarize(self, res, thr_mod=0.5):
        """推理输出：实例级变化清单（地图坐标）。"""
        out = []
        b = res["m_a"].shape[0]
        for bi in range(b):
            A, B = res["A"], res["B"]
            new = A["mask"][bi] & (res["m_a"][bi] < 0) & (res["id_a"][bi].argmax(-1) == 1)
            dem = B["mask"][bi] & (res["m_b"][bi] < 0) & (B["vis"][bi] > 0.5) & (res["id_b"][bi].argmax(-1) == 1)
            mod = res["geom"]["pair_mask"][bi] & (torch.sigmoid(res["geom"]["mod_logit"][bi]) > thr_mod)
            out.append({"new": new.nonzero()[:, 0].tolist(), "demolished": dem.nonzero()[:, 0].tolist(),
                        "modified": res["pairs"][bi, mod].tolist(), "T": res["go"]["x"][bi].tolist()})
        return out

    # ------------------------------------------------------------ 训练用整段前向
    def forward(self, frames, metas, hist, label_fn=None, gt_pairs_fn=None, hist_label_fn=None):
        H = self.encode_history(hist, hist_label_fn)
        b, t = frames.shape[:2]
        state = self.init_state(b, frames.device)
        steps = []
        for i in range(t):
            o, state = self.step(frames[:, i], metas[:, i], state,
                                 (lambda det, i=i: label_fn(det, i)) if label_fn else None)
            steps.append(o)
        gt_pairs = gt_pairs_fn(state, H) if gt_pairs_fn else None
        res = self.associate(state, H, gt_pairs)
        return {"H": H, "steps": steps, "res": res, "state": state}
