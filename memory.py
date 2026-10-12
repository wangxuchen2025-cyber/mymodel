"""⑤ 二维视频记忆：实例级跟踪记忆（每栋当前建筑一个槽）。

每个槽保存：外观嵌入、融合后的基底多边形（地图 GNSS 系）、多视角三角化的正规方程、高度 h、
观测次数、未观测帧数。三件事：
1) 预测：按透视径向位移律 R'_t = N_t + H_t/(H_t-h)·(F-N_t) 预测轨迹在新帧的屋顶位置，反投影到图像
   得到跟踪先验框 -> 作为检测器的跟踪查询（MOTR 式），稳定轨迹无需重新"发现"；
2) 二维多视角三角化：每次观测给出 H_t·F + (R'_t-N_t)·h = H_t·R'_t（对 (F,h) 线性），
   累加加权正规方程后闭式求解；单帧 η 作为 h 的弱先验。天底点随航线移动形成基线，
   基底位置与高度的不确定度随观测增多而收缩（协方差 = σ²(AᵀWA)⁻¹），供 GRAIL 使用；
3) 稳定性与计算复用：稳定轨迹（观测≥阈值、协方差小）只做轻量更新；关键帧门控在天底点移动
   不足时跳过整帧计算（见 model.py）。
"""
from typing import Dict

import torch
import torch.nn as nn

from .config import GrailCDCfg
from .geometry import camera_from_meta, cyclic_align, ground_to_img, sym2_clamp


class TrackMemory(nn.Module):
    def __init__(self, cfg: GrailCDCfg):
        super().__init__()
        self.cfg = cfg
        e = cfg.det.emb_dim
        self.gru = nn.GRUCell(e, e)
        self.sigma_obs = 0.6                                                    # 单次观测的地面投影噪声（米）

    def init(self, b, device) -> Dict[str, torch.Tensor]:
        T, V, E = self.cfg.mem.max_tracks, self.cfg.det.n_poly, self.cfg.det.emb_dim
        z = lambda *s, dt=torch.float32: torch.zeros(b, T, *s, device=device, dtype=dt)
        return {"alive": z(dt=torch.bool), "emb": z(E), "shape": z(V, 2), "wsum": z(), "AtA": z(3, 3), "Atb": z(3),
                "F": z(2), "h": z(), "cov": z(2, 2) + torch.eye(2, device=device) * 100, "n_obs": z(), "age": z(),
                "conf": z(), "gt_id": z(dt=torch.long) - 1}

    # ------------------------------------------------------------ 预测（跟踪先验查询）
    def track_queries(self, st, meta, img_hw, max_q):
        H, W = img_hw
        K, R, C = camera_from_meta(meta, img_hw)
        nad, Hc = C[:, :2], C[:, 2]
        foot = st["shape"] + st["F"][:, :, None]
        k = Hc[:, None, None, None] / (Hc[:, None, None, None] - st["h"][..., None, None]).clamp(min=1.0)
        roof_pred = nad[:, None, None] + (foot - nad[:, None, None]) * k
        uv, front = ground_to_img(roof_pred, K, R, C)
        uvn = uv / torch.tensor([W, H], device=uv.device)
        lo, hi = uvn.amin(2), uvn.amax(2)
        inside = (hi > 0.02).all(-1) & (lo < 0.98).all(-1) & front.all(-1)
        lo, hi = lo.clamp(0, 1), hi.clamp(0, 1)
        box = torch.cat([(lo + hi) / 2, (hi - lo).clamp(min=0.01)], -1)
        ok = st["alive"] & inside & (st["age"] < self.cfg.mem.max_age)
        score = ok.float() * (1 + st["n_obs"])
        idx = score.topk(min(max_q, score.shape[1]), 1)[1]
        g = lambda t: t.gather(1, idx.view(*idx.shape, *([1] * (t.dim() - 2))).expand(-1, -1, *t.shape[2:]))
        return {"emb": g(st["emb"]), "box": g(box), "mask": g(ok), "slot": idx}

    # ------------------------------------------------------------ 导出当前集合（供跨时相 GRAIL）
    def export_set(self, st):
        mask = st["alive"] & (st["n_obs"] >= 1)
        return {"poly": st["shape"] + st["F"][:, :, None], "emb": st["emb"], "mask": mask,
                "pos_cov": st["cov"] + 0.25 * torch.eye(2, device=st["cov"].device), "h": st["h"],
                "n_obs": st["n_obs"], "gt_id": st["gt_id"]}

    # ------------------------------------------------------------ 更新
    @torch.autocast("cuda", enabled=False)
    def update(self, st, obs: Dict, assign, birth, meta, lam_h=1.0):
        """obs：to_map 的输出 + 检测输出（emb, eta, eta_logsig, obj, gt_id 可选）；
        assign (B,Q) 槽索引或 -1；birth (B,Q) bool 需新建轨迹的检测。"""
        st = {k: v.clone() for k, v in st.items()}
        obs = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in obs.items()}
        b, q = assign.shape
        dev = assign.device
        nad, Hc = obs["nadir"], obs["H"]
        for bi in range(b):
            free = (~st["alive"][bi]).nonzero()[:, 0]
            nb = int(birth[bi].sum())
            slots = assign[bi].clone()
            if nb > 0 and len(free) > 0:
                bi_idx = birth[bi].nonzero()[:, 0][:len(free)]
                slots[bi_idx] = free[:len(bi_idx)]
                st["alive"][bi, free[:len(bi_idx)]] = True
                st["shape"][bi, free[:len(bi_idx)]] = 0
                st["emb"][bi, free[:len(bi_idx)]] = obs["emb"][bi, bi_idx].float()
                if "gt_id" in obs:
                    st["gt_id"][bi, free[:len(bi_idx)]] = obs["gt_id"][bi, bi_idx]
            qi = (slots >= 0).nonzero()[:, 0]
            if len(qi) == 0:
                st["age"][bi] += 1
                continue
            ti = slots[qi]
            w = torch.sigmoid(obs["obj"][bi, qi].float()) * obs["valid"][bi, qi].float()
            roof = obs["roof"][bi, qi]
            rc = roof.mean(1)
            r = (rc - nad[bi]) / Hc[bi]
            # 正规方程：每次观测两行 [1,0,r_x; 0,1,r_y]·[F;h] = R'_c（已按 H 归一），以及 h 的弱先验
            A = torch.zeros(len(qi), 2, 3, device=dev)
            A[:, 0, 0] = 1
            A[:, 1, 1] = 1
            A[:, :, 2] = r
            AtA = torch.einsum("nji,njk->nik", A, A) * w[:, None, None]
            Atb = torch.einsum("nji,nj->ni", A, rc) * w[:, None]
            h_obs = obs["eta"][bi, qi].float() * Hc[bi]
            sh = (obs["eta_logsig"][bi, qi].float().exp() * Hc[bi]).clamp(min=0.5)
            ph = lam_h * (self.sigma_obs / sh) ** 2 * w
            AtA[:, 2, 2] += ph
            Atb[:, 2] += ph * h_obs
            st["AtA"][bi, ti] += AtA
            st["Atb"][bi, ti] += Atb
            reg = torch.diag(torch.tensor([1e-4, 1e-4, 1e-4], device=dev))
            sol = torch.linalg.solve(st["AtA"][bi, ti] + reg, st["Atb"][bi, ti][..., None])[..., 0]
            st["F"][bi, ti], st["h"][bi, ti] = sol[:, :2], sol[:, 2].clamp(min=0.0)
            cov = self.sigma_obs ** 2 * torch.linalg.inv(st["AtA"][bi, ti] + reg)[:, :2, :2]
            cov = 0.5 * (cov + cov.transpose(-1, -2))
            st["cov"][bi, ti] = sym2_clamp(cov, 1e-4, 1e4)                      # 数值上保证正定
            # 形状融合：用当前 h 把屋顶投影还原为基底，去质心后与已存形状按顶点对齐加权平均
            k = (1 - st["h"][bi, ti] / Hc[bi]).clamp(min=0.05)[:, None, None]
            foot = nad[bi] + (roof - nad[bi]) * k
            shp = foot - foot.mean(1, keepdim=True)
            old = st["shape"][bi, ti]
            has = st["wsum"][bi, ti] > 0
            shp = torch.where(has[:, None, None], cyclic_align(shp, old), shp)
            ws = st["wsum"][bi, ti]
            st["shape"][bi, ti] = (old * ws[:, None, None] + shp * w[:, None, None]) / (ws + w).clamp(min=1e-6)[:, None, None]
            st["wsum"][bi, ti] = ws + w
            st["emb"][bi, ti] = nn.functional.normalize(self.gru(obs["emb"][bi, qi].float(), st["emb"][bi, ti]), dim=-1)
            st["n_obs"][bi, ti] += 1
            st["conf"][bi, ti] = torch.maximum(st["conf"][bi, ti], w)
            seen = torch.zeros_like(st["alive"][bi])
            seen[ti] = True
            st["age"][bi] = torch.where(seen, torch.zeros_like(st["age"][bi]), st["age"][bi] + 1)
        return st
