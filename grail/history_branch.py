"""② 历史分支：把不同来源编码成统一的建筑集合（离线，任意源可缺）。

- SHP：多边形直接入集合（外观嵌入用源嵌入代替）；
- 航摄 / 卫星 / 地图栅格：共享的正射集合提取器（FiLM 注入源类别与 GSD）输出建筑多边形 + 嵌入；
- 历史倾斜视频：复用当前分支 + 跟踪记忆离线运行，导出其轨迹集合（自带多视角协方差）；
- 融合：按优先级 SHP > 航摄 > 历史视频 > 卫星 > 地图，以 GRAIL（源间先验）把每个新源对齐并关联到
  已融合集合：匹配 -> 多边形按顶点对齐加权平均、存在证据 +r_s；新源独有 -> 追加；已融合但新源中
  缺失（且在其覆盖范围内）-> 反证据 +r_s。存在信度 b = (1+正证据)/(2+正+反)，缺源天然不贡献证据。
"""
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from geomac.decoder import DecLayer, inv_sigmoid
from geomac.layers import Mlp, ResBlock, conv_gn, fourier_encode
from .config import HIST_SOURCES, GrailCDCfg
from .geometry import apply_sim, cyclic_align, poly_moments
from .grail_layer import GRAIL, extract_matches

RASTER_SOURCES = ("aerial", "sat", "map")


class OrthoSetExtractor(nn.Module):
    """正射栅格 -> 建筑集合（查询式，多边形以框相对坐标逐层细化）。"""

    def __init__(self, cfg: GrailCDCfg, n_q=96, n_layers=2):
        super().__init__()
        d, V = cfg.det.dim, cfg.det.n_poly
        self.V, self.n_q = V, n_q
        self.s4 = nn.Sequential(conv_gn(3, 32, 3, 2), conv_gn(32, 64, 3, 2), ResBlock(64))
        self.s8 = nn.Sequential(conv_gn(64, d, 3, 2), ResBlock(d))
        self.s16 = nn.Sequential(conv_gn(d, d, 3, 2), ResBlock(d))
        self.p4 = nn.Conv2d(64, d, 1)
        self.film = nn.Linear(len(RASTER_SOURCES) + 1, 2 * d)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.heat = nn.Conv2d(d, 1, 3, 1, 1)
        nn.init.constant_(self.heat.bias, -4.0)
        self.size = nn.Conv2d(d, 2, 1)
        self.sel = nn.Linear(d, d)
        self.pos = Mlp(4 * 2 * 8, d, d)
        self.layers = nn.ModuleList(DecLayer(d, 8, 3, 4) for _ in range(n_layers))
        self.box_delta, self.poly_delta = Mlp(d, d, 4), Mlp(d, d, V * 2)
        for m in (self.box_delta, self.poly_delta):
            nn.init.zeros_(m.layers[-1].weight)
            nn.init.zeros_(m.layers[-1].bias)
        self.obj = nn.Linear(d, 1)
        nn.init.constant_(self.obj.bias, -2.0)
        self.emb = Mlp(d, d, cfg.det.emb_dim)
        t = torch.linspace(0, 1, V + 1)[:-1] * 2 * torch.pi
        self.register_buffer("ring", torch.stack([0.5 + 0.5 * torch.cos(t), 0.5 + 0.5 * torch.sin(t)], -1), persistent=False)

    def forward(self, img, src: str, log_gsd=None):
        x4 = self.s4(img)
        x8 = self.s8(x4)
        cond = torch.zeros(img.shape[0], len(RASTER_SOURCES) + 1, device=img.device)
        cond[:, RASTER_SOURCES.index(src)] = 1
        if log_gsd is not None:
            cond[:, -1] = log_gsd
        g, bt = self.film(cond).chunk(2, -1)
        x8 = x8 * (1 + g[..., None, None]) + bt[..., None, None]
        x16 = self.s16(x8)
        v4 = self.p4(x4)
        b, d, h, w = x8.shape
        heat = self.heat(x8)
        k = min(self.n_q, h * w)
        idx = heat.flatten(1).detach().topk(k, 1)[1]
        sel = x8.flatten(2).transpose(1, 2).gather(1, idx[..., None].expand(-1, -1, d))
        cxy = torch.stack([(idx % w).float() + 0.5, (idx // w).float() + 0.5], -1) / torch.tensor([w, h], device=img.device)
        wh = 0.01 + 0.15 * torch.sigmoid(self.size(x8)).flatten(2).transpose(1, 2).gather(1, idx[..., None].expand(-1, -1, 2))
        q, box = self.sel(sel), torch.cat([cxy, wh], -1)
        poly = self.ring[None, None].expand(b, k, -1, -1)
        kpm = torch.zeros(b, k, dtype=torch.bool, device=img.device)
        for layer in self.layers:
            q = layer(q, self.pos(fourier_encode(box, 8)), box.detach(), [v4, x8, x16], kpm)
            box = torch.sigmoid(inv_sigmoid(box.detach()) + self.box_delta(q))
            poly = (poly.detach() + 0.1 * torch.tanh(self.poly_delta(q).view(b, k, self.V, 2))).clamp(0, 1)
        return {"obj": self.obj(q)[..., 0], "box": box, "poly": box[:, :, None, :2] + (poly - 0.5) * box[:, :, None, 2:],
                "emb": F.normalize(self.emb(q), dim=-1), "heat": heat}


class HistoryBranch(nn.Module):
    def __init__(self, cfg: GrailCDCfg):
        super().__init__()
        self.cfg = cfg
        self.extractor = OrthoSetExtractor(cfg)
        self.src_emb = nn.Parameter(torch.randn(len(HIST_SOURCES), cfg.det.emb_dim) * 0.1)
        self.reliab = nn.Parameter(torch.zeros(len(HIST_SOURCES)))
        self.fuse_grail = GRAIL(cfg.grail)
        self.src_sigma = {"shp": 2.0, "aerial": 1.5, "hist_uav": 4.0, "sat": 3.0, "map": 3.0}

    def raster_to_map(self, poly_n):
        H, W = self.cfg.raster_hw
        r = self.cfg.raster_res
        return torch.stack([(poly_n[..., 0] - 0.5) * W * r, (0.5 - poly_n[..., 1]) * H * r], -1)

    def encode_sources(self, hist: Dict, obj_thr=0.4, label_fn=None) -> Dict[str, Dict]:
        sets, raw = {}, {}
        dev = next(self.parameters()).device
        if hist.get("shp") is not None:
            p, m = hist["shp"]["poly"], hist["shp"]["mask"]
            e = F.normalize(self.src_emb[0].expand(*m.shape, -1), dim=-1)
            sets["shp"] = {"poly": p, "emb": e, "mask": m, "pos_cov": self._cov(m, "shp")}
        for s in RASTER_SOURCES:
            if hist.get(s) is not None:
                o = self.extractor(hist[s], s, hist.get(f"{s}_log_gsd"))
                raw[s] = o
                m = torch.sigmoid(o["obj"]) > obj_thr
                if label_fn is not None:                                         # 训练：教师强制
                    m = m | (label_fn(s, o) >= 0)
                e = F.normalize(o["emb"] + self.src_emb[HIST_SOURCES.index(s)], dim=-1)
                sets[s] = {"poly": self.raster_to_map(o["poly"]), "emb": e, "mask": m, "pos_cov": self._cov(m, s)}
        if hist.get("hist_uav") is not None:
            u = hist["hist_uav"]
            sets["hist_uav"] = {"poly": u["poly"], "emb": u["emb"], "mask": u["mask"], "pos_cov": u["pos_cov"]}
        return sets, raw

    def _cov(self, m, s):
        return torch.eye(2, device=m.device).expand(*m.shape, 2, 2) * self.src_sigma[s] ** 2 * 0.25

    def fuse(self, sets: Dict[str, Dict]) -> Dict:
        names = [s for s in HIST_SOURCES if s in sets]
        assert names, "至少需要一个历史源"
        U = {k: v.clone() for k, v in sets[names[0]].items()}
        b, n = U["mask"].shape
        r0 = F.softplus(self.reliab[HIST_SOURCES.index(names[0])]) + 0.5
        U["pos"] = U["mask"].float() * r0
        U["neg"] = torch.zeros_like(U["pos"])
        U["wpoly"] = U["mask"].float()
        logs = {}
        for s in names[1:]:
            A = sets[s]
            c0 = (poly_moments(A["poly"])[1] * A["mask"][..., None]).sum(1) / A["mask"].sum(1, keepdim=True).clamp(min=1)
            sg = self.src_sigma[s]
            pm = torch.zeros(b, 4, device=c0.device)
            pv = torch.tensor([sg ** 2, sg ** 2, 0.02 ** 2, 0.02 ** 2], device=c0.device).expand(b, 4)
            o = self.fuse_grail(A, {**U, "vis": torch.ones_like(U["pos"])}, c0, pm, pv)
            logs[s] = o
            m_a, m_b = extract_matches(o["logP"], A["mask"], U["mask"])
            r = F.softplus(self.reliab[HIST_SOURCES.index(s)]) + 0.5
            pa = apply_sim(o["x"], A["poly"], c0) + o["delta"][:, :, None]
            new_U = {k: v.clone() for k, v in U.items()}
            for bi in range(b):
                ia = (m_a[bi] >= 0).nonzero()[:, 0]
                jb = m_a[bi, ia]
                if len(ia):
                    al = cyclic_align(pa[bi, ia], U["poly"][bi, jb])
                    wu = U["wpoly"][bi, jb][:, None, None]
                    new_U["poly"][bi, jb] = (U["poly"][bi, jb] * wu + al) / (wu + 1)
                    new_U["wpoly"][bi, jb] = U["wpoly"][bi, jb] + 1
                    new_U["emb"][bi, jb] = F.normalize(U["emb"][bi, jb] + A["emb"][bi, ia], dim=-1)
                    new_U["pos"][bi, jb] = U["pos"][bi, jb] + r
                unm_u = U["mask"][bi] & (m_b[bi] < 0)
                new_U["neg"][bi] = U["neg"][bi] + r * unm_u.float()
            # 新源独有的建筑追加到集合
            add = A["mask"] & (m_a < 0)
            na = int(add.sum(1).max())
            if na > 0:
                order = add.float().argsort(1, descending=True)[:, :na]
                g = lambda t: t.gather(1, order.view(*order.shape, *([1] * (t.dim() - 2))).expand(-1, -1, *t.shape[2:]))
                ext = {"poly": g(pa), "emb": g(A["emb"]), "mask": g(add), "pos_cov": g(A["pos_cov"])}
                ext["pos"] = ext["mask"].float() * r
                ext["neg"] = torch.zeros_like(ext["pos"])
                ext["wpoly"] = ext["mask"].float()
                new_U = {k: torch.cat([new_U[k], ext[k]], 1) for k in new_U}
            U = new_U
        U["belief"] = (1 + U["pos"]) / (2 + U["pos"] + U["neg"])
        nsrc = U["wpoly"].clamp(min=1)
        U["pos_cov"] = U["pos_cov"] / nsrc[..., None, None]
        return U, logs

    def forward(self, hist: Dict, label_fn=None):
        sets, raw = self.encode_sources(hist, label_fn=label_fn)
        U, logs = self.fuse(sets)
        mmax = self.cfg.max_hist
        if U["mask"].shape[1] > mmax:                                           # 截断到上限（按信度）
            order = (U["belief"] * U["mask"]).argsort(1, descending=True)[:, :mmax]
            U = {k: v.gather(1, order.view(*order.shape, *([1] * (v.dim() - 2))).expand(-1, -1, *v.shape[2:]))
                 for k, v in U.items()}
        return U, sets, raw, logs
