"""④ 变化解码器：分别学习"建筑身份"与"几何变化"。

- 身份头（逐实例）：以 GRAIL 的上下文 token、软分配统计（落入垃圾箱的概率、最大匹配概率）、
  观测次数、历史存在信度、覆盖率为输入，判定 当前建筑∈{保持, 新建}、历史建筑∈{保持, 拆除}；
  严格一对一由互为最大匹配保证，未覆盖的历史建筑不参与拆除判定；
- 几何头（逐配对）：在历史建筑的规范坐标系（以其质心为原点、主轴为 x 轴、√面积 归一化）中，
  对"对齐后的当前基底"（T + δ）与历史基底做软栅格化（两通道 + 差分）-> 小 CNN，
  同时把两组顶点作为 token 送入小 Transformer；输出 改扩建概率、子类（扩建/缩减/改形）、IoU、
  当前基底的精修顶点。由于对齐已由 GRAIL 完成，这里学到的是"纯几何变化"，与身份判别解耦。
"""
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from geomac.layers import Mlp, conv_gn, fourier_encode
from .config import GrailCDCfg
from geomac.synth import resample_ring
from .geometry import apply_sim, poly_moments, principal_angle, rot2, soft_raster


def boundary_retention(pa, pb, eps=(1.0, 2.0), n=48):
    """双向边界保留率：历史轮廓上落在当前轮廓 eps 米以内的比例（及反向）。
    改扩建保留原有部分外墙（保留率高），原址重建整体替换（保留率低）——区分二者的关键几何线索。
    pa/pb (..., V, 2) 米。返回 (..., 2*len(eps))。"""
    shp = pa.shape[:-2]
    A = resample_ring(pa.reshape(-1, *pa.shape[-2:]), n)
    B = resample_ring(pb.reshape(-1, *pb.shape[-2:]), n)

    def dist(pts, poly):
        a, b = poly[:, None], torch.roll(poly, -1, 1)[:, None]
        p = pts[:, :, None]
        ab, ap = b - a, p - a
        t = ((ap * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-8)).clamp(0, 1)
        return (ap - t[..., None] * ab).norm(dim=-1).min(-1)[0]
    db, da = dist(B, pa.reshape(-1, *pa.shape[-2:])), dist(A, pb.reshape(-1, *pb.shape[-2:]))
    feats = [(db < e).float().mean(-1) for e in eps] + [(da < e).float().mean(-1) for e in eps]
    return torch.stack(feats, -1).view(*shp, 2 * len(eps))


class ChangeDecoder(nn.Module):
    def __init__(self, cfg: GrailCDCfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.grail.dim
        V = cfg.det.n_poly
        self.cur_id = Mlp(d + 4, d, 2)                                         # 保持 / 新建
        self.hist_id = Mlp(d + 4, d, 2)                                        # 保持 / 拆除
        n = cfg.pair_patch
        self.cnn = nn.Sequential(conv_gn(3, 32, 3, 2), conv_gn(32, 64, 3, 2), conv_gn(64, 64, 3, 2),
                                 nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Linear(256, d))
        self.vtx_in = nn.Linear(2 * 2 * 6 + 2 + 1, d)
        enc = nn.TransformerEncoderLayer(d, 4, 2 * d, batch_first=True, norm_first=True)
        self.vtx = nn.TransformerEncoder(enc, 2)
        self.pair = Mlp(4 * d + 4, d, d)
        self.rebuild = nn.Linear(d, 1)                                          # 原址重建（应拆为 拆除+新建）
        self.use_retention = True                                               # 消融开关
        self.mod = nn.Linear(d, 1)
        self.sub = nn.Linear(d, 3)
        self.iou = nn.Linear(d, 1)
        self.refine = nn.Linear(d, V * 2)
        nn.init.zeros_(self.refine.weight)
        nn.init.zeros_(self.refine.bias)
        g = (torch.arange(n) + 0.5) / n - 0.5
        gy, gx = torch.meshgrid(g, g, indexing="ij")
        self.register_buffer("grid", torch.stack([gx, -gy], -1).view(-1, 2) * 3.0, persistent=False)

    @torch.autocast("cuda", enabled=False)
    def identity(self, go: Dict, A: Dict, B: Dict):
        logP = go["logP"]
        m, n = logP.shape[1] - 1, logP.shape[2] - 1
        cl = lambda t: t.clamp(-20, 0)
        lpa_d, lpa_m = cl(logP[:, :m, n]), cl(logP[:, :m, :n].max(2)[0]) if n > 0 else cl(logP[:, :m, n])
        lpb_d, lpb_m = cl(logP[:, m, :n]), cl(logP[:, :m, :n].max(1)[0]) if m > 0 else cl(logP[:, m, :n])
        na = A.get("n_obs", torch.ones_like(lpa_d)).float()
        fa = torch.stack([lpa_d / 5, lpa_m / 5, torch.log1p(na), torch.log(A["pos_cov"].diagonal(dim1=-2, dim2=-1).sum(-1).clamp(min=1e-3, max=1e4))], -1)
        vb = B.get("vis", torch.ones_like(lpb_d))
        bl = B.get("belief", torch.ones_like(lpb_d))
        fb = torch.stack([lpb_d / 5, lpb_m / 5, vb, bl], -1)
        return (self.cur_id(torch.cat([go["tok_a"].float(), fa.float()], -1)),
                self.hist_id(torch.cat([go["tok_b"].float(), fb.float()], -1)))

    def geometry(self, go: Dict, A: Dict, B: Dict, c0, pairs, pair_mask):
        """pairs (B,P,2)=[当前索引, 历史索引]。返回逐配对几何变化输出与规范坐标信息。"""
        b, p, _ = pairs.shape
        ia, jb = pairs[..., 0].clamp(min=0), pairs[..., 1].clamp(min=0)
        g = lambda t, idx: t.gather(1, idx.view(b, p, *([1] * (t.dim() - 2))).expand(-1, -1, *t.shape[2:]))
        pa = apply_sim(go["x"], A["poly"], c0) + go["delta"][:, :, None]                 # 对齐后的当前基底
        pa, pb = g(pa, ia), g(B["poly"], jb)
        area, cent, cov = poly_moments(pb)
        phi = principal_angle(cov)
        sc = area.clamp(min=4.0).sqrt()
        R = rot2(-phi)
        can = lambda P: torch.einsum("bpij,bpvj->bpvi", R, P - cent[:, :, None]) / sc[..., None, None]
        ca, cb = can(pa), can(pb)
        flat = lambda t: t.reshape(b * p, *t.shape[2:])
        tau = 0.08
        grid = self.grid[None].expand(b * p, -1, -1)
        ra = soft_raster(flat(ca), grid, tau).view(b * p, 1, self.cfg.pair_patch, self.cfg.pair_patch)
        rb = soft_raster(flat(cb), grid, tau).view(b * p, 1, self.cfg.pair_patch, self.cfg.pair_patch)
        fc = self.cnn(torch.cat([ra, rb, ra - rb], 1)).view(b, p, -1)
        V = ca.shape[2]
        tok = torch.cat([torch.cat([fourier_encode((ca / 6 + 0.5).clamp(0, 1), 6), ca, torch.zeros_like(ca[..., :1])], -1),
                         torch.cat([fourier_encode((cb / 6 + 0.5).clamp(0, 1), 6), cb, torch.ones_like(cb[..., :1])], -1)], 2)
        fv = self.vtx(self.vtx_in(flat(tok))).mean(1).view(b, p, -1)
        ret = boundary_retention(pa, pb)
        if not self.use_retention:
            ret = torch.zeros_like(ret)
        h = self.pair(torch.cat([fc, fv, g(go["tok_a"], ia), g(go["tok_b"], jb), ret], -1))
        ref = (ca + 0.1 * torch.tanh(self.refine(h)).view(b, p, V, 2))
        inv = rot2(phi)
        ref_map = torch.einsum("bpij,bpvj->bpvi", inv, ref * sc[..., None, None]) + cent[:, :, None]
        inter = (ra * rb).flatten(1).sum(-1)
        iou_soft = (inter / (ra.flatten(1).sum(-1) + rb.flatten(1).sum(-1) - inter).clamp(min=1e-6)).view(b, p)
        return {"rebuild_logit": self.rebuild(h)[..., 0], "retention": ret, "mod_logit": self.mod(h)[..., 0], "sub_logit": self.sub(h), "iou_pred": torch.sigmoid(self.iou(h)[..., 0]),
                "iou_soft": iou_soft, "refined": ref_map, "aligned_cur": pa, "pair_mask": pair_mask}
