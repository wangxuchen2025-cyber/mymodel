"""GRAIL：Geometric-Relational Alignment & Instance Linking（空间约束对齐关联层，本文研究重点）。

同时回答两个问题：
  (1) 关联：当前实例 i 与历史实例 j 是不是同一栋？（严格一对一 + 垃圾箱 = 新建/拆除/未覆盖）
  (2) 对齐：两套集合之间的全局相似变换 T=[t,θ,log s] 与逐栋残差 δ_i 是多少？

空间关系的四个层次（都对 GNSS 平移偏差不变，对旋转偏差等变/近似不变）：
  - 一元：形状 Fourier 描述子（平移/旋转/起点不变）、面积、长宽比、紧致度 + 外观嵌入；
  - 二元（组内）：成对距离、绝对/相对主轴方位、朝向差、面积比 -> 几何自注意力（GeoTransformer 式
    几何结构嵌入）。每栋楼由"它的邻居在哪里、长什么样"来刻画，从而区分成排的相似建筑；
  - 二阶（跨组）：邻域一致性 S_nc —— (i,j) 是好匹配，当且仅当 i 的邻居经 T 旋转缩放后的相对向量
    能在 j 的邻居中找到匹配（DGMC / PointDSC 的二阶一致性思想），按上一轮软分配加权；
  - 全局：所有建筑共享同一个 T（GNSS/IMU 先验约束），由 Hough 投票初始化，再以 EM 交替：
        E 步：S = 外观 + 位置似然(马氏距离, 逐栋残差边缘化) + 形状重叠(log Bhattacharyya) + 邻域一致性
              -> 带垃圾箱 Sinkhorn 得到软分配 P；
        M 步：变化感知加权 Umeyama（形状变化大的配对降权，避免改扩建拉偏对齐）+ GNSS 先验 MAP 融合。
最终逐栋残差 δ_i 取高斯先验下的 MAP（收缩估计），吸收地形起伏与屋顶→基底换算误差。
同一层用于三处：多视角跟踪关联（紧先验）、历史多源集合融合（源间先验）、跨时相变化关联（GNSS 先验）。
"""
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from geomac.layers import Mlp, fourier_encode
from .config import GrailCfg
from .topology import align_period_axes, lattice_agreement, lattice_coords
from .geometry import (apply_sim, apply_sim_cov, log_bhattacharyya, log_sinkhorn_dustbin, poly_moments,
                       principal_angle, rot2, sym2_eig, weighted_umeyama, wrap)


def shape_descriptor(poly, n_desc=8):
    """Fourier 形状描述子：复数轮廓 z 的 DFT 幅值 |Z_k|/|Z_1|（平移、旋转、起点不变）+ 标量形状量。"""
    z = torch.complex(poly[..., 0], poly[..., 1])
    Z = torch.fft.fft(z - z.mean(-1, keepdim=True), dim=-1)
    mag = Z.abs()
    ref = torch.maximum(mag[..., 1], mag[..., -1]).clamp(min=1e-6)
    ks = list(range(2, n_desc + 2))
    desc = torch.stack([mag[..., k] / ref for k in ks] + [mag[..., -k] / ref for k in ks], -1)
    A, _, cov = poly_moments(poly)
    l1, l2, _ = sym2_eig(cov)
    ev = torch.stack([l1, l2], -1).clamp(min=1e-6)
    per = (torch.roll(poly, -1, -2) - poly).norm(dim=-1).sum(-1).clamp(min=1e-6)
    scal = torch.stack([torch.log(A.clamp(min=1.0)) / 5, 0.5 * torch.log(ev[..., 1] / ev[..., 0]),
                        4 * math.pi * A / per ** 2], -1)
    return torch.cat([desc, scal], -1)


def relation_features(cent, phi, area, sigma_d, n_freq=6):
    """组内成对空间关系 (B,N,N,F)：距离正弦编码、绝对方位、相对主轴方位、朝向差、面积比。"""
    d = cent[:, None, :, :] - cent[:, :, None, :]                             # i->j
    dist = d.norm(dim=-1)
    bear = torch.atan2(d[..., 1], d[..., 0])
    rel = bear - phi[:, :, None]
    dphi = phi[:, None, :] - phi[:, :, None]
    la = torch.log(area.clamp(min=1.0))
    f = [fourier_encode((dist / (sigma_d * 20)).clamp(max=1)[..., None], n_freq),
         torch.stack([bear.cos(), bear.sin(), (2 * rel).cos(), (2 * rel).sin(),
                      (4 * dphi).cos(), (4 * dphi).sin(), (la[:, None, :] - la[:, :, None]) / 2,
                      torch.exp(-dist / sigma_d)], -1)]
    return torch.cat(f, -1)


class GeoSelfAttention(nn.Module):
    """几何自注意力：logit_ij = q_i·(k_j + r_ij)/√d，r_ij 来自成对空间关系。"""

    def __init__(self, dim, heads, rel_dim):
        super().__init__()
        self.h, self.dh = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.rk = nn.Linear(rel_dim, dim)
        self.rb = nn.Linear(rel_dim, heads)
        self.out = nn.Linear(dim, dim)
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ffn = Mlp(dim, 2 * dim, dim)

    def forward(self, x, rel, mask):
        b, n, c = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, n, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        r = self.rk(rel).view(b, n, n, self.h, self.dh).permute(0, 3, 1, 2, 4)        # b,h,n,n,dh
        logit = (q @ k.transpose(-1, -2) + (q[:, :, :, None, :] * r).sum(-1)) / self.dh ** 0.5
        logit = logit + self.rb(rel).permute(0, 3, 1, 2)
        logit = logit.masked_fill(~mask[:, None, None, :], -1e4)
        a = logit.softmax(-1)
        o = (a @ v).transpose(1, 2).reshape(b, n, c)
        x = x + self.out(o)
        return x + self.ffn(self.n2(x))


class CrossAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ffn = Mlp(dim, 2 * dim, dim)

    def forward(self, x, y, mask_y):
        kpm = ~mask_y
        kpm = kpm & ~(kpm.all(1, keepdim=True))                                   # 对端为空时避免全遮挡
        x = x + self.attn(self.n1(x), y, y, key_padding_mask=kpm)[0]
        return x + self.ffn(self.n2(x))


def shape_uniqueness(desc, area, mask, tau):
    """集合内形状独特性 u_i = 1 / (1 + Σ_{i'≠i} 相似度)：成排复制楼型 u 小，端头/转角/公共建筑 u 大。"""
    la = torch.log(area.clamp(min=1.0))
    dd = torch.cdist(desc, desc, p=1) / desc.shape[-1] ** 0.5 + (la[..., :, None] - la[..., None, :]).abs()
    sim = torch.exp(-(dd / tau) ** 2) * mask[..., None, :] * mask[..., :, None]
    sim = sim - torch.diag_embed(torch.diagonal(sim, dim1=-2, dim2=-1))
    return 1.0 / (1.0 + sim.sum(-1))


def knn_idx(cent, mask, k):
    d = torch.cdist(cent, cent)
    d = d.masked_fill(~mask[:, None, :], 1e6)
    d = d + torch.eye(d.shape[1], device=d.device) * 1e6
    k = min(k, max(d.shape[1] - 1, 1))
    val, idx = d.topk(k, -1, largest=False)
    return idx, val < 1e5


class GRAIL(nn.Module):
    def __init__(self, cfg: GrailCfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        nd = 2 * 8 + 3
        self.shape_a = Mlp(nd + cfg.emb_dim + 1, d, d)
        self.shape_b = Mlp(nd + cfg.emb_dim + 1, d, d)
        rel_in = 2 * cfg.shape_freq + 8
        self.rel = Mlp(rel_in, 64, 32)
        self.self_layers = nn.ModuleList(GeoSelfAttention(d, cfg.heads, 32) for _ in range(cfg.n_layers))
        self.cross_ab = nn.ModuleList(CrossAttention(d, cfg.heads) for _ in range(cfg.n_layers))
        self.cross_ba = nn.ModuleList(CrossAttention(d, cfg.heads) for _ in range(cfg.n_layers))
        self.proj_a, self.proj_b = nn.Linear(d, d), nn.Linear(d, d)
        self.match_a, self.match_b = nn.Linear(d, 1), nn.Linear(d, 1)
        self.dustbin = nn.Parameter(torch.tensor(1.0))
        self.vis_bias = nn.Parameter(torch.tensor(2.0))
        # 各项打分的权重（softplus 保证非负）与变化感知 M 步的形状降权指数
        self.w_raw = nn.Parameter(torch.tensor([0.5, 0.5, 1.0]))               # 位置, 形状, 邻域一致性
        self.gamma_raw = nn.Parameter(torch.tensor(0.0))
        self.w_tor_raw = nn.Parameter(torch.tensor(1.0))                       # TOR 一致性项权重

    # ------------------------------------------------------------ 编码
    def encode(self, A: Dict, B: Dict):
        cfg = self.cfg
        out = {}
        for key, S, mlp in (("a", A, self.shape_a), ("b", B, self.shape_b)):
            area, cent, cov = poly_moments(S["poly"])
            phi = principal_angle(cov)
            desc = shape_descriptor(S["poly"])
            vis = S.get("vis", torch.ones_like(area))
            tok = mlp(torch.cat([desc, S["emb"], vis[..., None]], -1))
            rel = self.rel(relation_features(cent, phi, area, cfg.sigma_d, cfg.shape_freq))
            if not cfg.use_rel:
                rel = torch.zeros_like(rel)
            out[key] = {"tok": tok, "rel": rel, "area": area, "cent": cent, "cov": cov, "phi": phi, "desc": desc}
        xa, xb = out["a"]["tok"], out["b"]["tok"]
        for sl, cab, cba in zip(self.self_layers, self.cross_ab, self.cross_ba):
            xa = sl(xa, out["a"]["rel"], A["mask"])
            xb = sl(xb, out["b"]["rel"], B["mask"])
            xa, xb = cab(xa, xb, B["mask"]), cba(xb, xa, A["mask"])
        out["a"]["tok"], out["b"]["tok"] = xa, xb
        return out

    # ------------------------------------------------------------ Hough 投票初始化（多假设）
    def hough_init(self, ea, eb, A, B, c0, prior_mu, prior_var, S_feat, n_hyp, tor_agr=None, tor_valid=None):
        """稠密候选投票：GNSS 门限内的全部 (i,j) 配对，以 外观相似 + 形状相似 加权，
        每个配对按朝向差（模 90°）给出 θ，按质心给出 t，在 (θ,tx,ty) 网格上核密度累积 + 先验；
        3D 非极大抑制取前 K 个峰，峰邻域 soft-argmax 亚格网精化。"""
        cfg = self.cfg
        b, m, n = S_feat.shape
        st = (prior_var[:, :2].mean(1)).sqrt().clamp(min=2.0)
        sr = prior_var[:, 2].sqrt().clamp(min=math.radians(2))
        tmax, rmax = 3 * st, (3 * sr).clamp(max=math.pi / 4)
        cap = apply_sim(prior_mu, ea["cent"], c0)
        dist = torch.cdist(cap, eb["cent"])
        gate = (dist < (tmax + 10.0)[:, None, None]) & A["mask"][..., None] & B["mask"][:, None]
        la, lb = torch.log(ea["area"].clamp(min=1.0)), torch.log(eb["area"].clamp(min=1.0))
        shape_sim = -2.0 * (la[..., None] - lb[:, None]).abs() - (ea["desc"][:, :, None] - eb["desc"][:, None]).abs().sum(-1)
        sc = (S_feat + shape_sim).masked_fill(~gate, -1e4)
        w = sc.softmax(-1) * gate.any(-1, keepdim=True) * gate                 # b,m,n
        if tor_agr is not None:                                                 # 拓扑序矛盾的配对不投票（打掉错位峰）
            w = w * ((1 - tor_valid) + tor_valid * tor_agr.clamp(min=1e-3) ** cfg.tor_vote_pow)
        if cfg.idf_vote:                                                        # 独特建筑的投票更有区分力
            ua = shape_uniqueness(ea["desc"], ea["area"], A["mask"], cfg.idf_tau) ** getattr(cfg, "idf_pow", 0.5)
            w = w * (ua / ua.masked_fill(~A["mask"], 0).amax(1, keepdim=True).clamp(min=1e-6))[..., None]
        th = wrap(eb["phi"][:, None] - ea["phi"][..., None], math.pi / 2)      # 4 重对称（矩形）
        ca = ea["cent"][:, :, None] - c0[:, None, None]
        t = eb["cent"][:, None] - c0[:, None, None] - torch.einsum("bmnij,bmj->bmni", rot2(th), ca[:, :, 0])
        nt, nr = cfg.hough_bins_t, cfg.hough_bins_theta
        gt = torch.linspace(-1, 1, nt, device=c0.device)[None] * tmax[:, None]   # b,nt
        gr = torch.linspace(-1, 1, nr, device=c0.device)[None] * rmax[:, None]
        ht, hr = (2 * tmax / (nt - 1)).view(-1, 1, 1), (2 * rmax / (nr - 1)).view(-1, 1, 1)
        rel = (t - prior_mu[:, None, None, :2]).flatten(1, 2)                     # b,p,2
        dth = wrap(th - prior_mu[:, None, None, 2], math.pi / 2).flatten(1)
        wf = w.flatten(1)
        keep = wf.topk(min(4096, wf.shape[1]), 1)[1]                             # 只保留权重最高的配对，控制开销
        rel, dth, wf = rel.gather(1, keep[..., None].expand(-1, -1, 2)), dth.gather(1, keep), wf.gather(1, keep)
        kw = getattr(cfg, "hough_kernel", 1.5)
        Kx = torch.exp(-0.5 * ((rel[..., 0:1] - gt[:, None]) / (kw * ht)) ** 2)
        Ky = torch.exp(-0.5 * ((rel[..., 1:2] - gt[:, None]) / (kw * ht)) ** 2)
        Kr = torch.exp(-0.5 * ((dth[..., None] - gr[:, None]) / (1.5 * hr)) ** 2)
        dens = torch.einsum("bp,bpr,bpy,bpx->bryx", wf, Kr, Ky, Kx)
        lp = (-0.5 * (gr[:, :, None, None] ** 2 / prior_var[:, 2, None, None, None]
                      + gt[:, None, :, None] ** 2 / prior_var[:, 1, None, None, None]
                      + gt[:, None, None, :] ** 2 / prior_var[:, 0, None, None, None]))
        score = torch.log(dens + 1e-6) + lp
        nw = getattr(cfg, "hough_nms_t", 5)
        peak = score == F.max_pool3d(score[:, None], (3, nw, nw), 1, (1, nw // 2, nw // 2))[:, 0]
        flat = score.masked_fill(~peak, -1e9).flatten(1)
        tops = flat.topk(n_hyp, 1)[1]                                             # b,K
        offs = torch.arange(-1, 2, device=c0.device)
        xs = []
        for kk in range(n_hyp):
            top = tops[:, kk]
            ir, iy, ix = top // (nt * nt), (top // nt) % nt, top % nt
            rr = (ir[:, None] + offs).clamp(0, nr - 1)
            yy = (iy[:, None] + offs).clamp(0, nt - 1)
            xx = (ix[:, None] + offs).clamp(0, nt - 1)
            bi = torch.arange(b, device=c0.device)[:, None, None, None]
            loc = score[bi, rr[:, :, None, None], yy[:, None, :, None], xx[:, None, None, :]]
            p = loc.flatten(1).softmax(-1).view(b, 3, 3, 3)
            er = (p.sum((2, 3)) * gr.gather(1, rr)).sum(1)
            ey = (p.sum((1, 3)) * gt.gather(1, yy)).sum(1)
            ex = (p.sum((1, 2)) * gt.gather(1, xx)).sum(1)
            xs.append(torch.stack([prior_mu[:, 0] + ex, prior_mu[:, 1] + ey, prior_mu[:, 2] + er, prior_mu[:, 3]], 1))
        return torch.stack(xs, 1), score

    # ------------------------------------------------------------ 主过程
    @torch.autocast("cuda", enabled=False)
    def forward(self, A: Dict, B: Dict, c0, prior_mu, prior_var, use_hough=True, n_hyp=3) -> Dict:
        """
        A：当前集合（GNSS 系）{poly (B,M,V,2) 米, emb (B,M,E), mask (B,M) bool, pos_cov (B,M,2,2) 质心不确定度}
        B：历史集合（地图系）{poly, emb, mask, pos_cov, vis (B,N) 当前视频覆盖概率}
        c0 (B,2) 变换参考点；prior_mu/prior_var (B,4) 变换先验（GNSS/IMU 或跟踪模式下的紧先验）。
        """
        cfg = self.cfg
        f32 = lambda S: {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in S.items()}
        A, B = f32(A), f32(B)                                                   # 几何/Sinkhorn 一律 fp32
        for S in (A, B):                                                        # 非有限多边形视为无效
            ok = torch.isfinite(S["poly"]).all(-1).all(-1) & torch.isfinite(S["emb"]).all(-1)
            S["mask"] = S["mask"] & ok
            S["poly"] = torch.nan_to_num(S["poly"], 0.0, 0.0, 0.0)
            S["emb"] = torch.nan_to_num(S["emb"], 0.0, 0.0, 0.0)
        c0, prior_mu, prior_var = c0.float(), prior_mu.float(), prior_var.float()
        enc = self.encode(A, B)
        ea, eb = enc["a"], enc["b"]
        b, m = A["mask"].shape
        n = B["mask"].shape[1]
        fa = F.normalize(self.proj_a(ea["tok"]), dim=-1)
        fb = F.normalize(self.proj_b(eb["tok"]), dim=-1)
        S_feat = 10.0 * fa @ fb.transpose(1, 2)
        vis = B.get("vis", torch.ones(b, n, device=S_feat.device))
        alpha_a = self.dustbin - self.match_a(ea["tok"])[..., 0]
        alpha_b = self.dustbin - self.match_b(eb["tok"])[..., 0] + self.vis_bias * (1 - vis)
        tor_agr = tor_valid = tb = None
        if getattr(cfg, "use_tor", False) or getattr(cfg, "lattice_expand", False):
            with torch.no_grad():                                               # 组合量，无梯度
                tb = lattice_coords(B["poly"], B["mask"], B.get("cov_img"), B.get("cov_bbox"))
                if getattr(cfg, "use_tor", False):
                    ta = lattice_coords(A["poly"], A["mask"], A.get("cov_img"), A.get("cov_bbox"))
                    tor_agr, tor_valid = lattice_agreement(ta, align_period_axes(ta, tb, prior_mu[:, 2]), cfg.tor_sigma)
        if use_hough and cfg.use_hough:
            X0, _ = self.hough_init(ea, eb, A, B, c0, prior_mu, prior_var, S_feat, n_hyp, tor_agr, tor_valid)
        else:
            X0 = prior_mu[:, None]
        if getattr(cfg, "lattice_expand", False) and tb is not None:
            # 格网补全：周期布局的真解常是某个峰沿周期向量平移整数户的位置（诊断中的"提议失败"）
            ext = []
            for kk in range(min(2, X0.shape[1])):
                for p in range(tb["V"].shape[1]):
                    for sgn in (-1.0, 1.0):
                        xe = X0[:, kk].clone()
                        xe[:, :2] = xe[:, :2] + sgn * tb["V"][:, p] * tb["V_valid"][:, p, None].float()
                        ext.append(xe)
            X0 = torch.cat([X0, torch.stack(ext, 1)], 1)
        K = X0.shape[1]
        rep = lambda t: t.repeat_interleave(K, 0)
        ctx = {"cent_a": rep(ea["cent"]), "cent_b": rep(eb["cent"]), "cov_a": rep(ea["cov"]), "cov_b": rep(eb["cov"]),
               "phi_a": rep(ea["phi"]), "phi_b": rep(eb["phi"]), "mask_a": rep(A["mask"]), "mask_b": rep(B["mask"]),
               "pcov_a": rep(A["pos_cov"]), "pcov_b": rep(B["pos_cov"]), "S_feat": rep(S_feat),
               "alpha_a": rep(alpha_a), "alpha_b": rep(alpha_b), "c0": rep(c0), "pm": rep(prior_mu), "pv": rep(prior_var)}
        if tor_agr is not None and getattr(cfg, "tor_estep", False):
            ctx["tor"] = rep(tor_valid * torch.log(tor_agr.clamp(min=1e-3)))
        if tor_agr is not None and getattr(cfg, "tor_select", True):
            ctx["tor_sel"] = rep((1 - tor_valid) + tor_valid * tor_agr)
        if cfg.idf_vote:
            ctx["u_a"] = rep(shape_uniqueness(ea["desc"], ea["area"], A["mask"], cfg.idf_tau) ** getattr(cfg, "idf_pow", 0.5))
        res = self.em(X0.flatten(0, 1), ctx)
        # 多假设择优：软内点数（纯几何，未训练时同样可靠）+ 邻域一致性 + 变换先验对数似然
        lpri = -0.5 * (((res["x"] - ctx["pm"]) ** 2) / ctx["pv"]).sum(1)
        hscore = res["inlier"] + 0.5 * lpri
        best = hscore.view(b, K).argmax(1)
        sel = lambda t: t.view(b, K, *t.shape[1:])[torch.arange(b, device=t.device), best]
        out = {k: sel(v) for k, v in res.items() if torch.is_tensor(v)}
        out["iters"] = [{kk: sel(vv) for kk, vv in it.items()} for it in res["iters"]]
        out["hyp_x"], out["hyp_score"] = X0, hscore.view(b, K)
        out["hyp_x_final"] = res["x"].view(b, K, 4)                             # 各假设 EM 收敛后的变换（用于分析）
        out.update({"S_feat": S_feat, "tok_a": ea["tok"], "tok_b": eb["tok"], "cent_a": ea["cent"],
                    "cent_b": eb["cent"], "cov_a": ea["cov"], "cov_b": eb["cov"], "phi_a": ea["phi"], "phi_b": eb["phi"]})
        nref = getattr(cfg, "hard_refine", 0)
        if nref > 0:
            out["x"], out["delta"] = self.hard_refine(out, A, B, c0, nref)
        return out

    def em(self, x, c) -> Dict:
        """对应–变换 EM 展开（批维已按假设展开）。"""
        cfg = self.cfg
        b, m = c["mask_a"].shape
        n = c["mask_b"].shape[1]
        wpos, wshape, wnc = F.softplus(self.w_raw)
        gamma = F.softplus(self.gamma_raw)
        if not cfg.use_nc:
            wnc = wnc * 0
        if not cfg.use_shape:
            wshape, gamma = wshape * 0, gamma * 0
        ea = {"cent": c["cent_a"], "phi": c["phi_a"]}
        eb = {"cent": c["cent_b"], "phi": c["phi_b"]}
        nA, okA = knn_idx(c["cent_a"], c["mask_a"], cfg.knn)
        nB, okB = knn_idx(c["cent_b"], c["mask_b"], cfg.knn)
        sd2 = cfg.sigma_delta ** 2
        eye = torch.eye(2, device=x.device)
        P, iters = None, []
        for it in range(cfg.em_iters):
            ca = apply_sim(x, c["cent_a"], c["c0"])
            Sig = apply_sim_cov(x, c["pcov_a"])[:, :, None] + c["pcov_b"][:, None] + sd2 * eye    # 残差 δ 边缘化
            r = c["cent_b"][:, None] - ca[:, :, None]
            Sinv = torch.linalg.inv_ex(Sig)[0]
            maha = torch.einsum("bmni,bmnij,bmnj->bmn", r, Sinv, r)
            ldet = torch.log((Sig[..., 0, 0] * Sig[..., 1, 1] - Sig[..., 0, 1] ** 2).clamp(min=1e-6))
            S_pos = (-0.5 * maha - 0.5 * ldet).clamp(min=-50)
            S_shape = log_bhattacharyya(ca[:, :, None], apply_sim_cov(x, c["cov_a"])[:, :, None],
                                        c["cent_b"][:, None], c["cov_b"][:, None]).clamp(min=-20)
            if not cfg.use_nc:                                                 # 关闭时直接跳过（省显存）
                S_nc = torch.zeros_like(S_pos)
            elif P is None:
                P0 = (c["S_feat"] + wpos * S_pos).masked_fill(~c["mask_b"][:, None], -1e4).softmax(-1)
                S_nc = self.consensus(P0, ea, eb, nA, nB, okA, okB, x)
            else:
                S_nc = self.consensus(P, ea, eb, nA, nB, okA, okB, x)
            S = c["S_feat"] + wpos * S_pos + wshape * S_shape + wnc * 10.0 * S_nc
            if "tor" in c:
                S = S + F.softplus(self.w_tor_raw) * c["tor"]
            logP = log_sinkhorn_dustbin(S, c["alpha_a"], c["alpha_b"], c["mask_a"], c["mask_b"], cfg.sinkhorn_iters)
            P = logP[:, :m, :n].exp()
            # —— 变化感知 M 步：形状变化大的配对降权，避免改扩建拉偏对齐 ——
            wp = P * torch.exp(gamma * S_shape.detach().clamp(min=-5))
            wi = wp.sum(-1)
            tgt = (wp[..., None] * c["cent_b"][:, None]).sum(2) / wi[..., None].clamp(min=1e-6)
            x, var, _ = weighted_umeyama(c["cent_a"], tgt, wi * c["mask_a"], c["c0"], c["pm"], c["pv"])
            iters.append({"logP": logP, "x": x, "var": var})
        # —— 逐栋残差 MAP（收缩估计；形状变化大的配对降权，使改扩建的几何差异留给变化解码器）——
        ca = apply_sim(x, c["cent_a"], c["c0"])
        wp = P * torch.exp(gamma * S_shape.detach().clamp(min=-5))
        wi = wp.sum(-1)
        tgt = (wp[..., None] * c["cent_b"][:, None]).sum(2) / wi[..., None].clamp(min=1e-6)
        s_i = 0.5 * apply_sim_cov(x, c["pcov_a"]).diagonal(dim1=-2, dim2=-1).sum(-1)
        delta = (tgt - ca) * ((sd2 / (sd2 + s_i)) * wi.clamp(max=1))[..., None]
        inl = P * torch.exp(-0.5 * maha) * (1 + S_nc)
        if "tor_sel" in c:                                                      # 择优：序一致的内点才算数
            inl = inl * c["tor_sel"]
        if "u_a" in c:                                                          # 择优时同样按独特性加权
            inl = inl * (c["u_a"] / c["u_a"].amax(1, keepdim=True).clamp(min=1e-6))[..., None]
        inlier = inl.sum((1, 2))
        return {"logP": logP, "P": P, "S": S, "x": x, "var": var, "delta": delta, "S_shape": S_shape,
                "S_nc": S_nc, "iters": iters, "inlier": inlier}

    def hard_refine(self, out, A, B, c0, n_iter, gate=3.0):
        """软到硬精化：取互为最大的硬匹配（P>0.2），迭代地用残差门限内的配对做无先验加权 Umeyama。
        软分配在密集布局中会把质量分给相邻建筑，使加权目标被"拉向邻居"；硬匹配可消除这种偏差。"""
        m_a, _ = extract_matches(out["logP"], A["mask"], B["mask"])
        ok = m_a >= 0
        ca, cb = out["cent_a"], out["cent_b"]
        tgt = cb.gather(1, m_a.clamp(min=0)[..., None].expand(-1, -1, 2))
        x = out["x"]
        for _ in range(n_iter):
            r = (tgt - apply_sim(x, ca, c0)).norm(dim=-1)
            w = (ok & (r < gate)).float()
            xn, _, _ = weighted_umeyama(ca, tgt, w, c0)
            enough = (w.sum(1) >= 3)[:, None]
            x = torch.where(enough, xn, x)
        sd2 = self.cfg.sigma_delta ** 2
        res = (tgt - apply_sim(x, ca, c0)) * ok[..., None]
        delta = res * (sd2 / (sd2 + 1.0))
        return x, delta

    def consensus(self, P, ea, eb, nA, nB, okA, okB, x):
        """二阶邻域一致性：S_nc[i,j] = (1/k) Σ_{i'∈N(i)} max_{j'∈N(j)} P[i',j']·κ(sR·d_ii' , d_jj')·ω(朝向)。"""
        cfg = self.cfg
        b, m, k = nA.shape
        n = nB.shape[1]
        kb = nB.shape[2]
        dA = ea["cent"].gather(1, nA.view(b, -1, 1).expand(-1, -1, 2)).view(b, m, k, 2) - ea["cent"][:, :, None]
        dA = torch.exp(x[:, 3])[:, None, None, None] * torch.einsum("bij,bmkj->bmki", rot2(x[:, 2]), dA)
        dB = eb["cent"].gather(1, nB.view(b, -1, 1).expand(-1, -1, 2)).view(b, n, kb, 2) - eb["cent"][:, :, None]
        diff = dA[:, :, None, :, None] - dB[:, None, :, None, :]
        kap = torch.exp(-0.5 * (diff ** 2).sum(-1) / cfg.sigma_rel ** 2)                     # b,m,n,k,kb
        pa = ea["phi"].gather(1, nA.view(b, -1)).view(b, m, k)
        pb = eb["phi"].gather(1, nB.view(b, -1)).view(b, n, kb)
        om = 0.5 * (1 + torch.cos(4 * (pb[:, None, :, None, :] - pa[:, :, None, :, None] - x[:, 2, None, None, None, None])))
        bi = torch.arange(b, device=P.device)[:, None, None, None, None]
        Pn = P[bi, nA[:, :, None, :, None], nB[:, None, :, None, :]]
        valid = (okA[:, :, None, :, None] & okB[:, None, :, None, :]).float()
        return (Pn * kap * om * valid).amax(-1).sum(-1) / k


def extract_matches(logP, mask_a, mask_b, thr=0.2):
    """互为最大 + 阈值 -> 严格一对一；返回 m_a (B,M) 历史索引或 -1，m_b (B,N) 当前索引或 -1。"""
    P = logP[:, :-1, :-1].exp()
    P = P * mask_a[..., None] * mask_b[:, None]
    ma_val, ma = P.max(2)
    mb_val, mb = P.max(1)
    b = P.shape[0]
    mutual_a = mb.gather(1, ma) == torch.arange(P.shape[1], device=P.device)[None]
    keep_a = mutual_a & (ma_val > thr) & mask_a
    m_a = torch.where(keep_a, ma, torch.full_like(ma, -1))
    m_b = torch.full_like(mb, -1)
    for bi in range(b):
        ia = keep_a[bi].nonzero()[:, 0]
        m_b[bi, ma[bi, ia]] = ia
    return m_a, m_b
