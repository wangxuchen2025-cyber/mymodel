"""TOR：拓扑序关系（Topological Ordinal Relations）。

动机：局部度量关系（近邻距离/方位/朝向差）在一维周期布局中对"整体平移一户"不变，因而无法消除周期性歧义；
能打破平移对称性的信息只存在于结构边界（排端、街坊转角）。TOR 把建筑组织成"排"（链），给每栋楼一个
离散的拓扑序坐标（本排第 k 户、共 n 户），它对平移/旋转/尺度不变、对米级噪声不敏感，而整体错位一户时
序号必然全部失配。序号只有在排端被真实观测到时才可信，因此结合覆盖范围（相机足迹 / 历史数据范围）判定
每条链两端的可靠性。

排的判定（对每栋楼 i 的两条主轴 k∈{长轴, 短轴} 分别进行）：
  朝向相容（模 90° 差 < ang_tol）、沿该轴的投影间隙 gap < g_max、横向错位 < lat_tol·短边，
  在该轴方向上的邻接图取连通分量即为一条链；链内按投影排序得到序号。
"""
import math
from typing import Dict

import torch
import torch.nn.functional as F

from .geometry import poly_moments, sym2_eig


def _components(adj, mask, iters=64):
    """邻接矩阵 (B,N,N) bool -> 连通分量标签 (B,N)（最小下标传播）。"""
    b, n, _ = adj.shape
    lab = torch.arange(n, device=adj.device).expand(b, n).clone()
    lab = torch.where(mask, lab, torch.full_like(lab, n + 1))
    big = torch.full_like(adj, n + 1, dtype=lab.dtype)
    for _ in range(iters):
        nb = torch.where(adj, lab[:, None, :].expand(b, n, n), big).amin(-1)
        new = torch.minimum(lab, nb)
        if torch.equal(new, lab):
            break
        lab = new
    return lab


def sample_cov(cov_img, bbox, pts):
    """覆盖栅格 (B,1,H,W)、bbox (B,4)=[xmin,ymin,xmax,ymax]（y 向上）-> pts (B,...,2) 处的覆盖值。"""
    b = pts.shape[0]
    shp = pts.shape
    p = pts.reshape(b, -1, 2)
    gx = (p[..., 0] - bbox[:, None, 0]) / (bbox[:, None, 2] - bbox[:, None, 0]).clamp(min=1e-3) * 2 - 1
    gy = (bbox[:, None, 3] - p[..., 1]) / (bbox[:, None, 3] - bbox[:, None, 1]).clamp(min=1e-3) * 2 - 1
    v = F.grid_sample(cov_img, torch.stack([gx, gy], -1)[:, :, None], align_corners=False, padding_mode="zeros")
    return v[:, 0, :, 0].reshape(shp[:-1])


def chain_coords(poly, mask, cov_img=None, bbox=None, gap_max=3.0, ang_tol=math.radians(12), lat_tol=0.35) -> Dict:
    """返回每栋楼在两条主轴方向上的拓扑序坐标：
    k (B,N,2) 从链"起点"数起的序号，n (B,N,2) 链长，rel_s / rel_e (B,N,2) 起/终端是否可靠（真实结构端点），
    以及链标签 lab (B,N,2)。轴 0 = 长轴，轴 1 = 短轴；方向符号做了规范化（x 分量为正）。"""
    area, cent, cov = poly_moments(poly)
    l1, l2, phi = sym2_eig(cov)
    Lmaj, Lmin = (12 * l2.clamp(min=1e-3)).sqrt(), (12 * l1.clamp(min=1e-3)).sqrt()
    a0 = torch.stack([phi.cos(), phi.sin()], -1)
    flip = (a0[..., 0] < -1e-6) | ((a0[..., 0].abs() <= 1e-6) & (a0[..., 1] < 0))
    a0 = torch.where(flip[..., None], -a0, a0)
    a1 = torch.stack([-a0[..., 1], a0[..., 0]], -1)
    a1 = torch.where((a1[..., 0] < -1e-6)[..., None], -a1, a1)
    d = cent[:, None, :, :] - cent[:, :, None, :]                               # d[i,j] = c_j - c_i
    dphi = (phi[:, None, :] - phi[:, :, None]).remainder(math.pi / 2)
    dphi = torch.minimum(dphi, math.pi / 2 - dphi)
    pair_ok = mask[:, :, None] & mask[:, None, :] & (dphi < ang_tol)
    eye = torch.eye(mask.shape[1], dtype=torch.bool, device=mask.device)[None]
    ks, ns, rss, res, labs = [], [], [], [], []
    for ax, (A, Lal, Lpe) in enumerate(((a0, Lmaj, Lmin), (a1, Lmin, Lmaj))):
        along = (d * A[:, :, None]).sum(-1)
        perp = torch.stack([-A[..., 1], A[..., 0]], -1)
        across = (d * perp[:, :, None]).sum(-1)
        # j 沿 i 的该轴方向的半长：按 j 自身朝向投影近似
        ej = 0.5 * ((Lmaj[:, None, :] * (a0[:, None, :] * A[:, :, None]).sum(-1).abs())
                    + (Lmin[:, None, :] * (a1[:, None, :] * A[:, :, None]).sum(-1).abs()))
        gap = along.abs() - 0.5 * Lal[:, :, None] - ej
        lat = lat_tol * torch.minimum(Lpe[:, :, None], Lpe[:, None, :]) + 1.0
        adj = pair_ok & (gap < gap_max) & (gap > -0.5 * torch.minimum(Lal[:, :, None], ej)) & (across.abs() < lat) & ~eye
        adj = adj & adj.transpose(1, 2)
        lab = _components(adj, mask)
        same = (lab[:, :, None] == lab[:, None, :]) & mask[:, :, None] & mask[:, None, :]
        s = (cent * A).sum(-1)                                                  # 沿自身轴的投影
        # 链内用链首成员的轴作为统一方向更稳；此处以每栋楼自身轴近似（同链朝向相容）
        proj = (cent[:, None, :, :] * A[:, :, None, :]).sum(-1)                 # proj[i,j] = c_j · a_i
        before = same & (proj < proj.diagonal(dim1=1, dim2=2)[..., None] - 1e-3)
        k = before.sum(-1)
        n = same.sum(-1)
        # 端点可靠性：在链端外侧一个"楼宽+间隙"处是否仍在覆盖范围内
        if cov_img is not None:
            is_start = k == 0
            is_end = k == n - 1
            step = (0.5 * Lal + gap_max + 0.5 * Lal)[..., None]
            p_s = cent - A * step
            p_e = cent + A * step
            cs, ce = sample_cov(cov_img, bbox, p_s), sample_cov(cov_img, bbox, p_e)
            # 链的起/终端可靠性 = 端点楼外侧是否被观测到；广播到链内所有成员
            big = torch.zeros_like(cs)
            rel_start_member = torch.where(is_start, (cs > 0.5).float(), big)
            rel_end_member = torch.where(is_end, (ce > 0.5).float(), big)
            rs = (same.float() * rel_start_member[:, None, :]).amax(-1)
            re = (same.float() * rel_end_member[:, None, :]).amax(-1)
        else:
            rs = torch.ones_like(k, dtype=torch.float)
            re = torch.ones_like(k, dtype=torch.float)
        ks.append(k), ns.append(n), rss.append(rs), res.append(re), labs.append(lab)
    st = lambda xs: torch.stack(xs, -1)
    return {"k": st(ks).float(), "n": st(ns).float(), "rel_s": st(rss), "rel_e": st(res), "lab": st(labs)}


def tor_agreement(ta: Dict, tb: Dict, sigma=0.35):
    """拓扑序一致性 (B,M,N) ∈[0,1] 与有效性权重 (B,M,N)。
    对每条轴：若两侧起端都可靠，比较"距起端序号"；若两侧终端都可靠，比较"距终端序号"；
    轴方向符号可能因旋转/规范化翻转，故同时考虑"起↔终"的交叉配对并取较优者。
    两条轴（长/短）按"链长>1 才有信息"加权平均；无可靠信息时有效性为 0（中性，不惩罚）。"""
    ka, na, rsa, rea = ta["k"][:, :, None], ta["n"][:, :, None], ta["rel_s"][:, :, None], ta["rel_e"][:, :, None]
    kb, nb, rsb, reb = tb["k"][:, None], tb["n"][:, None], tb["rel_s"][:, None], tb["rel_e"][:, None]
    ea, eb = na - 1 - ka, nb - 1 - kb
    g = lambda x: torch.exp(-0.5 * (x / sigma) ** 2)

    def score(ka_s, ka_e, ra_s, ra_e, kb_s, kb_e, rb_s, rb_e):
        w1, w2 = ra_s * rb_s, ra_e * rb_e
        num = w1 * g(ka_s - kb_s) + w2 * g(ka_e - kb_e)
        den = w1 + w2
        return num, den
    n1, d1 = score(ka, ea, rsa, rea, kb, eb, rsb, reb)                          # 同向
    n2, d2 = score(ka, ea, rsa, rea, eb, kb, reb, rsb)                          # 反向（轴符号翻转）
    agr1 = n1 / d1.clamp(min=1e-6)
    agr2 = n2 / d2.clamp(min=1e-6)
    use2 = (agr2 > agr1) & (d2 > 0)
    num = torch.where(use2, n2, n1)
    den = torch.where(use2, d2, d1)
    info = ((na > 1) & (nb > 1)).float()                                        # 单栋"链"无序信息
    den = den * info
    num = num * info
    agr = num.sum(-1) / den.sum(-1).clamp(min=1e-6)                             # 两条轴合并
    valid = (den.sum(-1) > 0).float()
    return agr, valid


# ---------------------------------------------------------------------------------------------
# 推广：格网（周期）序坐标。"相邻成排"只覆盖连排住宅；独栋郊区、板楼小区等是"等间距重复"，
# 而周期歧义恰恰来自主导周期向量 v 的整数倍平移。因此直接从布局中估计 v，沿 v 建立周期链与序号。
# ---------------------------------------------------------------------------------------------
def _shape_sim_mask(area, phi, mask, la_tol=0.25, ang_tol=math.radians(12)):
    la = torch.log(area.clamp(min=1.0))
    dphi = (phi[:, None, :] - phi[:, :, None]).remainder(math.pi / 2)
    dphi = torch.minimum(dphi, math.pi / 2 - dphi)
    return (mask[:, :, None] & mask[:, None, :] & ((la[:, None, :] - la[:, :, None]).abs() < la_tol) & (dphi < ang_tol))


def period_vectors(cent, area, phi, mask, n_periods=2, k=6, dmin=3.0, dmax=60.0, rel_tol=0.12, min_support=4):
    """从"形状相似近邻之间的位移"中投票出主导周期向量（规范符号：x>0 或 x≈0 且 y>0）。
    返回 V (B,P,2)、有效位 (B,P)、支持数 (B,P)。"""
    b, n, _ = cent.shape
    sim = _shape_sim_mask(area, phi, mask)
    d = torch.cdist(cent, cent).masked_fill(~sim, 1e9)
    d = d + torch.eye(n, device=cent.device) * 1e9
    kk = min(k, max(n - 1, 1))
    val, nb = d.topk(kk, largest=False)
    D = cent.gather(1, nb.view(b, -1, 1).expand(-1, -1, 2)).view(b, n, kk, 2) - cent[:, :, None]
    ok = (val < 1e8) & (val > dmin) & (val < dmax)
    flip = (D[..., 0] < -1e-6) | ((D[..., 0].abs() <= 1e-6) & (D[..., 1] < 0))
    D = torch.where(flip[..., None], -D, D).reshape(b, -1, 2)
    ok = ok.reshape(b, -1)
    V = torch.zeros(b, n_periods, 2, device=cent.device)
    valid = torch.zeros(b, n_periods, dtype=torch.bool, device=cent.device)
    sup = torch.zeros(b, n_periods, device=cent.device)
    for bi in range(b):
        Db = D[bi][ok[bi]]
        if len(Db) < min_support:
            continue
        tol = rel_tol * Db.norm(dim=-1)[:, None] + 0.8
        close = torch.cdist(Db, Db) < tol
        support = close.sum(-1).float()
        taken = torch.zeros(len(Db), dtype=torch.bool, device=cent.device)
        for p in range(n_periods):
            s = support.masked_fill(taken, -1)
            j = int(s.argmax())
            if s[j] < min_support:
                break
            v = Db[close[j]].mean(0)
            V[bi, p], valid[bi, p], sup[bi, p] = v, True, s[j]
            # 去掉与已选向量共线（含整数倍）的候选，保证第二个周期方向不同
            u = v / v.norm().clamp(min=1e-6)
            cosang = (Db @ u).abs() / Db.norm(dim=-1).clamp(min=1e-6)
            taken = taken | (cosang > 0.9)
    return V, valid, sup


def lattice_coords(poly, mask, cov_img=None, bbox=None, n_periods=2, tol_rel=0.15) -> Dict:
    """沿每个主导周期向量 v 建立周期链：i→j 当 |c_j - (c_i + v)| < tol 且形状相似；链内按沿 v 的投影排序得序号。
    端点可靠性：链端外侧下一个周期位置（c_end ± v）在覆盖范围内 => 该处"确实没有同型建筑"，是真实结构端点。"""
    area, cent, cov = poly_moments(poly)
    _, _, phi = sym2_eig(cov)
    V, vvalid, sup = period_vectors(cent, area, phi, mask, n_periods)
    sim = _shape_sim_mask(area, phi, mask, la_tol=0.6)                        # 连链时放宽形状容差（改扩建不断链）
    b, n, _ = cent.shape
    ks, ns, rss, res, labs = [], [], [], [], []
    for p in range(n_periods):
        v = V[:, p]                                                              # b,2
        vn = v.norm(dim=-1).clamp(min=1e-3)
        u = v / vn[:, None]
        up = torch.stack([-u[:, 1], u[:, 0]], -1)
        d = cent[:, None, :, :] - cent[:, :, None, :]                            # d[i,j] = c_j - c_i
        along = (d * u[:, None, None]).sum(-1)
        lat = (d * up[:, None, None]).sum(-1).abs()
        # 对噪声稳健的后继：沿 v 投影在 [0.5, 1.5]·|v|、横向偏移 < 0.35|v|+1.5 m 的同型建筑中，取沿向最近者
        cand = sim & (along > 0.5 * vn[:, None, None]) & (along < 2.5 * vn[:, None, None]) & \
            (lat < 0.35 * vn[:, None, None] + 1.5) & vvalid[:, p, None, None]
        key = torch.where(cand, along + 0.5 * lat, torch.full_like(along, 1e9))
        best = key.argmin(-1)
        nxt = torch.zeros_like(cand)
        nxt.scatter_(-1, best[..., None], cand.gather(-1, best[..., None]))
        # 去掉"多对一"：每个 j 只保留最近的前驱
        keyT = torch.where(nxt, key, torch.full_like(key, 1e9))
        bestp = keyT.argmin(1)
        keep = torch.zeros_like(nxt)
        keep.scatter_(1, bestp[:, None, :], nxt.gather(1, bestp[:, None, :]))
        nxt = nxt & keep
        adj = nxt | nxt.transpose(1, 2)
        lab = _components(adj, mask)
        same = (lab[:, :, None] == lab[:, None, :]) & mask[:, :, None] & mask[:, None, :]
        s = (cent * u[:, None]).sum(-1)
        # 几何序号：以周期长度为单位的沿向距离（跨过缺失/拆除的空位时序号不错位）
        big = torch.full_like(s, 1e9)
        s_min = torch.where(same, s[:, None, :].expand_as(same), big[:, None, :].expand_as(same)).amin(-1)
        s_max = torch.where(same, s[:, None, :].expand_as(same), -big[:, None, :].expand_as(same)).amax(-1)
        k = ((s - s_min) / vn[:, None]).round()
        nn_ = ((s_max - s_min) / vn[:, None]).round() + 1
        before = k
        has_next = nxt.any(-1)
        has_prev = nxt.any(1)
        if cov_img is not None:
            cs = sample_cov(cov_img, bbox, cent - v[:, None])
            ce = sample_cov(cov_img, bbox, cent + v[:, None])
            rel_start_member = ((~has_prev) & (cs > 0.5)).float()
            rel_end_member = ((~has_next) & (ce > 0.5)).float()
        else:
            rel_start_member = (~has_prev).float()
            rel_end_member = (~has_next).float()
        # 链的端点 = 链内序号最小/最大者；其可靠性广播到链内所有成员
        is_s, is_e = k == 0, k == nn_ - 1
        rs = (same.float() * (rel_start_member * is_s)[:, None, :]).amax(-1)
        re = (same.float() * (rel_end_member * is_e)[:, None, :]).amax(-1)
        valid_p = vvalid[:, p, None].float()
        ks.append(k.float()), ns.append(nn_.float() * valid_p + (1 - valid_p))  # noqa
        rss.append(rs * valid_p), res.append(re * valid_p), labs.append(lab)
    st = lambda xs: torch.stack(xs, -1)
    return {"k": st(ks), "n": st(ns), "rel_s": st(rss), "rel_e": st(res), "lab": st(labs), "V": V, "V_valid": vvalid}


def align_period_axes(ta: Dict, tb: Dict, theta=None, cos_min=0.95, len_tol=0.18) -> Dict:
    """把 B 的周期轴按方向/长度配对到 A 的周期轴（考虑旋转 θ 与符号），返回重排后的 tb（符号相反的轴交换起/终端）。"""
    Va, Vb = ta["V"], tb["V"]
    if theta is not None:
        c, s = theta.cos()[:, None], theta.sin()[:, None]
        Va = torch.stack([c * Va[..., 0] - s * Va[..., 1], s * Va[..., 0] + c * Va[..., 1]], -1)
    na_, nb_ = Va.norm(dim=-1).clamp(min=1e-6), Vb.norm(dim=-1).clamp(min=1e-6)
    cosm = (Va[:, :, None] * Vb[:, None]).sum(-1) / (na_[:, :, None] * nb_[:, None])
    lr = (na_[:, :, None] / nb_[:, None]).log().abs()
    good = (cosm.abs() > cos_min) & (lr < len_tol) & ta["V_valid"][:, :, None] & tb["V_valid"][:, None]
    score = cosm.abs() * good
    best = score.argmax(-1)                                                       # b,P  A 轴 -> B 轴
    ok = good.gather(-1, best[..., None])[..., 0]
    sign = cosm.gather(-1, best[..., None])[..., 0] < 0
    out = {}
    for key in ("k", "n", "rel_s", "rel_e"):
        out[key] = tb[key].gather(-1, best[:, None, :].expand(-1, tb[key].shape[1], -1))
    n = out["n"]
    k_rev = n - 1 - out["k"]
    sg = sign[:, None, :]
    out["k"] = torch.where(sg, k_rev, out["k"])
    out["rel_s"], out["rel_e"] = torch.where(sg, out["rel_e"], out["rel_s"]), torch.where(sg, out["rel_s"], out["rel_e"])
    okm = ok[:, None, :].float()
    out["rel_s"], out["rel_e"] = out["rel_s"] * okm, out["rel_e"] * okm
    return out


def lattice_agreement(ta: Dict, tb_aligned: Dict, sigma=0.35):
    """格网序一致性 (B,M,N)∈[0,1] 与有效位。逐周期轴：两侧起端都可靠 -> 比较"距起端序号"；两侧终端都可靠 ->
    比较"距终端序号"；两者都有时取较优者（变化会把一排打断或延长，但至少一端的序号仍正确；
    而整体错位一户时两端同时失配）。各轴取平均（仅计有信息的轴）。"""
    ka, na, rsa, rea = ta["k"][:, :, None], ta["n"][:, :, None], ta["rel_s"][:, :, None], ta["rel_e"][:, :, None]
    kb, nb, rsb, reb = tb_aligned["k"][:, None], tb_aligned["n"][:, None], tb_aligned["rel_s"][:, None], tb_aligned["rel_e"][:, None]
    g = lambda x: torch.exp(-0.5 * (x / sigma) ** 2)
    w1, w2 = rsa * rsb, rea * reb
    gs, ge = g(ka - kb), g((na - 1 - ka) - (nb - 1 - kb))
    both = (w1 > 0.5) & (w2 > 0.5)
    ax_agr = torch.where(both, torch.maximum(gs, ge), torch.where(w1 > 0.5, gs, ge))
    has = ((w1 > 0.5) | (w2 > 0.5)) & ((na > 1) | (nb > 1))
    hf = has.float()
    agr = (ax_agr * hf).sum(-1) / hf.sum(-1).clamp(min=1e-6)
    return agr, (hf.sum(-1) > 0).float()
