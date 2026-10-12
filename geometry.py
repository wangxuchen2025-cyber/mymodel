"""GRAIL-CD 的二维几何：地平面单应投影、透视径向位移（屋顶↔基底）、多边形矩、
加权 Umeyama 相似变换（含先验融合）、带垃圾箱的对数域 Sinkhorn、可微软栅格化。

关键关系（透视径向位移律，倾斜视角下的核心空间约束）：
相机中心 C=(N, H)，屋顶点 (F, h) 经地平面单应投到地面得
    R' = N + H/(H-h) · (F - N)       （屋顶沿"天底点→基底"射线向外偏移）
=>  F  = N + (1-η)(R' - N),  η = h/H  （η 是图像内在的射影量，每栋建筑 1 个自由度）
多帧：H_t·F + (R'_t - N_t)·h = H_t·R'_t  对 (F, h) 线性 -> 闭式二维多视角三角化。
"""
import math

import torch
import torch.nn.functional as F

from geomac.geometry import intrinsics, rot_world_from_cam  # noqa: F401  (复用第一版相机模型)


# ---------------------------------------------------------------- 相机与地平面单应
def camera_from_meta(meta, img_hw):
    """meta (B,8)=[e,n,u,yaw,pitch,roll,f/W,valid] -> K, R_wc, C。"""
    K = intrinsics(meta[:, 6], img_hw)
    R = rot_world_from_cam(meta[:, 3], meta[:, 4], meta[:, 5])
    return K, R, meta[:, :3]


def img_to_ground(uv, K, R, C, ground_z=0.0):
    """uv (B,...,2) 像素 -> 地面 z=ground_z 上的地图坐标 (B,...,2)，以及是否与地面相交。"""
    shp = uv.shape
    p = torch.cat([uv.reshape(shp[0], -1, 2), torch.ones_like(uv.reshape(shp[0], -1, 2)[..., :1])], -1)
    d = torch.einsum("bij,bnj->bni", R @ torch.linalg.inv(K), p)
    dz = d[..., 2]
    ok = dz < -1e-4
    t = (ground_z - C[:, None, 2]) / torch.where(ok, dz, -torch.ones_like(dz))
    xy = C[:, None, :2] + t[..., None] * d[..., :2]
    return xy.reshape(shp), ok.reshape(shp[:-1])


def ground_to_img(xy, K, R, C, z=0.0):
    """地图坐标（地面高程 z） -> 像素。"""
    shp = xy.shape
    X = torch.cat([xy.reshape(shp[0], -1, 2), torch.full_like(xy.reshape(shp[0], -1, 2)[..., :1], z)], -1)
    pc = torch.einsum("bji,bnj->bni", R, X - C[:, None])                      # R^T (X - C)
    uvw = torch.einsum("bij,bnj->bni", K, pc)
    uv = uvw[..., :2] / uvw[..., 2:].clamp(min=1e-3)
    return uv.reshape(shp), (pc[..., 2] > 0.1).reshape(shp[:-1])


def relief_footprint(roof_xy, nadir, eta):
    """屋顶的地面投影 R' -> 基底 F = N + (1-η)(R'-N)。roof_xy (B,Q,V,2), nadir (B,2), eta (B,Q)。"""
    n = nadir[:, None, None]
    return n + (1 - eta)[..., None, None] * (roof_xy - n)


def relief_roof(foot_xy, nadir, eta):
    n = nadir[:, None, None]
    return n + (foot_xy - n) / (1 - eta).clamp(min=0.05)[..., None, None]


# ---------------------------------------------------------------- 多边形
def poly_moments(p, eps=1e-6):
    """p (...,V,2) 简单多边形 -> 面积(>0)、质心 (...,2)、二阶中心矩协方差 (...,2,2)。退化时回退到顶点统计。"""
    o = p.mean(-2, keepdim=True)
    q = p - o
    x, y = q[..., 0], q[..., 1]
    xn, yn = torch.roll(x, -1, -1), torch.roll(y, -1, -1)
    c = x * yn - xn * y
    A = 0.5 * c.sum(-1)
    sgn = torch.where(A >= 0, torch.ones_like(A), -torch.ones_like(A))
    A_ = A.abs().clamp(min=eps)
    cx = ((x + xn) * c).sum(-1) / (6 * A * sgn).abs().clamp(min=eps) * sgn
    cy = ((y + yn) * c).sum(-1) / (6 * A * sgn).abs().clamp(min=eps) * sgn
    ixx = ((x * x + x * xn + xn * xn) * c).sum(-1) / 12 * sgn               # ∫x² dA
    iyy = ((y * y + y * yn + yn * yn) * c).sum(-1) / 12 * sgn
    ixy = ((x * yn + 2 * x * y + 2 * xn * yn + xn * y) * c).sum(-1) / 24 * sgn
    cxx = ixx / A_ - cx ** 2
    cyy = iyy / A_ - cy ** 2
    cxy = ixy / A_ - cx * cy
    deg = A.abs() < 1e-3
    vx, vy = x.var(-1, unbiased=False), y.var(-1, unbiased=False)
    vxy = ((x - x.mean(-1, keepdim=True)) * (y - y.mean(-1, keepdim=True))).mean(-1)
    cxx, cyy, cxy = torch.where(deg, vx, cxx), torch.where(deg, vy, cyy), torch.where(deg, vxy, cxy)
    cent = torch.stack([torch.where(deg, x.mean(-1), cx), torch.where(deg, y.mean(-1), cy)], -1) + o[..., 0, :]
    cov = torch.stack([torch.stack([cxx, cxy], -1), torch.stack([cxy, cyy], -1)], -2)
    cov = cov + eps * torch.eye(2, device=p.device, dtype=p.dtype)
    return A.abs(), cent, cov


def sym2_eig(cov):
    """2x2 对称矩阵的闭式特征分解：返回 (λ_min, λ_max) 与主方向角（无迭代，不会"不收敛"）。"""
    a, b, c = cov[..., 0, 0], cov[..., 0, 1], cov[..., 1, 1]
    m = 0.5 * (a + c)
    r = torch.sqrt((0.5 * (a - c)) ** 2 + b ** 2 + 1e-12)
    return m - r, m + r, 0.5 * torch.atan2(2 * b, a - c)


def sym2_clamp(cov, lo, hi):
    """把 2x2 对称矩阵投影到特征值 ∈ [lo, hi] 的正定矩阵。"""
    l1, l2, th = sym2_eig(cov)
    R = rot2(th)
    D = torch.diag_embed(torch.stack([l2.clamp(lo, hi), l1.clamp(lo, hi)], -1))
    return R @ D @ R.transpose(-1, -2)


def principal_angle(cov):
    """主轴方向 φ ∈ (-π/2, π/2]（模 π）。"""
    return 0.5 * torch.atan2(2 * cov[..., 0, 1], cov[..., 0, 0] - cov[..., 1, 1])


def wrap(a, period=2 * math.pi):
    return torch.remainder(a + period / 2, period) - period / 2


def log_bhattacharyya(m1, S1, m2, S2, eps=1e-4):
    """两个二维高斯（多边形矩近似）的 log Bhattacharyya 系数 ∈ (-∞, 0]，作为可微的形状重叠度。
    用带正则的 2x2 闭式逆，并按理论界截断到 ≤ 0（避免退化多边形导致的数值异常）。"""
    I = torch.eye(2, device=S1.device, dtype=S1.dtype)
    S1, S2 = S1 + eps * I, S2 + eps * I
    S = 0.5 * (S1 + S2)
    det = lambda M: (M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]).clamp(min=1e-8)
    dS = det(S)
    d = m1 - m2
    q = (S[..., 1, 1] * d[..., 0] ** 2 - 2 * S[..., 0, 1] * d[..., 0] * d[..., 1] + S[..., 0, 0] * d[..., 1] ** 2) / dS
    t1 = 0.125 * q.clamp(min=0)
    t2 = 0.5 * (torch.log(dS) - 0.5 * (torch.log(det(S1)) + torch.log(det(S2))))
    return -(t1 + t2.clamp(min=0))


def cyclic_align(src, ref):
    """把 src (...,V,2) 的顶点顺序循环平移/反向到与 ref 最一致。"""
    V = src.shape[-2]
    cands = torch.stack([torch.roll(src, k, -2) for k in range(V)] +
                        [torch.roll(src.flip(-2), k, -2) for k in range(V)], -3)       # ...,2V,V,2
    d = (cands - ref[..., None, :, :]).norm(dim=-1).mean(-1)
    idx = d.argmin(-1)
    return cands.gather(-3, idx[..., None, None, None].expand(*idx.shape, 1, V, 2)).squeeze(-3)


# ---------------------------------------------------------------- 相似变换
def rot2(theta):
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)


def apply_sim(x, p, c0):
    """x=[tx,ty,θ,log s] (B,4)，绕参考点 c0 (B,2)：q = c0 + t + s R (p - c0)。p (B,...,2)。"""
    b = p.shape[0]
    shp = p.shape
    pf = p.reshape(b, -1, 2) - c0[:, None]
    q = torch.exp(x[:, 3])[:, None, None] * torch.einsum("bij,bnj->bni", rot2(x[:, 2]), pf)
    return (q + c0[:, None] + x[:, None, :2]).reshape(shp)


def apply_sim_cov(x, cov):
    R = rot2(x[:, 2]) * torch.exp(x[:, 3])[:, None, None]
    shp = cov.shape
    c = cov.reshape(shp[0], -1, 2, 2)
    return (R[:, None] @ c @ R[:, None].transpose(-1, -2)).reshape(shp)


def weighted_umeyama(src, dst, w, c0, prior_mu=None, prior_var=None, eps=1e-6):
    """加权二维相似变换闭式解（绕 c0 参数化），并与高斯先验做参数空间的精度加权融合（MAP 近似）。

    src/dst (B,N,2)，w (B,N)≥0。返回 x=[tx,ty,θ,log s] (B,4)、对角方差 (B,4)、残差标准差 (B,)。
    """
    ws = w.sum(1).clamp(min=eps)
    ms = (w[..., None] * src).sum(1) / ws[:, None]
    md = (w[..., None] * dst).sum(1) / ws[:, None]
    X, Y = src - ms[:, None], dst - md[:, None]
    a = (w * (X * Y).sum(-1)).sum(1)
    bb = (w * (X[..., 0] * Y[..., 1] - X[..., 1] * Y[..., 0])).sum(1)
    var_s = (w * (X ** 2).sum(-1)).sum(1).clamp(min=eps)
    theta = torch.atan2(bb, a)
    s = torch.sqrt(a ** 2 + bb ** 2 + eps) / var_s
    t = md - c0 - s[:, None] * torch.einsum("bij,bj->bi", rot2(theta), ms - c0)
    x = torch.cat([t, theta[:, None], torch.log(s.clamp(min=1e-3))[:, None]], 1)
    res = dst - apply_sim(x, src, c0)
    sig2 = (w * (res ** 2).sum(-1)).sum(1) / (2 * ws) + 0.05
    n_eff = ws ** 2 / (w ** 2).sum(1).clamp(min=eps)                          # 有效样本数
    var = torch.stack([sig2 / n_eff, sig2 / n_eff, sig2 / var_s * ws / n_eff, sig2 / var_s * ws / n_eff], 1)
    lowinfo = (n_eff < 2.0)[:, None]
    var = torch.where(lowinfo, torch.full_like(var, 1e4), var)
    if prior_mu is not None:
        prec = 1 / var + 1 / prior_var
        x = (x / var + prior_mu / prior_var) / prec
        var = 1 / prec
    return x, var, sig2.sqrt()


# ---------------------------------------------------------------- 带垃圾箱的 Sinkhorn（SuperGlue 式）
def log_sinkhorn_dustbin(S, alpha_a, alpha_b, mask_a, mask_b, iters=20):
    """S (B,M,N) 相似度；alpha_a (B,M)/alpha_b (B,N) 每个实例落入垃圾箱的得分；mask 为有效位。
    返回 log P (B,M+1,N+1)，行/列和约束：有效实例为 1，垃圾箱容量为对方有效个数。"""
    b, m, n = S.shape
    NEG = -1e4
    Z = torch.cat([torch.cat([S, alpha_a[..., None]], 2),
                   torch.cat([alpha_b[:, None], torch.zeros(b, 1, 1, device=S.device, dtype=S.dtype)], 2)], 1)
    va = torch.cat([mask_a, torch.ones(b, 1, device=S.device, dtype=torch.bool)], 1)
    vb = torch.cat([mask_b, torch.ones(b, 1, device=S.device, dtype=torch.bool)], 1)
    Z = Z.masked_fill(~va[:, :, None] | ~vb[:, None, :], NEG)
    ma, nb = mask_a.sum(1).float(), mask_b.sum(1).float()
    norm = -torch.log(ma + nb).clamp(min=-1e4)[:, None]
    log_mu = torch.where(va, torch.zeros_like(va, dtype=S.dtype), torch.full_like(va, NEG, dtype=S.dtype))
    log_mu[:, -1] = torch.log(nb.clamp(min=1e-6))
    log_nu = torch.where(vb, torch.zeros_like(vb, dtype=S.dtype), torch.full_like(vb, NEG, dtype=S.dtype))
    log_nu[:, -1] = torch.log(ma.clamp(min=1e-6))
    log_mu, log_nu = log_mu + norm, log_nu + norm
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v[:, None, :], 2)
        v = log_nu - torch.logsumexp(Z + u[:, :, None], 1)
    return Z + u[:, :, None] + v[:, None, :] - norm[:, :, None]


# ---------------------------------------------------------------- 软栅格化
def soft_raster(poly, grid, tau=0.5):
    """poly (N,V,2)，grid (N,P,2) 采样点（同坐标单位）-> (N,P) ∈(0,1) 软占据。
    符号由绕数给出，幅值为到边界的可微距离（sigmoid(±d/τ)）。"""
    a = poly[:, None]                                                         # N,1,V,2
    b = torch.roll(poly, -1, 1)[:, None]
    p = grid[:, :, None]                                                      # N,P,1,2
    ab, ap = b - a, p - a
    t = ((ap * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-8)).clamp(0, 1)
    d = (ap - t[..., None] * ab).norm(dim=-1).min(-1)[0]
    va, vb = a - p, b - p
    ang = torch.atan2(va[..., 0] * vb[..., 1] - va[..., 1] * vb[..., 0], (va * vb).sum(-1))
    inside = (ang.sum(-1).abs() > math.pi).float()
    return torch.sigmoid((2 * inside - 1) * d / tau)


def poly_iou_raster(pa, pb, n=32, tau=0.3):
    """两组多边形在其联合外接框上的软栅格 IoU（训练监控/几何头输入）。pa, pb (N,V,2)。"""
    lo = torch.minimum(pa.amin(1), pb.amin(1))
    hi = torch.maximum(pa.amax(1), pb.amax(1))
    span = (hi - lo).amax(-1, keepdim=True).clamp(min=1e-3) * 1.1
    c = (lo + hi) / 2
    g = (torch.arange(n, device=pa.device, dtype=pa.dtype) + 0.5) / n - 0.5
    gy, gx = torch.meshgrid(g, g, indexing="ij")
    grid = c[:, None] + torch.stack([gx, -gy], -1).view(1, -1, 2) * span[:, None]
    ra, rb = soft_raster(pa, grid, tau * span[:, :1] / n), soft_raster(pb, grid, tau * span[:, :1] / n)
    inter = (ra * rb).sum(-1)
    return inter / (ra.sum(-1) + rb.sum(-1) - inter).clamp(min=1e-6), ra.view(-1, n, n), rb.view(-1, n, n)
