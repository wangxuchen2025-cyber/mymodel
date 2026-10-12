"""非学习基线：(1) GNSS 直接投影 + 质心距离门限 Hungarian（GIS 矢量匹配/合并的常规做法）；
(2) 实例级 ICP：最近邻互匹配 + Umeyama 迭代（门限由粗到细）+ 末轮 Hungarian。"""
import torch

from .geometry import apply_sim, poly_moments, weighted_umeyama

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


def _hungarian_gate(ca, cb, mask_a, mask_b, gate):
    b, m = mask_a.shape
    out = torch.full((b, m), -1, dtype=torch.long, device=ca.device)
    for bi in range(b):
        ia = mask_a[bi].nonzero()[:, 0]
        ib = mask_b[bi].nonzero()[:, 0]
        if len(ia) == 0 or len(ib) == 0:
            continue
        d = torch.cdist(ca[bi, ia], cb[bi, ib])
        cost = d.clone()
        cost[d > gate] = 1e6
        r, c = linear_sum_assignment(cost.cpu().numpy())
        r, c = torch.as_tensor(r, device=ca.device), torch.as_tensor(c, device=ca.device)
        ok = d[r, c] <= gate
        out[bi, ia[r[ok]]] = ib[c[ok]]
    return out


def gnss_hungarian(A, B, gate=5.0):
    _, ca, _ = poly_moments(A["poly"])
    _, cb, _ = poly_moments(B["poly"])
    m = _hungarian_gate(ca, cb, A["mask"], B["mask"], gate)
    x = torch.zeros(ca.shape[0], 4, device=ca.device)
    return x, m


def object_icp(A, B, c0, iters=15, gate0=15.0, gate1=4.0):
    _, ca, _ = poly_moments(A["poly"])
    _, cb, _ = poly_moments(B["poly"])
    b = ca.shape[0]
    x = torch.zeros(b, 4, device=ca.device)
    for k in range(iters):
        gate = gate0 + (gate1 - gate0) * k / max(iters - 1, 1)
        cat = apply_sim(x, ca, c0)
        d = torch.cdist(cat, cb)
        d = d.masked_fill(~B["mask"][:, None], 1e6).masked_fill(~A["mask"][..., None], 1e6)
        nn_ab = d.argmin(2)
        nn_ba = d.argmin(1)
        mutual = nn_ba.gather(1, nn_ab) == torch.arange(ca.shape[1], device=ca.device)[None]
        ok = mutual & (d.gather(2, nn_ab[..., None])[..., 0] < gate) & A["mask"]
        tgt = cb.gather(1, nn_ab[..., None].expand(-1, -1, 2))
        x, _, _ = weighted_umeyama(ca, tgt, ok.float(), c0)
    m = _hungarian_gate(apply_sim(x, ca, c0), cb, A["mask"], B["mask"], gate1)
    return x, m
