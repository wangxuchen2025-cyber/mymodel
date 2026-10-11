"""Module F - Change reasoning: Rigid-Residual Decomposition (RRD) + pair classifier.

Planimetric expansion / shrinkage must be told apart from residual
misregistration. Misregistration is spatially SMOOTH and shared by
neighbouring buildings, a true footprint change is building-specific. For
each matched (T1, T2) pair we therefore estimate the rigid component from its
NEIGHBOURS ONLY (leave-one-out, Gaussian-weighted, IRLS-robust weighted
Procrustes on boundary correspondences), remove it, and describe the
remaining non-rigid boundary displacement field with signed distances in both
directions. The classifier fuses these interpretable geometric cues with the
learned T1/T2 instance features, plus a physics-informed area-ratio evidence
term added to the logits.

Decisions for unmatched nodes (in MARSCD.match_and_reason):
  T1 unmatched & observed  -> demolished;  T1 unmatched & not observed -> unobserved;
  T2 unmatched & confirmed -> new.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..config import MARSConfig
from ..geometry import boundary_samples, polygon_area, polygon_centroid, polygon_iou_paired, signed_distance_to_polygon
from .common import MLP

PAIR_CLASSES = ("unchanged", "expanded", "shrunk")
REPORT_KINDS = ("unchanged", "expanded", "shrunk", "demolished", "new", "unobserved")


def closest_point_on_polygon(P: Tensor, V: Tensor) -> Tensor:
    """P: [B, Q, 2], V: [B, N, 2] -> closest boundary points [B, Q, 2]."""
    a, b = V, V.roll(-1, 1)
    ab = (b - a)[:, None]
    ap = P[:, :, None] - a[:, None]
    t = ((ap * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-12)).clamp(0, 1)
    proj = a[:, None] + t.unsqueeze(-1) * ab  # [B,Q,N,2]
    d = (P[:, :, None] - proj).norm(dim=-1)
    idx = d.argmin(-1)
    return torch.gather(proj, 2, idx[..., None, None].expand(-1, -1, 1, 2))[:, :, 0]


def rot2(theta: Tensor) -> Tensor:
    c, s = theta.cos(), theta.sin()
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)


def loo_local_rigid(p: Tensor, q: Tensor, w: Tensor, cen: Tensor, sigma: float, iters: int = 3,
                    huber: float = 1.0, groups: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Leave-one-out, locally weighted 2D Procrustes as kernel regression on sufficient statistics.

    p, q: [P, K, 2] boundary correspondences of P (candidate) pairs, w: [P] pair weights, cen: [P, 2] pair
    centroids. Returns (theta [P], t [P, 2]) - the rigid motion at pair i predicted from all OTHER pairs
    j != i (and, if `groups` is given, from no pair sharing pair i's group, e.g. alternative candidates of
    the same T1 building), with Gaussian spatial kernel g_ij = w_j exp(-|c_i - c_j|^2 / 2 sigma^2).

    The weighted Procrustes solution only needs the per-pair sums  S0 = sum_k r, Sp = sum_k r p, Sq = sum_k r q,
    Spq = sum_k r (p . q), Sx = sum_k r (p x q)  (r = robust IRLS weight of each correspondence, evaluated
    under its own pair's current local motion), so the kernel-weighted fit is a handful of P x P mat-vec
    products: O(P^2) memory/time instead of materialising [P, P, K] tensors.
    """
    Pn, K, _ = p.shape
    g = torch.exp(-((cen[:, None] - cen[None]) ** 2).sum(-1) / (2 * sigma**2)) * w[None]
    g = g * (1 - torch.eye(Pn, device=p.device, dtype=p.dtype))  # leave-one-out
    if groups is not None:
        g = g * (groups[:, None] != groups[None]).to(p.dtype)
    rob = torch.ones(Pn, K, device=p.device, dtype=p.dtype)
    pq = (p * q).sum(-1)
    px = p[..., 0] * q[..., 1] - p[..., 1] * q[..., 0]
    theta = p.new_zeros(Pn)
    t = p.new_zeros(Pn, 2)
    for _ in range(iters):
        s0 = rob.sum(1)
        sp = (rob[..., None] * p).sum(1)
        sq = (rob[..., None] * q).sum(1)
        Ws = (g @ s0).clamp_min(1e-9)
        pm = (g @ sp) / Ws[:, None]
        qm = (g @ sq) / Ws[:, None]
        a = g @ (rob * pq).sum(1) - Ws * (pm * qm).sum(-1)
        b = g @ (rob * px).sum(1) - Ws * (pm[:, 0] * qm[:, 1] - pm[:, 1] * qm[:, 0])
        theta = torch.atan2(b, a)
        R = rot2(theta)
        t = qm - (R @ pm[..., None])[..., 0]
        res = (torch.einsum("pab,pkb->pka", R, p) + t[:, None] - q).norm(dim=-1)  # own-pair residuals [P, K]
        rob = (huber / res.clamp_min(1e-6)).clamp(max=1.0)
    has_nb = g.sum(1) > 1e-6
    theta = torch.where(has_nb, theta, torch.zeros_like(theta))
    t = torch.where(has_nb[:, None], t, torch.zeros_like(t))
    return theta, t


class ChangeReasoner(nn.Module):
    N_GEO = 12

    def __init__(self, cfg: MARSConfig):
        super().__init__()
        C = cfg.d_model
        self.cfg = cfg
        from .atlas import N_ATLAS_FEATS
        self.n_extra = N_ATLAS_FEATS
        self.cls = MLP(4 * C + self.N_GEO + self.n_extra, C, len(PAIR_CLASSES), n_layers=3)
        self.evidence_scale = nn.Parameter(torch.tensor(1.0))
        self.rigid_sigma = 60.0  # m, spatial correlation length of residual misregistration
        self.icp_iters = 5

    def geo_features(self, PA: Tensor, PB: Tensor) -> Tensor:
        K = self.cfg.boundary_samples
        sa, _ = boundary_samples(PA, K)
        sb, _ = boundary_samples(PB, K)
        da = signed_distance_to_polygon(sa, PB)  # > 0: T1 boundary outside T2 -> shrink evidence
        db = signed_distance_to_polygon(sb, PA)  # > 0: T2 boundary outside T1 -> expansion evidence
        Aa = polygon_area(PA).abs().clamp_min(1.0)
        Ab = polygon_area(PB).abs().clamp_min(1.0)
        feats = [
            da.relu().mean(-1), da.relu().amax(-1), (da > 1.0).float().mean(-1), (-da).relu().mean(-1),
            db.relu().mean(-1), db.relu().amax(-1), (db > 1.0).float().mean(-1), (-db).relu().mean(-1),
            polygon_iou_paired(PA, PB, res=20), torch.log(Ab / Aa), torch.log(Aa) / 5.0,
            (polygon_centroid(PA) - polygon_centroid(PB)).norm(dim=-1) / 5.0,
        ]
        return torch.stack(feats, -1)

    def forward(self, fA: Tensor, fB: Tensor, PA: Tensor, PB: Tensor, w: Tensor, extra_fn=None) -> dict:
        """extra_fn(PA_aligned) -> [P, n_extra] view-rectified atlas comparison features (v2-III)."""
        if fA.shape[0] == 0:
            return dict(logits=fA.new_zeros(0, len(PAIR_CLASSES)), theta=fA.new_zeros(0), t=fA.new_zeros(0, 2),
                        PA_aligned=PA, geo=fA.new_zeros(0, self.N_GEO))
        with torch.no_grad():  # ICP: re-estimate closest-point correspondences under the current LOO motion
            K = self.cfg.boundary_samples
            pa, _ = boundary_samples(PA, K)
            theta, t = PA.new_zeros(PA.shape[0]), PA.new_zeros(PA.shape[0], 2)
            for _ in range(self.icp_iters):
                pa_al = torch.einsum("pab,pkb->pka", rot2(theta), pa) + t[:, None]
                qb = closest_point_on_polygon(pa_al, PB)
                theta, t = loo_local_rigid(pa, qb, w.detach(), polygon_centroid(PA), self.rigid_sigma)
        R = rot2(theta)
        PA_al = torch.einsum("pab,pnb->pna", R, PA) + t[:, None]
        geo = self.geo_features(PA_al, PB)
        extra = extra_fn(PA_al) if extra_fn is not None else geo.new_zeros(geo.shape[0], self.n_extra)
        x = torch.cat([fA, fB, (fA - fB).abs(), fA * fB, geo, extra], -1)
        logits = self.cls(x)
        ev = self.evidence_scale * 3.0 * torch.tanh(geo[:, 9] / 0.3)  # bounded area-ratio evidence
        logits = logits + torch.stack([-ev.abs() * 0.5, ev, -ev], -1)
        return dict(logits=logits, theta=theta, t=t, PA_aligned=PA_al, geo=geo, atlas=extra)
