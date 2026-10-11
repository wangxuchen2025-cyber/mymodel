"""v2-II  Hierarchical Spatial-Relation reasoning (HSR) for cross-temporal instance association.

Derivation (docs/research_iteration.md, Layer 2-II). Treat the correspondence sigma between the T1 graph
(SHP-anchored instances) and the T2 graph (video tracks) as the MAP state of a pairwise MRF with birth /
death states:

    log p(sigma) = sum_j u(j, sigma_j)                      first order : appearance, shape, provenance (RIGM)
                 + sum_{j<k} psi((j, sigma_j), (k, sigma_k))  second order: preservation of spatial relations
                 + log p(G | sigma)                          global      : one smooth rigid motion field G

with dustbins (sigma_j = 0  <=> death = demolished;  unmatched T2  <=> birth = new). The second-order term
is Lawler's QAP; we solve it on a sparse candidate set with NEURAL GRADUATED ASSIGNMENT
(Gold & Rangarajan) whose compatibility is learned on top of an analytic rigid-invariant prior, sharpened
by the second-order spatial-compatibility trick of SC^2-PCR (C2 = C * (C @ C)). The global term is an EM
step: the leave-one-out locally-rigid motion of each candidate is re-estimated from the soft assignment and
its alignment residual enters as the log-likelihood ratio  kappa - e^2 / (2 sigma_g^2)  between an inlier
(Gaussian residual) and an outlier (uniform, Coherent-Point-Drift-style) hypothesis, which rewards
geometrically verified candidates instead of only penalising bad ones. CHANGE IS THE OUTLIER CLASS of this joint
registration-association problem: dustbin mass = posterior probability of birth / death.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..config import MARSConfig
from ..geometry import boundary_samples, distance_to_polygon_boundary, min_boundary_gap, polygon_area, polygon_centroid
from .change_head import closest_point_on_polygon, loo_local_rigid, rot2
from .common import MLP
from .graph_matcher import log_optimal_transport

N_PAIR_FEATS = 9


class RelationalAssignment(nn.Module):
    def __init__(self, cfg: MARSConfig):
        super().__init__()
        self.cfg = cfg
        self.k = cfg.hsr_topk
        self.iters = cfg.hsr_iters
        self.outer = cfg.hsr_outer
        self.compat = MLP(N_PAIR_FEATS, 32, 1)
        nn.init.zeros_(self.compat.net[-1].weight)
        nn.init.constant_(self.compat.net[-1].bias, 3.0)  # start close to the analytic prior
        self.log_sd = nn.Parameter(torch.tensor(math.log(3.0)))    # distance-preservation tolerance (m)
        self.log_lam = nn.Parameter(torch.tensor(math.log(1.0)))   # weight of the pairwise support
        self.log_sg = nn.Parameter(torch.tensor(math.log(3.0)))    # motion-field residual tolerance (m)
        self.kappa = nn.Parameter(torch.tensor(2.0))               # log(1 / (u_out sqrt(2 pi) sigma_g)): inlier-vs-outlier LLR offset
        self.radius = 80.0                                          # relational neighbourhood (m)

    # ----------------------------------------------------------------- candidates
    def candidates(self, S: Tensor, gate: Tensor, dist: Tensor, forced: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Union of unary top-k and GEOMETRIC k-NN (rows and columns): spatial relations must take part in
        candidate generation, otherwise true pairs of look-alike buildings are pruned before reasoning."""
        M, N = S.shape
        mask = torch.zeros_like(gate)
        rows = torch.arange(M, device=S.device)
        cols = torch.arange(N, device=S.device)
        for score in (S.masked_fill(~gate, float("-inf")), (-dist).masked_fill(~gate, float("-inf"))):
            k1, k2 = min(self.k, N), min(self.k, M)
            mask[rows[:, None].expand(-1, k1), score.topk(k1, 1).indices] = True
            mask[score.topk(k2, 0).indices, cols[None].expand(k2, -1)] = True
        mask &= gate
        if forced is not None:
            ok = forced >= 0
            mask[ok.nonzero(as_tuple=True)[0], forced[ok]] = True
        ia, ib = mask.nonzero(as_tuple=True)
        return ia, ib

    # -------------------------------------------------------------- compatibility
    def compatibility(self, ia: Tensor, ib: Tensor, PA: Tensor, PB: Tensor) -> tuple[Tensor, Tensor]:
        """Second-order relational compatibility C2 [K, K] (rigid-invariant, change-tolerant) and A-degree [K]."""
        cA, cB = polygon_centroid(PA), polygon_centroid(PB)
        rA = polygon_area(PA).abs().clamp_min(1.0).sqrt()
        gA, gB = min_boundary_gap(PA, 8), min_boundary_gap(PB, 8)
        lA, lB = torch.log(rA), torch.log(polygon_area(PB).abs().clamp_min(1.0).sqrt())
        a, b = ia, ib
        vA = cA[a][None] - cA[a][:, None]
        vB = cB[b][None] - cB[b][:, None]
        dA, dB = vA.norm(dim=-1), vB.norm(dim=-1)
        tol = self.log_sd.exp() + 0.25 * (rA[a][:, None] + rA[a][None])  # expansions shift centroids
        dd = (dA - dB) / tol
        prior = torch.exp(-0.5 * dd**2)
        far = (dA > 1.0) & (dB > 1.0)
        cosd = torch.where(far, (vA * vB).sum(-1) / (dA * dB).clamp_min(1e-6), torch.ones_like(dA))
        sind = torch.where(far, (vA[..., 0] * vB[..., 1] - vA[..., 1] * vB[..., 0]) / (dA * dB).clamp_min(1e-6),
                           torch.zeros_like(dA))
        adjA = (gA[a][:, a] < 2.0).float()
        adjB = (gB[b][:, b] < 2.0).float()
        size = (lA[a][:, None] - lA[a][None]) - (lB[b][:, None] - lB[b][None])
        feats = torch.stack([dd, dd.abs(), cosd, sind, far.float(), adjA, adjB,
                             (gA[a][:, a] - gB[b][:, b]).clamp(-20, 20) / 10.0, size], -1)
        conflict = (a[:, None] == a[None]) | (b[:, None] == b[None])
        near = dA < self.radius
        C = prior * torch.sigmoid(self.compat(feats)[..., 0]) * (~conflict) * near
        common = C @ C  # SC^2: number of candidates compatible with both m and n
        C2 = C * common / common.amax().clamp_min(1e-6)  # suppresses accidental pairwise compatibility
        # normalise by the number of DISTINCT neighbouring T1 buildings (each carries <= 1 unit of mass)
        dA_nodes = torch.cdist(cA, cA)
        deg = ((dA_nodes[a] < self.radius).float().sum(1) - 1).clamp_min(1.0)
        return C2, deg.sqrt()  # votes of k neighbours: signal ~ k, noise ~ sqrt(k) -> z-score normalisation

    # ------------------------------------------------------------- motion field
    @torch.no_grad()
    def motion_residual(self, ia: Tensor, ib: Tensor, PA: Tensor, PB: Tensor, w: Tensor, icp: int = 3) -> Tensor:
        """Mean boundary residual of each candidate after its leave-one-out local rigid motion (metres)."""
        pa, _ = boundary_samples(PA[ia], 12)
        PBc = PB[ib]
        cen = polygon_centroid(PA[ia])
        theta, t = pa.new_zeros(len(ia)), pa.new_zeros(len(ia), 2)
        for _ in range(icp):
            al = torch.einsum("pab,pkb->pka", rot2(theta), pa) + t[:, None]
            qb = closest_point_on_polygon(al, PBc)
            theta, t = loo_local_rigid(pa, qb, w, cen, 60.0, groups=ia)
        al = torch.einsum("pab,pkb->pka", rot2(theta), pa) + t[:, None]
        return distance_to_polygon_boundary(al, PBc).mean(-1)

    # ------------------------------------------------------------------ forward
    def forward(self, S: Tensor, alpha: Tensor, beta: Tensor, corner: Tensor, gate: Tensor, PA: Tensor,
                PB: Tensor, forced: Tensor | None = None) -> dict:
        M, N = S.shape
        log_P0 = log_optimal_transport(S, alpha, beta, corner, self.cfg.sinkhorn_iters)
        if M < 2 or N < 2 or not gate.any():
            return dict(log_P=log_P0, log_P_unary=log_P0, cand=None)
        dist = torch.cdist(polygon_centroid(PA), polygon_centroid(PB))
        ia, ib = self.candidates(S.detach(), gate, dist, forced)
        C2, deg = self.compatibility(ia, ib, PA, PB)
        S_c = S[ia, ib]
        P_c = log_P0[ia, ib].exp()
        lam, sg = self.log_lam.exp(), self.log_sg.exp()
        llr = torch.zeros_like(S_c)  # global term: inlier (Gaussian residual) vs outlier (uniform) log-likelihood ratio
        betas = torch.linspace(1.0, 2.0, self.iters, device=S.device)
        log_P = log_P0
        for _ in range(self.outer):
            for bt in betas:
                Q = S_c + lam * (C2 @ P_c) / deg + llr
                Sf = torch.full_like(S, -1e4).index_put((ia, ib), Q)
                log_P = log_optimal_transport(Sf * bt, alpha * bt, beta * bt, corner * bt, self.cfg.sinkhorn_iters)
                P_c = log_P[ia, ib].exp()
            e = self.motion_residual(ia, ib, PA.detach(), PB.detach(), P_c.detach())
            llr = self.kappa - 0.5 * (e / sg) ** 2
        return dict(log_P=log_P, log_P_unary=log_P0, cand=(ia, ib), compat=C2, residual=e)
