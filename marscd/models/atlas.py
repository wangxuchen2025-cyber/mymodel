"""v2-III  Prism Texture Atlas (PTA): view-rectified, surface-indexed instance memory.

Oblique appearance of a building depends on which facades are visible, so comparing view-dependent
descriptors across epochs (different trajectories) confounds viewpoint with change. PTA indexes evidence
by SURFACE coordinates of the prism instead of by view:

  wall cells  (s, tau): s = arc-length along the footprint ring (canonical start), tau = normalised height;
  roof cells  (u, v)  : regular grid in the principal-axis frame of the footprint (masked to the interior).

Each frame back-projects image features to every visible cell with weight
    w = 1[in image] * 1[front-facing] * 1[not occluded (z-buffer)] * cos(view, normal)   (foreshortening),
and the atlas keeps running sums (A, W) -> constant memory per instance, independent of video length.
Cross-epoch comparison is then a cell-to-cell comparison after rigid alignment (RRD), restricted to cells
CO-OBSERVED in both epochs, plus "unmatched-wall" fractions that directly evidence expansion / shrinkage.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import MARSConfig
from ..geometry import Pose, boundary_samples, points_in_polygon_paired, polygon_centroid, principal_axis_angle, project
from ..geometry.visibility import ZBuffer, point_visibility

N_ATLAS_FEATS = 8


class PrismAtlas(nn.Module):
    def __init__(self, cfg: MARSConfig):
        super().__init__()
        C = cfg.d_model
        self.ns, self.nt, self.nr = cfg.atlas_s, cfg.atlas_tau, cfg.atlas_roof
        self.ca = cfg.atlas_dim
        self.proj = nn.Conv2d(C, self.ca, 1)
        self.cell_embed = nn.Parameter(torch.randn(self.n_cells, C) * 0.02)
        self.to_mem = nn.Linear(self.ca, C)

    @property
    def n_wall(self) -> int:
        return self.ns * self.nt

    @property
    def n_cells(self) -> int:
        return self.n_wall + self.nr * self.nr

    # ------------------------------------------------------------- geometry
    def cells(self, polys: Tensor, base: Tensor, height: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """-> X [N, Nc, 3], normals [N, Nc, 3], valid [N, Nc]."""
        N = polys.shape[0]
        dev = polys.device
        pts, nrm = boundary_samples(polys, self.ns)  # [N,ns,2]
        tau = (torch.arange(self.nt, device=dev) + 0.5) / self.nt
        z = base[:, None, None] + tau[None, None] * height[:, None, None]  # [N,1,nt]
        Xw = torch.cat([pts[:, :, None].expand(N, self.ns, self.nt, 2), z[..., None].expand(N, self.ns, self.nt, 1)], -1)
        Nw = torch.cat([nrm, torch.zeros_like(nrm[..., :1])], -1)[:, :, None].expand(N, self.ns, self.nt, 3)
        cen = polygon_centroid(polys)
        th = principal_axis_angle(polys)
        a1 = torch.stack([th.cos(), th.sin()], -1)
        a2 = torch.stack([-th.sin(), th.cos()], -1)
        rel = polys - cen[:, None]
        u, v = (rel * a1[:, None]).sum(-1), (rel * a2[:, None]).sum(-1)
        g = (torch.arange(self.nr, device=dev) + 0.5) / self.nr
        gu = u.min(1).values[:, None] + g[None] * (u.max(1).values - u.min(1).values)[:, None]
        gv = v.min(1).values[:, None] + g[None] * (v.max(1).values - v.min(1).values)[:, None]
        rxy = cen[:, None, None] + gu[:, :, None, None] * a1[:, None, None] + gv[:, None, :, None] * a2[:, None, None]
        rxy = rxy.reshape(N, -1, 2)
        rvalid = points_in_polygon_paired(rxy, polys)
        Xr = torch.cat([rxy, (base + height)[:, None, None].expand(N, rxy.shape[1], 1)], -1)
        Nr = torch.zeros_like(Xr)
        Nr[..., 2] = 1.0
        X = torch.cat([Xw.reshape(N, -1, 3), Xr], 1)
        Nn = torch.cat([Nw.reshape(N, -1, 3), Nr], 1)
        valid = torch.cat([torch.ones(N, self.n_wall, dtype=torch.bool, device=dev), rvalid], 1)
        return X, Nn, valid

    # ---------------------------------------------------------- observation
    def observe(self, fmap: Tensor, polys: Tensor, base: Tensor, height: Tensor, pose: Pose, hw,
                zb: ZBuffer, own_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Per-cell features [N, Nc, Ca] (already multiplied by weight) and weights [N, Nc]."""
        N = polys.shape[0]
        if N == 0:
            return fmap.new_zeros(0, self.n_cells, self.ca), fmap.new_zeros(0, self.n_cells)
        X, Nn, valid = self.cells(polys, base, height)
        to_cam = F.normalize(pose.center - X, dim=-1)
        cosang = (Nn * to_cam).sum(-1).clamp_min(0.0)
        vis = point_visibility(X, pose, hw, zb, own_ids=own_ids[:, None].expand(N, X.shape[1]), tol=1.5)
        w = cosang * vis * valid
        H, W = hw
        uv, _ = project(X, pose)
        g = torch.stack([uv[..., 0] / W * 2 - 1, uv[..., 1] / H * 2 - 1], -1)
        f = F.grid_sample(self.proj(fmap), g.reshape(1, 1, -1, 2), mode="bilinear", padding_mode="zeros",
                          align_corners=False).reshape(self.ca, N, -1).permute(1, 2, 0)
        return f * w[..., None], w

    def tokens(self, A: Tensor, W: Tensor) -> tuple[Tensor, Tensor]:
        """Atlas sums -> decoder memory tokens [N, Nc, C] and validity [N, Nc]."""
        mean = A / W[..., None].clamp_min(1e-6)
        return self.to_mem(mean) + self.cell_embed[None], W > 1e-3

    def pooled(self, A: Tensor, W: Tensor) -> Tensor:
        mean = A / W[..., None].clamp_min(1e-6)
        w = W / W.sum(1, keepdim=True).clamp_min(1e-6)
        return self.to_mem((mean * w[..., None]).sum(1))

    # ------------------------------------------------------------ comparison
    def compare(self, A1: Tensor, W1: Tensor, P1_aligned: Tensor, h1: Tensor, A2: Tensor, W2: Tensor, P2: Tensor,
                h2: Tensor, base: Tensor, d_wall: float = 2.0, d_roof: float = 3.0) -> Tensor:
        """Co-observation-aware atlas comparison of matched pairs. -> [P, N_ATLAS_FEATS].

        P1_aligned: T1 footprints already mapped into the T2 frame by the RRD rigid motion, so cell
        correspondences are found by proximity of surface positions (same tau level for walls).
        """
        P = P2.shape[0]
        if P == 0:
            return A1.new_zeros(0, N_ATLAS_FEATS)
        X1, _, v1 = self.cells(P1_aligned, base, h1)
        X2, _, v2 = self.cells(P2, base, h2)
        m1 = A1 / W1[..., None].clamp_min(1e-6)
        m2 = A2 / W2[..., None].clamp_min(1e-6)
        o1, o2 = (W1 > 1e-3) & v1, (W2 > 1e-3) & v2
        nw = self.n_wall
        feats = []
        cos_terms = []
        for sl, dthr in ((slice(0, nw), d_wall), (slice(nw, None), d_roof)):
            x1, x2 = X1[:, sl, :2], X2[:, sl, :2]
            d = torch.cdist(x2, x1)  # [P, n2, n1]
            if sl.start == 0:  # walls: only compare equal height levels
                lvl = torch.arange(nw, device=d.device) % self.nt
                d = d + 1e3 * (lvl[:, None] != lvl[None]).float()
            dmin, j = d.min(-1)
            near = dmin < dthr
            mm1 = torch.gather(m1[:, sl], 1, j[..., None].expand(-1, -1, m1.shape[-1]))
            oo1 = torch.gather(o1[:, sl], 1, j)
            co = near & o2[:, sl] & oo1
            cosd = 1 - F.cosine_similarity(mm1, m2[:, sl], dim=-1)
            wco = co.float()
            cos_terms.append((cosd * wco).sum(1) / wco.sum(1).clamp_min(1.0))
            if sl.start == 0:
                obs2 = o2[:, sl].float()
                feats.append((co.float().sum(1) / obs2.sum(1).clamp_min(1.0)))               # co-observed fraction
                feats.append(((~near) & o2[:, sl]).float().sum(1) / obs2.sum(1).clamp_min(1.0))  # new walls -> expansion
                dr = torch.cdist(x1, x2)
                dr = dr + 1e3 * (lvl[:, None] != lvl[None]).float()
                near_r = dr.min(-1).values < dthr
                obs1 = o1[:, sl].float()
                feats.append(((~near_r) & o1[:, sl]).float().sum(1) / obs1.sum(1).clamp_min(1.0))  # lost walls -> shrink
        feats = cos_terms + feats + [torch.log1p(W1.sum(1)), torch.log1p(W2.sum(1)), (o1.float().mean(1) * o2.float().mean(1))]
        return torch.stack(feats, -1)
