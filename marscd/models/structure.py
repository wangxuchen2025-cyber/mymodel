"""v2-I  Height-decoupled oblique structure: Ground-Contact Lines (GCL) + roof edges + instance height sweep.

Derivation (docs/research_iteration.md, Layer 2-I). For a camera at height H above the local ground and a
building at horizontal distance r, inverse-perspective (ground-plane homography) lifting of a ROOF point of
height h misplaces it by  r*h/(H-h)  (r=150 m, H=80 m, h=20 m -> 50 m), whereas the facade-ground CONTACT
line lies on the ground plane and is lifted EXACTLY, independently of the unknown height. We therefore
split the oblique evidence by its height dependence:

  * GCL map   (height-free)  -> footprint edges, registration against SHP edges, edge-level association;
  * roof-edge map (height-dependent) -> 1-D per-instance HEIGHT SWEEP accumulated over frames
                                        (a multi-view, single-parameter plane sweep with uncertainty).

Targets for both maps are rendered automatically from GT footprints/heights/poses with z-buffer visibility,
so no extra annotation is needed.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import MARSConfig
from ..geometry import Pose, boundary_samples, project
from ..geometry.visibility import build_zbuffer, point_visibility
from .common import MLP, ConvBNAct


class StructureHead(nn.Module):
    """Stride-4 logits: channel 0 = facade-ground contact line, channel 1 = roof outline edge."""

    def __init__(self, cfg: MARSConfig):
        super().__init__()
        C = cfg.d_model
        self.up = nn.Sequential(ConvBNAct(C, C // 2, 3), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                ConvBNAct(C // 2, C // 2, 3))
        self.out = nn.Conv2d(C // 2, 2, 1)
        nn.init.constant_(self.out.bias, -2.19)

    def forward(self, feats: list[Tensor]) -> Tensor:
        return self.out(self.up(feats[0]))[0]  # [2, H/4, W/4]


@torch.no_grad()
def render_structure_targets(polys: Tensor, base: Tensor, height: Tensor, pose: Pose, hw: tuple[int, int],
                             stride: int = 4, per_edge: int = 24) -> Tensor:
    """Visible contact lines / roof edges of GT prisms rasterised at `stride`. -> [2, h, w] in {0,1}."""
    H, W = hw
    h, w = (H + stride - 1) // stride, (W + stride - 1) // stride
    out = torch.zeros(2, h * w, device=polys.device)
    if polys.shape[0] == 0:
        return out.view(2, h, w)
    N, Nv, _ = polys.shape
    zb = build_zbuffer(polys, base, height, pose, hw, stride=stride)
    t = (torch.arange(per_edge, device=polys.device) + 0.5) / per_edge
    a, b = polys, polys.roll(-1, 1)
    xy = a[:, :, None] + (b - a)[:, :, None] * t[None, None, :, None]  # [N,Nv,E,2]
    e = b - a
    nrm = torch.stack([e[..., 1], -e[..., 0]], -1) / e.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    C = pose.center
    ids = torch.arange(N, device=polys.device)[:, None, None].expand(N, Nv, per_edge)
    for ch, z in ((0, base), (1, base + height)):
        X = torch.cat([xy, z[:, None, None, None].expand(N, Nv, per_edge, 1)], -1)
        vis = point_visibility(X, pose, hw, zb, own_ids=ids, tol=1.0)
        if ch == 0:  # a contact line is only seen if its wall faces the camera
            facing = ((nrm[:, :, None] * (C[:2] - xy)).sum(-1) > 0)
            vis = vis & facing
        uv, _ = project(X, pose)
        px = (uv[..., 0] / stride).floor().long().clamp(0, w - 1)
        py = (uv[..., 1] / stride).floor().long().clamp(0, h - 1)
        lin = (py * w + px)[vis]
        out[ch, lin] = 1.0
    out = out.view(2, h, w)
    return F.max_pool2d(out[None], 3, 1, 1)[0]  # 1-cell tolerance band


def sample_map(logits: Tensor, X: Tensor, pose: Pose, hw: tuple[int, int]) -> tuple[Tensor, Tensor]:
    """Bilinear lookup of a [h, w] map at the projections of world points X [..., 3]."""
    H, W = hw
    uv, dep = project(X, pose)
    g = torch.stack([uv[..., 0] / W * 2 - 1, uv[..., 1] / H * 2 - 1], -1)
    v = F.grid_sample(logits[None, None], g.reshape(1, 1, -1, 2), mode="bilinear", padding_mode="border",
                      align_corners=False).reshape(X.shape[:-1])
    ok = (dep > 0.5) & (g.abs() < 1).all(-1)
    return v, ok


class HeightSweep(nn.Module):
    """Per-instance 1-D plane sweep over height hypotheses using roof-edge evidence.

    log-likelihood increment per frame:  dL_m = beta * (s_m - mean_m s_m) * v,
      s_m = mean_k sigmoid(RE(pi(roof outline point k at height h_m)))  (visible points only),
      v   = fraction of visible outline points (frames that barely see the roof contribute little).
    The posterior over heights is accumulated in the track state (constant memory, calibrated spread).
    """

    def __init__(self, cfg: MARSConfig, n_bins: int = 24, h_min: float = 3.0, h_max: float = 120.0):
        super().__init__()
        self.register_buffer("bins", torch.exp(torch.linspace(math.log(h_min), math.log(h_max), n_bins)), persistent=False)
        self.log_beta = nn.Parameter(torch.tensor(math.log(8.0)))

    @property
    def n(self) -> int:
        return self.bins.numel()

    def increment(self, roof_logits: Tensor, polys: Tensor, base: Tensor, pose: Pose, hw) -> Tensor:
        N, Nv, _ = polys.shape
        if N == 0:
            return polys.new_zeros(0, self.n)
        mid = 0.5 * (polys + polys.roll(-1, 1))
        xy = torch.cat([polys, mid], 1)  # [N, 2Nv, 2]
        z = base[:, None, None] + self.bins[None, :, None]  # [N, B, 1]
        X = torch.cat([xy[:, None].expand(N, self.n, 2 * Nv, 2), z[..., None].expand(N, self.n, 2 * Nv, 1)], -1)
        v, ok = sample_map(roof_logits, X, pose, hw)
        p = torch.sigmoid(v) * ok
        cnt = ok.float().sum(-1)
        s = p.sum(-1) / cnt.clamp_min(1.0)  # [N, B]
        vis = (cnt / (2 * Nv)).mean(-1, keepdim=True)
        return self.log_beta.exp() * (s - s.mean(-1, keepdim=True)) * vis

    def estimate(self, loglik: Tensor) -> tuple[Tensor, Tensor]:
        p = torch.softmax(loglik, -1)
        mu = (p * self.bins).sum(-1)
        var = (p * (self.bins - mu[:, None]) ** 2).sum(-1)
        return mu, var.sqrt()


class ContactSnake(nn.Module):
    """GCL-guided vertex refinement: a learned 1-D 'snake' step along each vertex normal on the ground plane.

    The GCL logit profile sampled at offsets o in [-r, r] metres along the outward normal of every footprint
    vertex (at the base elevation, hence height-free) is mapped to a normal displacement and a confidence.
    """

    def __init__(self, n_off: int = 7, radius: float = 3.0):
        super().__init__()
        self.register_buffer("off", torch.linspace(-radius, radius, n_off), persistent=False)
        self.mlp = MLP(n_off + 1, 32, 2)
        nn.init.zeros_(self.mlp.net[-1].weight)
        nn.init.zeros_(self.mlp.net[-1].bias)
        self.radius = radius

    def forward(self, gcl_logits: Tensor, polys: Tensor, base: Tensor, pose: Pose, hw) -> tuple[Tensor, Tensor]:
        N, Nv, _ = polys.shape
        if N == 0:
            return polys.new_zeros(0, Nv, 2), polys.new_zeros(0, Nv)
        e_prev = polys - polys.roll(1, 1)
        e_next = polys.roll(-1, 1) - polys
        n_prev = torch.stack([e_prev[..., 1], -e_prev[..., 0]], -1)
        n_next = torch.stack([e_next[..., 1], -e_next[..., 0]], -1)
        nrm = F.normalize(F.normalize(n_prev, dim=-1) + F.normalize(n_next, dim=-1), dim=-1)  # outward (CCW)
        P = polys[:, :, None] + self.off[None, None, :, None] * nrm[:, :, None]
        X = torch.cat([P, base[:, None, None, None].expand(N, Nv, len(self.off), 1)], -1)
        v, ok = sample_map(gcl_logits, X, pose, hw)
        prof = torch.sigmoid(v) * ok
        out = self.mlp(torch.cat([prof, ok.float().mean(-1, keepdim=True)], -1))
        step = torch.tanh(out[..., 0]) * self.radius
        conf = torch.sigmoid(out[..., 1]) * ok.float().mean(-1)
        return step[..., None] * nrm * conf[..., None], conf


def shp_edge_map(polys: Tensor, grid, sigma_cells: float = 1.0, k: int = 64) -> Tensor:
    """Soft SHP boundary raster exp(-d^2 / 2 sigma^2) on the BEV grid (offline, label-free). -> [1, h, w]."""
    ys = grid.y0 - (torch.arange(grid.H, device=polys.device) + 0.5) * grid.gsd
    xs = grid.x0 + (torch.arange(grid.W, device=polys.device) + 0.5) * grid.gsd
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    cells = torch.stack([xx, yy], -1).reshape(-1, 2)
    if polys.shape[0] == 0:
        return torch.zeros(1, grid.H, grid.W, device=polys.device)
    pts, _ = boundary_samples(polys, k)
    pts = pts.reshape(-1, 2)
    d = torch.cat([torch.cdist(c, pts).min(1).values for c in cells.split(8192)])
    s = sigma_cells * grid.gsd
    return torch.exp(-(d**2) / (2 * s * s)).view(1, grid.H, grid.W)


def structural_score_volume(line_logits: Tensor, polys: Tensor, z: Tensor, prior: Pose, hw, center: Tensor,
                            alphas: Tensor, offsets: Tensor, k: int = 24, max_pts: int = 4096) -> Tensor:
    """Image-space render-and-compare of SHP boundaries for every planar hypothesis G(alpha, delta).

    Under hypothesis G the true pose is G o prior, so a world point X projects to pi(G^-1(X); prior).
    The score is the mean line probability at the projected CAMERA-FACING boundary samples. Evaluating the
    likelihood in sensor space keeps the detector noise isotropic; lifting the line map to the ground
    instead blurs it anisotropically with range (a stride-4 pixel covers ~3 m x 6 m at 150 m range).
    Returns [n_yaw, n_off, n_off] (rows: dy = -offsets, cols: dx = offsets, same layout as CARR).
    """
    nA, nO = alphas.numel(), offsets.numel()
    if polys.shape[0] == 0:
        return line_logits.new_zeros(nA, nO, nO)
    pts, nrm = boundary_samples(polys, k)
    zz = z[:, None].expand(-1, k)
    C = prior.center[:2]
    facing = ((nrm * (C - pts)).sum(-1) > 0)
    pts, zz = pts[facing], zz[facing]
    if pts.shape[0] > max_pts:
        sel = torch.randperm(pts.shape[0], device=pts.device)[:max_pts]
        pts, zz = pts[sel], zz[sel]
    q = pts - center
    ca, sa = alphas.cos()[:, None], alphas.sin()[:, None]
    dx = offsets[None, None, :, None]
    dy = -offsets[None, :, None, None]
    # G^-1(X) = R(-alpha)(X - c - delta) + c
    qx = q[None, None, None, :, 0] - dx
    qy = q[None, None, None, :, 1] - dy
    rx = ca[:, None, None] * qx + sa[:, None, None] * qy + center[0]
    ry = -sa[:, None, None] * qx + ca[:, None, None] * qy + center[1]
    X = torch.stack([rx, ry, zz.expand_as(rx)], -1)  # [nA, nO, nO, P, 3]
    v, ok = sample_map(line_logits, X, prior, hw)
    p = torch.sigmoid(v) * ok
    return p.sum(-1) / ok.float().sum(-1).clamp_min(1.0)
