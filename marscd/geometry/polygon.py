"""Batched polygon utilities (torch) for building footprints in the world frame.

Footprints are represented as closed rings with a fixed number of vertices
N_v (uniform arc-length resampling), counter-clockwise, without the repeated
closing vertex. A fixed N_v makes the representation compatible with
MapTR-style hierarchical (instance + vertex) queries.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor


def polygon_area(V: Tensor) -> Tensor:
    """Signed shoelace area; V: [..., N, 2] -> [...] (positive for CCW)."""
    x, y = V[..., 0], V[..., 1]
    return 0.5 * (x * y.roll(-1, -1) - x.roll(-1, -1) * y).sum(-1)


def polygon_centroid(V: Tensor) -> Tensor:
    """Area-weighted centroid with a vertex-mean fallback; [..., N, 2] -> [..., 2]."""
    x, y = V[..., 0], V[..., 1]
    xn, yn = x.roll(-1, -1), y.roll(-1, -1)
    cross = x * yn - xn * y
    A = 0.5 * cross.sum(-1)
    # robust formulation (handles tiny / degenerate rings)
    denom = 6.0 * A
    safe = denom.abs() > 1e-6
    cx = torch.where(safe, ((x + xn) * cross).sum(-1) / torch.where(safe, denom, torch.ones_like(denom)), x.mean(-1))
    cy = torch.where(safe, ((y + yn) * cross).sum(-1) / torch.where(safe, denom, torch.ones_like(denom)), y.mean(-1))
    return torch.stack([cx, cy], -1)


def polygon_perimeter(V: Tensor) -> Tensor:
    return (V.roll(-1, -2) - V).norm(dim=-1).sum(-1)


def canonicalize_polygon(V: Tensor) -> Tensor:
    """Make a ring CCW and start at the vertex closest to the bbox lower-left corner."""
    if polygon_area(V) < 0:
        V = V.flip(0)
    ll = V.min(0).values
    start = int(((V - ll) ** 2).sum(-1).argmin())
    return V.roll(-start, 0)


def resample_polygon(V, n: int) -> Tensor:
    """Uniform arc-length resampling of a closed ring to n vertices (CCW, canonical start).

    Corners are preserved when n is large enough by always keeping the original
    vertices if their count is <= n and distributing the remaining samples along
    edges proportionally to edge length.
    """
    V = torch.as_tensor(np.asarray(V, dtype=np.float32)) if not isinstance(V, Tensor) else V.float()
    if V.shape[0] > 1 and torch.allclose(V[0], V[-1]):
        V = V[:-1]
    if polygon_area(V) < 0:
        V = V.flip(0)
    m = V.shape[0]
    seg = V.roll(-1, 0) - V
    L = seg.norm(dim=-1)
    if m <= n:
        # keep corners; allocate extra points to edges by length (largest remainder)
        extra = n - m
        share = L / L.sum().clamp_min(1e-9) * extra
        k = share.floor().long()
        rem = extra - int(k.sum())
        if rem > 0:
            order = torch.argsort(share - k.float(), descending=True)[:rem]
            k[order] += 1
        pts = []
        for i in range(m):
            for j in range(int(k[i]) + 1):
                pts.append(V[i] + seg[i] * (j / (int(k[i]) + 1)))
        out = torch.stack(pts, 0)
    else:
        cum = torch.cat([L.new_zeros(1), L.cumsum(0)])
        total = cum[-1]
        t = torch.arange(n, dtype=V.dtype) * total / n
        idx = torch.searchsorted(cum[1:], t, right=True).clamp(max=m - 1)
        frac = (t - cum[idx]) / L[idx].clamp_min(1e-9)
        out = V[idx] + seg[idx] * frac.unsqueeze(-1)
    return canonicalize_polygon(out)


def _edges(V: Tensor) -> tuple[Tensor, Tensor]:
    return V, V.roll(-1, -2)


def points_in_polygons(P: Tensor, V: Tensor) -> Tensor:
    """All-vs-all crossing-number test. P: [Q, 2], V: [M, N, 2] -> bool [M, Q]."""
    a, b = _edges(V)  # [M,N,2]
    px = P[None, :, None, 0]
    py = P[None, :, None, 1]
    ax, ay = a[:, None, :, 0], a[:, None, :, 1]
    bx, by = b[:, None, :, 0], b[:, None, :, 1]
    cond = (ay > py) != (by > py)
    xint = (bx - ax) * (py - ay) / torch.where((by - ay).abs() < 1e-12, torch.full_like(by, 1e-12), by - ay) + ax
    crosses = cond & (px < xint)
    return (crosses.sum(-1) % 2) == 1


def points_in_polygon_paired(P: Tensor, V: Tensor) -> Tensor:
    """Paired crossing-number test. P: [B, Q, 2], V: [B, N, 2] -> bool [B, Q]."""
    a, b = _edges(V)
    px, py = P[:, :, None, 0], P[:, :, None, 1]
    ax, ay = a[:, None, :, 0], a[:, None, :, 1]
    bx, by = b[:, None, :, 0], b[:, None, :, 1]
    cond = (ay > py) != (by > py)
    xint = (bx - ax) * (py - ay) / torch.where((by - ay).abs() < 1e-12, torch.full_like(by, 1e-12), by - ay) + ax
    return ((cond & (px < xint)).sum(-1) % 2) == 1


def boundary_samples(V: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """Uniformly sample k boundary points and outward normals. V: [B, N, 2]."""
    a, b = _edges(V)
    seg = b - a
    L = seg.norm(dim=-1)  # [B,N]
    cum = torch.cat([L.new_zeros(L.shape[0], 1), L.cumsum(-1)], -1)
    total = cum[:, -1:]
    t = (torch.arange(k, dtype=V.dtype, device=V.device) + 0.5)[None] * total / k  # [B,k]
    idx = torch.searchsorted(cum[:, 1:].contiguous(), t.contiguous(), right=True).clamp(max=V.shape[1] - 1)
    a_i = torch.gather(a, 1, idx[..., None].expand(-1, -1, 2))
    s_i = torch.gather(seg, 1, idx[..., None].expand(-1, -1, 2))
    L_i = torch.gather(L, 1, idx).clamp_min(1e-9)
    c_i = torch.gather(cum, 1, idx)
    pts = a_i + s_i * ((t - c_i) / L_i).unsqueeze(-1)
    d = s_i / L_i.unsqueeze(-1)
    normals = torch.stack([d[..., 1], -d[..., 0]], -1)  # outward for CCW rings
    return pts, normals


def distance_to_polygon_boundary(P: Tensor, V: Tensor) -> Tensor:
    """Unsigned distance from points to ring boundary. P: [B, Q, 2], V: [B, N, 2] -> [B, Q]."""
    a, b = _edges(V)
    ab = (b - a)[:, None]  # [B,1,N,2]
    ap = P[:, :, None] - a[:, None]  # [B,Q,N,2]
    t = ((ap * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-12)).clamp(0, 1)
    proj = a[:, None] + t.unsqueeze(-1) * ab
    return (P[:, :, None] - proj).norm(dim=-1).min(-1).values


def signed_distance_to_polygon(P: Tensor, V: Tensor) -> Tensor:
    """Signed distance (negative inside). P: [B, Q, 2], V: [B, N, 2] -> [B, Q]."""
    d = distance_to_polygon_boundary(P, V)
    inside = points_in_polygon_paired(P, V)
    return torch.where(inside, -d, d)


def polygon_iou_paired(VA: Tensor, VB: Tensor, res: int = 24) -> Tensor:
    """Grid-sampled IoU for paired polygons. VA, VB: [B, N, 2] -> [B]."""
    lo = torch.minimum(VA.min(1).values, VB.min(1).values)
    hi = torch.maximum(VA.max(1).values, VB.max(1).values)
    g = (torch.arange(res, dtype=VA.dtype, device=VA.device) + 0.5) / res
    gy, gx = torch.meshgrid(g, g, indexing="ij")
    grid = torch.stack([gx, gy], -1).reshape(1, -1, 2)
    P = lo[:, None] + grid * (hi - lo)[:, None]
    ia = points_in_polygon_paired(P, VA)
    ib = points_in_polygon_paired(P, VB)
    inter = (ia & ib).float().sum(-1)
    union = (ia | ib).float().sum(-1).clamp_min(1.0)
    return inter / union


def principal_axis_angle(V: Tensor) -> Tensor:
    """Orientation (mod pi) of the principal axis of the vertex cloud; [..., N, 2] -> [...]."""
    c = V - V.mean(-2, keepdim=True)
    cov = c.transpose(-1, -2) @ c / V.shape[-2]
    return 0.5 * torch.atan2(2 * cov[..., 0, 1], cov[..., 0, 0] - cov[..., 1, 1])


def shape_descriptors(V: Tensor, n_fourier: int = 6) -> Tensor:
    """Rotation/translation-invariant shape descriptors; [M, N, 2] -> [M, 4 + n_fourier]."""
    A = polygon_area(V).abs().clamp_min(1e-3)
    P = polygon_perimeter(V).clamp_min(1e-3)
    compact = 4 * math.pi * A / P**2
    c = V - V.mean(-2, keepdim=True)
    cov = c.transpose(-1, -2) @ c / V.shape[-2]
    ev = torch.linalg.eigvalsh(cov).clamp_min(1e-6)
    elong = torch.log(ev[..., 1] / ev[..., 0])
    r = (V - polygon_centroid(V).unsqueeze(-2)).norm(dim=-1)
    F = torch.fft.rfft(r / r.mean(-1, keepdim=True).clamp_min(1e-6), dim=-1).abs()[..., 1:n_fourier + 1]
    if F.shape[-1] < n_fourier:
        F = torch.cat([F, F.new_zeros(*F.shape[:-1], n_fourier - F.shape[-1])], -1)
    return torch.cat([torch.log(A)[..., None], torch.log(P)[..., None], compact[..., None], elong[..., None],
                      F / V.shape[-2]], -1)


def min_boundary_gap(V: Tensor, k: int = 16) -> Tensor:
    """Approximate pairwise minimum boundary distance between polygons; [M, N, 2] -> [M, M]."""
    pts, _ = boundary_samples(V, k)  # [M,k,2]
    pts = torch.cat([pts, V], 1)
    d = torch.cdist(pts.reshape(-1, 2), pts.reshape(-1, 2)).reshape(V.shape[0], pts.shape[1], V.shape[0], pts.shape[1])
    return d.amin(dim=(1, 3))
