"""Coarse z-buffer visibility for prism scenes (shared by observability, GCL targets and the texture atlas)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .camera import Pose, project
from .polygon import polygon_centroid


@dataclass
class ZBuffer:
    zmin: Tensor   # [h*w] nearest depth
    zid: Tensor    # [h*w] id of the nearest prism (-1 = none)
    h: int
    w: int
    stride: int


def prism_surface_samples(polys: Tensor, base: Tensor, height: Tensor, n_edge: int = 3, n_z: int = 5) -> Tensor:
    """Dense points on walls (edge x height) and roof of each prism. -> [N, P, 3]."""
    N, Nv, _ = polys.shape
    dev = polys.device
    te = torch.linspace(0, 1, n_edge + 1, device=dev)[:-1]
    tz = torch.linspace(0, 1, n_z, device=dev)
    a, b = polys, polys.roll(-1, 1)
    exy = a[:, :, None] + (b - a)[:, :, None] * te[None, None, :, None]  # [N,Nv,E,2]
    z = (base[:, None, None, None] + tz[None, None, None] * height[:, None, None, None])  # [N,1,1,Z]
    wall = torch.cat([exy[:, :, :, None].expand(N, Nv, n_edge, n_z, 2),
                      z[..., None].expand(N, Nv, n_edge, n_z, 1)], -1).reshape(N, -1, 3)
    cen = polygon_centroid(polys)
    rxy = torch.cat([cen[:, None], cen[:, None] + 0.5 * (polys - cen[:, None])], 1)
    roof = torch.cat([rxy, (base + height)[:, None, None].expand(N, rxy.shape[1], 1)], -1)
    return torch.cat([wall, roof], 1)


def build_zbuffer(polys: Tensor, base: Tensor, height: Tensor, pose: Pose, hw: tuple[int, int],
                  stride: int = 8) -> ZBuffer:
    H, W = hw
    h, w = (H + stride - 1) // stride, (W + stride - 1) // stride
    dev = pose.R_cw.device
    zmin = torch.full((h * w,), float("inf"), device=dev)
    zid = torch.full((h * w,), -1, dtype=torch.long, device=dev)
    if polys.shape[0] == 0:
        return ZBuffer(zmin, zid, h, w, stride)
    pts = prism_surface_samples(polys, base, height)
    ids = torch.arange(polys.shape[0], device=dev)[:, None].expand(-1, pts.shape[1]).reshape(-1)
    uv, dep = project(pts.reshape(-1, 3), pose)
    px, py = (uv[:, 0] / stride).floor().long(), (uv[:, 1] / stride).floor().long()
    ok = (dep > 0.5) & (px >= 0) & (px < w) & (py >= 0) & (py < h)
    lin = py[ok] * w + px[ok]
    zmin = zmin.scatter_reduce(0, lin, dep[ok], "amin")
    front = dep[ok] <= zmin[lin] + 1e-4
    zid[lin[front]] = ids[ok][front]
    return ZBuffer(zmin, zid, h, w, stride)


def point_visibility(X: Tensor, pose: Pose, hw: tuple[int, int], zb: ZBuffer, owner: Tensor | None = None,
                     own_ids: Tensor | None = None, tol: float = 2.0) -> Tensor:
    """Visibility of world points X [..., 3].

    A point is visible if it projects inside the image, in front of the camera, and either no surface is
    nearer (up to `tol`) or the nearest surface belongs to the point's own prism (`own_ids`, broadcast to X[..., 0]).
    `owner` maps z-buffer prism ids to the id space of `own_ids`.
    """
    H, W = hw
    uv, dep = project(X, pose)
    inimg = (dep > 0.5) & (uv[..., 0] >= 0) & (uv[..., 0] < W) & (uv[..., 1] >= 0) & (uv[..., 1] < H)
    px = (uv[..., 0] / zb.stride).floor().long().clamp(0, zb.w - 1)
    py = (uv[..., 1] / zb.stride).floor().long().clamp(0, zb.h - 1)
    lin = py * zb.w + px
    clear = dep <= zb.zmin[lin] + tol
    if own_ids is not None:
        zi = zb.zid[lin]
        if owner is not None and owner.numel():
            zi = torch.where(zi >= 0, owner[zi.clamp_min(0)], zi)
        clear = clear | ((zi == own_ids) & (own_ids >= 0))
    return inimg & clear
