"""Procedural synthetic scenes for unit/smoke testing of the full MARS-CD pipeline.

A small town of prism buildings on a flat ground with a road grid. The T2
epoch is derived from T1 by demolitions, planimetric expansions (annexes ->
L-shapes), shrinkages (partial demolition) and new constructions (including
rebuilds on demolished lots). We render:
  * T1 ortho rasters: satellite-like DOM (with shadows), aerial DOM, raster map;
  * oblique UAV frames of both epochs along different orbits, with true
    perspective / relief displacement and per-face shading;
  * consumer-grade GNSS: slowly varying bias + white noise on position/yaw.

This is NOT a substitute for real data; it exercises every code path with
geometrically consistent ground truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, box
from shapely.affinity import rotate, translate

from ..geometry import Pose, pose_from_gnss, resample_polygon
from ..models.marscd import SequenceTargets
from ..models.ortho_encoder import GeoGrid
from ..models.registration import GNSSReading

ROADS = (-42.0, 0.0, 42.0)
ROAD_W = 7.0
HALF = 80.0


@dataclass
class SynthScene:
    t1_polys: torch.Tensor
    t1_heights: torch.Tensor
    t2_polys: torch.Tensor
    t2_heights: torch.Tensor
    t1_to_t2: torch.Tensor
    t1_change: torch.Tensor
    t2_is_new: torch.Tensor
    rasters: dict
    grid: GeoGrid
    t1_video: list
    t1_poses: list
    t2_video: list
    t2_poses: list

    def t1_targets(self) -> SequenceTargets:
        M = self.t1_polys.shape[0]
        return SequenceTargets(self.t1_polys, self.t1_heights, torch.arange(M), torch.zeros(M, dtype=torch.long),
                               self.t1_poses)

    def t2_targets(self) -> SequenceTargets:
        return SequenceTargets(self.t2_polys, self.t2_heights, self.t1_to_t2, self.t1_change, self.t2_poses)


# ----------------------------------------------------------------------------- layout
def _blocks():
    edges = [-HALF] + [r + s * ROAD_W / 2 for r in ROADS for s in (-1, 1)] + [HALF]
    xs = [(edges[i], edges[i + 1]) for i in range(0, len(edges), 2)]
    return [(x0, x1, y0, y1) for x0, x1 in xs for y0, y1 in xs]


def _rand_building(rng, blk, occupied, tries=30):
    x0, x1, y0, y1 = blk
    for _ in range(tries):
        w, l = rng.uniform(8, 15), rng.uniform(8, 18)
        th = rng.uniform(-8, 8)
        cx, cy = rng.uniform(x0 + 6, x1 - 6), rng.uniform(y0 + 6, y1 - 6)
        p = rotate(translate(box(-w / 2, -l / 2, w / 2, l / 2), cx, cy), th, origin=(cx, cy))
        if not box(x0 + 1.5, y0 + 1.5, x1 - 1.5, y1 - 1.5).contains(p):
            continue
        if any(p.buffer(2.0).intersects(o) for o in occupied):
            continue
        return p
    return None


def _largest(g):
    if g.geom_type == "MultiPolygon":
        g = max(g.geoms, key=lambda q: q.area)
    return g


def _expand(rng, p: Polygon) -> Polygon:
    """Attach an annex rectangle to one side (planimetric expansion)."""
    mrr = p.minimum_rotated_rectangle
    c = list(mrr.exterior.coords)[:4]
    i = rng.integers(4)
    a, b = np.array(c[i]), np.array(c[(i + 1) % 4])
    e = b - a
    L = np.linalg.norm(e)
    n = np.array([e[1], -e[0]]) / L
    if Polygon(c).exterior.is_ccw:
        pass
    else:
        n = -n
    frac = rng.uniform(0.5, 1.0)
    depth = rng.uniform(4, 8)
    s0 = rng.uniform(0, 1 - frac)
    p0 = a + e * s0
    p1 = a + e * (s0 + frac)
    annex = Polygon([p0, p1, p1 + n * depth, p0 + n * depth])
    return _largest(p.union(annex.buffer(0.01)).simplify(0.05))


def _shrink(rng, p: Polygon) -> Polygon:
    """Remove a slab from one end (partial demolition)."""
    mrr = p.minimum_rotated_rectangle
    c = np.array(list(mrr.exterior.coords)[:4])
    i = rng.integers(4)
    a, b, d = c[i], c[(i + 1) % 4], c[(i + 3) % 4]
    cut = rng.uniform(0.3, 0.45)
    q = Polygon([a, b, b + (d - a) * cut, a + (d - a) * cut]).buffer(0.01)
    return _largest(p.difference(q).simplify(0.05))


def _ring(p: Polygon, nv: int) -> torch.Tensor:
    return resample_polygon(np.asarray(p.exterior.coords)[:-1], nv)


# ----------------------------------------------------------------------------- rendering
def _ground_color(x, y):
    road = np.zeros_like(x, dtype=bool)
    for r in ROADS:
        road |= (np.abs(x - r) < ROAD_W / 2) | (np.abs(y - r) < ROAD_W / 2)
    tex = 0.5 + 0.25 * np.sin(0.37 * x) * np.sin(0.29 * y) + 0.15 * np.sin(0.11 * x + 0.07 * y)
    grass = np.stack([0.30 + 0.10 * tex, 0.42 + 0.12 * tex, 0.22 + 0.06 * tex], -1)
    dash = (np.abs(np.sin(0.5 * (x + y))) > 0.8)
    asphalt = np.stack([0.38 + 0.1 * dash, 0.38 + 0.1 * dash, 0.40 + 0.1 * dash], -1)
    col = np.where(road[..., None], asphalt, grass)
    far = (np.abs(x) > HALF + 30) | (np.abs(y) > HALF + 30)
    return np.where(far[..., None], col * 0.85, col)


def render_oblique(polys, heights, colors, pose: Pose, H: int, W: int, illum: float, rng) -> torch.Tensor:
    K, Rwc, C = pose.K.numpy(), pose.R_wc.numpy(), pose.center.numpy()
    v, u = np.mgrid[0:H, 0:W] + 0.5
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1) @ Rwc.T
    down = rays[..., 2] < -1e-3
    s = np.where(down, -C[2] / np.where(down, rays[..., 2], -1), 0)
    gx, gy = C[0] + s * rays[..., 0], C[1] + s * rays[..., 1]
    img = _ground_color(gx, gy)
    img[~down] = np.array([0.70, 0.80, 0.92])
    pim = Image.fromarray((np.clip(img * illum, 0, 1) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pim)
    light = np.array([0.6, 0.5, 0.62])
    faces = []
    for P, h, col in zip(polys, heights, colors):
        P = np.asarray(P)
        n = len(P)
        for i in range(n):
            a, b = P[i], P[(i + 1) % n]
            quad = np.array([[*a, 0], [*b, 0], [*b, h], [*a, h]])
            e = b - a
            nrm = np.array([e[1], -e[0], 0.0]) / (np.linalg.norm(e) + 1e-9)
            ctr = quad.mean(0)
            if nrm @ (C - ctr) <= 0:
                continue
            shade = 0.55 + 0.4 * max(0.0, float(nrm @ light))
            faces.append((np.linalg.norm(C - ctr), quad, np.asarray(col) * 0.8 * shade))
        roof = np.concatenate([P, np.full((n, 1), h)], 1)
        faces.append((np.linalg.norm(C - roof.mean(0)) - 1e-3, roof, np.asarray(col)))
    faces.sort(key=lambda f: -f[0])
    for _, X, col in faces:
        Xc = (X - C) @ Rwc
        if (Xc[:, 2] < 0.5).any():
            continue
        uv = Xc[:, :2] / Xc[:, 2:] * np.array([K[0, 0], K[1, 1]]) + K[:2, 2]
        draw.polygon([tuple(p) for p in uv], fill=tuple(int(c) for c in np.clip(col * illum, 0, 1) * 255))
    arr = np.asarray(pim).astype(np.float32) / 255.0
    arr = np.clip(arr + rng.normal(0, 0.015, arr.shape), 0, 1)
    return torch.from_numpy(arr).permute(2, 0, 1).float()


def render_rasters(polys, heights, colors, gsd: float):
    n = int(round(2 * HALF / gsd))
    v, u = np.mgrid[0:n, 0:n] + 0.5
    x, y = -HALF + u * gsd, HALF - v * gsd
    to_px = lambda P: [((px + HALF) / gsd, (HALF - py) / gsd) for px, py in np.asarray(P)]  # noqa: E731
    out = {}
    # satellite-like DOM (with cast shadows)
    sat = Image.fromarray((np.clip(_ground_color(x, y) * 0.95, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(sat)
    for P, h in zip(polys, heights):
        d.polygon(to_px(np.asarray(P) + np.array([0.25, -0.25]) * h), fill=(40, 45, 40))
    for P, col in zip(polys, colors):
        d.polygon(to_px(P), fill=tuple(int(c * 200) for c in col))
    out["sat"] = sat
    # aerial DOM (different radiometry, no shadows)
    aer = Image.fromarray((np.clip(_ground_color(x, y) * np.array([1.05, 0.95, 1.1]), 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(aer)
    for P, col in zip(polys, colors):
        d.polygon(to_px(P), fill=tuple(int(min(1, c * 1.1) * 255) for c in col))
    out["aerial"] = aer
    # raster map (cartographic symbology)
    road = np.zeros_like(x, dtype=bool)
    for r in ROADS:
        road |= (np.abs(x - r) < ROAD_W / 2) | (np.abs(y - r) < ROAD_W / 2)
    base = np.where(road[..., None], np.array([1.0, 1.0, 1.0]), np.array([0.96, 0.93, 0.85]))
    mp = Image.fromarray((base * 255).astype(np.uint8))
    d = ImageDraw.Draw(mp)
    for P in polys:
        d.polygon(to_px(P), fill=(225, 180, 180), outline=(90, 60, 60))
    out["map"] = mp
    return {k: torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0).permute(2, 0, 1) for k, im in out.items()}, \
        GeoGrid(-HALF, HALF, gsd, n, n)


# ----------------------------------------------------------------------------- trajectories
def _orbit(rng, n, H, W, f, radius, alt, phi0, dphi, bias, noise):
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    poses, gnss = [], []
    for k in range(n):
        phi = phi0 + k * dphi
        C = np.array([radius * math.cos(phi), radius * math.sin(phi), alt])
        look = rng.normal(0, 8, 2)
        v = look - C[:2]
        yaw = math.atan2(v[0], v[1])
        pitch = -math.atan2(alt, np.linalg.norm(v)) + rng.normal(0, 0.02)
        roll = rng.normal(0, 0.01)
        pose = pose_from_gnss(torch.tensor(C, dtype=torch.float32), torch.tensor(yaw), torch.tensor(pitch),
                              torch.tensor(roll), K)
        pos_m = C + np.array([bias[0], bias[1], bias[2]]) + rng.normal(0, noise[0], 3) * np.array([1, 1, 0.5])
        g = GNSSReading(torch.tensor(pos_m, dtype=torch.float32), torch.tensor(yaw + bias[3] + rng.normal(0, noise[1])),
                        torch.tensor(pitch + rng.normal(0, noise[1])), torch.tensor(roll + rng.normal(0, noise[1])))
        poses.append(pose)
        gnss.append(g)
    return poses, gnss, K


def make_scene(seed: int = 0, nv: int = 8, gsd: float = 0.5, H: int = 160, W: int = 256, f: float = 180.0,
               n_t1: int = 5, n_t2: int = 7, gnss_bias_m: float = 2.5, gnss_bias_deg: float = 2.5) -> SynthScene:
    rng = np.random.default_rng(seed)
    occupied, t1, hts, cols = [], [], [], []
    for blk in _blocks():
        for _ in range(rng.integers(1, 3)):
            p = _rand_building(rng, blk, occupied)
            if p is not None:
                occupied.append(p)
                t1.append(p)
                hts.append(rng.uniform(6, 30))
                cols.append(rng.uniform(0.25, 0.95, 3))
    M = len(t1)
    t2, t2h, t2c, t1_to_t2, change, is_new = [], [], [], [], [], []
    demolished_sites = []
    for j in range(M):
        r = rng.uniform()
        if r < 0.15:
            t1_to_t2.append(-1)
            change.append(3)
            demolished_sites.append(t1[j])
            continue
        if r < 0.30:
            g, c = _expand(rng, t1[j]), 1
        elif r < 0.40:
            g, c = _shrink(rng, t1[j]), 2
        else:
            g, c = t1[j], 0
        t1_to_t2.append(len(t2))
        change.append(c)
        t2.append(g)
        t2h.append(hts[j])
        t2c.append(cols[j])
        is_new.append(False)
    occ2 = list(t2)
    blocks = _blocks()
    n_new = 0
    for site in demolished_sites:  # some demolished lots are rebuilt with a different building
        if rng.uniform() < 0.5:
            cx, cy = site.centroid.x, site.centroid.y
            blk = next((b for b in blocks if b[0] <= cx <= b[1] and b[2] <= cy <= b[3]), None)
            p = _rand_building(rng, blk, occ2) if blk else None
            if p is not None:
                occ2.append(p)
                t2.append(p)
                t2h.append(rng.uniform(6, 30))
                t2c.append(rng.uniform(0.25, 0.95, 3))
                is_new.append(True)
                n_new += 1
    for _ in range(40):
        if n_new >= 3:
            break
        p = _rand_building(rng, blocks[rng.integers(len(blocks))], occ2)
        if p is not None:
            occ2.append(p)
            t2.append(p)
            t2h.append(rng.uniform(6, 30))
            t2c.append(rng.uniform(0.25, 0.95, 3))
            is_new.append(True)
            n_new += 1

    t1_rings = [np.asarray(p.exterior.coords)[:-1] for p in t1]
    t2_rings = [np.asarray(p.exterior.coords)[:-1] for p in t2]
    rasters, grid = render_rasters(t1_rings, hts, cols, gsd)

    def video(rings, heights, colors, radius, alt, phi0, dphi, n, illum):
        bias = (*rng.normal(0, gnss_bias_m, 2), rng.normal(0, 1.0), math.radians(rng.normal(0, gnss_bias_deg)))
        poses, gnss, K = _orbit(rng, n, H, W, f, radius, alt, phi0, dphi, bias, (0.3, math.radians(0.3)))
        frames = [(render_oblique(rings, heights, colors, p, H, W, illum, rng), g, K) for p, g in zip(poses, gnss)]
        return frames, poses

    phi1 = rng.uniform(0, 2 * math.pi)
    t1_video, t1_poses = video(t1_rings, hts, cols, rng.uniform(95, 120), rng.uniform(65, 85), phi1,
                               math.radians(rng.uniform(55, 70)), n_t1, rng.uniform(0.85, 1.1))
    t2_video, t2_poses = video(t2_rings, t2h, t2c, rng.uniform(95, 120), rng.uniform(65, 85),
                               phi1 + math.radians(rng.uniform(20, 60)), math.radians(rng.uniform(45, 55)), n_t2,
                               rng.uniform(0.85, 1.1))
    stack = lambda rings: torch.stack([_ring(Polygon(r), nv) for r in rings]) if rings else torch.zeros(0, nv, 2)  # noqa: E731
    return SynthScene(
        t1_polys=stack(t1_rings), t1_heights=torch.tensor(hts, dtype=torch.float32),
        t2_polys=stack(t2_rings), t2_heights=torch.tensor(t2h, dtype=torch.float32),
        t1_to_t2=torch.tensor(t1_to_t2, dtype=torch.long), t1_change=torch.tensor(change, dtype=torch.long),
        t2_is_new=torch.tensor(is_new, dtype=torch.bool), rasters=rasters, grid=grid,
        t1_video=t1_video, t1_poses=t1_poses, t2_video=t2_video, t2_poses=t2_poses)
