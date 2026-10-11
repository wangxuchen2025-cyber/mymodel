"""Oracle tests of the v2/v3 mechanisms in isolation (each test checks the claim made in its derivation)."""

import math

import numpy as np
import torch

from marscd.config import MARSConfig
from marscd.geometry import polygon_centroid, pose_from_gnss, resample_polygon
from marscd.geometry.visibility import build_zbuffer
from marscd.models.atlas import PrismAtlas
from marscd.models.evidential import accumulate, edl_beta_loss, opinion
from marscd.models.graph_matcher import decode_matches, log_optimal_transport
from marscd.models.relational_assignment import RelationalAssignment
from marscd.models.structure import HeightSweep, render_structure_targets

CFG = MARSConfig.tiny()
H, W, F_ = 160, 256, 180.0
K = torch.tensor([[F_, 0, W / 2], [0, F_, H / 2], [0, 0, 1.0]])


def _square(cx, cy, s=10.0):
    return resample_polygon([[cx - s / 2, cy - s / 2], [cx + s / 2, cy - s / 2], [cx + s / 2, cy + s / 2],
                             [cx - s / 2, cy + s / 2]], CFG.n_vertices)


def _orbit_pose(phi, radius=110.0, alt=80.0):
    C = torch.tensor([radius * math.cos(phi), radius * math.sin(phi), alt])
    yaw = torch.atan2(-C[0], -C[1])
    pitch = -torch.atan2(torch.tensor(alt), torch.tensor(radius))
    return pose_from_gnss(C, yaw, pitch, torch.tensor(0.0), K)


def test_contact_line_targets_follow_visibility():
    polys = torch.stack([_square(0, 0, 14)])
    base, h = torch.zeros(1), torch.tensor([20.0])
    pose = _orbit_pose(-math.pi / 2)  # camera south of the building, looking north
    t = render_structure_targets(polys, base, h, pose, (H, W))
    assert t[0].sum() > 0 and t[1].sum() > 0
    # only the south-facing wall's contact line is visible: its pixels lie below the roof-edge pixels
    ys_c = t[0].nonzero()[:, 0].float().mean()
    ys_r = t[1].nonzero()[:, 0].float().mean()
    assert ys_c > ys_r


def test_height_sweep_recovers_height_with_oracle_roof_edges():
    sweep = HeightSweep(CFG, n_bins=CFG.height_bins)
    polys = torch.stack([_square(-20, 0, 12), _square(15, 10, 10)])
    base, h_true = torch.zeros(2), torch.tensor([9.0, 27.0])
    L = torch.zeros(2, sweep.n)
    for phi in np.linspace(-2.2, -0.9, 5):
        pose = _orbit_pose(float(phi))
        roof = render_structure_targets(polys, base, h_true, pose, (H, W))[1]
        logits = (roof - 0.5) * 8.0  # oracle roof-edge detector
        L = L + sweep.increment(logits, polys, base, pose, (H, W))
    mu, sd = sweep.estimate(L)
    assert ((mu - h_true).abs() / h_true < 0.2).all(), (mu, h_true)


def test_atlas_weights_only_front_facing_visible_walls():
    atlas = PrismAtlas(CFG)
    polys = torch.stack([_square(0, 0, 14)])
    base, h = torch.zeros(1), torch.tensor([20.0])
    pose = _orbit_pose(-math.pi / 2)
    zb = build_zbuffer(polys, base, h, pose, (H, W))
    fmap = torch.randn(1, CFG.d_model, H // 8, W // 8)
    X, Nn, _ = atlas.cells(polys, base, h)
    _, w = atlas.observe(fmap, polys, base, h, pose, (H, W), zb, torch.zeros(1, dtype=torch.long))
    wall_w = w[0, :atlas.n_wall]
    facing = (Nn[0, :atlas.n_wall, :2] * (pose.center[:2] - X[0, :atlas.n_wall, :2])).sum(-1) > 0
    assert (wall_w[~facing] == 0).all() and (wall_w[facing] > 0).any()
    assert (w[0, atlas.n_wall:] > 0).any()  # roof seen from above


def test_hsr_spatial_relations_fix_ambiguous_unary():
    """5x5 block of identical buildings, residual misregistration, noisy appearance scores, one demolition
    and one new building: second-order relational reasoning must beat unary-only optimal transport."""
    torch.manual_seed(0)
    hsr = RelationalAssignment(CFG)
    pa = torch.stack([_square(x, y) for x in range(0, 100, 22) for y in range(0, 100, 22)])
    M = pa.shape[0]
    R = torch.tensor([[math.cos(0.03), -math.sin(0.03)], [math.sin(0.03), math.cos(0.03)]])
    pb = pa @ R.t() + torch.tensor([4.0, -3.0])
    keep = torch.ones(M, dtype=torch.bool)
    keep[12] = False  # demolished
    pb = torch.cat([pb[keep], _square(120, 40)[None]])
    gt = torch.full((M,), -1, dtype=torch.long)
    gt[keep] = torch.arange(int(keep.sum()))
    N = pb.shape[0]
    cA, cB = polygon_centroid(pa), polygon_centroid(pb)
    d = torch.cdist(cA, cB)
    S = torch.randn(M, N) * 1.2                       # appearance is ambiguous for identical buildings
    S[gt >= 0, gt[gt >= 0]] += 1.0                     # weakly informative correct pairs
    S = S - 0.02 * d                                   # weak proximity cue (pose uncertainty ~ metres)
    gate = d < 60
    S = S.masked_fill(~gate, -1e4)
    alpha, beta, corner = torch.zeros(M), torch.zeros(N), torch.tensor(0.0)
    out = hsr(S, alpha, beta, corner, gate, pa, pb)
    m0, _ = decode_matches(log_optimal_transport(S, alpha, beta, corner, 30), 0.2)
    m1, _ = decode_matches(out["log_P"], 0.2)
    acc0 = (m0 == gt).float().mean()
    acc1 = (m1 == gt).float().mean()
    assert acc1 > acc0 + 0.5 and acc1 >= 0.9, (acc0, acc1)
    assert int(m1[12]) == -1  # the demolished building goes to the dustbin (change = outlier class)


def test_evidential_uncertainty_and_loss():
    r, s = torch.zeros(3), torch.zeros(3)
    idx = torch.tensor([0, 1])
    for _ in range(6):  # instance 0: well observed & empty; 1: barely observed; 2: never observed
        r, s = accumulate(r, s, idx, torch.tensor([-4.0, -4.0]), torch.tensor([1.0, 0.05]))
    b, dis, u = opinion(r, s)
    assert dis[0] > 0.7 and u[0] < 0.3 and u[1] > 0.6 and u[2] == 1.0
    logit = torch.tensor([2.0], requires_grad=True)
    r1, s1 = accumulate(torch.zeros(1), torch.zeros(1), torch.tensor([0]), logit, torch.ones(1))
    loss = edl_beta_loss(r1, s1, torch.zeros(1))  # GT: empty -> gradient must push the logit down
    loss.backward()
    assert logit.grad > 0


def _gcl_registration(channel: int, seed: int = 0, hw=(544, 960), f=768.0):
    """Register a texture-less scene using ONLY structure lines vs ground-level SHP boundaries (image space)."""
    from dataclasses import replace

    from marscd.geometry import apply_world_rigid2d, ground_intersection
    from marscd.models.registration import CARR
    from marscd.models.structure import structural_score_volume

    g = torch.Generator().manual_seed(seed)
    cfg = replace(CFG, reg_window=96, reg_search=8, reg_yaw_bins=9, reg_yaw_range=math.radians(8))
    carr = CARR(cfg)
    polys, hts = [], []
    for x in range(-60, 61, 30):
        for y in range(-40, 41, 25):
            w, l = 8 + 8 * torch.rand(2, generator=g)
            polys.append(resample_polygon([[x - w / 2, y - l / 2], [x + w / 2, y - l / 2], [x + w / 2, y + l / 2],
                                           [x - w / 2, y + l / 2]], CFG.n_vertices))
            hts.append(float(8 + 30 * torch.rand(1, generator=g)))
    polys, hts = torch.stack(polys), torch.tensor(hts)
    base = torch.zeros(len(polys))
    Hh, Ww = hw
    Kd = torch.tensor([[f, 0, Ww / 2], [0, f, Hh / 2], [0, 0, 1.0]])
    prior = pose_from_gnss(torch.tensor([10.0, -130.0, 80.0]), torch.tensor(0.05), torch.tensor(math.radians(-38)),
                           torch.tensor(0.0), Kd)
    c = ground_intersection(prior, Kd[:2, 2], 0.0)[:2]
    a_true, d_true = math.radians(2.5), torch.tensor([3.0, -2.0])
    gt = apply_world_rigid2d(prior, torch.tensor(a_true), c, d_true)
    maps = render_structure_targets(polys, base, hts, gt, hw)  # oracle detector output at the TRUE pose
    logits = (maps[channel] - 0.5) * 8.0
    R, r, S = cfg.reg_search, 1.0, cfg.reg_window
    offsets = (torch.arange(2 * R + 1, dtype=torch.float32) - R) * r
    vol = structural_score_volume(logits, polys, torch.zeros(len(polys)), prior, hw, c, carr.alphas(c.device, c.dtype),
                                  offsets)
    cell_xy = CARR._window(c, S, r)
    _, _, alpha, delta, _ = carr.correlate(torch.zeros(cfg.match_dim, S, S), torch.zeros(1, S, S),
                                           torch.zeros(cfg.match_dim, S + 2 * R, S + 2 * R), cell_xy, c, r,
                                           torch.tensor([100.0, 100.0, 1.0]), torch.tensor(1.0), vol, torch.tensor(30.0))
    return (delta - d_true).norm().item(), abs(float(alpha) - a_true) * 180 / math.pi


def test_contact_lines_register_textureless_scene_roof_edges_are_biased():
    e_gcl, a_gcl = _gcl_registration(channel=0)
    e_roof, a_roof = _gcl_registration(channel=1)
    assert e_gcl < 1.0 and a_gcl < 1.0, (e_gcl, a_gcl)
    assert e_roof > 2 * e_gcl, (e_roof, e_gcl)  # relief displacement biases height-dependent evidence


def test_novelty_discount_prevents_overconfidence_from_correlated_frames():
    from marscd.models.evidential import novelty
    n_cells = 10
    # (a) 8 near-identical consecutive frames seeing the same 3 cells; (b) 8 frames each seeing new surface
    ra = sa = rb = sb = torch.zeros(1)
    Wa = torch.zeros(1, n_cells)
    Wb = torch.zeros(1, n_cells)
    for t in range(8):
        dWa = torch.zeros(1, n_cells)
        dWa[0, :3] = 1.0
        dWb = torch.zeros(1, n_cells)
        dWb[0, t % n_cells] = 1.0
        na, nb = novelty(dWa, Wa), novelty(dWb, Wb)
        ra, sa = accumulate(ra, sa, torch.tensor([0]), torch.tensor([-3.0]), na)
        rb, sb = accumulate(rb, sb, torch.tensor([0]), torch.tensor([-3.0]), nb)
        Wa, Wb = Wa + dWa, Wb + dWb
    ua, ub = opinion(ra, sa)[2], opinion(rb, sb)[2]
    assert ua > 1.8 * ub  # correlated frames leave substantially more uncertainty than diverse views
