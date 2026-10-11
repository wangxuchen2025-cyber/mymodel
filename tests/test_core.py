"""Unit tests for the geometric core and the key mechanisms of MARS-CD."""

import math

import torch

from marscd.config import MARSConfig
from marscd.geometry import (
    Pose,
    apply_world_rigid2d,
    ground_intersection,
    points_in_polygons,
    polygon_area,
    polygon_iou_paired,
    pose_from_gnss,
    project,
    projection_jacobian,
    resample_polygon,
    signed_distance_to_polygon,
)
from marscd.losses import cyclic_polygon_l1, ot_nll, registration_targets
from marscd.models.change_head import ChangeReasoner, loo_local_rigid, rot2
from marscd.models.graph_matcher import log_optimal_transport
from marscd.models.marscd import TrackBank

K = torch.tensor([[200.0, 0, 128], [0, 200, 80], [0, 0, 1]])


def _pose(yaw=0.3, pitch=-0.7, roll=0.02, c=(5.0, -100.0, 80.0)):
    return pose_from_gnss(torch.tensor(c), torch.tensor(yaw), torch.tensor(pitch), torch.tensor(roll), K)


def test_project_backproject_roundtrip():
    pose = _pose()
    uv = torch.tensor([[30.0, 120.0], [128.0, 80.0], [220.0, 140.0]])
    X = ground_intersection(pose, uv, 0.0)
    uv2, depth = project(X, pose)
    assert torch.allclose(uv, uv2, atol=1e-3) and (depth > 0).all()


def test_projection_jacobian_numeric():
    torch.set_default_dtype(torch.float64)
    try:
        pose = _pose()
        pose = Pose(pose.R_cw.double(), pose.t_cw.double(), K.double())
        X = torch.randn(6, 3) * 10 + torch.tensor([0.0, -20.0, 0.0])
        J = projection_jacobian(pose.to_camera(X), pose.K)
        num = []
        for i in range(6):
            d = torch.zeros(6)
            d[i] = 1e-6
            num.append((project(X, pose.left_update(d))[0] - project(X, pose.left_update(-d))[0]) / 2e-6)
        assert ((J - torch.stack(num, -1)).abs().max() / J.abs().max()) < 1e-6
    finally:
        torch.set_default_dtype(torch.float32)


def test_registration_target_inverts_world_rigid():
    prior = _pose()
    c = torch.tensor([3.0, -20.0])
    gt = apply_world_rigid2d(prior, torch.tensor(0.05), c, torch.tensor([2.0, -1.5]))
    a, d = registration_targets(prior, gt, c)
    assert abs(float(a) - 0.05) < 1e-5 and torch.allclose(d, torch.tensor([2.0, -1.5]), atol=1e-4)


def test_polygon_ops():
    sq = resample_polygon([[0, 0], [10, 0], [10, 5], [0, 5]], 8)
    assert abs(float(polygon_area(sq)) - 50.0) < 1e-4
    inside = points_in_polygons(torch.tensor([[5.0, 2.5], [11.0, 1.0]]), sq[None])
    assert inside.tolist() == [[True, False]]
    sd = signed_distance_to_polygon(torch.tensor([[[5.0, 2.5], [12.0, 2.5]]]), sq[None])
    assert torch.allclose(sd, torch.tensor([[-2.5, 2.0]]), atol=1e-4)
    iou = polygon_iou_paired(sq[None], (sq + torch.tensor([5.0, 0]))[None], res=64)
    assert abs(float(iou) - 1 / 3) < 0.03


def test_cyclic_polygon_loss_is_shift_invariant():
    sq = resample_polygon([[0, 0], [10, 0], [10, 5], [0, 5]], 8)
    l, _ = cyclic_polygon_l1(sq[None], sq.roll(3, 0)[None])
    assert float(l) < 1e-6


def test_sinkhorn_marginals():
    torch.manual_seed(0)
    M, N = 5, 7
    lp = log_optimal_transport(torch.randn(M, N), torch.randn(M), torch.randn(N), torch.tensor(0.0), 200)
    P = lp.exp()
    assert torch.allclose(P[:M].sum(1), torch.ones(M), atol=1e-3)
    assert torch.allclose(P[:, :N].sum(0), torch.ones(N), atol=1e-3)
    assert float(ot_nll(lp, torch.tensor([0, -1, 2, -1, 4]))) > 0


def test_loo_rigid_separates_misregistration_from_expansion():
    """A shared rigid offset is absorbed; a building-specific annex is NOT."""
    torch.manual_seed(0)
    cfg = MARSConfig.tiny()
    reasoner = ChangeReasoner(cfg)
    base = resample_polygon([[0, 0], [10, 0], [10, 10], [0, 10]], cfg.n_vertices)
    offsets = torch.tensor([[0, 0], [30, 0], [0, 30], [30, 30], [60, 0], [60, 30]], dtype=torch.float32)
    A = base[None] + offsets[:, None]
    R = rot2(torch.tensor(0.02))
    B = torch.einsum("ab,pnb->pna", R, A) + torch.tensor([1.5, -1.0])  # misregistration
    # building 0 additionally expands by 5 m to the east
    B0 = B[0].clone()
    east = B0[:, 0] > B0[:, 0].mean()
    B0[east, 0] += 5.0
    B[0] = B0
    geo_raw = reasoner.geo_features(A, B)
    out = reasoner(torch.zeros(6, cfg.d_model), torch.zeros(6, cfg.d_model), A, B, torch.ones(6))
    geo = out["geo"]
    # after LOO rigid removal unchanged buildings align well, the expanded one keeps its residual
    assert geo[1:, 8].min() > 0.95 > 0.8 > geo_raw[1:, 8].max()
    assert torch.allclose(out["t"][1], torch.tensor([1.5, -1.0]), atol=0.15)
    assert geo[0, 4] > 1.0 and geo[0, 9] > 0.3


def test_trackbank_atlas_accumulates():
    tb = TrackBank(4, 8, 6, 3, 5, "cpu")
    poly = resample_polygon([[0, 0], [10, 0], [10, 10], [0, 10]], 8)[None]
    z = torch.zeros(1)
    tb.add(torch.ones(1, 4), poly, z, z + 10, z, torch.tensor([-1]), torch.tensor([-1]), torch.ones(1, 6, 3),
           torch.ones(1, 6), torch.zeros(1, 5))
    idx = torch.tensor([0, 0])  # duplicate updates in one frame (track + merged detection) must both count
    tb.update(idx, torch.ones(2, 4), poly.expand(2, -1, -1), z.expand(2), (z + 10).expand(2), z.expand(2), torch.ones(2),
              torch.ones(2, 6, 3) * 2, torch.ones(2, 6), torch.zeros(2, 5))
    assert torch.allclose(tb.atlas_w[0], torch.full((6,), 3.0)) and torch.allclose(tb.atlas_sum[0], torch.full((6, 3), 5.0))
