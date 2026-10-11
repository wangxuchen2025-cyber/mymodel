"""Oracle test of CARR's geometric chain (lifting -> yaw-rotated templates -> correlation -> pose correction).

Learned feature heads are bypassed: raw RGB of a rendered textured ground is used both for the oblique image
and the ortho reference, so a correct implementation must recover the injected planar rigid error.
"""

import math
from dataclasses import replace

import numpy as np
import torch
import torch.nn.functional as F

from marscd.config import MARSConfig
from marscd.data.synthetic import _ground_color, render_oblique
from marscd.geometry import apply_world_rigid2d, ground_intersection, pose_from_gnss
from marscd.models.registration import CARR


def test_carr_correlation_recovers_planar_error():
    cfg = replace(MARSConfig.tiny(), reg_window=64, reg_search=8, reg_yaw_bins=9, reg_yaw_range=math.radians(8))
    carr = CARR(cfg)
    H, W, f = 160, 256, 180.0
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    prior = pose_from_gnss(torch.tensor([10.0, -110.0, 80.0]), torch.tensor(0.1), torch.tensor(math.radians(-40)),
                           torch.tensor(0.0), K)
    c = ground_intersection(prior, K[:2, 2], 0.0)[:2]
    a_true, d_true = math.radians(3.0), torch.tensor([2.5, -1.5])
    gt = apply_world_rigid2d(prior, torch.tensor(a_true), c, d_true)
    img = render_oblique([], [], [], gt, H, W, 1.0, np.random.default_rng(0))  # ground only

    r, S, R = 1.0, cfg.reg_window, cfg.reg_search
    cell_xy = CARR._window(c, S, r)
    mean = img.mean((1, 2), keepdim=True)
    T, valid = carr._lift(img - mean, prior, cell_xy, 0.0, (H, W))
    T = F.normalize(T, dim=0)
    ref_xy = CARR._window(c, S + 2 * R, r).numpy()
    ref = torch.from_numpy(_ground_color(ref_xy[..., 0], ref_xy[..., 1])).float().permute(2, 0, 1) - mean
    ref = F.normalize(ref, dim=0)
    log_post, alphas, alpha, delta, var = carr.correlate(T, valid[None].float(), ref, cell_xy, c, r,
                                                         torch.tensor([100.0, 100.0, 1.0]), torch.tensor(20.0))
    assert abs(float(alpha) - a_true) < math.radians(1.0), float(alpha)
    assert (delta - d_true).abs().max() < 0.75, delta
    corrected = apply_world_rigid2d(prior, alpha, c, delta)
    assert (corrected.center - gt.center)[:2].norm() < 0.5 * (prior.center - gt.center)[:2].norm()
