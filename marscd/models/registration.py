"""Module C - Change-Aware Robust geo-Registration (CARR).

Consumer GNSS + gimbal angles give metre-level position error and degree-level
yaw error that is dominated by a slowly varying BIAS (multipath, magnetometer
offset, timestamp offset). CARR therefore estimates a per-frame correction
against the T1 geo-prior and tracks the bias with a Kalman filter:

  C1  ground lifting: image features are lifted onto the local ground plane
      under the prior pose, weighted by a learned "ground-contact" score;
  C2  exhaustive (x, y, yaw) BEV correlation against the cached T1 matching
      map, fused with the Gaussian bias prior (Bayesian posterior volume);
  C3  feature-metric Levenberg-Marquardt refinement of the full 6-DoF pose
      (unrolled, differentiable; PixLoc-style);
  C4  GNSS bias filter -> prior of the next frame.

Change awareness: both the correlation reference and the LM residuals are
down-weighted by (i) the learned change-stable structure map of T1 and
(ii) the CURRENT change beliefs of the stream (instances believed to be
new/demolished/modified are excluded) - a streaming EM coupling between
registration and change detection at zero extra cost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import MARSConfig
from ..geometry import Pose, apply_world_rigid2d, ground_intersection, points_in_polygons, polygon_centroid, pose_from_gnss, project
from ..geometry.camera import projection_jacobian
from .common import ConvBNAct, grid_sample_points
from .ortho_encoder import GeoPriorMemory, sample_bev
from .structure import structural_score_volume


@dataclass
class GNSSReading:
    position: Tensor  # [3] world (projected lat/lon + altitude above local ground)
    yaw: Tensor       # compass heading of the gimbal (rad)
    pitch: Tensor
    roll: Tensor


class GNSSBiasFilter:
    """Random-walk Kalman filter over the bias b = (bx, by, bz, b_yaw)."""

    def __init__(self, q: tuple, p0: tuple, device=None):
        self.b = torch.zeros(4, device=device)
        self.P = torch.diag(torch.tensor(p0, device=device) ** 2)
        self.Q = torch.diag(torch.tensor(q, device=device) ** 2)

    def predict(self) -> tuple[Tensor, Tensor]:
        self.P = self.P + self.Q
        return self.b, self.P.diagonal().sqrt()

    def update(self, z: Tensor, r_var: Tensor) -> None:
        z, r_var = z.detach(), r_var.detach()
        S = self.P + torch.diag(r_var)
        Kg = self.P @ torch.linalg.inv(S)
        innov = z - self.b
        innov[3] = torch.atan2(innov[3].sin(), innov[3].cos())
        self.b = self.b + Kg @ innov
        self.P = (torch.eye(4, device=self.P.device) - Kg) @ self.P

    def prior_pose(self, g: GNSSReading, K: Tensor) -> Pose:
        return pose_from_gnss(g.position + self.b[:3], g.yaw + self.b[3], g.pitch, g.roll, K)


@dataclass
class RegistrationOutput:
    pose: Pose
    pose_coarse: Pose
    pose_prior: Pose
    log_post: Tensor     # [n_yaw, 2R+1, 2R+1]
    alphas: Tensor       # [n_yaw] CCW world rotation hypotheses
    center: Tensor       # [2] window centre
    cell: float
    alpha: Tensor
    delta: Tensor        # [2]
    var: Tensor          # [3] posterior variances (x, y, yaw)
    lifted: Tensor       # [C, S, S] image content lifted to ground (refined pose)
    lifted_valid: Tensor  # [S, S]
    cell_xy: Tensor      # [S, S, 2]


def soft_union_weight(xy: Tensor, polys: Tensor | None, probs: Tensor | None) -> Tensor:
    """prod_k (1 - p_k * 1[xy in poly_k]) : down-weights areas believed to have changed."""
    if polys is None or polys.shape[0] == 0:
        return xy.new_ones(xy.shape[:-1])
    inside = points_in_polygons(xy.reshape(-1, 2), polys).float()  # [K, Q]
    w = torch.prod(1 - probs[:, None] * inside, 0)
    return w.reshape(xy.shape[:-1])


class CARR(nn.Module):
    def __init__(self, cfg: MARSConfig):
        super().__init__()
        C = cfg.d_model
        self.cfg = cfg
        self.img_match = nn.Conv2d(C, cfg.match_dim, 1)
        self.img_ground = nn.Sequential(ConvBNAct(C, C // 2, 3), nn.Conv2d(C // 2, 1, 1))
        self.img_fine = nn.Conv2d(C, cfg.fine_dim, 1)
        self.lift_refine = nn.Sequential(nn.Conv2d(cfg.match_dim, cfg.match_dim, 3, padding=1), nn.GELU(),
                                         nn.Conv2d(cfg.match_dim, cfg.match_dim, 3, padding=1))
        self.log_temp = nn.Parameter(torch.tensor(math.log(10.0)))
        self.log_temp_gcl = nn.Parameter(torch.tensor(math.log(10.0 * cfg.gcl_corr_weight)))
        self.log_lambda = nn.Parameter(torch.tensor(math.log(1e-2)))
        self.log_robust = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("dkernel", torch.tensor([-0.5, 0.0, 0.5]), persistent=False)

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _window(center: Tensor, n: int, r: float) -> Tensor:
        idx = (torch.arange(n, device=center.device, dtype=center.dtype) - (n - 1) / 2) * r
        ys, xs = torch.meshgrid(center[1] - idx, center[0] + idx, indexing="ij")
        return torch.stack([xs, ys], -1)  # [n, n, 2], row 0 = north

    def _lift(self, fmap: Tensor, pose: Pose, cell_xy: Tensor, ground_z: float, hw) -> tuple[Tensor, Tensor]:
        X = torch.cat([cell_xy, torch.full_like(cell_xy[..., :1], ground_z)], -1)
        uv, depth = project(X, pose)
        f, inb = grid_sample_points(fmap, uv, hw)
        valid = inb & (depth > 1.0)
        return f.permute(2, 0, 1) * valid, valid

    # ---------------------------------------------------------------- forward
    def forward(self, feats: list[Tensor], hw: tuple[int, int], prior: Pose, mem: GeoPriorMemory,
                ground_z: float, prior_sigma: Tensor, change_polys: Tensor | None = None,
                change_probs: Tensor | None = None, struct_logits: Tensor | None = None) -> RegistrationOutput:
        cfg = self.cfg
        P8 = feats[0][0]
        H, W = hw
        S, R, r = cfg.reg_window, cfg.reg_search, mem.grid.gsd
        dev, dt = P8.device, P8.dtype

        # window centre: ground point of the principal ray, kept within 1.5x altitude
        pp = prior.K[:2, 2]
        c = ground_intersection(prior, pp, ground_z)[:2]
        C_xy, alt = prior.center[:2], (prior.center[2] - ground_z).clamp_min(1.0)
        v = c - C_xy
        c = torch.where(v.norm() > 1.5 * alt, C_xy + v / v.norm().clamp_min(1e-6) * 1.5 * alt, c).detach()
        cell_xy = self._window(c, S, r)

        # C1 ground lifting
        m_img = F.normalize(self.img_match(P8.unsqueeze(0))[0], dim=0)
        g_img = torch.sigmoid(self.img_ground(P8.unsqueeze(0))[0])
        T, valid = self._lift(torch.cat([m_img, g_img]), prior, cell_xy, ground_z, hw)
        wT = T[-1:] * valid  # [1,S,S]
        T = F.normalize(self.lift_refine(T[:-1].unsqueeze(0))[0], dim=0)

        # reference crop with stability + change-aware weights
        ref_xy = self._window(c, S + 2 * R, r)
        ref = sample_bev(mem.bev_match, mem.grid, ref_xy).permute(2, 0, 1)
        ref_w = sample_bev(mem.stability, mem.grid, ref_xy)[..., 0] * soft_union_weight(ref_xy, change_polys, change_probs)
        ref = ref * ref_w

        # v2-I: height-free structural channel - image-space render-and-compare of camera-facing SHP
        # boundaries (at ground level) against the contact-line map; changed buildings are excluded.
        struct = None
        if struct_logits is not None:
            struct = self.structural_volume(struct_logits[0], mem, c, S, R, r, prior, hw, ground_z, change_polys, change_probs)

        # C2 rotated templates + correlation + Bayesian fusion with the bias prior
        log_post, alphas, alpha, delta, var = self.correlate(T, wT, ref, cell_xy, c, r, prior_sigma, self.log_temp.exp(),
                                                             struct, self.log_temp_gcl.exp())
        pose_c = apply_world_rigid2d(prior, alpha, c, delta)

        # C3 feature-metric LM refinement (6-DoF)
        pose_f = self._refine(P8, hw, pose_c, mem, cell_xy, ground_z, change_polys, change_probs)

        lifted, lv = self._lift(P8, pose_f.detach(), cell_xy, ground_z, hw)
        return RegistrationOutput(pose_f, pose_c, prior, log_post, alphas, c, r, alpha, delta, var, lifted, lv, cell_xy)

    def alphas(self, device, dtype) -> Tensor:
        cfg = self.cfg
        return torch.linspace(-cfg.reg_yaw_range, cfg.reg_yaw_range, cfg.reg_yaw_bins, device=device, dtype=dtype)

    def structural_volume(self, gcl_logits: Tensor, mem: GeoPriorMemory, c: Tensor, S: int, R: int, r: float,
                          prior: Pose, hw, ground_z: float, change_polys=None, change_probs=None) -> Tensor:
        half = (S / 2 + R) * r
        sel = (mem.polys - c).abs().amax((-1, -2)) < half
        polys = mem.polys[sel]
        if polys.shape[0] and change_polys is not None and change_polys.shape[0]:
            keep = soft_union_weight(polygon_centroid(polys), change_polys, change_probs) > 0.5
            polys = polys[keep]
        offsets = (torch.arange(2 * R + 1, device=c.device, dtype=c.dtype) - R) * r
        z = torch.full((polys.shape[0],), ground_z, device=c.device)
        return structural_score_volume(gcl_logits, polys, z, prior, hw, c, self.alphas(c.device, c.dtype), offsets)

    def correlate(self, T: Tensor, wT: Tensor, ref: Tensor, cell_xy: Tensor, c: Tensor, r: float,
                  prior_sigma: Tensor, temp: Tensor, struct: Tensor | None = None, temp_g: Tensor | None = None):
        """Exhaustive (yaw, x, y) matching of the lifted template against the reference crop.

        T [Cm,S,S] lifted (under the prior pose) features, wT [1,S,S] weights, ref [Cm,S+2R,S+2R].
        Hypothesis G(alpha, delta): T(p) ~ ref(R_alpha (p - c) + c + delta). Returns the log posterior
        volume [n_yaw, 2R+1, 2R+1], the yaw grid, the local soft-argmax (alpha, delta) and variances.
        """
        cfg = self.cfg
        S = T.shape[-1]
        R = (ref.shape[-1] - S) // 2
        dev, dt = T.device, T.dtype
        alphas = self.alphas(dev, dt)
        q = cell_xy - c
        ca, sa = alphas.cos()[:, None, None], alphas.sin()[:, None, None]
        px = ca * q[..., 0] + sa * q[..., 1]       # R(-alpha) q
        py = -sa * q[..., 0] + ca * q[..., 1]
        grid = torch.stack([px / (S * r) * 2, -py / (S * r) * 2], -1)  # [n_yaw, S, S, 2]
        src = torch.cat([T * wT, wT], 0).unsqueeze(0).expand(len(alphas), -1, -1, -1)
        rot = F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        Tk, wk = rot[:, :-1], rot[:, -1:]
        scores = F.conv2d(ref.unsqueeze(0), Tk)[0] / wk.sum((1, 2, 3)).clamp_min(1.0)[:, None, None]
        logits = scores * temp
        if struct is not None:  # height-free structural likelihood (image space)
            logits = logits + temp_g * struct
        off = (torch.arange(2 * R + 1, device=dev, dtype=dt) - R) * r
        dxg = off[None, None, :]
        dyg = -off[None, :, None]
        sx, sy, sa_ = prior_sigma[0].clamp_min(r), prior_sigma[1].clamp_min(r), prior_sigma[2].clamp_min(1e-3)
        log_prior = -0.5 * (dxg**2 / sx**2 + dyg**2 / sy**2 + alphas[:, None, None] ** 2 / sa_**2)
        log_post = torch.log_softmax((logits + log_prior).flatten(), 0).view_as(logits)

        # local soft-argmax around the MAP hypothesis
        post = log_post.exp()
        k, i, j = torch.unravel_index(post.argmax(), post.shape)
        k0, k1 = max(int(k) - 1, 0), min(int(k) + 2, len(alphas))
        i0, i1 = max(int(i) - 2, 0), min(int(i) + 3, 2 * R + 1)
        j0, j1 = max(int(j) - 2, 0), min(int(j) + 3, 2 * R + 1)
        loc = post[k0:k1, i0:i1, j0:j1]
        loc = loc / loc.sum().clamp_min(1e-12)
        alpha = (loc.sum((1, 2)) * alphas[k0:k1]).sum()
        dx = (loc.sum((0, 1)) * off[j0:j1]).sum()
        dy = (loc.sum((0, 2)) * -off[i0:i1]).sum()
        delta = torch.stack([dx, dy])
        mx = (post.sum((0, 1)) * off).sum()
        my = (post.sum((0, 2)) * -off).sum()
        ma = (post.sum((1, 2)) * alphas).sum()
        var = torch.stack([
            (post.sum((0, 1)) * (off - mx) ** 2).sum(),
            (post.sum((0, 2)) * (-off - my) ** 2).sum(),
            (post.sum((1, 2)) * (alphas - ma) ** 2).sum(),
        ]).clamp_min(r * r / 12)
        return log_post, alphas, alpha, delta, var

    def _refine(self, P8: Tensor, hw, pose: Pose, mem: GeoPriorMemory, cell_xy: Tensor, ground_z: float,
                change_polys, change_probs) -> Pose:
        cfg = self.cfg
        f_img = self.img_fine(P8.unsqueeze(0))[0]  # [Cf, h, w]
        stride = hw[1] / f_img.shape[-1]
        Cf = f_img.shape[0]
        kx = self.dkernel.view(1, 1, 1, 3).expand(Cf, 1, 1, 3)
        ky = self.dkernel.view(1, 1, 3, 1).expand(Cf, 1, 3, 1)
        gu = F.conv2d(F.pad(f_img.unsqueeze(0), (1, 1, 0, 0), mode="replicate"), kx, groups=Cf)[0] / stride
        gv = F.conv2d(F.pad(f_img.unsqueeze(0), (0, 0, 1, 1), mode="replicate"), ky, groups=Cf)[0] / stride
        maps = torch.cat([f_img, gu, gv], 0)

        # reference points: stable ground cells + SHP base vertices inside the window
        xy = cell_xy.reshape(-1, 2)
        half = (cell_xy[0, -1] - cell_xy[0, 0]).abs().max() / 2 + 1.0
        cen = cell_xy.mean((0, 1))
        sel = (mem.polys - cen).abs().amax((-1, -2)) < half
        if sel.any():
            xy = torch.cat([xy, mem.polys[sel].reshape(-1, 2)], 0)
        w = sample_bev(mem.stability, mem.grid, xy)[..., 0] * soft_union_weight(xy, change_polys, change_probs)
        X = torch.cat([xy, torch.full_like(xy[:, :1], ground_z)], -1)
        uv, depth = project(X, pose)
        _, inb = grid_sample_points(f_img[:1], uv, hw)
        w = w * inb * (depth > 1.0)
        n = min(cfg.lm_points, int((w > 0).sum()))
        if n < 12:
            return pose
        top = w.topk(n).indices
        X, w = X[top], w[top]
        f_ref = sample_bev(mem.bev_fine, mem.grid, X[:, :2])
        lam, c2 = self.log_lambda.exp(), self.log_robust.exp() ** 2

        def evaluate(p: Pose):
            uv, depth = project(X, p)
            s, ok = grid_sample_points(maps, uv, hw)
            f, du, dv = s.split(Cf, -1)
            res = f - f_ref
            valid = w * ok * (depth > 1.0)
            cost = (valid * torch.log1p((res**2).sum(-1) / c2)).sum()  # Cauchy robust cost
            return res, du, dv, valid, cost

        res, du, dv, valid, cost = evaluate(pose)
        for _ in range(cfg.lm_iters):
            wr = valid / (1 + (res**2).sum(-1) / c2)  # IRLS weights of the Cauchy cost
            Juv = projection_jacobian(pose.to_camera(X), pose.K)  # [N,2,6]
            J = du[..., None] * Juv[:, None, 0] + dv[..., None] * Juv[:, None, 1]  # [N,Cf,6]
            Hm = torch.einsum("n,nci,ncj->ij", wr, J, J)
            g = torch.einsum("n,nci,nc->i", wr, J, res)
            Hd = Hm + lam * torch.diag(Hm.diagonal()) + 1e-6 * torch.eye(6, device=Hm.device, dtype=Hm.dtype)
            step = -torch.linalg.solve(Hd, g)
            step = torch.cat([step[:3].clamp(-0.05, 0.05), step[3:].clamp(-2.0, 2.0)])  # trust region
            cand = pose.left_update(step)
            res_c, du_c, dv_c, valid_c, cost_c = evaluate(cand)
            acc = cost_c < cost  # LM acceptance: never increase the robust feature-metric cost
            pose = Pose(torch.where(acc, cand.R_cw, pose.R_cw), torch.where(acc, cand.t_cw, pose.t_cw), pose.K)
            res, du, dv, valid, cost = (torch.where(acc, a, b) for a, b in
                                        ((res_c, res), (du_c, du), (dv_c, dv), (valid_c, valid), (cost_c, cost)))
            lam = torch.where(acc, lam * 0.5, lam * 4.0)
        return pose
