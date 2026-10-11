"""Training objectives of MARS-CD.

L = l_reg (CARR: posterior NLL + pose L1/geodesic)
  + l_struct (v2-I contact-line / roof-edge maps, targets rendered from GT prisms, no extra labels)
  + sum_layers l_det (PASID: focal existence, permutation-equivalent polygon L1 + edge direction, height)
  + l_heat (discovery BEV heatmap, CenterNet focal)
  + l_intra (RIGM mode 1: OT NLL with dustbins)
  + l_match (v2-II: OT NLL of the relational (HSR) and unary assignments, dustbins = change)
  + l_evidential (v3: Beta evidential loss on observation-weighted occupancy evidence)
  + l_change (pair classifier CE)
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import MARSConfig
from .geometry import Pose, points_in_polygons, polygon_centroid
from .models.evidential import edl_beta_loss
from .models.marscd import FrameOutput, SequenceTargets
from .models.structure import render_structure_targets


def sigmoid_focal(logits: Tensor, targets: Tensor, alpha: float = 0.25, gamma: float = 2.0) -> Tensor:
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = p * targets + (1 - p) * (1 - targets)
    a = alpha * targets + (1 - alpha) * (1 - targets)
    return (a * (1 - pt) ** gamma * ce).sum()


def cyclic_polygon_l1(pred: Tensor, gt: Tensor) -> tuple[Tensor, Tensor]:
    """MapTR-style permutation-equivalent L1 over cyclic shifts (both rings CCW). -> ([P], shifted gt)."""
    Nv = gt.shape[1]
    shifts = torch.stack([gt.roll(-s, 1) for s in range(Nv)], 1)  # [P, Nv, Nv, 2]
    d = (pred[:, None] - shifts).abs().mean((-1, -2))
    best = d.argmin(1)
    return d.gather(1, best[:, None])[:, 0], shifts[torch.arange(gt.shape[0]), best]


def edge_direction_loss(pred: Tensor, gt: Tensor) -> Tensor:
    ep = F.normalize(pred.roll(-1, 1) - pred, dim=-1)
    eg = F.normalize(gt.roll(-1, 1) - gt, dim=-1)
    return (1 - (ep * eg).sum(-1)).mean(-1)


def ot_nll(log_P: Tensor, row_target: Tensor) -> Tensor:
    """NLL of the GT partial assignment. row_target [M]: column index or -1 (dustbin)."""
    M, N = log_P.shape[0] - 1, log_P.shape[1] - 1
    if M == 0 and N == 0:
        return log_P.sum() * 0
    terms = []
    if M:
        col = torch.where(row_target >= 0, row_target, torch.full_like(row_target, N))
        terms.append(-log_P[torch.arange(M, device=log_P.device), col])
    if N:
        hit = torch.zeros(N, dtype=torch.bool, device=log_P.device)
        if M and (row_target >= 0).any():
            hit[row_target[row_target >= 0]] = True
        if (~hit).any():
            terms.append(-log_P[M, (~hit).nonzero(as_tuple=True)[0]])
    return torch.cat(terms).mean()


def heatmap_focal(heat_logit: Tensor, cell_xy: Tensor, centers: Tensor, sigma: float = 2.0) -> Tensor:
    """CenterNet focal loss on the lifted BEV window."""
    if centers.shape[0] == 0:
        tgt = torch.zeros_like(heat_logit)
    else:
        d2 = ((cell_xy[:, :, None] - centers[None, None]) ** 2).sum(-1)
        tgt = torch.exp(-d2 / (2 * sigma**2)).amax(-1)
        flat = d2.flatten(0, 1).argmin(0)  # exact peak cell per centre
        tgt.view(-1)[flat] = 1.0
    p = torch.sigmoid(heat_logit).clamp(1e-4, 1 - 1e-4)
    pos = tgt >= 1.0
    pos_l = -((1 - p) ** 2 * torch.log(p))[pos].sum()
    neg_l = -((1 - tgt) ** 4 * p**2 * torch.log(1 - p))[~pos].sum()
    return (pos_l + neg_l) / pos.sum().clamp_min(1)


def rotation_angle(R1: Tensor, R2: Tensor) -> Tensor:
    R = R1 @ R2.transpose(-1, -2)
    cos = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.acos(cos)


def registration_targets(prior: Pose, gt: Pose, center: Tensor) -> tuple[Tensor, Tensor]:
    """(alpha*, delta*) such that gt ~= G(alpha*, delta*) o prior (see apply_world_rigid2d)."""
    dyaw = prior.yaw - gt.yaw
    alpha = torch.atan2(dyaw.sin(), dyaw.cos())
    c, s = alpha.cos(), alpha.sin()
    v = prior.center[:2] - center
    rot = torch.stack([c * v[0] - s * v[1], s * v[0] + c * v[1]])
    return alpha, gt.center[:2] - rot - center


def registration_loss(reg, gt: Pose) -> dict:
    """CARR: NLL of the GT (yaw, x, y) cell in the posterior volume + metric pose errors (coarse & refined)."""
    L = {}
    a, d = registration_targets(reg.pose_prior, gt, reg.center)
    R = (reg.log_post.shape[1] - 1) // 2
    k = int((a - reg.alphas).abs().argmin())
    j = int(torch.round(d[0] / reg.cell)) + R
    i = R - int(torch.round(d[1] / reg.cell))
    if 0 <= i <= 2 * R and 0 <= j <= 2 * R:
        L["reg_nll"] = -reg.log_post[k, i, j]
    L["reg_pose"] = ((reg.pose.center - gt.center).abs().sum() + (reg.pose_coarse.center[:2] - gt.center[:2]).abs().sum()) / 4 \
        + (rotation_angle(reg.pose.R_cw, gt.R_cw) + rotation_angle(reg.pose_coarse.R_cw, gt.R_cw)) * 180 / math.pi / 4
    return L


class MARSCriterion:
    def __init__(self, cfg: MARSConfig, w: dict | None = None):
        self.cfg = cfg
        self.w = dict(reg_nll=0.5, reg_pose=1.0, exist=2.0, poly=1.0, edge=0.5, height=0.2, base=0.1, heat=1.0,
                      intra=0.5, match=1.0, change=1.0, shp_aux=0.5, struct=1.0, evidential=0.5)
        if w:
            self.w.update(w)

    def frame_losses(self, fo: FrameOutput, tgt: SequenceTargets) -> dict:
        L = {}
        reg = fo.reg
        dev = reg.log_post.device
        if fo.gt_pose is not None:
            L.update(registration_loss(reg, fo.gt_pose))
        if fo.target is not None and fo.dec.layers:
            t = fo.target
            pos = t >= 0
            ex, po, ed, he, ba = [], [], [], [], []
            for lay in fo.dec.layers:
                ex.append(sigmoid_focal(lay["exist_logit"], pos.float()) / max(int(pos.sum()), 1))
                if pos.any():
                    pl, gshift = cyclic_polygon_l1(lay["polys"][pos], tgt.polys[t[pos]])
                    po.append(pl.mean() / 2.0)
                    ed.append(edge_direction_loss(lay["polys"][pos], gshift).mean())
                    he.append((lay["height"][pos] - tgt.heights[t[pos]]).abs().mean() / 5.0)
                    gb = tgt.bases[t[pos]] if tgt.bases is not None else torch.zeros_like(lay["base"][pos])
                    ba.append((lay["base"][pos] - gb).abs().mean())
            L["exist"] = sum(ex) / len(ex)
            if po:
                L["poly"], L["edge"], L["height"] = sum(po) / len(po), sum(ed) / len(ed), sum(he) / len(he)
                L["base"] = sum(ba) / len(ba)
        if fo.proposals.get("heat_logit") is not None and fo.visible_gt is not None:
            cen = polygon_centroid(tgt.polys[fo.visible_gt]) if fo.visible_gt.numel() else torch.zeros(0, 2, device=dev)
            L["heat"] = heatmap_focal(fo.proposals["heat_logit"], reg.cell_xy, cen)
        if fo.intra is not None and fo.intra.get("target") is not None:
            L["intra"] = ot_nll(fo.intra["log_P"], fo.intra["target"])
        if "struct" in fo.extras and fo.gt_pose is not None:  # v2-I: auto-rendered contact-line / roof-edge targets
            bases = tgt.bases if tgt.bases is not None else torch.zeros_like(tgt.heights)
            st_t = render_structure_targets(tgt.polys, bases, tgt.heights, fo.gt_pose, fo.extras["hw"])
            logit = fo.extras["struct"]
            L["struct"] = sigmoid_focal(logit, st_t, alpha=0.75) / st_t.sum().clamp_min(1.0)
        return L

    def match_losses(self, mo: dict) -> dict:
        L = {}
        if "gt_match" in mo:  # v2-II deep supervision: relational (final) and unary assignments
            L["match"] = ot_nll(mo["log_P"], mo["gt_match"]) + 0.5 * ot_nll(mo["log_P_unary"], mo["gt_match"])
        if "pair_labels" in mo and mo["pair_labels"].numel():
            L["change"] = F.cross_entropy(mo["change"]["logits"], mo["pair_labels"])
        ev = mo.get("evidence", {})
        if "y" in ev:  # v3 evidential occupancy loss
            L["evidential"] = edl_beta_loss(ev["r"], ev["s"], ev["y"], ev["mask"])
        return L

    @staticmethod
    def shp_consistency(memory) -> Tensor:
        """Auxiliary: the fused T1 BEV must reproduce the SHP footprint raster (cheap, label-free)."""
        g = memory.grid
        ys = g.y0 - (torch.arange(g.H, device=memory.bev.device) + 0.5) * g.gsd
        xs = g.x0 + (torch.arange(g.W, device=memory.bev.device) + 0.5) * g.gsd
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        pts = torch.stack([xx, yy], -1).reshape(-1, 2)
        if memory.polys.shape[0]:
            mask = points_in_polygons(pts, memory.polys).any(0).float().view(g.H, g.W)
        else:
            mask = torch.zeros(g.H, g.W, device=pts.device)
        return F.binary_cross_entropy_with_logits(memory.bld_logit[0], mask)

    def __call__(self, frames: list[FrameOutput], tgt: SequenceTargets, match_out: dict | None = None,
                 extra_frames: list[tuple[list[FrameOutput], SequenceTargets]] = (), memory=None) -> tuple[Tensor, dict]:
        acc: dict[str, list] = {}
        if memory is not None:
            acc["shp_aux"] = [self.shp_consistency(memory)]
        groups = [(frames, tgt)] + list(extra_frames)
        for fr, tg in groups:
            for fo in fr:
                for k, v in self.frame_losses(fo, tg).items():
                    acc.setdefault(k, []).append(v)
        if match_out is not None:
            for k, v in self.match_losses(match_out).items():
                acc.setdefault(k, []).append(v)
        logs = {k: torch.stack(v).mean() for k, v in acc.items()}
        total = sum(self.w[k] * v for k, v in logs.items())
        return total, {k: float(v.detach()) for k, v in logs.items()}
