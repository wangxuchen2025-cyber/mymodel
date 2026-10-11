"""Evaluation utilities (instance-level, geo-referenced).

Metrics follow the evaluation protocol proposed in docs/research_proposal.md:
  * registration: horizontal position error (m) and yaw error (deg) of the GNSS
    prior, the BEV-correlation estimate and the LM-refined pose;
  * T1-side change decisions (unchanged / expanded / shrunk / demolished) vs GT,
    with "unobserved" reported separately (abstention, not an error);
  * T2-side new-building detection: precision / recall at footprint IoU >= 0.25;
  * footprint quality: mean best IoU of GT T2 footprints w.r.t. confirmed tracks.
"""

from __future__ import annotations

import math

import torch

from .geometry import polygon_iou_paired
from .models.change_head import REPORT_KINDS

GT_KINDS = ("unchanged", "expanded", "shrunk", "demolished")


def _best_iou(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """For every polygon in P, best IoU against any polygon in Q. [n,Nv,2],[m,Nv,2] -> [n]."""
    if P.shape[0] == 0 or Q.shape[0] == 0:
        return P.new_zeros(P.shape[0])
    n, m = P.shape[0], Q.shape[0]
    iou = polygon_iou_paired(P[:, None].expand(n, m, -1, -1).reshape(n * m, -1, 2),
                             Q[None].expand(n, m, -1, -1).reshape(n * m, -1, 2), res=24).view(n, m)
    return iou.max(1).values


def evaluate_scene(sc, st, frame_outs, match_out) -> dict:
    out: dict = {}
    # ---- registration --------------------------------------------------------
    errs = {"prior": [], "coarse": [], "refined": []}
    yaw = {"prior": [], "refined": []}
    for fo, gt in zip(frame_outs, sc.t2_poses):
        for k, p in (("prior", fo.reg.pose_prior), ("coarse", fo.reg.pose_coarse), ("refined", fo.reg.pose)):
            errs[k].append(float((p.center[:2] - gt.center[:2]).norm()))
        for k, p in (("prior", fo.reg.pose_prior), ("refined", fo.reg.pose)):
            d = p.yaw - gt.yaw
            yaw[k].append(abs(float(torch.atan2(d.sin(), d.cos()))) * 180 / math.pi)
    out["reg_xy_err_m"] = {k: sum(v) / len(v) for k, v in errs.items()}
    out["reg_yaw_err_deg"] = {k: sum(v) / len(v) for k, v in yaw.items()}

    # ---- T1-side decisions ----------------------------------------------------
    rep = match_out["report"]
    pred_t1 = {r["t1"]: r["kind"] for r in rep if r["t1"] >= 0}
    conf = {g: {p: 0 for p in REPORT_KINDS} for g in GT_KINDS}
    for j, g in enumerate(sc.t1_change.tolist()):
        if j in pred_t1:
            conf[GT_KINDS[g]][pred_t1[j]] += 1
    decided = sum(conf[g][p] for g in GT_KINDS for p in REPORT_KINDS if p not in ("unobserved", "new"))
    correct = sum(conf[g][g] for g in GT_KINDS)
    out["t1_decision_acc"] = correct / max(decided, 1)
    out["t1_unobserved"] = sum(conf[g]["unobserved"] for g in GT_KINDS)
    out["t1_evaluated"] = len(pred_t1)
    out["t1_confusion"] = conf

    # ---- new buildings ------------------------------------------------------------
    gt_new = sc.t2_polys[sc.t2_is_new]
    pred_new = [torch.tensor(r["t2_polygon"]) for r in rep if r["kind"] == "new"]
    pred_new = torch.stack(pred_new) if pred_new else sc.t2_polys.new_zeros(0, sc.t2_polys.shape[1], 2)
    tp_p = int((_best_iou(pred_new, gt_new) >= 0.25).sum()) if len(pred_new) else 0
    tp_r = int((_best_iou(gt_new, pred_new) >= 0.25).sum()) if len(gt_new) else 0
    out["new_precision"] = tp_p / max(len(pred_new), 1)
    out["new_recall"] = tp_r / max(len(gt_new), 1)
    out["new_pred_count"] = len(pred_new)

    # ---- footprint quality of confirmed tracks -------------------------------------
    tb = st.tracks
    conf_tracks = tb.polys[tb.p_exist > 0.5] if len(tb) else sc.t2_polys.new_zeros(0, sc.t2_polys.shape[1], 2)
    out["t2_footprint_mIoU"] = float(_best_iou(sc.t2_polys, conf_tracks).mean()) if len(sc.t2_polys) else 0.0
    out["num_tracks"] = len(tb)
    return out
