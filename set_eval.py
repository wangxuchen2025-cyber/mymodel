"""集合级损失与指标（GRAIL 单独训练/评测）。"""
import math
from typing import Dict

import torch

from .geometry import apply_sim, poly_moments, wrap
from .synth_sets import gt_x_about


def to_sets(b, pos_cov_a=1.5, pos_cov_b=1.0):
    eye = torch.eye(2, device=b["A_poly"].device)
    A = {"poly": b["A_poly"], "emb": b["A_emb"], "mask": b["A_mask"],
         "pos_cov": eye.expand(*b["A_mask"].shape, 2, 2) * pos_cov_a}
    B = {"poly": b["B_poly"], "emb": b["B_emb"], "mask": b["B_mask"],
         "pos_cov": eye.expand(*b["B_mask"].shape, 2, 2) * pos_cov_b, "vis": b["B_vis"]}
    c0 = (poly_moments(b["A_poly"])[1] * b["A_mask"][..., None]).sum(1) / b["A_mask"].sum(1, keepdim=True).clamp(min=1)
    if "A_cov" in b:                                                            # 覆盖栅格（相机足迹 / 历史数据范围）
        A["cov_img"], A["cov_bbox"] = b["A_cov"], b["A_cov_bbox"]
        B["cov_img"], B["cov_bbox"] = b["B_cov"], b["B_cov_bbox"]
    return A, B, c0


def assignment_nll(logP, corr_a, corr_b, mask_a, mask_b):
    b, m1, n1 = logP.shape
    m, n = m1 - 1, n1 - 1
    ia = corr_a.clamp(min=0)
    lp_match = logP[:, :m].gather(2, ia[..., None])[..., 0]
    mm = (corr_a >= 0) & mask_a
    ua = (corr_a < 0) & mask_a
    ub = (corr_b < 0) & mask_b
    l1 = -(lp_match * mm).sum() / mm.sum().clamp(min=1)
    l2 = -(logP[:, :m, n] * ua).sum() / ua.sum().clamp(min=1)
    l3 = -(logP[:, m, :n] * ub).sum() / ub.sum().clamp(min=1)
    return l1 + 0.5 * (l2 + l3)


def grail_set_loss(out, b, c0) -> Dict[str, torch.Tensor]:
    L = {}
    its = out["iters"]
    L["assign"] = sum((0.5 if k < len(its) - 1 else 1.0) * assignment_nll(it["logP"], b["corr_a"], b["corr_b"],
                                                                         b["A_mask"], b["B_mask"]) for k, it in enumerate(its))
    xg = gt_x_about(b["x_gt"], c0)
    d = out["x"] - xg
    d = torch.cat([d[:, :2], wrap(d[:, 2:3]), d[:, 3:]], 1)
    L["trans"] = 0.05 * (0.5 * d ** 2 / out["var"] + 0.5 * out["var"].log()).sum(1).mean()
    # 逐栋残差：仅对"未变化且匹配"的建筑监督（GT 残差相对于 GT 变换）
    ok = (b["corr_a"] >= 0) & (b["a_state"] == 0) & b["A_mask"]
    if ok.any():
        _, cb, _ = poly_moments(b["B_poly"])
        _, ca, _ = poly_moments(b["A_poly"])
        tgt = cb.gather(1, b["corr_a"].clamp(min=0)[..., None].expand(-1, -1, 2))
        dg = tgt - apply_sim(xg, ca, c0)
        L["delta"] = 0.2 * (out["delta"] - dg).abs().sum(-1)[ok].mean()
    return L


@torch.no_grad()
def set_metrics(m_a, x, b, c0) -> Dict[str, float]:
    """m_a (B,M) 预测的历史索引或 -1。"""
    va = b["A_mask"]
    gt = b["corr_a"]
    tp = ((m_a == gt) & (gt >= 0) & va).sum().item()
    pp = ((m_a >= 0) & va).sum().item()
    gp = ((gt >= 0) & va).sum().item()
    P, R = tp / max(pp, 1), tp / max(gp, 1)
    # 新建（当前侧未匹配）
    pn, gn = (m_a < 0) & va, (gt < 0) & va
    tpn = (pn & gn).sum().item()
    nP, nR = tpn / max(pn.sum().item(), 1), tpn / max(gn.sum().item(), 1)
    # 拆除（历史侧可见且未匹配）
    bsz, n = b["B_mask"].shape
    matched_b = torch.zeros(bsz, n, dtype=torch.bool, device=m_a.device)
    for bi in range(bsz):
        idx = m_a[bi][m_a[bi] >= 0]
        matched_b[bi, idx] = True
    visb = (b["B_vis"] > 0.5) & b["B_mask"]
    pd, gd = (~matched_b) & visb, (b["corr_b"] < 0) & visb
    tpd = (pd & gd).sum().item()
    dP, dR = tpd / max(pd.sum().item(), 1), tpd / max(gd.sum().item(), 1)
    xg = gt_x_about(b["x_gt"], c0)
    _, ca, _ = poly_moments(b["A_poly"])
    e = (apply_sim(x, ca, c0) - apply_sim(xg, ca, c0)).norm(dim=-1)
    rmse = ((e ** 2 * va).sum() / va.sum()).sqrt().item()
    per = ((e ** 2 * va).sum(1) / va.sum(1).clamp(min=1)).sqrt()                # 逐样本对齐 RMSE
    f1 = lambda p, r: 2 * p * r / max(p + r, 1e-9)
    return {"assoc_P": P, "assoc_R": R, "assoc_F1": f1(P, R), "new_F1": f1(nP, nR), "demol_F1": f1(dP, dR),
            "align_rmse_m": rmse, "align_med_m": per.median().item(), "align_succ_1m": (per < 1.0).float().mean().item(),
            "rot_err_deg": math.degrees(wrap(x[:, 2] - xg[:, 2]).abs().mean().item())}
