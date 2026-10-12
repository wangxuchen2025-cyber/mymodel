"""GRAIL-CD 端到端训练目标（合成视频数据上的监督版本；真实数据时以伪标签替换对应真值）。

① 检测：逐帧 Hungarian 匹配（屋顶），focal + 框 + 循环多边形 + 掩膜 + η 的 Laplace NLL；跨帧实例嵌入监督对比；
⑤ 跟踪关联：GRAIL(跟踪模式) 的带垃圾箱分配 NLL；
③ 跨时相：GRAIL 分配 NLL（各 EM 迭代深监督）+ 变换高斯 NLL；
② 历史：正射集合提取 Hungarian 损失 + 源间融合 GRAIL 分配 NLL；
④ 变化：身份 CE（当前：保持/新建；历史：保持/拆除，仅覆盖区）+ 几何（改扩建 BCE、子类无监督占位、IoU 回归、精修顶点）。
"""
from typing import Dict

import torch
import torch.nn.functional as F

from geomac.losses import cyclic_poly_l1, dice_loss, poly_box, sigmoid_focal
from .geometry import poly_iou_raster, poly_moments, wrap
from .set_eval import assignment_nll
from .synth_sets import gt_x_about

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


def hungarian(cost):
    r, c = linear_sum_assignment(cost.detach().float().cpu().numpy())
    return torch.as_tensor(r, device=cost.device), torch.as_tensor(c, device=cost.device)


def nearest_labels(cent_pred, mask, cent_gt, mask_gt, gate):
    """一对一最近邻标注：返回 (B,N) 的 GT 索引或 -1。"""
    b, n = mask.shape
    out = torch.full((b, n), -1, dtype=torch.long, device=mask.device)
    for bi in range(b):
        ia, ig = mask[bi].nonzero()[:, 0], mask_gt[bi].nonzero()[:, 0]
        if len(ia) == 0 or len(ig) == 0:
            continue
        d = torch.cdist(cent_pred[bi, ia].float(), cent_gt[bi, ig].float())
        r, c = hungarian(torch.where(d > gate, torch.full_like(d, 1e6), d))
        ok = d[r, c] <= gate
        out[bi, ia[r[ok]]] = ig[c[ok]]
    return out


class Labeler:
    """训练时在前向过程中做逐帧检测–GT 匹配，并把结果缓存给损失使用。"""

    def __init__(self, batch):
        self.b = batch
        self.cache = {}

    def __call__(self, det, t):
        gt = self.b["gt"]
        B, Q = det["obj"].shape
        lab = torch.full((B, Q), -1, dtype=torch.long, device=det["obj"].device)
        matches = []
        for bi in range(B):
            gi = gt["valid"][bi, t].nonzero()[:, 0]
            if len(gi) == 0:
                e = torch.zeros(0, dtype=torch.long, device=lab.device)
                matches.append((e, e))
                continue
            gbox = poly_box(gt["roof"][bi, t, gi])
            cost = -2 * torch.sigmoid(det["obj"][bi].detach().float())[:, None] + 5 * torch.cdist(det["box"][bi].detach().float(), gbox, p=1)
            cost = cost + (~det["qmask"][bi])[:, None].float() * 1e4
            qi, gj = hungarian(cost)
            matches.append((qi, gi[gj]))
            lab[bi, qi] = gt["bid"][bi, t, gi[gj]]
        self.cache[t] = matches
        return lab


class HistLabeler:
    """历史栅格提取结果 -> 源真值多边形的 Hungarian 匹配（训练时教师强制 + 损失复用）。"""

    def __init__(self, batch, cfg):
        self.b, self.cfg = batch, cfg
        self.cache = {}

    def __call__(self, s, o):
        si = ("aerial", "sat", "map").index(s)
        Hh, W = self.cfg.raster_hw
        r = self.cfg.raster_res
        gp = self.b["src_poly"][:, si]
        gpn = torch.stack([gp[..., 0] / (W * r) + 0.5, 0.5 - gp[..., 1] / (Hh * r)], -1)
        gm = self.b["hist_state"] >= 0
        lab = torch.full(o["obj"].shape, -1, dtype=torch.long, device=o["obj"].device)
        matches = []
        for bi in range(gp.shape[0]):
            gi = gm[bi].nonzero()[:, 0]
            if len(gi) == 0:
                e = torch.zeros(0, dtype=torch.long, device=lab.device)
                matches.append((e, e))
                continue
            gbox = poly_box(gpn[bi, gi])
            cost = -2 * torch.sigmoid(o["obj"][bi].detach().float())[:, None] + 5 * torch.cdist(o["box"][bi].detach().float(), gbox, p=1)
            qi, gj = hungarian(cost)
            matches.append((qi, gi[gj]))
            lab[bi, qi] = gi[gj]
        self.cache[s] = (matches, gpn)
        return lab


def det_losses(det, gt, t, matches, w_aux=0.5):
    L = {}

    def one(o, pref, w):
        B, Q = o["obj"].shape
        tgt = torch.zeros_like(o["obj"])
        lp = lb = le = lm = 0
        n = 0
        for bi, (qi, gk) in enumerate(matches):
            if len(qi) == 0:
                continue
            tgt[bi, qi] = 1
            roof = gt["roof"][bi, t, gk]
            lp = lp + cyclic_poly_l1(o["poly"][bi, qi].float(), roof).sum()
            lb = lb + F.l1_loss(o["box"][bi, qi].float(), poly_box(roof), reduction="sum") / 4
            sig = o["eta_logsig"][bi, qi].float().exp()
            le = le + ((o["eta"][bi, qi].float() - gt["eta"][bi, t, gk]).abs() / sig + sig.log()).sum()
            mt = gt["roof_mask"][bi, t, gk]
            ml = o["mask"][bi, qi].float()
            lm = lm + F.binary_cross_entropy_with_logits(ml, mt, reduction="none").mean((1, 2)).sum() + dice_loss(ml, mt) * len(qi)
            n += len(qi)
        n = max(n, 1)
        valid = torch.ones_like(tgt)
        L[f"{pref}obj"] = w * (sigmoid_focal(o["obj"].float(), tgt) * valid).sum() / n
        L[f"{pref}poly"] = w * 5 * lp / n
        L[f"{pref}box"] = w * 2 * lb / n
        L[f"{pref}eta"] = w * 0.5 * le / n
        L[f"{pref}mask"] = w * lm / n
    one(det, "det_", 1.0)
    for i, a in enumerate(det["aux"]):
        one(a, f"aux{i}_", w_aux)
    return L


def emb_contrastive(steps, labeler, tau=0.1):
    """跨帧实例嵌入监督对比（同一 GT 建筑为正样本）。"""
    feats, ids = [], []
    for t, st in enumerate(steps):
        for bi, (qi, gk) in enumerate(labeler.cache[t]):
            if len(qi):
                feats.append(st["det"]["emb"][bi, qi].float())
                ids.append(gk + bi * 10000)
    if not feats:
        return torch.zeros((), device=steps[0]["det"]["obj"].device)
    f, y = torch.cat(feats), torch.cat(ids)
    sim = f @ f.T / tau
    eye = torch.eye(len(y), device=f.device, dtype=torch.bool)
    pos = (y[:, None] == y[None]) & ~eye
    has = pos.any(1)
    if not has.any():
        return sim.sum() * 0
    logp = sim.masked_fill(eye, -1e4).log_softmax(1)
    return -(logp * pos).sum(1)[has].div(pos.sum(1)[has]).mean() * 0.2


def track_assign_loss(steps, labeler):
    tot, n = 0, 0
    for t, st in enumerate(steps):
        go = st["trk"]
        if go is None:
            continue
        lab_q = torch.full_like(st["assign"], -1)
        for bi, (qi, gk) in enumerate(labeler.cache[t]):
            lab_q[bi, qi] = gk
        Bset = st["trk_B"]
        gt_t = Bset["gt_id"]
        corr_a = torch.full_like(lab_q, -1)
        corr_b = torch.full_like(gt_t, -1)
        for bi in range(lab_q.shape[0]):
            for qi in st["trk_A"]["mask"][bi].nonzero()[:, 0].tolist():
                g = int(lab_q[bi, qi])
                if g < 0:
                    continue
                hit = ((gt_t[bi] == g) & Bset["mask"][bi]).nonzero()[:, 0]
                if len(hit):
                    corr_a[bi, qi] = hit[0]
                    corr_b[bi, hit[0]] = qi
        tot = tot + assignment_nll(go["logP"], corr_a, corr_b, st["trk_A"]["mask"], Bset["mask"])
        n += 1
    return tot / max(n, 1)


def label_unified(U, batch):
    """历史统一集合的每个元素 -> 真实历史建筑索引（最近质心一对一，门限 4 m）。"""
    _, cu, _ = poly_moments(U["poly"])
    hm = batch["hist_state"] >= 0
    _, ch, _ = poly_moments(batch["hist_poly"])
    return nearest_labels(cu, U["mask"], ch, hm, 4.0)


def make_gt_pairs(batch):
    def fn(state, H):
        A_gt = state["mem"]["gt_id"]
        alive = state["mem"]["alive"] & (state["mem"]["n_obs"] >= 1)
        u_lab = label_unified(H["U"], batch)
        b, P = A_gt.shape
        pairs = torch.zeros(b, P, 2, dtype=torch.long, device=A_gt.device)
        pairs[..., 0] = torch.arange(P, device=A_gt.device)
        pm = torch.zeros(b, P, dtype=torch.bool, device=A_gt.device)
        for bi in range(b):
            for i in (alive[bi] & (A_gt[bi] >= 0)).nonzero()[:, 0].tolist():
                h = int(batch["cur_src"][bi, A_gt[bi, i]])
                if h < 0:
                    continue
                hit = (u_lab[bi] == h).nonzero()[:, 0]
                if len(hit):
                    pairs[bi, i, 1] = hit[0]
                    pm[bi, i] = True
        return pairs, pm
    return fn


def cross_temporal_losses(res, H, batch):
    L = {}
    go, A, B = res["go"], res["A"], res["B"]
    u_lab = label_unified(H["U"], batch)
    pairs, pm = res["pairs"], res["geom"]["pair_mask"]
    b, m = A["mask"].shape
    corr_a = torch.where(pm, pairs[..., 1], torch.full_like(pairs[..., 1], -1))
    corr_b = torch.full_like(B["mask"], -1, dtype=torch.long)
    for bi in range(b):
        for i in pm[bi].nonzero()[:, 0].tolist():
            corr_b[bi, pairs[bi, i, 1]] = i
    its = go["iters"]
    L["x_assign"] = sum((0.5 if k < len(its) - 1 else 1.0) * assignment_nll(it["logP"], corr_a, corr_b, A["mask"], B["mask"])
                        for k, it in enumerate(its))
    # 变换真值：GNSS->真实（序列拟合）再叠加统一集合相对真值的整体偏移
    _, cu, _ = poly_moments(B["poly"])
    _, ch, _ = poly_moments(batch["hist_poly"])
    off = torch.zeros(b, 2, device=cu.device)
    for bi in range(b):
        ok = (u_lab[bi] >= 0) & B["mask"][bi]
        if ok.any():
            off[bi] = (cu[bi, ok] - ch[bi, u_lab[bi, ok]]).mean(0)
    xg = gt_x_about(batch["sim"] + torch.cat([off, torch.zeros_like(off)], 1), res["c0"])
    d = go["x"] - xg
    d = torch.cat([d[:, :2], wrap(d[:, 2:3]), d[:, 3:]], 1)
    L["x_trans"] = 0.05 * (0.5 * d ** 2 / go["var"] + 0.5 * go["var"].log()).sum(1).mean()
    # —— 变化身份 ——
    gt_id = A["gt_id"]
    ok_a = A["mask"] & (gt_id >= 0)
    if ok_a.any():
        is_new = (batch["cur_src"].gather(1, gt_id.clamp(min=0)) < 0).long()
        L["id_cur"] = F.cross_entropy(res["id_a"][ok_a].float(), is_new[ok_a])
    ok_b = B["mask"] & (u_lab >= 0) & (B["vis"] > 0.5)
    if ok_b.any():
        dem = (batch["hist_state"].gather(1, u_lab.clamp(min=0)) == 1).long()
        L["id_hist"] = F.cross_entropy(res["id_b"][ok_b].float(), dem[ok_b])
    # —— 几何变化（GT 配对上）——
    g = res["geom"]
    if pm.any():
        h_idx = u_lab.gather(1, pairs[..., 1])
        mod = (batch["hist_state"].gather(1, h_idx.clamp(min=0)) == 2).float()
        L["geo_mod"] = F.binary_cross_entropy_with_logits(g["mod_logit"][pm].float(), mod[pm])
        cur_true = batch["cur_poly"].gather(1, gt_id.clamp(min=0)[..., None, None].expand(-1, -1, *batch["cur_poly"].shape[2:]))
        cur_true = cur_true + off[:, None, None]
        hist_true = batch["hist_poly"].gather(1, h_idx.clamp(min=0)[..., None, None].expand(-1, -1, *batch["hist_poly"].shape[2:]))
        iou_gt, _, _ = poly_iou_raster(cur_true[pm], hist_true[pm] + off[:, None, None].expand_as(cur_true)[pm])
        L["geo_iou"] = (g["iou_pred"][pm].float() - iou_gt).abs().mean()
        L["geo_refine"] = 0.2 * cyclic_poly_l1(g["refined"][pm].float(), cur_true[pm]).mean()
    return L


def history_losses(H, hlab):
    L = {}
    for s, (matches, gpn) in hlab.cache.items():
        o = H["raw"][s]
        tgt = torch.zeros_like(o["obj"])
        lp = lb = 0
        n = 0
        for bi, (qi, gk) in enumerate(matches):
            if len(qi) == 0:
                continue
            tgt[bi, qi] = 1
            lp = lp + cyclic_poly_l1(o["poly"][bi, qi].float(), gpn[bi, gk]).sum()
            lb = lb + F.l1_loss(o["box"][bi, qi].float(), poly_box(gpn[bi, gk]), reduction="sum") / 4
            n += len(qi)
        n = max(n, 1)
        L[f"hist_{s}"] = sigmoid_focal(o["obj"].float(), tgt).sum() / n + 5 * lp / n + 2 * lb / n
    return L


def total_loss(out, batch, labeler, hlab, cfg):
    L = {}
    steps = out["steps"]
    for t, st in enumerate(steps):
        for k, v in det_losses(st["det"], batch["gt"], t, labeler.cache[t]).items():
            L[k] = L.get(k, 0) + v / len(steps)
    L["emb_nce"] = emb_contrastive(steps, labeler)
    L["trk_assign"] = track_assign_loss(steps, labeler)
    L.update(cross_temporal_losses(out["res"], out["H"], batch))
    L.update(history_losses(out["H"], hlab))
    total = sum(L.values())
    return total, {k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in L.items()}
