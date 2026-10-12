"""BONAI 真实离轴影像上的屋顶→基底偏移实验（验证"当前分支"的几何参数化）。

同一个 Mask R-CNN（torchvision v2，COCO 预训练）上并行训练三种偏移参数化，特征完全相同、对比公平：
  free   ：LOFT 式逐实例自由二维偏移回归（基线）；
  shared ：逐实例只回归标量幅值 m_i，方向由"影像级平行方向"给出；
  radial ：逐实例只回归 m_i，方向指向"影像级天底点" N（中心投影的径向位移律，平行是 N→∞ 的极限）。
影像级几何（方向 θ 或天底点 N）不由额外网络预测，而由"几何共识层"从所有实例的方向证据闭式估计：
  平行：加权圆均值；径向：直线束最小二乘交点（带"趋向平行"的正则），全程可微、无参数。
"""
import json
import math
import os
import zipfile
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss, maskrcnn_inference, maskrcnn_loss

OFF_SCALE = 50.0                     # 偏移归一化尺度（像素）


# ------------------------------------------------------------------------------------------- 数据
class BonaiDataset(torch.utils.data.Dataset):
    def __init__(self, img_src: str, ann_files: List[str], train=True, image_ids=None):
        """img_src：解压后的图像目录，或 zip 文件路径（按文件名在 zip 内查找）。"""
        self.train, self.img_src = train, img_src
        self.items = []
        for f in ann_files:
            d = json.load(open(f))
            anns = {}
            for a in d["annotations"]:
                anns.setdefault(a["image_id"], []).append(a)
            for im in d["images"]:
                if image_ids is not None and (os.path.basename(f), im["id"]) not in image_ids:
                    continue
                self.items.append((im["file_name"], anns.get(im["id"], []), os.path.basename(f), im["id"]))
        self._zip = None
        self._index = None

    def __len__(self):
        return len(self.items)

    def _read(self, name):
        if os.path.isdir(self.img_src):
            p = os.path.join(self.img_src, name)
            if not os.path.exists(p):
                p = self._index[name] if self._index else p
            return cv2.imread(p, cv2.IMREAD_COLOR)[:, :, ::-1]
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.img_src)
            self._index = {os.path.basename(n): n for n in self._zip.namelist()}
        buf = np.frombuffer(self._zip.read(self._index[os.path.basename(name)]), np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1]

    @staticmethod
    def _poly(seg):
        return np.array(seg[0] if isinstance(seg[0], list) else seg, np.float32).reshape(-1, 2)

    def __getitem__(self, i):
        name, anns, src, iid = self.items[i]
        img = np.ascontiguousarray(self._read(name))
        H, W = img.shape[:2]
        roofs, foots, offs, hts, ign = [], [], [], [], []
        for a in anns:
            # 注意：BONAI 测试集 json 中 segmentation 是"基底"、屋顶在 roof_mask；训练集 json 中 segmentation 是屋顶。
            # offset 为"基底→屋顶"（roof = footprint + offset）；这里统一改为"屋顶→基底"的向量 t = -offset。
            r = self._poly(a["roof_mask"]) if "roof_mask" in a else self._poly(a["segmentation"])
            off = -np.array(a["offset"], np.float32)
            fp = self._poly(a["footprint_mask"]) if "footprint_mask" in a else r + off
            if len(r) < 3:
                continue
            roofs.append(r), foots.append(fp), offs.append(off)
            hts.append(float(a.get("building_height", 0) or 0)), ign.append(int(a.get("ignore_flag", 0)))
        k = 0
        if self.train:                                                          # 随机翻转/旋转 90°（多边形与偏移同步变换）
            fx, fy, k = np.random.rand() < 0.5, np.random.rand() < 0.5, np.random.randint(4)
            if fx:
                img = img[:, ::-1]
                roofs = [np.stack([W - p[:, 0], p[:, 1]], 1) for p in roofs]
                foots = [np.stack([W - p[:, 0], p[:, 1]], 1) for p in foots]
                offs = [o * np.array([-1, 1], np.float32) for o in offs]
            if fy:
                img = img[::-1]
                roofs = [np.stack([p[:, 0], H - p[:, 1]], 1) for p in roofs]
                foots = [np.stack([p[:, 0], H - p[:, 1]], 1) for p in foots]
                offs = [o * np.array([1, -1], np.float32) for o in offs]
            for _ in range(k):                                                  # 顺时针 90°：(x,y)->(H-y, x)
                img = np.rot90(img, -1)
                roofs = [np.stack([H - p[:, 1], p[:, 0]], 1) for p in roofs]
                foots = [np.stack([H - p[:, 1], p[:, 0]], 1) for p in foots]
                offs = [np.array([-o[1], o[0]], np.float32) for o in offs]
            img = np.ascontiguousarray(img)
        keep = [j for j in range(len(roofs)) if (not self.train) or ign[j] == 0]
        boxes, masks = [], []
        for j in keep:
            p = roofs[j]
            x0, y0 = p.min(0)
            x1, y1 = p.max(0)
            boxes.append([max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)])
        good = [j for j, b in zip(keep, boxes) if b[2] - b[0] >= 2 and b[3] - b[1] >= 2]
        boxes = [b for b in boxes if b[2] - b[0] >= 2 and b[3] - b[1] >= 2]
        if self.train:
            for j in good:
                m = np.zeros((H, W), np.uint8)
                cv2.fillPoly(m, [np.round(roofs[j]).astype(np.int32)], 1)
                masks.append(m)
        t = {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
             "labels": torch.ones(len(good), dtype=torch.int64),
             "offsets": torch.tensor(np.array([offs[j] for j in good]), dtype=torch.float32).reshape(-1, 2),
             "heights": torch.tensor([hts[j] for j in good], dtype=torch.float32),
             "ignore": torch.tensor([ign[j] for j in good], dtype=torch.bool)}
        if self.train:
            t["masks"] = torch.from_numpy(np.stack(masks)) if masks else torch.zeros(0, H, W, dtype=torch.uint8)
        else:
            t["roof_polys"] = [roofs[j] for j in good]
            t["foot_polys"] = [foots[j] for j in good]
            t["name"] = name
        x = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0
        return x, t


def collate(b):
    return tuple(zip(*b))


# ------------------------------------------------------------------------------------------- 几何共识层
def consensus_parallel(dirs, w):
    """dirs (n,2) 单位向量，w (n,) -> 影像级方向单位向量 (2,)。加权圆均值。"""
    v = (w[:, None] * dirs).sum(0)
    return v / v.norm().clamp(min=1e-6)


def consensus_radial(centers, dirs, w, lam=1e-3, scale=1024.0):
    """直线束最小二乘交点：每个实例给出一条过 c_i、沿 d_i 的直线，求 N 使 Σ w_i·dist(N, line_i)² 最小。
    采用齐次参数化避免 N 在无穷远时数值发散：解 (I - d dᵀ) 加权和的最小特征向量方向会退化为平行情形，
    这里用 Tikhonov 正则把解拉向"影像中心 + 远处"，并在直线近乎平行时自动退化为平行共识。
    返回每个实例的方向 (n,2) 与 N (2,)、径向强度 rho∈[0,1)（0 表示平行）。"""
    c = centers / scale
    P = torch.eye(2, device=c.device)[None] - dirs[:, :, None] * dirs[:, None, :]           # 法向投影
    A = (w[:, None, None] * P).sum(0)
    bvec = (w[:, None, None] * P @ c[:, :, None]).sum(0)[:, 0]
    u = consensus_parallel(dirs, w)
    # 正则：N 先验位于中心沿 u 方向很远处（平行极限），强度 lam·Σw
    far = 0.5 + u * 50.0
    lamw = lam * w.sum()
    N = torch.linalg.solve(A + lamw * torch.eye(2, device=c.device), bvec + lamw * far)
    d = N[None] - c
    d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    rho = 1.0 / (1.0 + (N - 0.5).norm())
    return d, N * scale, rho


# ------------------------------------------------------------------------------------------- 偏移头与 RoIHeads
class OffsetHead(nn.Module):
    def __init__(self, cin=256):
        super().__init__()
        self.tower = nn.Sequential(*[m for _ in range(4) for m in (nn.Conv2d(cin, cin, 3, 1, 1), nn.ReLU(inplace=True))],
                                   nn.Conv2d(cin, cin, 3, 2, 1), nn.ReLU(inplace=True))
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(cin * 7 * 7, 1024), nn.ReLU(inplace=True))
        self.free = nn.Linear(1024, 2)               # 自由偏移（/OFF_SCALE）
        self.dir = nn.Linear(1024, 2)                # 方向证据
        self.mag = nn.Linear(1024, 1)                # 幅值（/OFF_SCALE）
        self.conf = nn.Linear(1024, 1)               # 方向证据置信度

    def forward(self, x):
        h = self.fc(self.tower(x))
        h = h.float()
        cl = self.conf(h)[:, 0]
        return {"free": self.free(h), "dir": F.normalize(self.dir(h), dim=-1),
                "mag": F.softplus(self.mag(h))[:, 0], "conf": torch.sigmoid(cl), "conf_logit": cl}


def enlarge_boxes(b, hw, factor=2.0, margin=32.0):
    c = (b[:, :2] + b[:, 2:]) / 2
    half = (b[:, 2:] - b[:, :2]) / 2 * factor + margin
    lo = (c - half).clamp(min=0)
    hi = torch.stack([(c[:, 0] + half[:, 0]).clamp(max=hw[1]), (c[:, 1] + half[:, 1]).clamp(max=hw[0])], -1)
    return torch.cat([lo, hi], -1)


class GlobalDirHead(nn.Module):
    """影像级倾倒方向：成像几何对整幅影像共享，立面外观在全图范围内提供更强的方向证据。"""

    def __init__(self, cin=256):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, 256, 3, 2, 1), nn.ReLU(inplace=True), nn.Conv2d(256, 256, 3, 1, 1),
                                 nn.ReLU(inplace=True))
        self.out = nn.Linear(256, 3)

    def forward(self, f):
        h = self.net(f.float()).mean((2, 3))
        o = self.out(h)
        return F.normalize(o[:, :2], dim=-1), torch.sigmoid(o[:, 2])


class OffsetRoIHeads(RoIHeads):
    """在 torchvision RoIHeads 上增加偏移分支（复用 mask_roi_pool 的 14×14 RoI 特征）。"""

    def setup_offset(self, use_global=False):
        self.offset_head = OffsetHead()
        self.use_global = use_global
        if use_global:
            self.global_dir = GlobalDirHead()

    @staticmethod
    @torch.autocast("cuda", enabled=False)
    def geometric_offsets(o, centers, img_index, n_img, gdir=None):
        """按影像做几何共识，返回 shared / radial 两种参数化下的偏移（像素）与影像级几何。"""
        o = {k: v.float() for k, v in o.items()}
        centers = centers.float()
        mag = o["mag"] * OFF_SCALE
        sh = torch.zeros(len(mag), 2, device=mag.device)
        ra = torch.zeros(len(mag), 2, device=mag.device)
        geo = []
        for k in range(n_img):
            sel = img_index == k
            if sel.sum() == 0:
                geo.append(None)
                continue
            w = o["conf"][sel] * (o["mag"][sel].detach() + 0.05)
            u = consensus_parallel(o["dir"][sel], w)
            if gdir is not None:                                                # 与影像级方向融合（等效为 Σw 倍权重的伪观测）
                gu, gc = gdir[0][k], gdir[1][k]
                u = F.normalize(u * w.sum() + gu * gc * w.sum().clamp(min=1.0), dim=-1)
            sh[sel] = mag[sel, None] * u[None]
            if sel.sum() >= 3:
                dirs_k = o["dir"][sel]
                if gdir is not None:                                            # 用融合后的方向修正逐栋证据（避免单栋噪声主导直线束交点）
                    dirs_k = F.normalize(dirs_k + u[None], dim=-1)
                d, N, rho = consensus_radial(centers[sel], dirs_k, w)
            else:
                d, N, rho = u[None].expand(int(sel.sum()), 2), None, torch.zeros(())
            ra[sel] = mag[sel, None] * d
            geo.append({"u": u, "N": N, "rho": rho})
        return sh, ra, geo

    def forward(self, features, proposals, image_shapes, targets=None):
        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = regression_targets = matched_idxs = None
        box_features = self.box_head(self.box_roi_pool(features, proposals, image_shapes))
        class_logits, box_regression = self.box_predictor(box_features)
        result, losses = [], {}
        if self.training:
            lc, lb = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            losses = {"loss_classifier": lc, "loss_box_reg": lb}
        else:
            boxes, scores, labs = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            result = [{"boxes": b, "labels": l, "scores": s} for b, s, l in zip(boxes, scores, labs)]
        if self.training:
            mask_proposals, pos_matched_idxs = [], []
            for i in range(len(proposals)):
                pos = torch.where(labels[i] > 0)[0]
                mask_proposals.append(proposals[i][pos])
                pos_matched_idxs.append(matched_idxs[i][pos])
        else:
            mask_proposals = [r["boxes"] for r in result]
        mfeat = self.mask_roi_pool(features, mask_proposals, image_shapes)
        mask_logits = self.mask_predictor(self.mask_head(mfeat))
        # 偏移分支在扩大的 RoI 上取特征：倾倒方向由立面揭示，而立面位于屋顶框之外
        big = [enlarge_boxes(p, image_shapes[i]) for i, p in enumerate(mask_proposals)]
        ofeat = self.mask_roi_pool(features, big, image_shapes)
        with torch.autocast("cuda", enabled=False):
            o = self.offset_head(ofeat.float())
        img_index = torch.cat([torch.full((len(p),), i, device=mfeat.device) for i, p in enumerate(mask_proposals)])
        boxes_all = torch.cat(mask_proposals)
        centers = (boxes_all[:, :2] + boxes_all[:, 2:]) / 2
        gdir = None
        if getattr(self, "use_global", False):
            with torch.autocast("cuda", enabled=False):
                gdir = self.global_dir(features["2"])
        sh, ra, geo = self.geometric_offsets(o, centers, img_index, len(mask_proposals), gdir)
        if self.training:
            losses["loss_mask"] = maskrcnn_loss(mask_logits, mask_proposals, [t["masks"] for t in targets],
                                                [t["labels"] for t in targets], pos_matched_idxs)
            gt = torch.cat([t["offsets"][m] for t, m in zip(targets, pos_matched_idxs)]) if len(boxes_all) else \
                torch.zeros(0, 2, device=mfeat.device)
            if len(gt):
                g = gt / OFF_SCALE
                gm = g.norm(dim=-1)
                losses["loss_off_free"] = F.smooth_l1_loss(o["free"], g, beta=0.1)
                losses["loss_off_shared"] = F.smooth_l1_loss(sh / OFF_SCALE, g, beta=0.1)
                losses["loss_off_radial"] = F.smooth_l1_loss(ra / OFF_SCALE, g, beta=0.1)
                losses["loss_off_mag"] = F.smooth_l1_loss(o["mag"], gm, beta=0.1)
                wd = (gm > 0.1).float()
                gd = F.normalize(g, dim=-1)
                losses["loss_off_dir"] = ((1 - (o["dir"] * gd).sum(-1)) * wd).sum() / wd.sum().clamp(min=1)
                # 置信度：方向证据与真值方向一致时置信度高
                tgt = ((o["dir"].detach() * gd).sum(-1) > math.cos(math.radians(10))).float()
                if gdir is not None:                                            # 影像级方向监督：真值偏移的幅值加权平均方向
                    lg = 0
                    for k, t in enumerate(targets):
                        to = t["offsets"]
                        mm = to.norm(dim=-1)
                        if (mm > 5).any():
                            ug = F.normalize((to * mm[:, None]).sum(0), dim=-1)
                            lg = lg + (1 - (gdir[0][k] * ug).sum())
                    losses["loss_off_gdir"] = lg / len(targets)
                losses["loss_off_conf"] = 0.2 * (F.binary_cross_entropy_with_logits(o["conf_logit"], tgt, reduction="none") * wd).sum() / wd.sum().clamp(min=1)
        else:
            probs = maskrcnn_inference(mask_logits, [r["labels"] for r in result])
            start = 0
            for k, (mp, r) in enumerate(zip(probs, result)):
                n = len(r["boxes"])
                r["masks"] = mp
                r["off_free"] = o["free"][start:start + n] * OFF_SCALE
                r["off_shared"] = sh[start:start + n]
                r["off_radial"] = ra[start:start + n]
                r["off_dir"], r["off_mag"], r["off_conf"] = o["dir"][start:start + n], o["mag"][start:start + n] * OFF_SCALE, o["conf"][start:start + n]
                r["geo"] = geo[k]
                start += n
        return result, losses


def build_model(pretrained_ckpt=None, max_det=400, use_global=False):
    m = maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, num_classes=91, min_size=1024, max_size=1024,
                                 box_detections_per_img=max_det, rpn_post_nms_top_n_train=3000,
                                 rpn_post_nms_top_n_test=2000, box_batch_size_per_image=512)
    if pretrained_ckpt and os.path.exists(pretrained_ckpt):
        m.load_state_dict(torch.load(pretrained_ckpt, map_location="cpu", weights_only=True))
    m.roi_heads.box_predictor = FastRCNNPredictor(1024, 2)
    m.roi_heads.mask_predictor = MaskRCNNPredictor(256, 256, 2)
    rh = m.roi_heads
    new = OffsetRoIHeads(rh.box_roi_pool, rh.box_head, rh.box_predictor, 0.5, 0.5, 512, 0.25, None,
                         rh.score_thresh, rh.nms_thresh, rh.detections_per_img,
                         mask_roi_pool=rh.mask_roi_pool, mask_head=rh.mask_head, mask_predictor=rh.mask_predictor)
    new.setup_offset(use_global)
    m.roi_heads = new
    return m


# ------------------------------------------------------------------------------------------- 评测
def _shift(mask, dx, dy):
    H, W = mask.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)


def _poly_mask(p, H, W):
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(p).astype(np.int32)], 1)
    return m


def match_f1(pred_masks, scores, gt_masks, gt_ignore, thr=0.5):
    """一对一贪心匹配（按得分降序），IoU≥thr 为 TP；与 ignore 真值匹配的预测不计 FP。返回 (tp, fp, fn)。"""
    n_gt = len(gt_masks)
    if len(pred_masks) == 0:
        return 0, 0, int((~gt_ignore).sum()) if n_gt else 0
    P = np.stack([m.reshape(-1) for m in pred_masks]).astype(np.float32)
    if n_gt == 0:
        return 0, len(pred_masks), 0
    G = np.stack([m.reshape(-1) for m in gt_masks]).astype(np.float32)
    inter = P @ G.T
    union = P.sum(1)[:, None] + G.sum(1)[None] - inter
    iou = inter / np.maximum(union, 1)
    used = np.zeros(n_gt, bool)
    tp = fp = 0
    for i in np.argsort(-scores):
        j = int(np.argmax(np.where(used, -1, iou[i])))
        if iou[i, j] >= thr and not used[j]:
            used[j] = True
            if not gt_ignore[j]:
                tp += 1
        else:
            fp += 1
    fn = int(((~used) & (~gt_ignore)).sum())
    return tp, fp, fn
