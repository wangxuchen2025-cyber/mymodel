"""实例级视频合成：在第一版高度场渲染器基础上，额外渲染实例 ID，并给出逐帧
屋顶/基底图像多边形、η=h/H、可见度、跨帧建筑 ID、GNSS→真实 的序列级相似变换真值、
各历史源（SHP/航摄/卫星/地图）的真值多边形，以及当前↔历史的实例对应与变化状态。"""
import math
from typing import Dict

import torch
import torch.nn.functional as F

from geomac.config import GeoMACCfg
from geomac.geometry import intrinsics, rot_world_from_cam, world_rays
from geomac.synth import SceneSynth, SynthCfg, _rects_to_corners, resample_ring
from .config import GrailCDCfg
from .geometry import ground_to_img


def _v1_cfg(cfg: GrailCDCfg) -> GeoMACCfg:
    c = GeoMACCfg()
    c.backbone = cfg.backbone
    c.canvas_hw = cfg.raster_hw
    c.bev_res = cfg.raster_res
    c.dec.n_poly = cfg.det.n_poly
    c.dec.max_prior = cfg.max_hist
    return c


class VideoSynth(SceneSynth):
    def __init__(self, cfg: GrailCDCfg, scfg: SynthCfg = SynthCfg(), device="cuda"):
        super().__init__(_v1_cfg(cfg), scfg, device)
        self.gcfg = cfg
        self._log = []

    def _shift(self, rects, sigma_m, rot_sigma=0.01):
        out = super()._shift(rects, sigma_m, rot_sigma)
        self._log.append(out)
        return out

    def _march_ids(self, field, ids_f, meta, hw):
        img, depth, surf = self._march(field, meta, hw)
        H, W = hw
        hc, wc = self.cfg.canvas_hw
        r = self.cfg.bev_res
        K = intrinsics(meta[6:7], hw)
        R = rot_world_from_cam(meta[3:4], meta[4:5], meta[5:6])
        rays = world_rays(K, R, (H, W), 1)[0]
        p = meta[:3] + depth[..., None] * rays
        gp = torch.stack([(p[..., 0] + wc * r / 2) / (wc * r) * 2 - 1, (hc * r / 2 - p[..., 1]) / (hc * r) * 2 - 1], -1)
        ids = F.grid_sample(ids_f[None, None], gp[None], mode="nearest", align_corners=False,
                            padding_mode="border")[0, 0].round().long()
        ids = torch.where((surf == 1) | (surf == 2), ids, torch.full_like(ids, -1))
        return img, depth, surf, ids

    def frame_gt(self, sc, meta_true, surf, ids, kmax):
        """逐帧实例真值（真实相机投影）。"""
        g = self.gcfg
        H, W = g.backbone.img_hw
        V = g.det.n_poly
        cur = sc["cur"]
        corners = _rects_to_corners(cur)                                        # n,4,2
        ring = resample_ring(corners, V)
        m = meta_true[None]
        K = intrinsics(m[:, 6], (H, W))
        R = rot_world_from_cam(m[:, 3], m[:, 4], m[:, 5])
        C = m[:, :3]
        hts = sc["cur_h"]
        roof_uv = torch.stack([ground_to_img(ring[k][None, None], K, R, C, z=float(hts[k]))[0][0, 0] for k in range(len(cur))])
        foot_uv = ground_to_img(ring[None], K, R, C, z=0.0)[0][0]
        nrm = torch.tensor([W, H], device=ring.device)
        roof_n, foot_n = roof_uv / nrm, foot_uv / nrm
        cnt = torch.bincount((ids + 1).flatten(), minlength=len(cur) + 1)[1:].float()
        top_cnt = torch.bincount((torch.where(surf == 1, ids, torch.full_like(ids, -1)) + 1).flatten(),
                                 minlength=len(cur) + 1)[1:].float()
        x, y = roof_uv[..., 0], roof_uv[..., 1]
        area = 0.5 * (x * torch.roll(y, -1, 1) - torch.roll(x, -1, 1) * y).sum(1).abs().clamp(min=1.0)
        vis = (top_cnt / area).clamp(0, 1)
        cen = roof_n.mean(1)
        ok = (vis > 0.15) & (cen > 0).all(-1) & (cen < 1).all(-1) & (cnt > 30)
        eta = hts / meta_true[2]
        h8, w8 = H // 8, W // 8
        roof_mask = torch.stack([F.adaptive_avg_pool2d(((ids == k) & (surf == 1)).float()[None, None], (h8, w8))[0, 0]
                                 for k in range(len(cur))])
        pad = lambda t, v=0: torch.cat([t, torch.full((kmax - len(t), *t.shape[1:]), v, device=t.device, dtype=t.dtype)])[:kmax]
        return {"roof": pad(roof_n), "foot": pad(foot_n), "eta": pad(eta), "valid": pad(ok, False),
                "bid": pad(torch.arange(len(cur), device=ring.device), -1), "roof_mask": pad(roof_mask)}

    def seq_sim(self, metas, true_metas, depths, surfs, stride=8):
        """序列级 GNSS->真实 相似变换（米，绕原点）：用真实深度在噪声/真实元数据下抬升的点对拟合。"""
        hw = self.gcfg.backbone.img_hw
        P, Q = [], []
        for t in range(len(metas)):
            d = depths[t][stride // 2::stride, stride // 2::stride]
            ok = surfs[t][stride // 2::stride, stride // 2::stride] < 3
            for meta, out in ((metas[t], P), (true_metas[t], Q)):
                K = intrinsics(meta[6:7], hw)
                R = rot_world_from_cam(meta[3:4], meta[4:5], meta[5:6])
                out.append((meta[:3] + d[..., None] * world_rays(K, R, d.shape, stride)[0])[ok][:, :2])
        P, Q = torch.cat(P), torch.cat(Q)
        mp, mq = P.mean(0), Q.mean(0)
        X, Y = P - mp, Q - mq
        a = (X * Y).sum()
        b = (X[:, 0] * Y[:, 1] - X[:, 1] * Y[:, 0]).sum()
        th = torch.atan2(b, a)
        s = torch.sqrt(a ** 2 + b ** 2) / (X ** 2).sum()
        R = torch.stack([torch.stack([th.cos(), -th.sin()]), torch.stack([th.sin(), th.cos()])])
        t = mq - s * R @ mp
        return torch.stack([t[0], t[1], th, s.log()])

    def sample(self, batch, n_frames, drop=True, kmax=24) -> Dict:
        g, dev = self.gcfg, self.dev
        V, mh = g.det.n_poly, g.max_hist
        outs = []
        for _ in range(batch):
            sc = self.scene()
            self._log = []
            hr = self.render_history(sc)
            aer_r, sat_r, map_r, shp_r = self._log[:4]
            # —— 当前视频（含实例 ID）——
            ids_f = self._id_map(sc["cur"]).float()
            fr, me, de, su, tm = self.render_frames(sc, hr, n_frames)
            ids_list, gts = [], []
            for t in range(n_frames):
                _, _, _, ids = self._march_ids(self._field_cache, ids_f, tm[t], g.backbone.img_hw)
                gts.append(self.frame_gt(sc, tm[t], su[t], ids, kmax))
            sim = self.seq_sim(me, tm, de, su)
            to_m = lambda rects: resample_ring(_rects_to_corners(rects), V) if len(rects) else torch.zeros(0, V, 2, device=dev)
            pad = lambda t, n, v=0: torch.cat([t, torch.full((n - len(t), *t.shape[1:]), v, device=dev, dtype=t.dtype)])[:n]
            nh = len(sc["hist"])
            outs.append({
                "frames": fr, "metas": me, "true_metas": tm,
                "gt": {k: torch.stack([gg[k] for gg in gts]) for k in gts[0]},
                "sim": sim,
                "aerial": hr["aerial"], "sat": hr["sat"], "map": hr["map"],
                "shp_poly": pad(to_m(shp_r), mh), "shp_mask": pad(torch.ones(nh, device=dev, dtype=torch.bool), mh, False),
                "src_poly": torch.stack([pad(to_m(r), mh) for r in (aer_r, sat_r, map_r)]),
                "hist_poly": pad(to_m(sc["hist"]), mh), "hist_state": pad(sc["state"], mh, -1),
                "cur_poly": pad(to_m(sc["cur"]), kmax), "cur_src": pad(sc["cur_src"], kmax, -2),
                "cur_h": pad(sc["cur_h"], kmax),
            })
        b = {}
        for k in outs[0]:
            if k == "gt":
                b["gt"] = {kk: torch.stack([o["gt"][kk] for o in outs]) for kk in outs[0]["gt"]}
            else:
                b[k] = torch.stack([o[k] for o in outs])
        hist = {"aerial": b["aerial"], "sat": b["sat"], "map": b["map"],
                "shp": {"poly": b["shp_poly"], "mask": b["shp_mask"]},
                "aerial_log_gsd": torch.full((batch,), math.log(0.5), device=dev),
                "sat_log_gsd": torch.full((batch,), math.log(2.0), device=dev)}
        if drop:
            names = ["aerial", "sat", "map", "shp"]
            keep = [n for n in names if torch.rand(1).item() > self.s.p_drop] or [names[torch.randint(0, 4, (1,)).item()]]
            for n in names:
                if n not in keep:
                    hist[n] = None
        b["hist"] = hist
        return b

    def render_frames(self, sc, hist_r, n_frames):
        # 缓存字段以便 ID 渲染复用（与父类完全相同的高度/颜色场）
        orig = self._march
        cache = {}

        def wrap(field, meta, hw):
            cache["f"] = field
            return orig(field, meta, hw)
        self._march = wrap
        try:
            out = super().render_frames(sc, hist_r, n_frames)
        finally:
            self._march = orig
        self._field_cache = cache["f"]
        return out
