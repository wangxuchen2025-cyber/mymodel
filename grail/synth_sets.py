"""集合级合成数据：城市式建筑布局（街区网格 + 成排相似建筑 + 零散建筑），施加变化、视频覆盖、
GNSS 全局相似变换偏差、倾斜径向残差、检测漏检/误检与外观嵌入噪声。

用途：在不依赖影像的条件下单独训练/评测 GRAIL 层（对齐 + 关联），并可直接替换为真实城市 SHP
布局（把 make_layout 换成读取真实矢量）——这是"空间关系"研究的低成本实验平台。
"""
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from geomac.synth import resample_ring


@dataclass
class SetSynthCfg:
    ext: float = 300.0                       # AOI 边长（米）
    n_poly: int = 16
    emb_dim: int = 64
    p_demolish: float = 0.12
    p_modify: float = 0.15
    new_frac: float = 0.12
    strip_w: Tuple[float, float] = (120.0, 200.0)   # 视频覆盖条带宽
    sigma_t: float = 6.0                     # GNSS 平移偏差（米）
    sigma_theta_deg: float = 3.0
    sigma_s: float = 0.03
    resid_m: float = 0.7                     # 逐栋残差
    radial_m: float = 0.8                    # 倾斜径向残差（垂直于航线方向）
    vertex_m: float = 0.3
    hist_offset_m: float = 1.0               # 历史数据系统偏移（如 SHP 偏移）
    p_miss: float = 0.05
    p_fp: float = 0.03
    p_hist_missing: float = 0.03             # 历史数据本身缺失的建筑
    emb_noise: float = 1.0                   # 外观噪声（相对信号 1.0）；越大外观越不可靠
    max_a: int = 160
    max_b: int = 200


def _rect(cx, cy, w, h, a):
    c, s = math.cos(a), math.sin(a)
    pts = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    return [(cx + x * c - y * s, cy + x * s + y * c) for x, y in pts]


def _lshape(cx, cy, w, h, a, fx, fy):
    """在矩形右上角挖去 (fx·w, fy·h) 的 L 形（逆时针）。"""
    c, s = math.cos(a), math.sin(a)
    x0, x1, y0, y1 = -w / 2, w / 2, -h / 2, h / 2
    xm, ym = x1 - fx * w, y1 - fy * h
    pts = [(x0, y0), (x1, y0), (x1, ym), (xm, ym), (xm, y1), (x0, y1)]
    return [(cx + x * c - y * s, cy + x * s + y * c) for x, y in pts]


def make_layout(ext, rng: random.Random) -> List[Dict]:
    U = rng.uniform
    psi = U(-math.pi / 4, math.pi / 4)
    bx, by = U(55, 90), U(45, 75)
    street = U(10, 16)
    out = []
    c, s = math.cos(psi), math.sin(psi)
    n = int(ext / min(bx, by)) + 2
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            if rng.random() < 0.15:                                           # 公园/空地
                continue
            ox, oy = i * bx, j * by
            for row in (0, 1):                                                # 街区前后两排
                w0, d0 = U(8, 18), U(9, 16)
                gap = U(2.5, 7)
                ry = oy - by / 2 + street / 2 + d0 / 2 + row * (by - street - d0) * U(0.85, 1.0)
                x = ox - bx / 2 + street / 2 + w0 / 2
                while x + w0 / 2 < ox + bx / 2 - street / 2:
                    w = w0 * U(0.93, 1.07)
                    d = d0 * U(0.93, 1.07)
                    lx, ly = x, ry
                    cx, cy = lx * c - ly * s, lx * s + ly * c
                    if abs(cx) < ext / 2 - 10 and abs(cy) < ext / 2 - 10:
                        a = psi + U(-0.03, 0.03)
                        if rng.random() < 0.12:
                            out.append({"pts": _lshape(cx, cy, w, d, a, U(0.3, 0.5), U(0.3, 0.5)), "c": (cx, cy),
                                        "w": w, "h": d, "a": a})
                        else:
                            out.append({"pts": _rect(cx, cy, w, d, a), "c": (cx, cy), "w": w, "h": d, "a": a})
                    x += w0 + gap
    # 零散建筑（任意朝向）
    for _ in range(int(U(3, 12))):
        cx, cy = U(-ext / 2 + 15, ext / 2 - 15), U(-ext / 2 + 15, ext / 2 - 15)
        if all(math.hypot(cx - b["c"][0], cy - b["c"][1]) > 18 for b in out):
            w, d, a = U(8, 25), U(8, 20), U(-math.pi, math.pi)
            out.append({"pts": _rect(cx, cy, w, d, a), "c": (cx, cy), "w": w, "h": d, "a": a})
    return out


def _ring(pts, n):
    t = torch.tensor(pts, dtype=torch.float32)[None]
    return resample_ring(t, n)[0]


def _modify(b, rng: random.Random):
    U = rng.uniform
    kind = rng.randrange(3)
    cx, cy, w, h, a = b["c"][0], b["c"][1], b["w"], b["h"], b["a"]
    if kind == 0:                                                             # 扩建：沿进深方向延伸一侧
        f = U(1.3, 1.9)
        d = h * (f - 1) / 2
        sg = 1 if rng.random() < 0.5 else -1
        return _rect(cx - sg * d * math.sin(a), cy + sg * d * math.cos(a), w, h * f, a)
    if kind == 1:                                                             # 缩减/部分拆除
        f = U(0.5, 0.75)
        d = h * (1 - f) / 2
        return _rect(cx + d * math.sin(a), cy - d * math.cos(a), w, h * f, a)
    return _lshape(cx, cy, w * U(1.0, 1.3), h, a, U(0.35, 0.5), U(0.35, 0.5))   # 改建为 L 形


class SetSynth:
    def __init__(self, cfg: SetSynthCfg = SetSynthCfg(), seed=0, device="cuda"):
        self.cfg, self.dev = cfg, device
        self.rng = torch.Generator().manual_seed(seed)
        self.r = random.Random(seed)

    def one(self, sigma_t=None, emb_noise=None):
        cfg, rng, r = self.cfg, self.rng, self.r
        U = r.uniform
        N = lambda s: r.gauss(0.0, 1.0) * s
        lay = make_layout(cfg.ext, r)
        nv = cfg.n_poly
        lat = torch.randn(len(lay), cfg.emb_dim, generator=rng)
        # —— 历史集合 ——
        hist_idx = [k for k in range(len(lay)) if r.random() > cfg.p_hist_missing]
        hoff = torch.tensor([N(cfg.hist_offset_m), N(cfg.hist_offset_m)])
        # —— 变化 ——
        state, cur = {}, []                                                   # cur: (poly, src_layout_idx or -1)
        for k, b in enumerate(lay):
            u = r.random()
            if u < cfg.p_demolish:
                state[k] = "demolished"
            elif u < cfg.p_demolish + cfg.p_modify:
                state[k] = "modified"
                cur.append((_ring(_modify(b, r), nv), k))
            else:
                state[k] = "unchanged"
                cur.append((_ring(b["pts"], nv), k))
        occupied = [b["c"] for b in lay]
        for _ in range(int(len(lay) * cfg.new_frac)):
            for _try in range(20):
                cx, cy = U(-cfg.ext / 2 + 12, cfg.ext / 2 - 12), U(-cfg.ext / 2 + 12, cfg.ext / 2 - 12)
                if all(math.hypot(cx - o[0], cy - o[1]) > 16 for o in occupied):
                    occupied.append((cx, cy))
                    cur.append((_ring(_rect(cx, cy, U(8, 22), U(8, 18), U(-math.pi, math.pi)), nv), -1))
                    break
        # —— 视频覆盖条带 ——
        ang = U(0, math.pi)
        dirv = torch.tensor([math.cos(ang), math.sin(ang)])
        nrm = torch.tensor([-dirv[1], dirv[0]])
        off = U(-40, 40)
        half = U(*cfg.strip_w) / 2
        def vis_of(c):
            dd = (c @ nrm) - off
            return ((half - dd.abs()) / 10).clamp(0, 1)
        # —— 当前集合（GNSS 系）——
        st = cfg.sigma_t if sigma_t is None else sigma_t
        en = cfg.emb_noise if emb_noise is None else emb_noise
        th = math.radians(N(cfg.sigma_theta_deg))
        ls = N(cfg.sigma_s)
        t_gt = torch.tensor([N(st), N(st)])
        A_poly, A_emb, A_src = [], [], []
        for poly, k in cur:
            c = poly.mean(0)
            if vis_of(c[None])[0] < 0.5 or r.random() < cfg.p_miss:
                continue
            dd = (c @ nrm) - off
            radial = nrm * (N(cfg.radial_m) + 0.004 * dd)                    # 倾斜径向残差（随离航线距离增大）
            res = torch.tensor([N(cfg.resid_m), N(cfg.resid_m)]) + radial
            p = poly + res + torch.randn(poly.shape, generator=rng) * cfg.vertex_m
            A_poly.append(p)
            if k >= 0:
                sig = lat[k] if state[k] == "unchanged" else 0.5 * lat[k] + 0.5 * torch.randn(cfg.emb_dim, generator=rng)
            else:
                sig = torch.randn(cfg.emb_dim, generator=rng)
            A_emb.append(sig + en * torch.randn(cfg.emb_dim, generator=rng))
            A_src.append(k)
        for _ in range(int(len(A_poly) * cfg.p_fp)):                           # 误检
            cx, cy = U(-cfg.ext / 2, cfg.ext / 2), U(-cfg.ext / 2, cfg.ext / 2)
            A_poly.append(_ring(_rect(cx, cy, U(4, 9), U(4, 9), U(-3, 3)), nv))
            A_emb.append(torch.randn(cfg.emb_dim, generator=rng) * (1 + en))
            A_src.append(-2)
        A_true = torch.stack(A_poly) if A_poly else torch.zeros(0, nv, 2)
        # GNSS 系 = T_gt^{-1}(真实系)，T_gt 绕原点：q = s R p + t
        R = torch.tensor([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        A_gnss = ((A_true - t_gt) @ R) / math.exp(ls)
        B_poly = torch.stack([_ring(lay[k]["pts"], nv) + hoff + torch.randn(nv, 2, generator=rng) * 0.2 for k in hist_idx])
        B_emb = torch.stack([lat[k] + en * torch.randn(cfg.emb_dim, generator=rng) for k in hist_idx])
        B_vis = vis_of(B_poly.mean(1))
        hist_pos = {k: j for j, k in enumerate(hist_idx)}
        corr_a = torch.tensor([hist_pos.get(k, -1) if k >= 0 else -1 for k in A_src], dtype=torch.long)
        corr_b = torch.full((len(hist_idx),), -1, dtype=torch.long)
        for i, j in enumerate(corr_a.tolist()):
            if j >= 0:
                corr_b[j] = i
        b_state = torch.tensor([{"unchanged": 0, "demolished": 1, "modified": 2}[state[k]] for k in hist_idx])
        a_state = torch.tensor([({"unchanged": 0, "modified": 2}.get(state.get(k, ""), 3) if k >= 0 else 3)
                                for k in A_src])                              # 3 = 新建/误检/历史缺失
        return {"A_poly": A_gnss, "A_true": A_true, "A_emb": torch.stack(A_emb) if A_emb else torch.zeros(0, cfg.emb_dim), "B_poly": B_poly,
                "B_emb": B_emb, "B_vis": B_vis, "corr_a": corr_a, "corr_b": corr_b, "b_state": b_state,
                "a_state": a_state, "x_gt": torch.tensor([t_gt[0] + hoff[0], t_gt[1] + hoff[1], th, ls]), "hoff": hoff}

    def batch(self, b, sigma_t=None, emb_noise=None) -> Dict:
        cfg, dev = self.cfg, self.dev
        items = [self.one(sigma_t, emb_noise) for _ in range(b)]
        M = min(max(len(it["A_poly"]) for it in items), cfg.max_a)
        Nn = min(max(len(it["B_poly"]) for it in items), cfg.max_b)
        nv = cfg.n_poly

        def pad(x, n, fill=0.0):
            x = x[:n]
            if len(x) < n:
                x = torch.cat([x, torch.full((n - len(x), *x.shape[1:]), fill, dtype=x.dtype)])
            return x
        out = {
            "A_poly": torch.stack([pad(it["A_poly"], M) for it in items]),
            "A_true": torch.stack([pad(it["A_true"], M) for it in items]),
            "A_emb": torch.stack([pad(it["A_emb"], M) for it in items]),
            "A_mask": torch.stack([pad(torch.ones(len(it["A_poly"]), dtype=torch.bool), M, False) for it in items]),
            "B_poly": torch.stack([pad(it["B_poly"], Nn) for it in items]),
            "B_emb": torch.stack([pad(it["B_emb"], Nn) for it in items]),
            "B_mask": torch.stack([pad(torch.ones(len(it["B_poly"]), dtype=torch.bool), Nn, False) for it in items]),
            "B_vis": torch.stack([pad(it["B_vis"], Nn) for it in items]),
            "corr_a": torch.stack([pad(torch.where(it["corr_a"] < Nn, it["corr_a"], -1), M, -1) for it in items]),
            "corr_b": torch.stack([pad(torch.where(it["corr_b"] < M, it["corr_b"], -1), Nn, -1) for it in items]),
            "b_state": torch.stack([pad(it["b_state"], Nn, -1) for it in items]),
            "a_state": torch.stack([pad(it["a_state"], M, -1) for it in items]),
            "x_gt": torch.stack([it["x_gt"] for it in items]),
        }
        out = {k: v.to(dev) for k, v in out.items()}
        out["A_poly"] = out["A_poly"].float()
        return out


def gt_x_about(x_gt, c0):
    """GT 相似变换（绕原点 q = sRp + t）-> 绕 c0 的参数。"""
    th, ls = x_gt[:, 2], x_gt[:, 3]
    s = ls.exp()
    c, sn = th.cos(), th.sin()
    Rc = torch.stack([c * c0[:, 0] - sn * c0[:, 1], sn * c0[:, 0] + c * c0[:, 1]], -1)
    t = x_gt[:, :2] + s[:, None] * Rc - c0
    return torch.cat([t, th[:, None], ls[:, None]], 1)


class SetStream(torch.utils.data.IterableDataset):
    """多进程无限流：每个 worker 用独立种子在 CPU 上生成 batch（训练时 DataLoader(num_workers=k, batch_size=None)）。"""

    def __init__(self, cfg: SetSynthCfg, batch: int, seed=0, sigma_t_range=(2.0, 15.0), emb_noise_range=(0.7, 2.5)):
        self.cfg, self.b, self.seed = cfg, batch, seed
        self.st, self.en = sigma_t_range, emb_noise_range

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else 0
        sy = SetSynth(self.cfg, seed=self.seed * 1000 + wid, device="cpu")
        while True:
            yield sy.batch(self.b, sigma_t=sy.r.uniform(*self.st), emb_noise=sy.r.uniform(*self.en))
