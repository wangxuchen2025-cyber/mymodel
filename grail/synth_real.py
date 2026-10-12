"""基于真实城市布局（OSM）的集合级合成：真实的建筑位置/形状/朝向/间距分布，施加与程序化平台相同的
变化与观测扰动，另外加入"拆除后原址新建"（位置重合、身份不同，检验严格一对一）。

新建建筑的形状从同一区域的真实建筑中抽取（与周边建筑同分布，比程序化矩形更难与历史建筑区分）。
窗口边长按区域建筑密度自适应，使每个样本约有 60–140 栋历史建筑。
"""
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from .osm_layouts import load_all, self_similarity

TRAIN_AREAS = ("tokyo_setagaya", "amsterdam_west", "berlin_marzahn", "sh_oldtown", "bj_tiantongyuan", "wh_estate")
TEST_AREAS = ("florence_old", "phoenix_suburb", "sh_pudong_estate", "sz_urbanvillage")


@dataclass
class RealSetCfg:
    target_n: Tuple[int, int] = (60, 140)
    side_range: Tuple[float, float] = (150.0, 420.0)
    emb_dim: int = 64
    p_demolish: float = 0.12
    p_rebuild: float = 0.3                   # 拆除中"原址新建"的比例
    p_modify: float = 0.15
    new_frac: float = 0.10
    strip_frac: Tuple[float, float] = (0.55, 0.85)   # 覆盖条带宽 / 窗口边长
    sigma_t: float = 6.0
    sigma_theta_deg: float = 3.0
    sigma_s: float = 0.03
    resid_m: float = 0.7
    radial_m: float = 0.8
    vertex_m: float = 0.3
    hist_offset_m: float = 1.0
    p_miss: float = 0.05
    p_fp: float = 0.03
    p_hist_missing: float = 0.03
    emb_noise: float = 1.0
    max_a: int = 200
    max_b: int = 240


def _principal(p):
    c = p.mean(0)
    q = p - c
    cov = q.T @ q / len(q)
    ev, evec = torch.linalg.eigh(cov)
    return c, evec[:, 1], evec[:, 0]                                            # 质心, 主轴, 次轴


def modify_poly(p, r: random.Random):
    """真实多边形的改扩建：沿主轴单侧拉伸（扩建）/ 压缩（缩减），或沿次轴单侧加宽（改形）。"""
    c, u, v = _principal(p)
    q = p - c
    a, b = q @ u, q @ v
    kind = r.randrange(3)
    if kind < 2:
        f = r.uniform(1.3, 1.8) if kind == 0 else r.uniform(0.55, 0.8)
        if r.random() < 0.5:
            a = a.min() + (a - a.min()) * f
        else:
            a = a.max() + (a - a.max()) * f
    else:
        f = r.uniform(1.3, 1.7)
        b = b.min() + (b - b.min()) * f
    return c + a[:, None] * u + b[:, None] * v


def _rot(p, th, c=None):
    c = p.mean(0) if c is None else c
    R = torch.tensor([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    return (p - c) @ R.T + c


class RealSetSynth:
    def __init__(self, areas, cfg: RealSetCfg = RealSetCfg(), seed=0, device="cpu", half: Optional[str] = None,
                 data_dir="data/osm"):
        """half：'west' / 'east' 只在区域的西半/东半取窗口（同一城市内空间不相交的训练/测试划分）。"""
        L = load_all(data_dir)
        self.L = {k: L[k] for k in areas if k in L}
        self.cfg, self.dev, self.half = cfg, device, half
        self.r = random.Random(seed)
        self.g = torch.Generator().manual_seed(seed)

    def _window(self, name):
        cfg, r = self.cfg, self.r
        a = self.L[name]
        x0, x1, y0, y1 = a["ext"]
        dens = len(a["poly"]) / ((x1 - x0) * (y1 - y0))
        side = min(max(math.sqrt(r.uniform(*cfg.target_n) / dens), cfg.side_range[0]), cfg.side_range[1])
        side = min(side, 0.5 * (x1 - x0) - 10, (y1 - y0) - 10)
        lo_x, hi_x = x0 + side / 2, x1 - side / 2
        if self.half == "west":
            hi_x = min(hi_x, -side / 2)
        elif self.half == "east":
            lo_x = max(lo_x, side / 2)
        cx, cy = r.uniform(lo_x, hi_x), r.uniform(y0 + side / 2, y1 - side / 2)
        c = a["cent"]
        m = ((c[:, 0] - cx).abs() < side / 2) & ((c[:, 1] - cy).abs() < side / 2)
        return a, m, torch.tensor([cx, cy]), side

    def one(self, sigma_t=None, emb_noise=None, area=None) -> Dict:
        cfg, r, g = self.cfg, self.r, self.g
        N = lambda s: r.gauss(0.0, 1.0) * s
        name = area or r.choice(list(self.L))
        for _ in range(20):
            a, m, wc, side = self._window(name)
            if m.sum() >= 15:
                break
        lay = a["poly"][m] - wc                                                  # 以窗口中心为原点
        areas = a["area"][m]
        pool = a["poly"][~m]
        n0 = len(lay)
        lat = torch.randn(n0, cfg.emb_dim, generator=g)
        hist_idx = [k for k in range(n0) if r.random() > cfg.p_hist_missing]
        hoff = torch.tensor([N(cfg.hist_offset_m), N(cfg.hist_offset_m)])
        state, cur, rebuilt = {}, [], {}
        cent = lay.mean(1)
        rad = (areas.sqrt() * 0.75).tolist()
        occupied = [(float(cent[k, 0]), float(cent[k, 1]), rad[k]) for k in range(n0)]

        def sample_shape(at, orient_ref=None):
            q = pool[r.randrange(len(pool))] if len(pool) else lay[r.randrange(n0)]
            q = q - q.mean(0) + at
            if orient_ref is not None:                                         # 与邻近建筑朝向对齐（±10°）
                _, u0, _ = _principal(q)
                _, u1, _ = _principal(orient_ref)
                th = math.atan2(float(u1[1]), float(u1[0])) - math.atan2(float(u0[1]), float(u0[0]))
                q = _rot(q, th + math.radians(r.uniform(-10, 10)), at)
            else:
                q = _rot(q, r.uniform(-math.pi, math.pi), at)
            return q
        for k in range(n0):
            u = r.random()
            if u < cfg.p_demolish:
                state[k] = "demolished"
                if r.random() < cfg.p_rebuild:                                  # 原址新建（不同建筑）
                    cur.append((sample_shape(cent[k], lay[k]), -1))
                    rebuilt[len(cur) - 1] = k
            elif u < cfg.p_demolish + cfg.p_modify:
                state[k] = "modified"
                cur.append((modify_poly(lay[k], r), k))
            else:
                state[k] = "unchanged"
                cur.append((lay[k].clone(), k))
        for _ in range(int(n0 * cfg.new_frac)):
            for _t in range(30):
                at = torch.tensor([r.uniform(-side / 2 + 8, side / 2 - 8), r.uniform(-side / 2 + 8, side / 2 - 8)])
                q = sample_shape(at, lay[int(torch.cdist(at[None], cent).argmin())])
                rq = math.sqrt(max(float(0.5 * (q[:, 0] * torch.roll(q[:, 1], -1) - torch.roll(q[:, 0], -1) * q[:, 1]).sum().abs()), 1.0)) * 0.75
                if all(math.hypot(float(at[0]) - o[0], float(at[1]) - o[1]) > rq + o[2] + 2 for o in occupied):
                    occupied.append((float(at[0]), float(at[1]), rq))
                    cur.append((q, -1))
                    break
        # 视频覆盖条带
        ang = r.uniform(0, math.pi)
        nrm = torch.tensor([-math.sin(ang), math.cos(ang)])
        off = r.uniform(-0.15, 0.15) * side
        half = r.uniform(*cfg.strip_frac) * side / 2
        vis_of = lambda c: ((half - ((c @ nrm) - off).abs()) / 10).clamp(0, 1)
        st = cfg.sigma_t if sigma_t is None else sigma_t
        en = cfg.emb_noise if emb_noise is None else emb_noise
        th, ls = math.radians(N(cfg.sigma_theta_deg)), N(cfg.sigma_s)
        t_gt = torch.tensor([N(st), N(st)])
        A_poly, A_emb, A_src, A_rb, A_rbsrc = [], [], [], [], []
        for ci, (poly, k) in enumerate(cur):
            c = poly.mean(0)
            if vis_of(c[None])[0] < 0.5 or r.random() < cfg.p_miss:
                continue
            dd = (c @ nrm) - off
            res = torch.tensor([N(cfg.resid_m), N(cfg.resid_m)]) + nrm * (N(cfg.radial_m) + 0.004 * dd)
            A_poly.append(poly + res + torch.randn(poly.shape, generator=g) * cfg.vertex_m)
            if k >= 0:
                sig = lat[k] if state[k] == "unchanged" else 0.5 * lat[k] + 0.5 * torch.randn(cfg.emb_dim, generator=g)
            else:
                sig = torch.randn(cfg.emb_dim, generator=g)
            A_emb.append(sig + en * torch.randn(cfg.emb_dim, generator=g))
            A_src.append(k)
            A_rb.append(ci in rebuilt)
            A_rbsrc.append(rebuilt.get(ci, -1))
        for _ in range(int(len(A_poly) * cfg.p_fp)):
            at = torch.tensor([r.uniform(-side / 2, side / 2), r.uniform(-side / 2, side / 2)])
            q = sample_shape(at) * 0.6 + at * 0.4
            A_poly.append(q)
            A_emb.append(torch.randn(cfg.emb_dim, generator=g) * (1 + en))
            A_src.append(-2)
            A_rb.append(False)
            A_rbsrc.append(-1)
        A_true = torch.stack(A_poly) if A_poly else torch.zeros(0, lay.shape[1], 2)
        R = torch.tensor([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        A_gnss = ((A_true - t_gt) @ R) / math.exp(ls)
        B_poly = torch.stack([lay[k] + hoff + torch.randn(lay.shape[1], 2, generator=g) * 0.2 for k in hist_idx])
        B_emb = torch.stack([lat[k] + en * torch.randn(cfg.emb_dim, generator=g) for k in hist_idx])
        hist_pos = {k: j for j, k in enumerate(hist_idx)}
        # 覆盖栅格：当前集合在 GNSS 系（= 相机足迹按同样有偏的元数据计算所得），历史集合为数据范围（窗口）
        G = 64
        hw = side / 2
        corners = torch.tensor([[-hw, -hw], [hw, -hw], [hw, hw], [-hw, hw]])
        cg = ((corners - t_gt) @ R) / math.exp(ls)
        lo, hi = cg.min(0)[0] - 10, cg.max(0)[0] + 10
        gx = lo[0] + (torch.arange(G) + 0.5) / G * (hi[0] - lo[0])
        gy = hi[1] - (torch.arange(G) + 0.5) / G * (hi[1] - lo[1])
        yy, xx = torch.meshgrid(gy, gx, indexing="ij")
        pg = torch.stack([xx, yy], -1).reshape(-1, 2)
        q = math.exp(ls) * pg @ R.T + t_gt                                      # GNSS 系 -> 真实系
        inside = (q.abs() < hw).all(-1) & (((q @ nrm) - off).abs() < half)
        A_cov = inside.float().view(1, G, G)
        A_cov_bbox = torch.tensor([lo[0], lo[1], hi[0], hi[1]])
        B_cov = torch.ones(1, G, G)
        B_cov_bbox = torch.tensor([-hw + hoff[0], -hw + hoff[1], hw + hoff[0], hw + hoff[1]])
        corr_a = torch.tensor([hist_pos.get(k, -1) if k >= 0 else -1 for k in A_src], dtype=torch.long)
        corr_b = torch.full((len(hist_idx),), -1, dtype=torch.long)
        for i, j in enumerate(corr_a.tolist()):
            if j >= 0:
                corr_b[j] = i
        return {"A_poly": A_gnss, "A_true": A_true, "A_emb": torch.stack(A_emb) if A_emb else torch.zeros(0, cfg.emb_dim),
                "B_poly": B_poly, "B_emb": B_emb, "B_vis": vis_of(B_poly.mean(1)), "corr_a": corr_a, "corr_b": corr_b,
                "b_state": torch.tensor([{"unchanged": 0, "demolished": 1, "modified": 2}[state[k]] for k in hist_idx]),
                "a_state": torch.tensor([({"unchanged": 0, "modified": 2}.get(state.get(k, ""), 3) if k >= 0 else 3) for k in A_src]),
                "x_gt": torch.tensor([t_gt[0] + hoff[0], t_gt[1] + hoff[1], th, ls]),
                "area": name, "periodicity": self_similarity(lay.mean(1), areas, tol=2.5),
                "a_rebuild": torch.tensor(A_rb, dtype=torch.bool),
                "A_cov": A_cov, "A_cov_bbox": A_cov_bbox, "B_cov": B_cov, "B_cov_bbox": B_cov_bbox,
                "a_rebuild_src": torch.tensor([hist_pos.get(k, -1) if k >= 0 else -1 for k in A_rbsrc], dtype=torch.long)}

    def batch(self, b, sigma_t=None, emb_noise=None, area=None) -> Dict:
        cfg = self.cfg
        items = [self.one(sigma_t, emb_noise, area) for _ in range(b)]
        M = min(max(len(it["A_poly"]) for it in items), cfg.max_a)
        Nn = min(max(len(it["B_poly"]) for it in items), cfg.max_b)

        def pad(x, n, fill=0.0):
            x = x[:n]
            if len(x) < n:
                x = torch.cat([x, torch.full((n - len(x), *x.shape[1:]), fill, dtype=x.dtype)])
            return x
        out = {
            "A_poly": torch.stack([pad(it["A_poly"], M) for it in items]).float(),
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
            "periodicity": torch.tensor([it["periodicity"] for it in items]),
            "a_rebuild": torch.stack([pad(it["a_rebuild"], M, False) for it in items]),
            "A_cov": torch.stack([it["A_cov"] for it in items]), "A_cov_bbox": torch.stack([it["A_cov_bbox"] for it in items]),
            "B_cov": torch.stack([it["B_cov"] for it in items]), "B_cov_bbox": torch.stack([it["B_cov_bbox"] for it in items]),
            "a_rebuild_src": torch.stack([pad(torch.where(it["a_rebuild_src"] < Nn, it["a_rebuild_src"], -1), M, -1) for it in items]),
        }
        out = {k: v.to(self.dev) for k, v in out.items()}
        out["areas"] = [it["area"] for it in items]
        return out


class RealSetStream(torch.utils.data.IterableDataset):
    def __init__(self, areas, batch, seed=0, half="west", sigma_t_range=(2.0, 15.0), emb_noise_range=(0.7, 3.0)):
        self.areas, self.b, self.seed, self.half = areas, batch, seed, half
        self.st, self.en = sigma_t_range, emb_noise_range

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else 0
        sy = RealSetSynth(self.areas, seed=self.seed * 1000 + wid, half=self.half)
        while True:
            b = sy.batch(self.b, sigma_t=sy.r.uniform(*self.st), emb_noise=sy.r.uniform(*self.en))
            b.pop("areas")
            yield b
