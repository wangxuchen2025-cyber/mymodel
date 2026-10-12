"""真实城市建筑布局（OpenStreetMap，© OpenStreetMap contributors，ODbL）。

把 Overpass 导出的 building way 转为局部米制坐标（以区域中心为原点的等距近似，1 km 量级误差可忽略），
清洗（闭合、面积 15–20000 m²）、统一为逆时针并沿周长重采样为 V 个顶点；按区域缓存为 .pt。
"""
import json
import math
import os
from typing import Dict

import torch

from geomac.synth import resample_ring

AREA_TYPES = {
    "tokyo_setagaya": "密集独栋住宅（东京）", "amsterdam_west": "联排街坊（阿姆斯特丹）",
    "florence_old": "欧洲老城（佛罗伦萨）", "berlin_marzahn": "板楼小区（柏林）",
    "phoenix_suburb": "郊区复制户型（凤凰城）", "sh_pudong_estate": "住宅小区（上海浦东）",
    "bj_tiantongyuan": "大型住宅区（北京天通苑）", "wh_estate": "住宅小区（武汉）",
    "sh_oldtown": "老城里弄（上海）", "sz_urbanvillage": "城中村（深圳）",
}


def load_area(path, n_poly=16, min_area=15.0, max_area=20000.0) -> Dict:
    d = json.load(open(path))
    pts_all = [[(p["lat"], p["lon"]) for p in el["geometry"]] for el in d["elements"]
               if el.get("type") == "way" and "geometry" in el]
    lats = [p[0] for pts in pts_all for p in pts]
    lons = [p[1] for pts in pts_all for p in pts]
    lat0, lon0 = (min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2
    kx, ky = math.cos(math.radians(lat0)) * 111320.0, 110540.0
    polys, areas = [], []
    for pts in pts_all:
        if len(pts) < 4 or pts[0] != pts[-1]:
            continue
        xy = torch.tensor([[(lo - lon0) * kx, (la - lat0) * ky] for la, lo in pts[:-1]], dtype=torch.float64)
        x, y = xy[:, 0], xy[:, 1]
        a = 0.5 * (x * torch.roll(y, -1) - torch.roll(x, -1) * y).sum().item()
        if not (min_area <= abs(a) <= max_area):
            continue
        if a < 0:
            xy = xy.flip(0)
        polys.append(resample_ring(xy.float()[None], n_poly)[0])
        areas.append(abs(a))
    P = torch.stack(polys)
    return {"poly": P, "area": torch.tensor(areas), "cent": P.mean(1),
            "ext": (P[..., 0].min().item(), P[..., 0].max().item(), P[..., 1].min().item(), P[..., 1].max().item()),
            "origin": (lat0, lon0)}


def load_all(data_dir="data/osm", n_poly=16, cache=True) -> Dict[str, Dict]:
    cp = os.path.join(data_dir, f"layouts_v{n_poly}.pt")
    if cache and os.path.exists(cp):
        return torch.load(cp, weights_only=False)
    out = {}
    for name in AREA_TYPES:
        f = os.path.join(data_dir, f"{name}.json")
        if os.path.exists(f):
            out[name] = load_area(f, n_poly)
    if cache:
        torch.save(out, cp)
    return out


def self_similarity(cent, area, radius=45.0, tol=1.5, k=6):
    """布局周期性指数 ∈[0,1]：最佳非零平移下，能找到"位置 tol 内、面积相近"对应者的建筑比例。
    1 表示整个布局平移一个楼距后几乎与自身重合（强周期性，关联最易出现整体错位）。"""
    n = cent.shape[0]
    if n < 4:
        return 0.0
    d = torch.cdist(cent, cent)
    d.fill_diagonal_(1e9)
    kk = min(k, n - 1)
    nb = d.topk(kk, largest=False)[1]
    T = (cent[nb] - cent[:, None]).reshape(-1, 2)
    T = T[(T.norm(dim=-1) < radius) & (T.norm(dim=-1) > 3.0)]
    if len(T) == 0:
        return 0.0
    la = torch.log(area.clamp(min=1.0))
    best = 0.0
    for chunk in T.split(256):
        sh = cent[None] + chunk[:, None]                                        # t,n,2
        dd = torch.cdist(sh, cent[None].expand(len(chunk), -1, -1))             # t,n,n
        ok = (dd < tol) & ((la[None, :, None] - la[None, None, :]).abs() < 0.25)
        best = max(best, ok.any(-1).float().mean(-1).max().item())
    return best
