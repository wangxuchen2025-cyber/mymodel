"""① 当前视频分支：在倾斜原始帧上独立发现建筑，输出屋顶多边形 + 相对高度 η=h/H。

- 骨干复用 MPG-ViT（元数据 token + 射线条件化），stride 8/16/32 金字塔；
- 查询 = 跟踪先验查询（由实例记忆按"径向位移律"预测其在本帧的屋顶框，见 memory.py）+ 新检测查询
  （屋顶中心热图 top-K，RT-DETR 式查询选择）；多尺度可变形注意力解码，逐层细化框与多边形；
- 屋顶→基底不回归二维偏移向量，而只回归标量 η：在地图系里基底 F = N + (1-η)(R' - N)，
  偏移方向由"天底点→基底"的径向几何确定（PolyFootNet 平行方向先验在透视下的推广）；
- 基底位置的不确定度沿径向各向异性：Σ = σ_η²·|R'-N|²·uuᵀ + σ0²I，供 GRAIL 的马氏距离使用。
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from geomac.backbone import MPGViT
from geomac.decoder import DecLayer, inv_sigmoid
from geomac.geometry import ground_plane_depth, world_rays
from geomac.layers import Mlp, conv_gn, fourier_encode
from .config import GrailCDCfg
from .geometry import camera_from_meta, img_to_ground, relief_footprint


def geo_prompt(meta, img_hw, stride=16):
    K, R, C = camera_from_meta(meta, img_hw)
    rays = world_rays(K, R, (img_hw[0] // stride, img_hw[1] // stride), stride)
    dp, v = ground_plane_depth(C, rays, torch.zeros(meta.shape[0], device=meta.device))
    return torch.cat([F.normalize(rays, dim=-1), (dp.log() / 6.0)[..., None], v[..., None].float()], -1).permute(0, 3, 1, 2)


class RoofDetector(nn.Module):
    def __init__(self, cfg: GrailCDCfg):
        super().__init__()
        self.cfg = cfg
        dc, bc = cfg.det, cfg.backbone
        d = dc.dim
        self.backbone = MPGViT(bc)
        self.p8 = nn.Conv2d(bc.out_ch, d, 1)
        self.p16 = nn.Conv2d(bc.out_ch, d, 1)
        self.p32 = conv_gn(bc.out_ch, d, 3, 2)
        self.heat = nn.Conv2d(d, 1, 3, 1, 1)
        self.size = nn.Conv2d(d, 2, 1)
        nn.init.constant_(self.heat.bias, -4.0)
        self.sel = nn.Linear(d, d)
        self.new_q = nn.Parameter(torch.randn(1, dc.n_new, d) * 0.02)
        self.trk_proj = nn.Linear(dc.emb_dim, d)
        self.type_emb = nn.Parameter(torch.randn(2, d) * 0.02)               # 0=新检测, 1=跟踪先验
        self.pos = Mlp(4 * 2 * 8, d, d)
        self.layers = nn.ModuleList(DecLayer(d, dc.heads, 3, dc.n_points) for _ in range(dc.n_layers))
        self.box_delta = Mlp(d, d, 4)
        self.poly_delta = Mlp(d, d, dc.n_poly * 2)
        for m in (self.box_delta, self.poly_delta):
            nn.init.zeros_(m.layers[-1].weight)
            nn.init.zeros_(m.layers[-1].bias)
        self.obj = nn.Linear(d, 1)
        nn.init.constant_(self.obj.bias, -2.0)
        self.eta = nn.Linear(d, 2)
        self.emb = Mlp(d, d, dc.emb_dim)
        self.mask_emb = Mlp(d, d, d)
        self.pix = nn.Conv2d(d, d, 1)
        t = torch.linspace(0, 1, dc.n_poly + 1)[:-1] * 2 * torch.pi
        self.register_buffer("ring", torch.stack([0.5 + 0.5 * torch.cos(t), 0.5 + 0.5 * torch.sin(t)], -1), persistent=False)
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, img, meta, track: Optional[Dict] = None) -> Dict:
        """img (B,3,H,W)∈[0,1]；track：{"emb" (B,T,E), "box" (B,T,4) 归一化 cxcywh, "mask" (B,T)} 跟踪先验。"""
        cfg, dc = self.cfg, self.cfg.det
        hw = cfg.backbone.img_hw
        x = (img - self.img_mean) / self.img_std
        bb = self.backbone(x, meta, geo_prompt(meta, hw).to(x.dtype))
        v8, v16, v32 = self.p8(bb["f8"]), self.p16(bb["f16"]), self.p32(bb["f16"])
        b, d, h8, w8 = v8.shape
        heat = self.heat(v8)
        k = min(dc.n_new, h8 * w8)
        idx = heat.flatten(1).detach().topk(k, 1)[1]
        feat = v8.flatten(2).transpose(1, 2)
        sel = feat.gather(1, idx[..., None].expand(-1, -1, d))
        cxy = torch.stack([(idx % w8).float() + 0.5, (idx // w8).float() + 0.5], -1) / torch.tensor([w8, h8], device=x.device)
        wh = 0.03 + 0.3 * torch.sigmoid(self.size(v8)).flatten(2).transpose(1, 2).gather(1, idx[..., None].expand(-1, -1, 2))
        q = self.sel(sel) + self.new_q[:, :k] + self.type_emb[0]
        box = torch.cat([cxy, wh], -1)
        poly = self.ring[None, None].expand(b, k, -1, -1)
        kpm = torch.zeros(b, k, dtype=torch.bool, device=x.device)
        n_trk = 0
        if track is not None and track["mask"].any():
            n_trk = track["emb"].shape[1]
            q = torch.cat([self.trk_proj(track["emb"]) + self.type_emb[1], q], 1)
            box = torch.cat([track["box"], box], 1)
            poly = torch.cat([self.ring[None, None].expand(b, n_trk, -1, -1), poly], 1)
            kpm = torch.cat([~track["mask"], kpm], 1)
        pix = self.pix(v8)
        values = [v8, v16, v32]
        aux = []
        for layer in self.layers:
            pos = self.pos(fourier_encode(box, 8))
            q = layer(q, pos, box.detach(), values, kpm)
            box = torch.sigmoid(inv_sigmoid(box.detach()) + self.box_delta(q))
            poly = (poly.detach() + 0.1 * torch.tanh(self.poly_delta(q).view(b, -1, dc.n_poly, 2))).clamp(0, 1)
            aux.append(self.heads(q, box, poly, pix))
        out = aux[-1]
        out.update({"heat": heat, "aux": aux[:-1], "n_trk": n_trk, "qmask": ~kpm})
        return out

    def heads(self, q, box, poly, pix):
        dc = self.cfg.det
        e = self.eta(q)
        return {"obj": self.obj(q)[..., 0], "box": box,
                "poly": box[:, :, None, :2] + (poly - 0.5) * box[:, :, None, 2:],        # 屋顶多边形（归一化像素）
                "eta": dc.eta_max * torch.sigmoid(e[..., 0]), "eta_logsig": e[..., 1].clamp(-6, 1),
                "emb": F.normalize(self.emb(q), dim=-1),
                "mask": torch.einsum("bqd,bdhw->bqhw", self.mask_emb(q), pix)}


def to_map(det: Dict, meta, img_hw, sigma0=0.5):
    """屋顶多边形（归一化像素）+ η -> 地图系（GNSS 系）的屋顶投影 R'、基底 F 及其各向异性协方差。"""
    H, W = img_hw
    K, R, C = camera_from_meta(meta, img_hw)
    uv = det["poly"] * torch.tensor([W, H], device=meta.device, dtype=det["poly"].dtype)
    roof, ok = img_to_ground(uv.float(), K, R, C)
    nadir = C[:, :2]
    eta = det["eta"].float()
    foot = relief_footprint(roof, nadir, eta)
    rc = roof.mean(2)
    radial = rc - nadir[:, None]
    rn = radial.norm(dim=-1, keepdim=True).clamp(min=1e-3)
    u = radial / rn
    sig_eta = det["eta_logsig"].float().exp()
    cov = (sig_eta * rn[..., 0])[..., None, None] ** 2 * (u[..., :, None] * u[..., None, :]) + \
        sigma0 ** 2 * torch.eye(2, device=meta.device)
    return {"roof": roof, "foot": foot, "cov": cov, "valid": ok.all(-1), "nadir": nadir, "H": C[:, 2]}
