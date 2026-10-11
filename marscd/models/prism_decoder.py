"""Module D - Prism-Anchored Streaming Instance Decoder (PASID).

Every building hypothesis is a geo-anchored PRISM: a footprint ring of N_v
vertices in the world frame, a base elevation and a height. Three query
families share one decoder:

  * prior queries      - one per T1 SHP footprint entering the frustum
                         (content = cached GeoPriorMemory token);
  * track queries      - persistent T2 instance states propagated through the
                         stream (static world => propagation is free: the anchor
                         stays in geo coordinates and only the pose changes);
  * discovery queries  - proposals for buildings absent from the prior, taken
                         from a CenterNet-style head on the ground-lifted BEV.

Hierarchical (instance + vertex) queries follow MapTR; the key difference is
the PRISM DEFORMABLE CROSS-ATTENTION: each vertex query samples image
features along its VERTICAL CORNER LINE (base -> roof, height-adaptive),
with learned 3D offsets. In oblique views the vertical building edges and the
facade/ground contact are visible, so the footprint corner is constrained by
evidence that a nadir view never sees. Multi-view instance association within
the video then reduces to geometric consistency of a geo-anchored state, and a
view-diverse memory accumulates evidence across viewpoints at constant cost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import MARSConfig
from ..geometry import Pose, polygon_centroid, project
from .common import FFN, MLP, ConvBNAct, FourierFeatures, RelationalAttention, sinusoid

PRIOR, TRACK, DISCOVERY = 0, 1, 2


@dataclass
class QuerySet:
    feat: Tensor       # [N, C]
    polys: Tensor      # [N, Nv, 2] world
    base: Tensor       # [N] base elevation
    height: Tensor     # [N]
    kind: Tensor       # [N] long (PRIOR / TRACK / DISCOVERY)
    mem: Tensor        # [N, Nc, C]  prism-texture-atlas tokens (surface-indexed memory, v2-III)
    mem_valid: Tensor  # [N, Nc] bool

    @property
    def n(self) -> int:
        return self.feat.shape[0]


@dataclass
class DecoderOutput:
    layers: list       # per-layer dict(polys, height, base, exist_logit)
    feat: Tensor       # [N, C]
    vert_feat: Tensor  # [N, Nv, C]
    view_dir: Tensor   # [N, 3] unit vector camera -> prism centre
    in_view: Tensor    # [N] fraction of prism samples projecting into the image

    @property
    def final(self) -> dict:
        return self.layers[-1]


def rect_polygon(center: Tensor, w: Tensor, l: Tensor, theta: Tensor, nv: int) -> Tensor:
    """Oriented rectangles -> CCW rings with nv/4 points per edge. center [K,2] -> [K,nv,2]."""
    m = nv // 4
    t = torch.arange(m, device=center.device, dtype=center.dtype) / m
    hw, hl = (w / 2)[:, None], (l / 2)[:, None]
    e0 = torch.stack([-hw + 2 * hw * t, -hl.expand_as(t * hw)], -1)
    e1 = torch.stack([hw.expand_as(t * hw), -hl + 2 * hl * t], -1)
    e2 = torch.stack([hw - 2 * hw * t, hl.expand_as(t * hw)], -1)
    e3 = torch.stack([-hw.expand_as(t * hw), hl - 2 * hl * t], -1)
    loc = torch.cat([e0, e1, e2, e3], 1)
    c, s = theta.cos()[:, None], theta.sin()[:, None]
    x = c * loc[..., 0] - s * loc[..., 1]
    y = s * loc[..., 0] + c * loc[..., 1]
    return torch.stack([x, y], -1) + center[:, None]


class DiscoveryProposer(nn.Module):
    """CenterNet-style BEV proposals for buildings that are NOT in the prior."""

    def __init__(self, cfg: MARSConfig):
        super().__init__()
        C = cfg.d_model
        self.cfg = cfg
        # input: ground-lifted image features + lifted contact-line probability (height-free evidence, v3)
        self.trunk = nn.Sequential(ConvBNAct(C + 1, C, 3), ConvBNAct(C, C, 3))
        self.heat = nn.Conv2d(C, 1, 1)
        nn.init.constant_(self.heat.bias, -2.19)
        self.box = nn.Conv2d(C, 4, 1)  # log-scale w, l and (cos 2t, sin 2t)
        nn.init.zeros_(self.box.weight)
        nn.init.zeros_(self.box.bias)
        self.content = nn.Conv2d(C, C, 1)

    def forward(self, lifted: Tensor, valid: Tensor, cell_xy: Tensor, covered: Tensor, k: int) -> dict:
        x = self.trunk(lifted.unsqueeze(0))
        heat = self.heat(x)[0, 0]
        score = torch.sigmoid(heat) * valid * (~covered)
        peak = score == F.max_pool2d(score[None, None], 5, 1, 2)[0, 0]
        score = (score * peak).flatten()
        k = min(k, int((score > 0).sum()))
        idx = score.topk(k).indices if k > 0 else score.new_zeros(0, dtype=torch.long)
        box = self.box(x)[0].flatten(1)[:, idx].t()  # [k,4]
        s0 = self.cfg.default_size
        w = s0 * torch.exp(torch.tanh(box[:, 0]) * 1.2)
        l = s0 * torch.exp(torch.tanh(box[:, 1]) * 1.2)
        theta = 0.5 * torch.atan2(box[:, 3], box[:, 2] + 1.0)
        centers = cell_xy.reshape(-1, 2)[idx]
        polys = rect_polygon(centers, w, l, theta, self.cfg.n_vertices)
        feats = self.content(x)[0].flatten(1)[:, idx].t()
        return dict(heat_logit=heat, centers=centers, polys=polys, feats=feats, scores=score[idx])


class PrismDeformableCrossAttention(nn.Module):
    """Vertex query -> samples along its height-adaptive vertical corner line (+ learned 3D offsets)."""

    def __init__(self, cfg: MARSConfig, n_levels: int = 2):
        super().__init__()
        C = cfg.d_model
        self.h, self.L, self.Kh, self.P = cfg.n_heads, n_levels, cfg.height_samples, cfg.deform_points
        n = self.h * self.L * self.Kh * self.P
        self.offsets = nn.Linear(C, n * 3)
        nn.init.zeros_(self.offsets.weight)
        with torch.no_grad():  # small deterministic spread around the corner line
            ang = torch.arange(self.P, dtype=torch.float32) * 2 * math.pi / max(self.P, 1)
            b = torch.zeros(self.h, self.L, self.Kh, self.P, 3)
            b[..., 0] = 0.3 * ang.cos()
            b[..., 1] = 0.3 * ang.sin()
            self.offsets.bias.copy_(b.flatten())
        self.weights = nn.Linear(C, n)
        nn.init.zeros_(self.weights.weight)
        nn.init.zeros_(self.weights.bias)
        self.out = nn.Linear(C, C)
        self.register_buffer("tau", torch.linspace(0.0, 1.0, self.Kh), persistent=False)

    def forward(self, q: Tensor, xy: Tensor, base: Tensor, height: Tensor, pose: Pose, values: list[Tensor],
                hw: tuple[int, int], offset_scale: float) -> Tensor:
        Q, C = q.shape
        H, L, Kh, P = self.h, self.L, self.Kh, self.P
        off = self.offsets(q).view(Q, H, L, Kh, P, 3)
        dxy = torch.tanh(off[..., :2]) * offset_scale
        dtau = torch.tanh(off[..., 2]) * (0.5 / max(Kh - 1, 1))
        tau = (self.tau.view(1, 1, 1, Kh, 1) + dtau).clamp(0.0, 1.0)
        X = torch.stack([
            xy[:, None, None, None, None, 0] + dxy[..., 0],
            xy[:, None, None, None, None, 1] + dxy[..., 1],
            base[:, None, None, None, None] + tau * height[:, None, None, None, None],
        ], -1)
        uv, depth = project(X, pose)
        Hi, Wi = hw
        g = torch.stack([uv[..., 0] / Wi * 2 - 1, uv[..., 1] / Hi * 2 - 1], -1)
        valid = (depth > 0.5) & (g.abs() < 1).all(-1)  # [Q,H,L,Kh,P]
        logits = self.weights(q).view(Q, H, L * Kh * P).masked_fill(~valid.view(Q, H, -1), float("-inf"))
        attn = torch.softmax(logits, -1).nan_to_num(0.0)
        sampled = []
        for l in range(L):
            v = values[l][0]  # [C,h,w]
            v = v.view(H, C // H, *v.shape[-2:])
            gl = g[:, :, l].reshape(Q, H, Kh * P, 2).permute(1, 0, 2, 3)  # [H,Q,KhP,2]
            s = F.grid_sample(v, gl, mode="bilinear", padding_mode="zeros", align_corners=False)  # [H,Ch,Q,KhP]
            sampled.append(s.permute(2, 0, 3, 1))  # [Q,H,KhP,Ch]
        sampled = torch.cat(sampled, 2)  # [Q,H,L*KhP,Ch]
        out = (attn.unsqueeze(-1) * sampled).sum(2).reshape(Q, C)
        return self.out(out)


class PASIDLayer(nn.Module):
    def __init__(self, cfg: MARSConfig, layer_idx: int):
        super().__init__()
        C, H = cfg.d_model, cfg.n_heads
        self.cfg = cfg
        self.step = cfg.max_vertex_step / (2 ** layer_idx)
        self.offset_scale = 2.0 / (layer_idx + 1)
        self.rel = MLP(32 + 2, C // 2, H)
        self.inst_sa = RelationalAttention(C, H)
        self.mem_ca = RelationalAttention(C, H)
        self.pdca = PrismDeformableCrossAttention(cfg)
        self.vert_sa = RelationalAttention(C, H)
        self.inst_to_vert = nn.Linear(C, C)
        self.vert_to_inst = nn.Linear(C, C)
        self.ffn_i, self.ffn_v = FFN(C), FFN(C)
        self.n = nn.ModuleList([nn.LayerNorm(C) for _ in range(6)])
        self.d_vert = nn.Linear(C, 2)
        self.d_height = nn.Linear(C, 1)
        self.d_base = nn.Linear(C, 1)
        for m in (self.d_vert, self.d_height, self.d_base):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self.exist = nn.Linear(C, 1)

    def forward(self, qi: Tensor, qv: Tensor, polys: Tensor, base: Tensor, height: Tensor, qs: QuerySet,
                view_pe: Tensor, vert_pe_fn, pose: Pose, values, hw, snake=None, gcl: Tensor | None = None):
        N, Nv, C = qv.shape
        # (1) geo-relational instance self-attention (neighbourhood layout context)
        cen = polygon_centroid(polys.detach())
        d = cen[:, None] - cen[None]
        dist = d.norm(dim=-1)
        dirn = d / dist.clamp_min(1e-3)[..., None]
        bias = self.rel(torch.cat([sinusoid(dist / 10.0, 32), dirn], -1)).permute(2, 0, 1).unsqueeze(0)
        x = self.n[0](qi)
        qi = qi + self.inst_sa(x[None], x[None], bias=bias)[0]
        # (2) cross-attention to the instance's prism texture atlas (multi-view evidence in surface coordinates)
        if qs.mem_valid.any():
            x = self.n[1](qi) + view_pe
            qi = qi + self.mem_ca(x[:, None], qs.mem, mask=qs.mem_valid[:, None, :])[:, 0]
        # (3) prism deformable cross-attention for every vertex query
        qv = qv + self.inst_to_vert(qi)[:, None] + vert_pe_fn(polys.detach(), cen)
        x = self.n[2](qv).reshape(N * Nv, C)
        xy = polys.detach().reshape(N * Nv, 2)
        qv = qv + self.pdca(x, xy, base.detach().repeat_interleave(Nv), height.detach().repeat_interleave(Nv),
                            pose, values, hw, self.offset_scale).view(N, Nv, C)
        # (4) intra-instance vertex self-attention (decoupled, MapTRv2-style)
        x = self.n[3](qv)
        qv = qv + self.vert_sa(x, x)
        qv = qv + self.ffn_v(self.n[4](qv))
        qi = qi + self.vert_to_inst(qv.mean(1))
        qi = qi + self.ffn_i(self.n[5](qi))
        # (5) iterative refinement of the prism anchor in geo coordinates
        new_polys = polys.detach() + torch.tanh(self.d_vert(qv)) * self.step
        if snake is not None and gcl is not None:  # v2-I: height-free contact-line attraction along vertex normals
            dv, _ = snake(gcl, polys.detach(), base.detach(), pose, hw)
            new_polys = new_polys + dv
        new_height = (height.detach() + 2.0 * self.d_height(qi)[:, 0]).clamp(2.0, 300.0)
        new_base = base.detach() + self.d_base(qi)[:, 0]
        pred = dict(polys=new_polys, height=new_height, base=new_base, exist_logit=self.exist(qi)[:, 0])
        return qi, qv, pred


class PASID(nn.Module):
    def __init__(self, cfg: MARSConfig):
        super().__init__()
        C = cfg.d_model
        self.cfg = cfg
        self.layers = nn.ModuleList([PASIDLayer(cfg, i) for i in range(cfg.dec_layers)])
        self.kind_embed = nn.Embedding(3, C)
        self.vert_embed = nn.Parameter(torch.randn(cfg.n_vertices, C) * 0.02)
        self.vert_pe = FourierFeatures(2, C, n_freq=6, min_period=1.0, max_period=64.0)
        self.view_pe = MLP(3, C, C)
        self.values = nn.ModuleList([nn.Conv2d(C, C, 1) for _ in range(2)])
        self.proposer = DiscoveryProposer(cfg)
        from .structure import ContactSnake
        self.snake = ContactSnake()
        # the instance state is propagated recurrently through the stream: normalise it, otherwise residual
        # updates accumulate frame after frame and the state norm grows without bound
        self.out_norm_i = nn.LayerNorm(C)
        self.out_norm_v = nn.LayerNorm(C)

    def _vert_pe(self, polys: Tensor, cen: Tensor) -> Tensor:
        return self.vert_pe(polys - cen[:, None]) + self.vert_embed[None]

    def forward(self, qs: QuerySet, feats: list[Tensor], pose: Pose, hw: tuple[int, int],
                struct_logits: Tensor | None = None) -> DecoderOutput:
        N, Nv, C = qs.n, self.cfg.n_vertices, self.cfg.d_model
        dev = qs.feat.device
        cen = polygon_centroid(qs.polys)
        mid = torch.cat([cen, (qs.base + 0.5 * qs.height)[:, None]], -1)
        view_dir = F.normalize(mid - pose.center, dim=-1)
        if N == 0:
            e = qs.feat.new_zeros(0)
            return DecoderOutput([], qs.feat, qs.feat.new_zeros(0, Nv, C), qs.feat.new_zeros(0, 3), e)
        values = [proj(f) for proj, f in zip(self.values, feats)]
        qi = qs.feat + self.kind_embed(qs.kind)
        qv = qi[:, None].expand(N, Nv, C).clone()
        view_pe = self.view_pe(view_dir)
        polys, base, height = qs.polys, qs.base, qs.height
        gcl = struct_logits[0] if struct_logits is not None else None
        outs = []
        for layer in self.layers:
            qi, qv, pred = layer(qi, qv, polys, base, height, qs, view_pe, self._vert_pe, pose, values, hw,
                                 self.snake, gcl)
            outs.append(pred)
            polys, base, height = pred["polys"], pred["base"], pred["height"]
        # fraction of corner-line samples inside the image (geometric in-view score)
        tau = torch.tensor([0.0, 1.0], device=dev)
        X = torch.cat([polys.detach()[:, :, None].expand(N, Nv, 2, 2),
                       (base.detach()[:, None, None] + tau * height.detach()[:, None, None])[..., None].expand(N, Nv, 2, 1)], -1)
        uv, depth = project(X, pose)
        Hi, Wi = hw
        inv = ((depth > 0.5) & (uv[..., 0] >= 0) & (uv[..., 0] < Wi) & (uv[..., 1] >= 0) & (uv[..., 1] < Hi))
        return DecoderOutput(outs, self.out_norm_i(qi), self.out_norm_v(qv), view_dir, inv.float().mean((1, 2)))
