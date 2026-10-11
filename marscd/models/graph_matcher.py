"""Module E - Relation-Invariant Geo-Graph Matcher (RIGM).

One matcher, two uses (mode embedding):
  * mode 0  cross-temporal: T1 graph (SHP-anchored, multi-source tokens) <-> T2 graph
            (instance tracks accumulated from the current video);
  * mode 1  intra-video   : new discovery detections <-> existing T2 tracks.

Spatial relationships drive the association:
  * node features are pose-free (appearance + rotation-invariant shape
    descriptors + height); absolute position is NOT a node feature;
  * intra-graph relational self-attention uses a geometric structure
    embedding (pairwise centroid distance, min boundary gap / adjacency,
    relative principal-axis angle, and GeoTransformer-style triplet angles
    over k-NN) which is invariant to the rigid misregistration that remains
    after CARR - neighbourhood layout disambiguates repetitive buildings;
  * cross-graph attention is biased by uncertainty-normalised geometry
    (Mahalanobis centroid offset under the registration covariance, footprint
    IoU, area ratio, anchor provenance) and gated to plausible pairs;
  * partial assignment = log-domain Sinkhorn with NODE-ADAPTIVE dustbins:
    the T1 dustbin is "demolished / unobserved", the T2 dustbin is "new".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..config import MARSConfig
from ..geometry import min_boundary_gap, polygon_area, polygon_centroid, polygon_iou_paired, principal_axis_angle, shape_descriptors
from .common import FFN, MLP, RelationalAttention, ScalarEmbedding


@dataclass
class GraphNodes:
    feat: Tensor          # [N, C]
    polys: Tensor         # [N, Nv, 2]
    height: Tensor        # [N]
    height_known: Tensor  # [N] bool
    exist_logit: Tensor | None = None  # [N] (T2 side)
    anchor_src: Tensor | None = None   # [N] long, index into the other graph or -1

    @property
    def n(self) -> int:
        return self.feat.shape[0]


def log_optimal_transport(scores: Tensor, alpha: Tensor, beta: Tensor, corner: Tensor, iters: int) -> Tensor:
    """SuperGlue-style log-Sinkhorn with per-node dustbin scores.

    scores [M, N], alpha [M] (dustbin column for rows), beta [N] (dustbin row for
    columns) -> log assignment [M+1, N+1].
    """
    M, N = scores.shape
    Z = torch.cat([torch.cat([scores, alpha[:, None]], 1),
                   torch.cat([beta[None], corner.view(1, 1)], 1)], 0)
    norm = -math.log(M + N)
    log_mu = torch.cat([scores.new_full((M,), norm), scores.new_full((1,), math.log(N) + norm)])
    log_nu = torch.cat([scores.new_full((N,), norm), scores.new_full((1,), math.log(M) + norm)])
    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v[None], 1)
        v = log_nu - torch.logsumexp(Z + u[:, None], 0)
    return Z + u[:, None] + v[None] - norm


def decode_matches(log_P: Tensor, thresh: float) -> tuple[Tensor, Tensor]:
    """Mutual-argmax decoding. Returns match index for each row ([M], -1 = dustbin) and probs."""
    P = log_P[:-1, :-1].exp()
    M, N = P.shape
    if M == 0 or N == 0:
        return torch.full((M,), -1, dtype=torch.long, device=log_P.device), P.new_zeros(M)
    v0, i0 = P.max(1)
    _, i1 = P.max(0)
    mutual = i1[i0] == torch.arange(M, device=P.device)
    ok = mutual & (v0 > thresh)
    return torch.where(ok, i0, torch.full_like(i0, -1)), v0


class RIGM(nn.Module):
    N_CROSS = 7

    def __init__(self, cfg: MARSConfig):
        super().__init__()
        C, H = cfg.d_model, cfg.n_heads
        self.cfg = cfg
        self.node_desc = MLP(10 + 2, C, C)
        self.side_embed = nn.Embedding(2, C)
        self.mode_embed = nn.Embedding(2, C)
        self.e_dist = ScalarEmbedding(C, sigma=10.0)
        self.e_gap = ScalarEmbedding(C, sigma=3.0)
        self.e_axis = nn.Linear(2, C)
        self.e_tri = ScalarEmbedding(C, sigma=math.radians(15.0))
        self.rel_norm = nn.LayerNorm(C)
        self.self_attn = nn.ModuleList([RelationalAttention(C, H, rel_dim=C) for _ in range(cfg.match_layers)])
        self.cross_attn = nn.ModuleList([RelationalAttention(C, H) for _ in range(cfg.match_layers)])
        self.cross_bias = nn.ModuleList([MLP(self.N_CROSS, C // 2, H) for _ in range(cfg.match_layers)])
        self.ffn = nn.ModuleList([FFN(C) for _ in range(cfg.match_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(C) for _ in range(3 * cfg.match_layers)])
        self.final = nn.Sequential(nn.LayerNorm(C), nn.Linear(C, C))
        self.log_scale = nn.Parameter(torch.tensor(math.log(5.0)))  # bounded cosine logits
        self.pair_score = MLP(self.N_CROSS, C // 2, 1)
        self.bin_a = MLP(C, C // 2, 1)
        self.bin_b = MLP(C + 1, C // 2, 1)
        self.corner = nn.Parameter(torch.tensor(0.0))

    # ------------------------------------------------------------ geometry
    def relation_embedding(self, polys: Tensor) -> Tensor:
        """Rigid-invariant geometric structure embedding r_ij: [N, N, C]."""
        N = polys.shape[0]
        cen = polygon_centroid(polys)
        d = cen[:, None] - cen[None]
        dist = d.norm(dim=-1)
        gap = min_boundary_gap(polys, 8)
        ax = principal_axis_angle(polys)
        dax = ax[:, None] - ax[None]
        rel = self.e_dist(dist) + self.e_gap(gap) + self.e_axis(torch.stack([(2 * dax).cos(), (2 * dax).sin()], -1))
        k = min(self.cfg.knn_angles, N - 1)
        if k > 0:
            nn_idx = dist.topk(k + 1, largest=False).indices[:, 1:]  # [N,k] neighbours of i
            v_ij = cen[None] - cen[:, None]               # [N(i),N(j),2]  c_j - c_i
            v_ik = cen[nn_idx] - cen[:, None]              # [N(i),k,2]
            cross = v_ij[:, :, None, 0] * v_ik[:, None, :, 1] - v_ij[:, :, None, 1] * v_ik[:, None, :, 0]
            dot = (v_ij[:, :, None] * v_ik[:, None]).sum(-1)
            ang = torch.atan2(cross, dot)                  # [N,N,k]
            rel = rel + self.e_tri(ang).amax(2)
        return self.rel_norm(rel)

    def cross_features(self, A: GraphNodes, B: GraphNodes, sigma: Tensor) -> tuple[Tensor, Tensor]:
        """Uncertainty-normalised cross-graph geometry [M, N, 7] and the plausibility gate [M, N]."""
        cA, cB = polygon_centroid(A.polys), polygon_centroid(B.polys)
        d = cA[:, None] - cB[None]
        maha = (d / sigma.clamp_min(0.5)).norm(dim=-1)
        dist = d.norm(dim=-1)
        rA = polygon_area(A.polys).abs().clamp_min(1.0).sqrt()
        rB = polygon_area(B.polys).abs().clamp_min(1.0).sqrt()
        gate = dist < (self.cfg.gate_radius + 3 * sigma.max() + 0.5 * (rA[:, None] + rB[None]))
        M, N = gate.shape
        iou = A.polys.new_zeros(M, N)
        if gate.any():
            ia, ib = gate.nonzero(as_tuple=True)
            iou[ia, ib] = polygon_iou_paired(A.polys[ia], B.polys[ib], res=16)
        lar = torch.log(rB[None] ** 2 / rA[:, None] ** 2)
        both = (A.height_known[:, None] & B.height_known[None]).float()
        dh = (A.height[:, None] - B.height[None]).abs() / 10.0 * both
        src = A.polys.new_zeros(M, N)
        if B.anchor_src is not None:
            ok = B.anchor_src >= 0
            if ok.any():
                jj = ok.nonzero(as_tuple=True)[0]
                src[B.anchor_src[jj], jj] = 1.0
        feats = torch.stack([torch.exp(-maha / 3.0), dist / 50.0, iou, lar, dh, both, src], -1)
        return feats, gate

    def node_embedding(self, G: GraphNodes, side: int, mode: int) -> Tensor:
        hfeat = torch.stack([torch.log1p(G.height) * G.height_known, G.height_known.float()], -1)
        x = G.feat + self.node_desc(torch.cat([shape_descriptors(G.polys), hfeat], -1))
        idx = torch.tensor(side, device=x.device)
        return x + self.side_embed(idx) + self.mode_embed(torch.tensor(mode, device=x.device))

    # ------------------------------------------------------------ forward
    def forward(self, A: GraphNodes, B: GraphNodes, mode: int, sigma: Tensor) -> dict:
        M, N = A.n, B.n
        if M == 0 or N == 0:
            log_P = A.feat.new_zeros(M + 1, N + 1)
            if M:
                log_P[:M, :] = -1e4
                log_P[:M, N] = 0.0
            if N:
                log_P[:, :N] = -1e4
                log_P[M, :N] = 0.0
            return dict(log_P=log_P, gate=A.feat.new_zeros(M, N, dtype=torch.bool), feat_a=A.feat, feat_b=B.feat)
        hA, hB = self.node_embedding(A, 0, mode), self.node_embedding(B, 1, mode)
        rA, rB = self.relation_embedding(A.polys), self.relation_embedding(B.polys)
        xf, gate = self.cross_features(A, B, sigma)
        gate_t = gate.t()
        for l in range(self.cfg.match_layers):
            n0, n1, n2 = self.norms[3 * l: 3 * l + 3]
            a, b = n0(hA), n0(hB)
            hA = hA + self.self_attn[l](a[None], a[None], rel=rA[None])[0]
            hB = hB + self.self_attn[l](b[None], b[None], rel=rB[None])[0]
            bias = self.cross_bias[l](xf).permute(2, 0, 1)  # [H, M, N]
            a, b = n1(hA), n1(hB)
            dA = self.cross_attn[l](a[None], b[None], bias=bias[None], mask=gate[None])[0]
            dB = self.cross_attn[l](b[None], a[None], bias=bias.transpose(1, 2)[None], mask=gate_t[None])[0]
            hA, hB = hA + dA, hB + dB
            hA = hA + self.ffn[l](n2(hA))
            hB = hB + self.ffn[l](n2(hB))
        fA = torch.nn.functional.normalize(self.final(hA), dim=-1)
        fB = torch.nn.functional.normalize(self.final(hB), dim=-1)
        S = self.log_scale.exp().clamp(max=30.0) * (fA @ fB.t()) + self.pair_score(xf)[..., 0]
        S = S.masked_fill(~gate, -1e4)
        ex = B.exist_logit if B.exist_logit is not None else B.feat.new_zeros(N)
        alpha = self.bin_a(hA)[:, 0]
        beta = self.bin_b(torch.cat([hB, -ex[:, None]], -1))[:, 0]
        log_P = log_optimal_transport(S, alpha, beta, self.corner, self.cfg.sinkhorn_iters)
        return dict(log_P=log_P, gate=gate, feat_a=hA, feat_b=hB, cross=xf, S=S, alpha=alpha, beta=beta)
