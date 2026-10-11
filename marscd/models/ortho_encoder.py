"""Module A - Multi-source Geo-Prior Encoder (MGPE), run OFFLINE on the T1 epoch.

All T1 sources are encoded once, before the T2 flight, into a cached
geo-anchored memory. This asymmetric offline/online split is what makes
near-real-time T2 inference possible:

  * co-registered ortho rasters (satellite DOM, aerial DOM, raster map) ->
    reliability-gated fused BEV feature map + change-stable structure map;
  * SHP footprints -> one token per historical building (shape + context);
  * (optional) historical oblique UAV video -> oblique evidence per SHP building
    (facade appearance, height, T1 observability) fused into the tokens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import MARSConfig
from ..geometry import boundary_samples, points_in_polygons, polygon_centroid, shape_descriptors
from .common import MLP, ConvBNAct, FourierFeatures, ResBlock
from .structure import shp_edge_map


@dataclass
class GeoGrid:
    """North-up raster georeference: pixel (r, c) centre at (x0 + (c+.5) gsd, y0 - (r+.5) gsd)."""

    x0: float
    y0: float
    gsd: float
    H: int
    W: int

    def to_norm(self, xy: Tensor) -> Tensor:
        gx = (xy[..., 0] - self.x0) / (self.W * self.gsd) * 2 - 1
        gy = (self.y0 - xy[..., 1]) / (self.H * self.gsd) * 2 - 1
        return torch.stack([gx, gy], -1)

    def to_pixel(self, xy: Tensor) -> Tensor:
        return torch.stack([(xy[..., 0] - self.x0) / self.gsd, (self.y0 - xy[..., 1]) / self.gsd], -1)


def sample_bev(fmap: Tensor, grid: GeoGrid, xy: Tensor) -> Tensor:
    """Bilinear lookup of a geo-referenced map [C, h, w] at world points xy [..., 2] -> [..., C]."""
    g = grid.to_norm(xy).reshape(1, 1, -1, 2).to(fmap.dtype)
    out = F.grid_sample(fmap.unsqueeze(0), g, mode="bilinear", padding_mode="zeros", align_corners=False)
    return out.reshape(fmap.shape[0], -1).t().reshape(*xy.shape[:-1], fmap.shape[0])


@dataclass
class GeoPriorMemory:
    """Cached T1 state consumed by the online T2 stream."""

    feats: Tensor            # [M, C]   per-building tokens
    vert_feats: Tensor       # [M, Nv, C]
    polys: Tensor            # [M, Nv, 2] SHP footprints (world)
    heights: Tensor          # [M]  height estimate (from T1 oblique video, else prior)
    height_known: Tensor     # [M]  bool
    t1_obs: Tensor           # [M]  T1 observability from the historical video (0 if none)
    bev: Tensor              # [C, h, w] fused BEV features
    bev_match: Tensor        # [Cm, h, w] L2-normalised cross-view matching embedding
    bev_fine: Tensor         # [Cf, h, w] feature-metric refinement embedding
    stability: Tensor        # [1, h, w] P(change-stable ground structure)
    bld_logit: Tensor        # [1, h, w] building footprint logits (aux., SHP-supervised)
    grid: GeoGrid            # geo-reference of the BEV feature maps
    aux: dict = field(default_factory=dict)
    shp_edge: Tensor | None = None    # [1, h, w] soft SHP boundary raster (v2-I, label-free)
    height_ll: Tensor | None = None   # [M, B] height-sweep log-likelihood from the T1 video (v2-I)
    atlas_sum: Tensor | None = None   # [M, Nc, Ca] T1 prism texture atlas (v2-III)
    atlas_w: Tensor | None = None     # [M, Nc]

    @property
    def num(self) -> int:
        return self.polys.shape[0]

    @property
    def centroids(self) -> Tensor:
        return polygon_centroid(self.polys)


class OrthoBEVEncoder(nn.Module):
    """Modality-specific stems + shared trunk + reliability-gated fusion."""

    def __init__(self, cfg: MARSConfig):
        super().__init__()
        C = cfg.d_model
        self.modalities = list(cfg.ortho_modalities)
        self.p_drop = cfg.modality_dropout
        n_down = int(round(math.log2(cfg.ortho_stride)))

        def stem():
            layers, c = [], 3
            for i in range(n_down):
                c_out = C if i == n_down - 1 else C // 2
                layers.append(ConvBNAct(c, c_out, 3, 2))
                c = c_out
            return nn.Sequential(*layers)

        self.stems = nn.ModuleDict({m: stem() for m in self.modalities})
        self.mod_embed = nn.Parameter(torch.zeros(len(self.modalities), C, 1, 1))
        self.trunk = nn.Sequential(ResBlock(C), ResBlock(C))
        self.gate = nn.Conv2d(C, 1, 1)  # per-cell reliability logit of each modality
        self.post = nn.Sequential(ResBlock(C), ConvBNAct(C, C, 3))

    def forward(self, rasters: dict[str, Tensor], available: dict[str, bool]) -> tuple[Tensor, Tensor]:
        feats, logits, keep = [], [], []
        mods = [m for m in self.modalities if m in rasters and available.get(m, True)]
        assert mods, "at least one T1 ortho source is required"
        if self.training and len(mods) > 1:  # modality dropout -> robustness to missing sources
            drop = torch.rand(len(mods)) < self.p_drop
            if drop.all():
                drop[torch.randint(len(mods), (1,))] = False
            mods = [m for m, d in zip(mods, drop) if not d]
        for m in mods:
            i = self.modalities.index(m)
            f = self.trunk(self.stems[m](rasters[m].unsqueeze(0)) + self.mod_embed[i])
            feats.append(f)
            logits.append(self.gate(f))
        w = torch.softmax(torch.cat(logits, 1), 1)  # [1, n_mod, h, w]
        fused = sum(w[:, i:i + 1] * f for i, f in enumerate(feats))
        return self.post(fused)[0], w[0]


class PolygonTokenizer(nn.Module):
    """SHP footprint -> building token (shape encoder + interior/ring context pooling)."""

    def __init__(self, cfg: MARSConfig):
        super().__init__()
        C = cfg.d_model
        self.vert_pe = FourierFeatures(2, C, n_freq=6, min_period=2.0, max_period=128.0)
        self.vert_ctx = nn.Linear(C, C)
        self.vert_attn = nn.TransformerEncoderLayer(C, cfg.n_heads, 2 * C, dropout=0.0, batch_first=True,
                                                    norm_first=True)
        self.desc = MLP(10, C, C)
        self.fuse = MLP(4 * C, C, C)
        self.n_interior = 6
        self.ring_dist = (2.0, 5.0)

    def forward(self, polys: Tensor, bev: Tensor, grid: GeoGrid) -> tuple[Tensor, Tensor]:
        M, Nv, _ = polys.shape
        if M == 0:
            C = bev.shape[0]
            return bev.new_zeros(0, C), bev.new_zeros(0, Nv, C)
        cen = polygon_centroid(polys)
        local = polys - cen[:, None]
        v = self.vert_pe(local) + self.vert_ctx(sample_bev(bev, grid, polys))
        v = self.vert_attn(v)
        shape = v.mean(1)
        # interior pooling on a per-polygon regular grid
        lo, hi = polys.min(1).values, polys.max(1).values
        g = (torch.arange(self.n_interior, device=polys.device, dtype=polys.dtype) + 0.5) / self.n_interior
        gy, gx = torch.meshgrid(g, g, indexing="ij")
        unit = torch.stack([gx, gy], -1).reshape(-1, 2)
        pts = lo[:, None] + unit[None] * (hi - lo)[:, None]  # [M, G, 2]
        inside = torch.stack([points_in_polygons(pts[i], polys[i:i + 1])[0] for i in range(M)], 0).float()
        inside[:, 0] = torch.where(inside.sum(1) == 0, torch.ones_like(inside[:, 0]), inside[:, 0])
        f_in = sample_bev(bev, grid, pts)
        interior = (f_in * inside[..., None]).sum(1) / inside.sum(1, keepdim=True)
        # ring pooling just outside the footprint (context: yards, roads, neighbours)
        bp, bn = boundary_samples(polys, 16)
        ring_pts = torch.cat([bp + d * bn for d in self.ring_dist], 1)
        ring = sample_bev(bev, grid, ring_pts).mean(1)
        tok = self.fuse(torch.cat([shape, interior, ring, self.desc(shape_descriptors(polys))], -1))
        return tok, v


class GeoPriorEncoder(nn.Module):
    """Module A: builds the cached GeoPriorMemory from T1 sources."""

    def __init__(self, cfg: MARSConfig):
        super().__init__()
        C = cfg.d_model
        self.cfg = cfg
        self.bev_enc = OrthoBEVEncoder(cfg)
        self.tokenizer = PolygonTokenizer(cfg)
        self.match_head = nn.Conv2d(C, cfg.match_dim, 1)
        self.fine_head = nn.Conv2d(C, cfg.fine_dim, 1)
        self.stab_head = nn.Sequential(ConvBNAct(C, C // 2, 3), nn.Conv2d(C // 2, 1, 1))
        self.bld_head = nn.Sequential(ConvBNAct(C, C // 2, 3), nn.Conv2d(C // 2, 1, 1))
        # fusion of historical oblique evidence into SHP tokens (A4)
        self.obl_gate = MLP(2 * C + 1, C, C)
        self.obl_val = MLP(C, C, C)

    def forward(self, rasters: dict[str, Tensor], grid: GeoGrid, polys: Tensor,
                available: dict[str, bool] | None = None, heights: Tensor | None = None) -> GeoPriorMemory:
        cfg = self.cfg
        bev, gate_w = self.bev_enc(rasters, available or {})
        s = cfg.ortho_stride
        h, w = bev.shape[-2:]
        fgrid = GeoGrid(grid.x0, grid.y0, grid.gsd * s, h, w)
        tokens, vfeats = self.tokenizer(polys, bev, fgrid)
        M = polys.shape[0]
        if heights is None:
            heights = polys.new_full((M,), cfg.default_height)
            known = torch.zeros(M, dtype=torch.bool, device=polys.device)
        else:
            known = torch.isfinite(heights)
            heights = torch.where(known, heights, torch.full_like(heights, cfg.default_height))
        return GeoPriorMemory(
            feats=tokens,
            vert_feats=vfeats,
            polys=polys,
            heights=heights,
            height_known=known,
            t1_obs=polys.new_zeros(M),
            bev=bev,
            bev_match=F.normalize(self.match_head(bev.unsqueeze(0))[0], dim=0),
            bev_fine=self.fine_head(bev.unsqueeze(0))[0],
            stability=torch.sigmoid(self.stab_head(bev.unsqueeze(0))[0]),
            bld_logit=self.bld_head(bev.unsqueeze(0))[0],
            grid=fgrid,
            aux={"modality_weights": gate_w},
            shp_edge=shp_edge_map(polys, fgrid),
        )

    def fuse_oblique(self, mem: GeoPriorMemory, obl_feats: Tensor, obs: Tensor, heights: Tensor,
                     height_ll: Tensor | None = None, atlas_sum: Tensor | None = None,
                     atlas_w: Tensor | None = None) -> GeoPriorMemory:
        """A4: inject evidence aggregated from the historical oblique video into the SHP tokens, and keep the
        T1 height posterior and T1 prism texture atlas for the T2 stream."""
        g = torch.sigmoid(self.obl_gate(torch.cat([mem.feats, obl_feats, obs[:, None]], -1))) * obs[:, None]
        feats = mem.feats + g * self.obl_val(obl_feats)
        seen = obs > 0.3
        new_h = torch.where(seen, heights, mem.heights)
        return replace(mem, feats=feats, heights=new_h, height_known=mem.height_known | seen, t1_obs=obs,
                       height_ll=height_ll, atlas_sum=atlas_sum, atlas_w=atlas_w)
