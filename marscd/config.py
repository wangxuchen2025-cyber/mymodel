"""Hyper-parameters of MARS-CD.

`MARSConfig()` is the deployment-scale configuration (single RTX 4090/3090,
1080p stream decoded to 960x544, >= 10 FPS). `MARSConfig.tiny()` is the
CPU-friendly configuration used by unit tests and the synthetic smoke test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace


@dataclass
class MARSConfig:
    # ---- global -------------------------------------------------------------
    d_model: int = 256
    n_heads: int = 8
    n_vertices: int = 16            # N_v vertices per footprint ring
    dropout: float = 0.0

    # ---- Module A: multi-source geo-prior encoder (offline, T1) --------------
    ortho_modalities: tuple = ("sat", "aerial", "map")
    ortho_gsd: float = 0.25         # m / px of the co-registered T1 rasters
    ortho_stride: int = 4           # BEV feature cell = ortho_gsd * ortho_stride (= 1 m)
    modality_dropout: float = 0.3   # training-time drop rate of each T1 source

    # ---- Module B: streaming oblique encoder (online, T2) --------------------
    img_channels: tuple = (64, 128, 256)  # stage widths of the light backbone (strides 8/16/32)

    # ---- Module C: change-aware robust geo-registration (CARR) ---------------
    match_dim: int = 32             # cross-view matching embedding
    reg_window: int = 96            # lifted BEV template size (cells)
    reg_search: int = 12            # +- search radius (cells) for (x, y)
    reg_yaw_bins: int = 9           # yaw hypotheses
    reg_yaw_range: float = math.radians(8.0)
    fine_dim: int = 16              # feature-metric refinement channels
    lm_iters: int = 3
    lm_points: int = 768
    gnss_bias_q: tuple = (0.05, 0.05, 0.05, math.radians(0.05))  # random-walk std per frame
    gnss_bias_p0: tuple = (5.0, 5.0, 3.0, math.radians(5.0))    # initial bias std

    # ---- Module D: prism-anchored streaming instance decoder (PASID) ---------
    dec_layers: int = 3
    height_samples: int = 4         # tau_k samples along each vertical corner line
    deform_points: int = 4          # learned offsets per (head, level, tau)
    memory_slots: int = 8           # view-diverse instance memory (constant cost)
    num_discovery: int = 32         # discovery queries per frame (new buildings)
    max_vertex_step: float = 4.0    # m, first-layer vertex refinement bound
    default_height: float = 12.0    # m, height prior for SHP anchors without height
    default_size: float = 12.0      # m, side of a discovery anchor
    max_range: float = 400.0        # m, prism must be closer than this to be queried
    spawn_thresh: float = 0.5
    drop_thresh: float = 0.15

    # ---- Module E: relation-invariant geo-graph matcher (RIGM) --------------
    match_layers: int = 3
    sinkhorn_iters: int = 30
    gate_radius: float = 40.0       # m, pairs farther than this (beyond pose sigma) are masked
    knn_angles: int = 4             # k for triplet-angle embedding
    match_thresh: float = 0.2

    # ---- Module F: change reasoning (RRD + classifier) ----------------------
    boundary_samples: int = 32
    observe_thresh: float = 0.5     # (v1) accumulated observability to call 'demolished'

    # ---- v2-I height-decoupled structure (GCL / roof edges / height sweep) --
    height_bins: int = 24
    gcl_corr_weight: float = 1.0    # initial weight of the GCL-to-SHP-edge correlation term in CARR

    # ---- v2-II hierarchical spatial-relation reasoning (HSR) ----------------
    hsr_topk: int = 4               # candidates per node (rows and columns)
    hsr_iters: int = 5              # graduated-assignment iterations (annealed)
    hsr_outer: int = 2              # motion-field EM iterations

    # ---- v2-III prism texture atlas (PTA) ------------------------------------
    atlas_s: int = 32               # cells along the footprint perimeter
    atlas_tau: int = 4              # cells along the height
    atlas_roof: int = 4             # roof grid (atlas_roof x atlas_roof)
    atlas_dim: int = 64

    # ---- v3 evidential decision -----------------------------------------------
    u_max: float = 0.4              # max subjective-logic uncertainty to commit to 'demolished'

    @staticmethod
    def tiny() -> "MARSConfig":
        return replace(
            MARSConfig(),
            d_model=64,
            n_heads=4,
            n_vertices=8,
            img_channels=(32, 48, 64),
            ortho_gsd=0.5,
            ortho_stride=2,
            match_dim=16,
            reg_window=48,
            reg_search=6,
            reg_yaw_bins=5,
            fine_dim=8,
            lm_iters=2,
            lm_points=256,
            dec_layers=2,
            height_samples=3,
            deform_points=2,
            memory_slots=4,
            num_discovery=8,
            match_layers=2,
            sinkhorn_iters=20,
            boundary_samples=16,
            max_range=600.0,
            height_bins=16,
            atlas_s=12,
            atlas_tau=3,
            atlas_roof=3,
            atlas_dim=16,
            hsr_iters=4,
        )
