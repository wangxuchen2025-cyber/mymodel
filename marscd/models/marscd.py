"""MARS-CD v2: Map-Anchored Relational Streaming Change Detection.

    offline (T1)                               online (T2, per frame)
    ------------                               ----------------------
    DOM sat / DOM aerial / raster map --+      frame -> B oblique encoder -> v2-I structure head (GCL, roof edges)
    SHP footprints ---------------------+-> A  geo-prior memory (+ SHP edge raster, T1 atlas, T1 height posterior)
    historical oblique video ------------+                           |
                                              GNSS+gimbal -> C CARR: bias KF + BEV feature correlation
                                                              + image-space GCL<->SHP render-and-compare + LM
                                                                     |
                                              v2-I height sweep (per-instance 1-D posterior from roof edges)
                                                                     |
                                              D PASID: prior / track / discovery prism queries, atlas memory,
                                                 GCL contact snake
                                                                     |
                                              track bank: v2-III prism texture atlas, height posterior,
                                              z-buffer observability -> v3 evidential (Beta) accumulation
                                                                     |
                               periodic / end:  E RIGM unary -> v2-II HSR (2nd-order QAP + motion-field EM)
                                                                     |
                                              F RRD + atlas comparison + classifier -> v3 evidential report
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from ..config import MARSConfig
from ..geometry import Pose, boundary_samples, points_in_polygons, polygon_centroid, polygon_iou_paired, project
from ..geometry.visibility import build_zbuffer, point_visibility
from . import evidential
from .atlas import PrismAtlas
from .change_head import PAIR_CLASSES, ChangeReasoner
from .graph_matcher import RIGM, GraphNodes, decode_matches
from .oblique_backbone import build_backbone
from .ortho_encoder import GeoGrid, GeoPriorEncoder, GeoPriorMemory
from .prism_decoder import DISCOVERY, PASID, PRIOR, TRACK, DecoderOutput, QuerySet
from .registration import CARR, GNSSBiasFilter, GNSSReading, RegistrationOutput
from .relational_assignment import RelationalAssignment
from .structure import HeightSweep, StructureHead


# --------------------------------------------------------------------------- state
@dataclass
class SequenceTargets:
    """Training-time ground truth for one stream (T2 video, or T1 video with SHP as GT)."""

    polys: Tensor        # [G, Nv, 2] instance footprints of the stream's epoch
    heights: Tensor      # [G]
    t1_to_gt: Tensor     # [M] index of the GT instance corresponding to SHP polygon j, -1 if none
    t1_change: Tensor    # [M] 0 unchanged / 1 expanded / 2 shrunk / 3 demolished
    poses: list          # GT Pose per frame (offline aerotriangulation)
    bases: Tensor | None = None  # [G] base elevation (DEM / photogrammetric DSM); None -> ground plane


class TrackBank:
    """Persistent T2 instance states: prism geometry, existence, height posterior and texture atlas."""

    FIELDS = ("feat", "polys", "base", "height", "logodds", "hits", "anchor_src", "gt_id", "atlas_sum",
              "atlas_w", "height_ll")

    def __init__(self, C: int, Nv: int, n_cells: int, ca: int, n_hbins: int, device):
        z = lambda *s, **kw: torch.zeros(*s, device=device, **kw)  # noqa: E731
        self.feat, self.polys = z(0, C), z(0, Nv, 2)
        self.base, self.height, self.logodds = z(0), z(0), z(0)
        self.hits = z(0, dtype=torch.long)
        self.anchor_src = z(0, dtype=torch.long)
        self.gt_id = z(0, dtype=torch.long)
        self.atlas_sum, self.atlas_w = z(0, n_cells, ca), z(0, n_cells)
        self.height_ll = z(0, n_hbins)

    def __len__(self) -> int:
        return self.feat.shape[0]

    @property
    def p_exist(self) -> Tensor:
        return torch.sigmoid(self.logodds)

    def add(self, feat, polys, base, height, logit, anchor_src, gt_id, atlas_sum, atlas_w, height_ll) -> Tensor:
        n0, n = len(self), feat.shape[0]
        new = dict(feat=feat, polys=polys, base=base, height=height, logodds=(0.5 * logit).clamp(-6, 6),
                   hits=torch.ones(n, dtype=torch.long, device=feat.device), anchor_src=anchor_src, gt_id=gt_id,
                   atlas_sum=atlas_sum, atlas_w=atlas_w, height_ll=height_ll)
        for k in self.FIELDS:
            setattr(self, k, torch.cat([getattr(self, k), new[k]]))
        return torch.arange(n0, n0 + n, device=feat.device)

    def update(self, idx, feat, polys, base, height, logit, weight, atlas_sum, atlas_w, height_ll) -> None:
        w = weight[:, None, None]
        self.feat = self.feat.index_put((idx,), feat)
        self.polys = self.polys.index_put((idx,), self.polys[idx] + w * (polys - self.polys[idx]))
        self.base = self.base.index_put((idx,), base)
        self.height = self.height.index_put((idx,), height)
        self.logodds = self.logodds.index_put((idx,), (self.logodds[idx] + 0.5 * logit * weight).clamp(-6, 6))
        self.hits = self.hits.index_put((idx,), self.hits[idx] + 1)
        self.atlas_sum = self.atlas_sum.index_add(0, idx, atlas_sum)
        self.atlas_w = self.atlas_w.index_add(0, idx, atlas_w)
        self.height_ll = self.height_ll.index_put((idx,), height_ll)

    def keep(self, mask: Tensor) -> None:
        for k in self.FIELDS:
            setattr(self, k, getattr(self, k)[mask])

    def detach(self, keep_atlas_grad: bool = False) -> None:
        """Truncate the graph between frames; atlas evidence may stay differentiable within a training clip so
        that the cross-temporal losses train the atlas projection (truncated BPTT over the clip)."""
        for k in self.FIELDS:
            v = getattr(self, k)
            if v.is_floating_point() and not (keep_atlas_grad and k == "atlas_sum"):
                setattr(self, k, v.detach())


@dataclass
class StreamState:
    memory: GeoPriorMemory
    tracks: TrackBank
    bias: GNSSBiasFilter
    t1_spawned: Tensor
    t1_obs: Tensor                         # noisy-or observability (diagnostic)
    t1_r: Tensor                           # v3 evidence: T1 footprint occupied at the stream epoch
    t1_s: Tensor                           # v3 evidence: T1 footprint empty
    ground_z: float
    mode: str = "t2"                       # "t1": historical video (no discovery), "t2": current video
    targets: SequenceTargets | None = None
    frame_idx: int = 0
    last_report: list | None = None


@dataclass
class FrameOutput:
    reg: RegistrationOutput
    dec: DecoderOutput
    kind: Tensor
    src: Tensor                 # prior index (PRIOR) / track index (TRACK) / -1
    target: Tensor | None       # GT instance per candidate (-1 negative)
    proposals: dict
    intra: dict | None = None
    gt_pose: Pose | None = None
    visible_gt: Tensor | None = None
    extras: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- model
class MARSCD(nn.Module):
    def __init__(self, cfg: MARSConfig | None = None):
        super().__init__()
        self.cfg = cfg or MARSConfig()
        cfg = self.cfg
        self.geo_prior = GeoPriorEncoder(cfg)        # A
        self.backbone = build_backbone(cfg)          # B
        self.structure = StructureHead(cfg)          # v2-I
        self.height_sweep = HeightSweep(cfg, n_bins=cfg.height_bins)
        self.carr = CARR(cfg)                        # C
        self.decoder = PASID(cfg)                    # D
        self.atlas = PrismAtlas(cfg)                 # v2-III
        self.matcher = RIGM(cfg)                     # E (unary)
        self.hsr = RelationalAssignment(cfg)         # v2-II
        self.change = ChangeReasoner(cfg)            # F
        self.pose_grad = False                       # gradients from instance losses into the pose
        self.teacher_pose_prob = 0.0                 # training curriculum: decode with (noisy) GT pose

    # ============================================================== offline (T1)
    def build_memory(self, rasters: dict, grid: GeoGrid, polys: Tensor, available: dict | None = None,
                     t1_video: list | None = None, t1_targets: SequenceTargets | None = None,
                     ground_z: float = 0.0) -> tuple[GeoPriorMemory, list]:
        mem = self.geo_prior(rasters, grid, polys, available)
        outs: list = []
        if t1_video:
            st = self.init_stream(mem, ground_z, mode="t1", targets=t1_targets)
            for img, gnss, K in t1_video:
                outs.append(self.step(st, img, gnss, K))
            M, tb = mem.num, st.tracks
            obl = mem.feats.new_zeros(M, self.cfg.d_model)
            hts = mem.heights.clone()
            A = mem.feats.new_zeros(M, self.atlas.n_cells, self.atlas.ca)
            Wt = mem.feats.new_zeros(M, self.atlas.n_cells)
            ll = mem.feats.new_zeros(M, self.height_sweep.n)
            ok = tb.anchor_src >= 0
            if ok.any():
                j = tb.anchor_src[ok]
                obl = obl.index_put((j,), self.atlas.pooled(tb.atlas_sum[ok], tb.atlas_w[ok]))
                hts = hts.index_put((j,), tb.height[ok])
                A = A.index_put((j,), tb.atlas_sum[ok])
                Wt = Wt.index_put((j,), tb.atlas_w[ok])
                ll = ll.index_put((j,), tb.height_ll[ok])
            mem = self.geo_prior.fuse_oblique(mem, obl, st.t1_obs, hts, ll, A, Wt)
        return mem, outs

    # ============================================================== online (T2)
    def init_stream(self, memory: GeoPriorMemory, ground_z: float = 0.0, mode: str = "t2",
                    targets: SequenceTargets | None = None) -> StreamState:
        cfg, dev = self.cfg, memory.feats.device
        M = memory.num
        tb = TrackBank(cfg.d_model, cfg.n_vertices, self.atlas.n_cells, self.atlas.ca, self.height_sweep.n, dev)
        z = torch.zeros(M, device=dev)
        return StreamState(memory, tb, GNSSBiasFilter(cfg.gnss_bias_q, cfg.gnss_bias_p0, dev),
                           torch.zeros(M, dtype=torch.bool, device=dev), z.clone(), z.clone(), z.clone(), ground_z,
                           mode, targets)

    def _in_view(self, polys: Tensor, base: Tensor, height: Tensor, pose: Pose, hw) -> Tensor:
        if polys.shape[0] == 0:
            return torch.zeros(0, dtype=torch.bool, device=polys.device)
        N, Nv, _ = polys.shape
        tau = torch.tensor([0.0, 1.0], device=polys.device)
        X = torch.cat([polys[:, :, None].expand(N, Nv, 2, 2),
                       (base[:, None, None] + tau * height[:, None, None])[..., None].expand(N, Nv, 2, 1)], -1)
        uv, depth = project(X, pose)
        H, W = hw
        ok = (depth > 0.5) & (depth < self.cfg.max_range) & (uv[..., 0] >= 0) & (uv[..., 0] < W) & (uv[..., 1] >= 0) & (uv[..., 1] < H)
        return ok.flatten(1).float().mean(1) > 0.25

    def _change_beliefs(self, st: StreamState) -> tuple[Tensor | None, Tensor | None]:
        """Regions excluded from registration (streaming EM): unexplained T2 tracks and T1 buildings whose
        anchored track currently believes the footprint is empty."""
        tb = st.tracks
        if len(tb) == 0:
            return None, None
        p = tb.p_exist.detach()
        new_like = tb.anchor_src < 0
        gone_like = (tb.anchor_src >= 0) & (p < 0.5)
        sel = (new_like & (p > 0.5)) | gone_like
        if not sel.any():
            return None, None
        prob = torch.where(new_like[sel], p[sel], 1 - p[sel])
        return tb.polys[sel].detach(), prob

    def _footprint_visibility(self, polys: Tensor, t1_idx: Tensor, zb, occ_owner: Tensor, pose: Pose, hw,
                              ground_z: float) -> Tensor:
        """Fraction of each T1 footprint's ground samples that is observed: no OTHER prism in front of it
        (the ground is visible, or the building's own prism is the front-most surface)."""
        if t1_idx.numel() == 0:
            return polys.new_zeros(0)
        P = polys[t1_idx]
        bp, _ = boundary_samples(P, 12)
        cen = polygon_centroid(P)
        xy = torch.cat([bp, cen[:, None], cen[:, None] + 0.5 * (P - cen[:, None])], 1)
        X = torch.cat([xy, torch.full_like(xy[..., :1], ground_z)], -1)
        vis = point_visibility(X, pose, hw, zb, owner=occ_owner, own_ids=t1_idx[:, None].expand(-1, X.shape[1]))
        return vis.float().mean(1)

    # ---------------------------------------------------------------- one frame
    def step(self, st: StreamState, img: Tensor, gnss: GNSSReading, K: Tensor) -> FrameOutput:
        cfg, mem, tb = self.cfg, st.memory, st.tracks
        hw = tuple(img.shape[-2:])
        dev = img.device
        tgt = st.targets
        feats = self.backbone(img)
        struct = self.structure(feats)  # v2-I: [2, H/4, W/4] contact-line / roof-edge logits

        # ---- C: registration (feature correlation + height-free structural likelihood) ------------------
        _, bsig = st.bias.predict()
        prior = st.bias.prior_pose(gnss, K)
        cp, cprob = self._change_beliefs(st)
        reg = self.carr(feats, hw, prior, mem, st.ground_z, torch.stack([bsig[0], bsig[1], bsig[3]]), cp, cprob,
                        struct_logits=struct)
        pose = reg.pose if self.pose_grad else reg.pose.detach()
        if self.training and tgt is not None and torch.rand(()) < self.teacher_pose_prob:
            pose = self._noisy_pose(tgt.poses[st.frame_idx])

        # ---- D: candidate queries ------------------------------------------------------------------------
        base_m = torch.full_like(mem.heights, st.ground_z)
        p_idx = (self._in_view(mem.polys, base_m, mem.heights, pose, hw) & ~st.t1_spawned).nonzero(as_tuple=True)[0]
        t_idx = self._in_view(tb.polys, tb.base, tb.height, pose, hw).nonzero(as_tuple=True)[0] if len(tb) else \
            torch.zeros(0, dtype=torch.long, device=dev)
        anchors = torch.cat([mem.polys[p_idx], tb.polys[t_idx]])
        if st.mode == "t2":
            covered = points_in_polygons(reg.cell_xy.reshape(-1, 2), anchors).any(0).view(reg.cell_xy.shape[:2]) \
                if anchors.shape[0] else torch.zeros(reg.cell_xy.shape[:2], dtype=torch.bool, device=dev)
            gcl_bev, _ = self.carr._lift(torch.sigmoid(struct[:1]), pose, reg.cell_xy, st.ground_z, hw)
            prop = self.decoder.proposer(torch.cat([reg.lifted, gcl_bev]), reg.lifted_valid, reg.cell_xy, covered,
                                         cfg.num_discovery)
        else:
            prop = dict(polys=mem.polys.new_zeros(0, cfg.n_vertices, 2), feats=mem.feats.new_zeros(0, cfg.d_model),
                        heat_logit=None)
        nP, nT, nD = len(p_idx), len(t_idx), prop["polys"].shape[0]
        polys0 = torch.cat([mem.polys[p_idx], tb.polys[t_idx], prop["polys"]])
        base0 = torch.cat([base_m[p_idx], tb.base[t_idx], torch.full((nD,), st.ground_z, device=dev)])

        # v2-I: per-instance height sweep (posterior carried by the prior / track, updated with this frame)
        Bh, Nc, Ca = self.height_sweep.n, self.atlas.n_cells, self.atlas.ca
        ll_p = mem.height_ll[p_idx] if mem.height_ll is not None else mem.feats.new_zeros(nP, Bh)
        ll = torch.cat([ll_p, tb.height_ll[t_idx], mem.feats.new_zeros(nD, Bh)])
        ll = ll + self.height_sweep.increment(struct[1], polys0, base0, pose, hw)
        h_sweep, _ = self.height_sweep.estimate(ll)

        # v2-III: atlas tokens as the decoder memory (T1 atlas for priors, own T2 atlas for tracks)
        A_p = mem.atlas_sum[p_idx] if mem.atlas_sum is not None else mem.feats.new_zeros(nP, Nc, Ca)
        W_p = mem.atlas_w[p_idx] if mem.atlas_w is not None else mem.feats.new_zeros(nP, Nc)
        tok, tok_valid = self.atlas.tokens(torch.cat([A_p, tb.atlas_sum[t_idx], mem.feats.new_zeros(nD, Nc, Ca)]),
                                           torch.cat([W_p, tb.atlas_w[t_idx], mem.feats.new_zeros(nD, Nc)]))
        qs = QuerySet(
            feat=torch.cat([mem.feats[p_idx], tb.feat[t_idx], prop["feats"]]),
            polys=polys0, base=base0, height=h_sweep,
            kind=torch.cat([torch.full((nP,), PRIOR), torch.full((nT,), TRACK), torch.full((nD,), DISCOVERY)]).to(dev),
            mem=tok, mem_valid=tok_valid,
        )
        src = torch.cat([p_idx, t_idx, torch.full((nD,), -1, device=dev, dtype=torch.long)])
        dec = self.decoder(qs, feats, pose, hw, struct)

        # ---- training targets (MOTR-style persistent identities) -----------------------------------------
        target, visible_gt = None, None
        if tgt is not None:
            target, visible_gt = self._assign_targets(st, qs, src, dec, pose, hw)
        out = FrameOutput(reg, dec, qs.kind, src, target, prop, gt_pose=tgt.poses[st.frame_idx] if tgt else None,
                          visible_gt=visible_gt, extras=dict(struct=struct, hw=hw))
        if qs.n == 0:
            self._finish_frame(st, reg, gnss, pose)
            return out
        fin = dec.final
        p_ex = torch.sigmoid(fin["exist_logit"]).detach()
        polys_f, base_f, height_f = fin["polys"].detach(), fin["base"].detach(), fin["height"].detach()

        # ---- shared z-buffer of confident prisms ----------------------------------------------------------
        owner = torch.full_like(src, -1)  # T1 building each candidate stands for
        owner[qs.kind == PRIOR] = src[qs.kind == PRIOR]
        if nT:
            owner[qs.kind == TRACK] = tb.anchor_src[src[qs.kind == TRACK]]
        occ = p_ex > 0.5
        zb = build_zbuffer(polys_f[occ], base_f[occ], height_f[occ], pose, hw)
        occ_id = torch.full_like(src, -1)
        occ_id[occ] = torch.arange(int(occ.sum()), device=dev)

        # ---- v2-III atlas observation (all candidates, final geometry) --------------------------------------
        A_new, W_new = self.atlas.observe(feats[0], polys_f, base_f, height_f, pose, hw, zb, occ_id)
        if not self.training:
            A_new = A_new.detach()
        W_new = W_new.detach()

        # ---- T1 observability + v3 novelty-discounted evidential accumulation --------------------------------
        t1_view = self._in_view(mem.polys, base_m, mem.heights, pose, hw).nonzero(as_tuple=True)[0]
        o = self._footprint_visibility(mem.polys, t1_view, zb, owner[occ], pose, hw, st.ground_z)
        st.t1_obs = st.t1_obs.index_put((t1_view,), 1 - (1 - st.t1_obs[t1_view]) * (1 - o))
        obs_full = mem.feats.new_zeros(mem.num).index_put((t1_view,), o)
        W_old = torch.cat([mem.feats.new_zeros(nP, Nc), tb.atlas_w[t_idx], mem.feats.new_zeros(nD, Nc)])
        nov = evidential.novelty(W_new, W_old)
        own = owner >= 0
        if own.any():
            st.t1_r, st.t1_s = evidential.accumulate(st.t1_r, st.t1_s, owner[own], fin["exist_logit"][own],
                                                     (obs_full[owner[own]] * nov[own]).detach())

        # ---- track bank update -----------------------------------------------------------------------------
        f_all = dec.feat
        g = lambda k: fin[k]  # noqa: E731
        if nP:
            sl = slice(0, nP)
            gt = target[sl] if target is not None else torch.full((nP,), -1, device=dev, dtype=torch.long)
            tb.add(f_all[sl], g("polys")[sl], g("base")[sl], g("height")[sl], g("exist_logit")[sl], p_idx, gt,
                   A_new[sl], W_new[sl], ll[sl].detach())
            st.t1_spawned[p_idx] = True
        if nT:
            sl = slice(nP, nP + nT)
            tb.update(t_idx, f_all[sl], g("polys")[sl], g("base")[sl], g("height")[sl], g("exist_logit")[sl],
                      dec.in_view[sl].detach(), A_new[sl], W_new[sl], ll[sl].detach())
        if nD:
            sl = slice(nP + nT, nP + nT + nD)
            keep = p_ex[sl] > cfg.spawn_thresh
            if self.training and target is not None:  # teacher forcing: spawn every positive proposal
                keep = keep | (target[sl] >= 0)
            d_ids = keep.nonzero(as_tuple=True)[0] + nP + nT
            out.intra = self._intra_associate(st, d_ids, t_idx, dec, target, reg, A_new, W_new, ll.detach())
        if len(tb):
            dead = (tb.anchor_src < 0) & (tb.hits >= 3) & (tb.p_exist < cfg.drop_thresh)
            if dead.any():
                tb.keep(~dead)
        self._finish_frame(st, reg, gnss, pose)
        return out

    @staticmethod
    def _noisy_pose(gt: Pose, sig_t: float = 0.5, sig_r: float = 0.005) -> Pose:
        xi = torch.cat([torch.randn(3) * sig_r, torch.randn(3) * sig_t]).to(gt.R_cw)
        return gt.left_update(xi)

    def _finish_frame(self, st: StreamState, reg: RegistrationOutput, gnss: GNSSReading, pose_used: Pose) -> None:
        pz = pose_used.detach()
        dyaw = pz.yaw - gnss.yaw
        z = torch.cat([pz.center - gnss.position, torch.atan2(dyaw.sin(), dyaw.cos())[None]])
        st.bias.update(z, torch.stack([reg.var[0], reg.var[1], reg.var.new_tensor(4.0), reg.var[2]]).detach())
        st.tracks.detach(keep_atlas_grad=self.training)
        st.frame_idx += 1

    def _intra_associate(self, st: StreamState, d_ids: Tensor, t_idx: Tensor, dec: DecoderOutput,
                         target: Tensor | None, reg: RegistrationOutput, A_new: Tensor, W_new: Tensor,
                         ll: Tensor) -> dict | None:
        """E (mode 1): new discovery detections <-> existing tracks; matched -> merge, else spawn."""
        tb, cfg, fin = st.tracks, self.cfg, dec.final
        if d_ids.numel() == 0:
            return None
        dev = d_ids.device
        if len(t_idx):  # only tracks that can possibly match a detection enter the graph (O(N^2) relations)
            dc = polygon_centroid(fin["polys"][d_ids].detach())
            tc = polygon_centroid(tb.polys[t_idx])
            t_idx = t_idx[torch.cdist(tc, dc).min(1).values < cfg.gate_radius]
        D = GraphNodes(dec.feat[d_ids], fin["polys"][d_ids], fin["height"][d_ids],
                       torch.ones(len(d_ids), dtype=torch.bool, device=dev), fin["exist_logit"][d_ids])
        A = GraphNodes(tb.feat[t_idx], tb.polys[t_idx], tb.height[t_idx],
                       torch.ones(len(t_idx), dtype=torch.bool, device=dev))
        res = self.matcher(A, D, mode=1, sigma=reg.var[:2].sqrt().detach())
        log_P = res["log_P"]
        with torch.no_grad():
            m_col, _ = decode_matches(log_P.t().contiguous(), cfg.match_thresh)  # per detection -> track
        matched = m_col >= 0
        gt_d = target[d_ids] if target is not None else torch.full((len(d_ids),), -1, device=dev, dtype=torch.long)
        if matched.any():
            ti, di = t_idx[m_col[matched]], d_ids[matched]
            tb.update(ti, dec.feat[di], tb.polys[ti], tb.base[ti], tb.height[ti], fin["exist_logit"][di],
                      torch.ones(len(ti), device=dev), A_new[di], W_new[di], tb.height_ll[ti])
        if (~matched).any():
            di = d_ids[~matched]
            tb.add(dec.feat[di], fin["polys"][di], fin["base"][di], fin["height"][di], fin["exist_logit"][di],
                   torch.full((len(di),), -1, device=dev, dtype=torch.long), gt_d[~matched], A_new[di], W_new[di], ll[di])
        tgt_rows = None
        if target is not None:  # row i (track) -> column (detection) with the same GT identity
            tgt_rows = torch.full((len(t_idx),), -1, dtype=torch.long, device=dev)
            for a, ga in enumerate(tb.gt_id[t_idx].tolist()):  # t_idx stays valid: add() only appends
                if ga >= 0:
                    hit = (gt_d == ga).nonzero(as_tuple=True)[0]
                    if hit.numel():
                        tgt_rows[a] = hit[0]
        return dict(log_P=log_P, target=tgt_rows, det_gt=gt_d)

    # ------------------------------------------------------------ targets
    @torch.no_grad()
    def _assign_targets(self, st: StreamState, qs: QuerySet, src: Tensor, dec: DecoderOutput, pose: Pose, hw):
        from scipy.optimize import linear_sum_assignment

        tgt, tb = st.targets, st.tracks
        dev = qs.feat.device
        G = tgt.polys.shape[0]
        N = qs.n
        target = torch.full((N,), -1, dtype=torch.long, device=dev)
        vis = self._in_view(tgt.polys, torch.full((G,), st.ground_z, device=dev), tgt.heights, pose, hw)
        claimed = torch.zeros(G, dtype=torch.bool, device=dev)
        t1_gts = set(int(x) for x in tgt.t1_to_gt.tolist() if x >= 0)
        for i in range(N):
            k = int(qs.kind[i])
            if k == TRACK:
                gi = int(tb.gt_id[src[i]])
            elif k == PRIOR:
                j = int(src[i])
                gi = int(tgt.t1_to_gt[j])
                if gi < 0:  # building j is gone: the anchor may now host a NEW building
                    ious = polygon_iou_paired(qs.polys[i:i + 1].expand(G, -1, -1), tgt.polys, res=16)
                    for g_ in ious.argsort(descending=True).tolist():
                        if ious[g_] < 0.3:
                            break
                        if g_ not in t1_gts and not claimed[g_]:
                            gi = g_
                            break
            else:
                continue
            if gi >= 0 and not claimed[gi]:
                target[i] = gi
                claimed[gi] = True
        d = (qs.kind == DISCOVERY).nonzero(as_tuple=True)[0]
        free = (vis & ~claimed).nonzero(as_tuple=True)[0]
        if d.numel() and free.numel():
            pc = polygon_centroid(dec.final["polys"][d])
            gc = polygon_centroid(tgt.polys[free])
            cost = torch.cdist(pc, gc)
            r, c = linear_sum_assignment(cost.cpu().numpy())
            for a, b in zip(r, c):
                if cost[a, b] < 15.0:
                    target[d[a]] = free[b]
        return target, vis.nonzero(as_tuple=True)[0]

    @torch.no_grad()
    def _occupancy_targets(self, st: StreamState, a_idx: Tensor) -> Tensor:
        """GT occupancy of T1 footprints at the stream epoch (same building OR a rebuild on the lot)."""
        tgt, mem = st.targets, st.memory
        y = (tgt.t1_to_gt[a_idx] >= 0).float()
        G = tgt.polys.shape[0]
        if G:
            for n, j in enumerate(a_idx.tolist()):
                if y[n] == 0:
                    iou = polygon_iou_paired(mem.polys[j:j + 1].expand(G, -1, -1), tgt.polys, res=16)
                    y[n] = float(iou.max() > 0.3)
        return y

    # ============================================================== cross-temporal
    def match_and_reason(self, st: StreamState, with_targets: bool = False) -> dict:
        """E unary -> v2-II HSR -> F change reasoning (+ atlas) -> v3 evidential geo change report."""
        cfg, mem, tb = self.cfg, st.memory, st.tracks
        dev = mem.feats.device
        a_idx = ((st.t1_obs > 0.05) | st.t1_spawned).nonzero(as_tuple=True)[0]
        b_idx = (tb.hits >= 1).nonzero(as_tuple=True)[0] if len(tb) else torch.zeros(0, dtype=torch.long, device=dev)
        remap = torch.full((mem.num,), -1, dtype=torch.long, device=dev)
        remap[a_idx] = torch.arange(len(a_idx), device=dev)
        src_b = tb.anchor_src[b_idx]
        src_b = torch.where(src_b >= 0, remap[src_b.clamp_min(0)], src_b)
        featA = mem.feats[a_idx]
        if mem.atlas_sum is not None:
            featA = featA + self.atlas.pooled(mem.atlas_sum[a_idx], mem.atlas_w[a_idx])
        A = GraphNodes(featA, mem.polys[a_idx], mem.heights[a_idx], mem.height_known[a_idx])
        B = GraphNodes(tb.feat[b_idx] + self.atlas.pooled(tb.atlas_sum[b_idx], tb.atlas_w[b_idx]), tb.polys[b_idx],
                       tb.height[b_idx], tb.hits[b_idx] >= 2, tb.logodds[b_idx], src_b)
        sigma = st.bias.P.diagonal()[:2].sqrt().clamp_min(0.5)
        res = self.matcher(A, B, mode=0, sigma=sigma)
        out = dict(a_idx=a_idx, b_idx=b_idx, log_P_unary=res["log_P"])

        gt_match = None
        if with_targets and st.targets is not None:
            gt_b = tb.gt_id[b_idx]
            gt_match = torch.full((len(a_idx),), -1, dtype=torch.long, device=dev)
            for i, gidx in enumerate(st.targets.t1_to_gt[a_idx].tolist()):
                if gidx >= 0:
                    hit = (gt_b == gidx).nonzero(as_tuple=True)[0]
                    if hit.numel():  # prefer the track anchored at this T1 building, then the most observed one
                        anchored = hit[src_b[hit] == i]
                        gt_match[i] = anchored[0] if anchored.numel() else hit[tb.hits[b_idx[hit]].argmax()]
            out["gt_match"] = gt_match
        if "S" in res:  # v2-II: second-order relational reasoning + motion-field EM
            h = self.hsr(res["S"], res["alpha"], res["beta"], self.matcher.corner, res["gate"], A.polys, B.polys,
                         forced=gt_match)
            log_P = h["log_P"]
        else:
            log_P = res["log_P"]
        out["log_P"] = log_P
        match, prob = decode_matches(log_P, cfg.match_thresh)
        out["match"] = match

        # --- change classification on predicted matches (inference) / GT pairs (training)
        pairs_a = (match >= 0).nonzero(as_tuple=True)[0]
        pairs_b = match[pairs_a]
        if gt_match is not None:
            pairs_a = (gt_match >= 0).nonzero(as_tuple=True)[0]
            pairs_b = gt_match[pairs_a]
            out["pair_labels"] = st.targets.t1_change[a_idx][pairs_a]
        w = log_P[pairs_a, pairs_b].exp() if len(pairs_a) else log_P.new_zeros(0)
        ja, ib = a_idx[pairs_a], b_idx[pairs_b]
        A1 = mem.atlas_sum[ja] if mem.atlas_sum is not None else None

        def atlas_feats(PA_al):
            if A1 is None:
                return PA_al.new_zeros(PA_al.shape[0], self.change.n_extra)
            f = self.atlas.compare(A1, mem.atlas_w[ja], PA_al, mem.heights[ja], tb.atlas_sum[ib], tb.atlas_w[ib],
                                   tb.polys[ib], tb.height[ib], torch.full((len(ja),), st.ground_z, device=dev))
            # v3: relative (scene-level) normalisation of the appearance residual. Confident, well co-observed
            # pairs act as pseudo-invariant features (the no-change set of relative radiometric normalisation),
            # removing global illumination / season shifts between epochs.
            reliable = (w.detach() > 0.5) & (f[:, 2] > 0.3)
            if int(reliable.sum()) >= 3:
                base_shift = f[reliable, :2].median(0).values
                f = torch.cat([f[:, :2] - base_shift, f[:, 2:]], -1)
            return f

        feat_a = res.get("feat_a", A.feat)
        feat_b = res.get("feat_b", B.feat)
        ch = self.change(feat_a[pairs_a], feat_b[pairs_b], A.polys[pairs_a], B.polys[pairs_b], w, atlas_feats)
        out.update(pairs_a=pairs_a, pairs_b=pairs_b, change=ch)

        # --- v3 evidential opinions of T1 occupancy
        b_, d_, u_ = evidential.opinion(st.t1_r[a_idx], st.t1_s[a_idx])
        out["evidence"] = dict(r=st.t1_r[a_idx], s=st.t1_s[a_idx], belief=b_, disbelief=d_, uncertainty=u_)
        if with_targets and st.targets is not None:
            out["evidence"]["y"] = self._occupancy_targets(st, a_idx)
            out["evidence"]["mask"] = (st.t1_r[a_idx] + st.t1_s[a_idx]).detach() > 1e-3
        out["report"] = self._report(st, a_idx, b_idx, match, prob, ch, pairs_a, gt_match is None, out["evidence"])
        st.last_report = out["report"]
        return out

    @torch.no_grad()
    def _report(self, st, a_idx, b_idx, match, prob, ch, pairs_a, use_pred_cls, ev) -> list[dict]:
        cfg, mem, tb = self.cfg, st.memory, st.tracks
        rep = []
        cls = {}
        if use_pred_cls and len(pairs_a):
            for i, a in enumerate(pairs_a.tolist()):
                cls[a] = int(ch["logits"][i].argmax())
        for i, a in enumerate(a_idx.tolist()):
            j = int(match[i])
            op = dict(belief=float(ev["belief"][i]), disbelief=float(ev["disbelief"][i]),
                      uncertainty=float(ev["uncertainty"][i]))
            if j >= 0:
                rep.append(dict(kind=PAIR_CLASSES[cls.get(i, 0)], t1=a, track=int(b_idx[j]), score=float(prob[i]),
                                t1_polygon=mem.polys[a].tolist(), t2_polygon=tb.polys[b_idx[j]].tolist(), **op))
            else:
                # v3: commit to 'demolished' only with low subjective-logic uncertainty
                kind = "demolished" if op["uncertainty"] < cfg.u_max else "unobserved"
                rep.append(dict(kind=kind, t1=a, track=-1, score=1 - op["uncertainty"], t1_polygon=mem.polys[a].tolist(),
                                t2_polygon=None, rebuilt_or_mismatch=op["belief"] > 0.5, **op))
        matched_b = set(int(m) for m in match.tolist() if m >= 0)
        for j, b in enumerate(b_idx.tolist()):
            if j in matched_b:
                continue
            if float(tb.p_exist[b]) > 0.5 and int(tb.hits[b]) >= 2:
                rep.append(dict(kind="new", t1=-1, track=b, score=float(tb.p_exist[b]), t1_polygon=None,
                                t2_polygon=tb.polys[b].tolist()))
        return rep
