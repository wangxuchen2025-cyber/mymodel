"""End-to-end test: offline T1 memory -> streaming T2 frames -> matching -> report -> loss.backward()."""

import torch

from marscd.config import MARSConfig
from marscd.data import make_scene
from marscd.losses import MARSCriterion
from marscd.models import MARSCD


def test_end_to_end_tiny():
    torch.manual_seed(0)
    cfg = MARSConfig.tiny()
    model = MARSCD(cfg)
    crit = MARSCriterion(cfg)
    sc = make_scene(seed=7, nv=cfg.n_vertices, gsd=cfg.ortho_gsd, n_t1=2, n_t2=3)
    model.train()
    model.teacher_pose_prob = 0.5
    mem, t1_outs = model.build_memory(sc.rasters, sc.grid, sc.t1_polys, t1_video=sc.t1_video,
                                      t1_targets=sc.t1_targets())
    assert mem.feats.shape == (sc.t1_polys.shape[0], cfg.d_model)
    st = model.init_stream(mem, targets=sc.t2_targets())
    outs = [model.step(st, img, g, K) for img, g, K in sc.t2_video]
    mo = model.match_and_reason(st, with_targets=True)
    loss, logs = crit(outs, sc.t2_targets(), mo, extra_frames=[(t1_outs, sc.t1_targets())], memory=mem)
    assert torch.isfinite(loss)
    loss.backward()
    assert {"reg_pose", "exist", "poly", "match"} <= set(logs)

    model.eval()
    with torch.no_grad():
        mem, _ = model.build_memory(sc.rasters, sc.grid, sc.t1_polys, t1_video=sc.t1_video)
        st = model.init_stream(mem)
        for img, g, K in sc.t2_video:
            model.step(st, img, g, K)
        rep = model.match_and_reason(st)["report"]
    kinds = {"unchanged", "expanded", "shrunk", "demolished", "new", "unobserved"}
    assert all(r["kind"] in kinds for r in rep)
