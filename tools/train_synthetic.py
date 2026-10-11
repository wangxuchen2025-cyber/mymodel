"""Train / smoke-test MARS-CD end-to-end on procedural synthetic scenes (CPU-friendly).

    PYTHONPATH=. python tools/train_synthetic.py --iters 40 --scenes 6

Exercises every module (A-F), the streaming state machine, all losses and the
pose teacher-forcing curriculum, then evaluates on a held-out scene. Numbers
obtained here only demonstrate that the pipeline trains; they say nothing
about real-data accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import torch

from marscd.config import MARSConfig
from marscd.data import make_scene
from marscd.evaluation import evaluate_scene
from marscd.losses import MARSCriterion
from marscd.models import MARSCD


def run_scene(model, crit, sc, train=True):
    mem, t1_outs = model.build_memory(sc.rasters, sc.grid, sc.t1_polys, t1_video=sc.t1_video,
                                      t1_targets=sc.t1_targets() if train else None)
    st = model.init_stream(mem, targets=sc.t2_targets() if train else None)
    outs = [model.step(st, img, g, K) for img, g, K in sc.t2_video]
    mo = model.match_and_reason(st, with_targets=train)
    if not train:
        return st, outs, mo
    loss, logs = crit(outs, sc.t2_targets(), mo, extra_frames=[(t1_outs, sc.t1_targets())], memory=mem)
    return loss, logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--scenes", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", type=str, default="")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    cfg = MARSConfig.tiny()
    model = MARSCD(cfg)
    crit = MARSCriterion(cfg)
    print(f"MARS-CD tiny: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters")
    scenes = [make_scene(seed=100 + i, nv=cfg.n_vertices, gsd=cfg.ortho_gsd) for i in range(args.scenes)]
    held_out = make_scene(seed=999, nv=cfg.n_vertices, gsd=cfg.ortho_gsd)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / 5) * 0.5 * (1 + math.cos(math.pi * i / args.iters)))
    history = []
    model.train()
    t0 = time.time()
    for it in range(args.iters):
        model.teacher_pose_prob = max(0.3, 1.0 - it / max(1, args.iters - 1))  # curriculum
        loss, logs = run_scene(model, crit, scenes[it % len(scenes)])
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        logs.update(total=float(loss), grad_norm=float(gn), it=it)
        history.append(logs)
        if it % 5 == 0 or it == args.iters - 1:
            keys = ("total", "reg_nll", "reg_pose", "exist", "poly", "heat", "intra", "match", "change")
            print(f"[{it:3d}] {time.time() - t0:6.1f}s " + " ".join(f"{k}={logs[k]:.3f}" for k in keys if k in logs))
    model.eval()
    with torch.no_grad():
        st, outs, mo = run_scene(model, crit, held_out, train=False)
        metrics = evaluate_scene(held_out, st, outs, mo)
    print("held-out scene metrics:", json.dumps(metrics, indent=1))
    first = sum(h["total"] for h in history[:5]) / min(5, len(history))
    last = sum(h["total"] for h in history[-5:]) / min(5, len(history))
    print(f"mean total loss: first-5 {first:.3f} -> last-5 {last:.3f}")
    if args.log:
        with open(args.log, "w") as f:
            json.dump(dict(history=history, metrics=metrics), f, indent=1)


if __name__ == "__main__":
    main()
