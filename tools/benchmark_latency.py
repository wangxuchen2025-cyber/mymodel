"""Per-module latency of the ONLINE T2 path (the offline T1 memory is built once and cached).

    PYTHONPATH=. python tools/benchmark_latency.py --config deploy --frames 20 [--half]

On an RTX 4090 run with --half (fp16 autocast) and a CUDA build of PyTorch.
On CPU the numbers are only indicative of relative module cost.
"""

from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict

import torch

from marscd.config import MARSConfig
from marscd.geometry import resample_polygon
from marscd.models import MARSCD
from marscd.models.ortho_encoder import GeoGrid
from marscd.models.registration import GNSSReading


def synthetic_inputs(cfg: MARSConfig, extent: float, n_side: int, H: int, W: int, device):
    gsd = cfg.ortho_gsd
    n = int(extent / gsd)
    rasters = {m: torch.rand(3, n, n, device=device) for m in cfg.ortho_modalities}
    grid = GeoGrid(-extent / 2, extent / 2, gsd, n, n)
    step = extent / n_side
    polys = []
    for i in range(n_side):
        for j in range(n_side):
            cx, cy = -extent / 2 + (i + 0.5) * step, -extent / 2 + (j + 0.5) * step
            s = 0.3 * step
            polys.append(resample_polygon([[cx - s, cy - s], [cx + s, cy - s], [cx + s, cy + s], [cx - s, cy + s]],
                                          cfg.n_vertices))
    polys = torch.stack(polys).to(device)
    K = torch.tensor([[0.8 * W, 0, W / 2], [0, 0.8 * W, H / 2], [0, 0, 1.0]], device=device)
    return rasters, grid, polys, K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["deploy", "tiny"], default="deploy")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--half", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = MARSConfig() if args.config == "deploy" else MARSConfig.tiny()
    H, W = (544, 960) if args.config == "deploy" else (160, 256)
    model = MARSCD(cfg).to(dev).eval()
    rasters, grid, polys, K = synthetic_inputs(cfg, 400.0 if args.config == "deploy" else 160.0, 14, H, W, dev)
    times = defaultdict(list)

    def timed(name, mod):
        def pre(*_):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            mod._t0 = time.perf_counter()

        def post(*_):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            times[name].append(time.perf_counter() - mod._t0)
        mod.register_forward_pre_hook(pre)
        mod.register_forward_hook(post)

    for name in ("backbone", "carr", "decoder", "matcher"):
        timed(name, getattr(model, name))
    amp = torch.autocast(dev.type, dtype=torch.float16 if dev.type == "cuda" else torch.bfloat16, enabled=args.half)
    with torch.no_grad(), amp:
        mem, _ = model.build_memory(rasters, grid, polys)
        times.clear()
        st = model.init_stream(mem)
        total = []
        for k in range(args.frames):
            phi = 0.05 * k
            C = torch.tensor([220 * math.cos(phi), 220 * math.sin(phi), 120.0], device=dev)
            yaw = torch.atan2(-C[0], -C[1])
            g = GNSSReading(C + torch.tensor([2.0, -2.0, 0.5], device=dev), yaw + 0.03,
                            torch.tensor(-math.radians(30.0), device=dev), torch.tensor(0.0, device=dev))
            img = torch.rand(3, H, W, device=dev)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            fo = model.step(st, img, g, K)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            total.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        model.match_and_reason(st)
        t_match = time.perf_counter() - t0
    skip = min(2, args.frames - 1)  # warm-up
    print(f"device={dev.type} config={args.config} input={W}x{H} candidates(last)={fo.kind.numel()} tracks={len(st.tracks)}")
    for name, v in times.items():
        v = v[skip:] if name != "matcher" else v
        if v:
            print(f"  {name:9s} {1000 * sum(v) / len(v):8.1f} ms/call ({len(v)} calls)")
    mean = sum(total[skip:]) / len(total[skip:])
    print(f"  step total {1000 * mean:8.1f} ms/frame  -> {1 / mean:.1f} FPS")
    print(f"  cross-temporal match+reason (periodic, not per frame): {1000 * t_match:.1f} ms")


if __name__ == "__main__":
    main()
