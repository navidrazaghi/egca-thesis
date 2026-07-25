"""Latency / FLOPs benchmark (Table 5-6): batch-1 forward on a single GPU."""
import argparse
import time

import torch

from ..config import load_config
from ..data.dataset import CarlaDrivingDataset, collate
from ..models import EGCAPolicy


def benchmark(cfg, device, iters=200, warmup=50):
    model = EGCAPolicy(cfg, sensor_dropout=0.0).to(device).eval()
    ds = CarlaDrivingDataset(cfg, [], synthetic_len=1)
    batch = {k: v.to(device) for k, v in collate([ds[0]]).items()}
    with torch.no_grad():
        for _ in range(warmup):
            model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    return ms, model.num_parameters() / 1e6


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/egca.yaml")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ms, mparams = benchmark(cfg, device)
    print(f"attention={cfg.model.fusion.attention}  params={mparams:.1f}M  "
          f"latency={ms:.1f} ms/frame  ({1000 / ms:.1f} FPS)")
