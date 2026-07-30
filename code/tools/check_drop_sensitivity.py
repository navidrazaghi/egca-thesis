"""How much does each trained model change when a sensor is taken away?

This is an open-loop probe of the claim behind modality dropout: a policy
trained with it should have learned to stand on either modality, so removing one
should move its trajectory less.  Nothing here needs a simulator, so it can be
measured on the checkpoints that already exist and it costs minutes rather than
the days a closed-loop robustness sweep costs.

It is a proxy, not the experiment.  A small trajectory shift under sensor loss
is necessary for robust driving but not sufficient -- a model that ignores LiDAR
entirely also scores perfectly here, which is why `camera_only` is reported
alongside as the degenerate reference.  The closed-loop matrix remains the real
test; this says whether it is worth running and roughly what to expect.

Usage:
    python tools/check_drop_sensitivity.py --frames 40 [--device cpu]
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", default="checkpoints")
    ap.add_argument("--config", default="configs/egca.yaml")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    from egca.config import load_config, Cfg
    from egca.models import EGCAPolicy
    from egca.data.dataset import CarlaDrivingDataset, collate

    device = torch.device(a.device)
    cfg = load_config(a.config, [])
    ds = CarlaDrivingDataset(cfg, cfg.data.towns_train, augment=False,
                             split="val")
    if len(ds) == 0:
        sys.exit("no validation frames")
    idx = np.linspace(0, len(ds) - 1, min(a.frames, len(ds))).astype(int)
    batches = [collate([ds[int(i)]]) for i in idx]
    batches = [{k: v.to(device) for k, v in b.items()} for b in batches]

    # Only frames where the expert was actually moving: a standstill label is
    # near-zero whatever the sensors say, so it would dilute the measurement
    # towards "nothing changes" for every model equally.
    moving = [b for b in batches
              if float((b["waypoints"][0, 1] - b["waypoints"][0, 0]).norm())
              / cfg.model.decoder.wp_dt > 2.0]
    print(f"frames: {len(batches)} loaded, {len(moving)} with the expert moving")
    if not moving:
        sys.exit("no moving frames in the sample")

    rows = []
    for p in sorted(glob.glob(os.path.join(a.checkpoints, "*", "best.pth"))):
        name = os.path.basename(os.path.dirname(p))
        ck = torch.load(p, map_location=device, weights_only=False)
        if "cfg" not in ck:
            continue
        mcfg = Cfg(ck["cfg"])
        mode = getattr(mcfg.model.fusion, "mode", "egca")
        model = EGCAPolicy(mcfg, sensor_dropout=0.0).to(device).eval()
        model.load_state_dict(ck["model"], strict=True)

        d_lidar, d_cam, base = [], [], []
        with torch.no_grad():
            for b in moving:
                w0 = model(b)["waypoints"]
                base.append(float(w0.norm(dim=-1).mean()))
                if mode != "camera_only":
                    w = model(b, force_drop="lidar")["waypoints"]
                    d_lidar.append(float((w - w0).norm(dim=-1).mean()))
                if mode != "lidar_only":
                    w = model(b, force_drop="cam")["waypoints"]
                    d_cam.append(float((w - w0).norm(dim=-1).mean()))
        rows.append((name, np.mean(base),
                     np.mean(d_lidar) if d_lidar else float("nan"),
                     np.mean(d_cam) if d_cam else float("nan")))

    print(f"\nmean waypoint shift when a modality is removed, metres\n")
    print(f"{'checkpoint':<14}{'|wp|':>8}{'drop lidar':>12}{'drop camera':>13}")
    print("-" * 47)
    for name, b, dl, dc in rows:
        f = lambda v: f"{v:>12.3f}" if v == v else f"{'n/a':>12}"
        print(f"{name:<14}{b:>8.2f}{f(dl)}{f(dc)[:12]:>13}")

    print("\nreading it: a model that leans entirely on one sensor barely moves "
          "when the\nother is removed, so compare against camera_only -- the "
          "degenerate case where\nremoving LiDAR is by construction free.")


if __name__ == "__main__":
    main()
