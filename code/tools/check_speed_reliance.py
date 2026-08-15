"""How much of the policy's output is decided by the speedometer alone?

The closed-loop failure is a standstill the policy cannot leave: traced over
three Longest6 routes it braked on 89% of frames and predicted a two-second
reach of 0.76 m. The mechanism is that the strongest predictor of the next speed
is the current one, which is true in the data and self-sustaining on the road.

This measures the dependence directly rather than inferring it: hold the scene
fixed, change only the speed reading, and see how far the predicted trajectory
moves. A policy that reads the road should barely notice. A policy that has
learned to echo the speedometer will swing from a standstill to full speed on
the same image.

Usage:
    PYTHONPATH=. python tools/check_speed_reliance.py \
        --ckpt checkpoints/tf_base/best.pth --root ~/transfuser/data
"""
import argparse
import os

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="~/transfuser/data")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from egca.config import load_config, Cfg
    from egca.data.transfuser_dataset import TransfuserDataset
    from egca.data.dataset import collate
    from egca.models import EGCAPolicy

    device = torch.device(a.device)
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    cfg = Cfg(ck["cfg"])
    model = EGCAPolicy(cfg, sensor_dropout=0.0).to(device).eval()
    model.load_state_dict(ck["model"], strict=True)
    print("checkpoint: %s (epoch %s)" % (a.ckpt, ck.get("epoch")))

    ds = TransfuserDataset(cfg, os.path.expanduser(a.root), ["Town05"],
                           augment=False, split="val")
    stride = max(len(ds) // a.frames, 1)
    idx = list(range(0, len(ds), stride))[:a.frames]
    dt = cfg.model.decoder.wp_dt

    # Only moving frames: on a frame where the expert is stopped, predicting a
    # standstill is correct and says nothing about whether the speedometer is
    # being echoed.
    speeds = [0.0, 2.0, 4.0, 6.0]
    reach = {v: [] for v in speeds}
    with torch.no_grad():
        for s in range(0, len(idx), a.batch):
            chunk = [ds[i] for i in idx[s:s + a.batch]]
            chunk = [c for c in chunk if float(c["speed"][0]) > 2.0]
            if not chunk:
                continue
            base = collate(chunk)
            for v in speeds:
                b = {k: t.to(device) for k, t in base.items()}
                b["speed"] = torch.full_like(b["speed"], v)
                wp = model(b)["waypoints"]
                # implied target speed, the quantity the controller reads
                step = (wp[:, 1] - wp[:, 0]).norm(dim=-1) / dt
                reach[v].extend(step.tolist())

    n = len(reach[speeds[0]])
    print("moving frames scored: %d\n" % n)
    print("  speed fed in    implied target speed out")
    means = []
    for v in speeds:
        mu = float(np.mean(reach[v]))
        means.append(mu)
        print("  %5.1f m/s        %6.2f m/s" % (v, mu))

    swing = max(means) - min(means)
    print("\n  swing across the range: %.2f m/s" % swing)
    print("  as a fraction of the speed range fed in: %.2f"
          % (swing / (max(speeds) - min(speeds))))
    print("\nA value near 1 means the network is echoing the speedometer and\n"
          "the scene is barely consulted; near 0 means the trajectory is\n"
          "decided by what the sensors see, which is what lets a stopped car\n"
          "start again.")


if __name__ == "__main__":
    main()
