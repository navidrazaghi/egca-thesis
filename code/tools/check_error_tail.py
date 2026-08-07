"""Where does the waypoint error live, and on what kind of frame?

The mean is the wrong statistic for a driving policy.  Closed-loop failure comes
from the tail: a model that predicts within three centimetres on 99% of frames
and is two metres out on the rest has an excellent average and hits something.
This project has direct evidence that the average does not carry: the model
trained on our own collection reached 0.133 m open-loop and scored DS 2.7.

So this reports the distribution rather than its mean, and splits it by the
situations a driving policy is actually judged on -- what the navigation command
asks for, and how fast the vehicle is going.  A tail concentrated in turns at
speed is a different problem from one concentrated in standstills.

Two metrics, because they answer different questions:

  mean L1     the quantity training reports, averaged over both coordinates
              and all four waypoints, so the numbers here are comparable to the
              validation figure in the log
  final L2    Euclidean error of the last waypoint, two seconds ahead.  This is
              the one the lateral controller actually aims at, and the one that
              grows when the model misreads a turn

That last one is then split into its two components, because they are not
equally dangerous and summing them hides which is which.  In this model's ego
frame x is forward and y is left, so the longitudinal error is a misjudged
speed -- the vehicle arrives somewhere along the right path early or late --
while the lateral error puts it in the wrong place across the road.  Two metres
of the first is untidy; two metres of the second is a lane departure.

Runs on CPU by default: the GPU is normally busy with the next rung of the
ladder, and its allocator holds nearly the whole card.

Usage:
    PYTHONPATH=. python tools/check_error_tail.py \
        --ckpt checkpoints/tf_base/best.pth --root ~/transfuser/data --frames 1500
"""
import argparse
import os

import numpy as np
import torch


CMD = {0: "left", 1: "right", 2: "straight", 3: "follow"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/egca.yaml")
    ap.add_argument("--root", default="~/transfuser/data")
    ap.add_argument("--towns", nargs="*", default=["Town05"])
    ap.add_argument("--frames", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    from egca.config import load_config, Cfg
    from egca.data.transfuser_dataset import TransfuserDataset
    from egca.data.dataset import collate
    from egca.models import EGCAPolicy

    device = torch.device(a.device)
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    mcfg = Cfg(ck["cfg"])
    model = EGCAPolicy(mcfg, sensor_dropout=0.0).to(device).eval()
    model.load_state_dict(ck["model"], strict=True)
    print(f"checkpoint: {a.ckpt}  (epoch {ck.get('epoch')})")

    ds = TransfuserDataset(mcfg, os.path.expanduser(a.root), a.towns,
                           augment=False, split="val")
    stride = max(len(ds) // a.frames, 1)
    idx = list(range(0, len(ds), stride))[:a.frames]

    mean_l1, final_l2, lon, lat, cmds, speeds = [], [], [], [], [], []
    with torch.no_grad():
        for s in range(0, len(idx), a.batch):
            chunk = [ds[i] for i in idx[s:s + a.batch]]
            b = {k: v.to(device) for k, v in collate(chunk).items()}
            pred = model(b)["waypoints"]
            tgt = b["waypoints"]
            mean_l1.extend((pred - tgt).abs().mean(dim=(1, 2)).tolist())
            d = pred[:, -1] - tgt[:, -1]
            final_l2.extend(d.norm(dim=-1).tolist())
            lon.extend(d[:, 0].abs().tolist())      # x is forward
            lat.extend(d[:, 1].abs().tolist())      # y is left
            cmds.extend(b["command"].tolist())
            speeds.extend(b["speed"].squeeze(-1).tolist())

    mean_l1 = np.array(mean_l1)
    final_l2 = np.array(final_l2)
    lon = np.array(lon)
    lat = np.array(lat)
    cmds = np.array(cmds)
    speeds = np.array(speeds)
    print(f"frames scored: {len(mean_l1)}  towns {a.towns}\n")

    print("distribution")
    print(f"  {'':<8}{'mean L1':>9}{'final L2':>10}{'longitudinal':>14}"
          f"{'lateral':>10}")
    for label, q in (("mean", None), ("p50", 50), ("p90", 90),
                     ("p95", 95), ("p99", 99), ("max", 100)):
        vals = []
        for arr in (mean_l1, final_l2, lon, lat):
            vals.append(arr.mean() if q is None else np.percentile(arr, q))
        print(f"  {label:<8}{vals[0]:>9.3f}{vals[1]:>10.3f}{vals[2]:>14.3f}"
              f"{vals[3]:>10.3f}")

    # The share of the final error that is lateral, on the frames where the
    # model is worst. If the tail is longitudinal it is a speed problem the
    # controller can absorb; if it is lateral the vehicle leaves its lane.
    worst = final_l2 >= np.percentile(final_l2, 99)
    print(f"\n  over the worst 1% of frames: "
          f"longitudinal {lon[worst].mean():.2f} m, "
          f"lateral {lat[worst].mean():.2f} m")

    # How much of the total error the worst 1% of frames account for. A policy
    # whose tail is a small slice of a smooth distribution is a different risk
    # from one whose tail is a handful of catastrophic frames.
    cut = np.percentile(final_l2, 99)
    share = final_l2[final_l2 >= cut].sum() / final_l2.sum()
    print(f"\n  worst 1% of frames carry {share * 100:.1f}% of the total "
          f"final-waypoint error")

    print("\nby navigation command")
    print(f"  {'':<10}{'n':>6}{'mean':>9}{'p99':>9}")
    for c in sorted(set(cmds.tolist())):
        m = final_l2[cmds == c]
        print(f"  {CMD.get(c, c):<10}{len(m):>6}{m.mean():>9.3f}"
              f"{np.percentile(m, 99):>9.3f}")

    print("\nby speed")
    print(f"  {'':<14}{'n':>6}{'mean':>9}{'p99':>9}")
    bins = [(-1, 0.5, "stopped"), (0.5, 3.0, "slow"),
            (3.0, 6.0, "moderate"), (6.0, 1e9, "fast")]
    for lo, hi, label in bins:
        sel = (speeds > lo) & (speeds <= hi)
        if sel.sum() == 0:
            continue
        m = final_l2[sel]
        print(f"  {label:<14}{len(m):>6}{m.mean():>9.3f}"
              f"{np.percentile(m, 99):>9.3f}")


if __name__ == "__main__":
    main()
