"""Compare what the leaderboard agent feeds the network against the training set.

Two of the three defects found in the evaluation harness so far were train/eval
input mismatches, and neither was visible in open-loop validation: the
look-ahead goal was 3-6x beyond anything in the dataset while the waypoint error
sat at a healthy 0.137 m.  A network cannot be blamed for an input it never saw,
so before touching the model this script asks the only question that separates
"the policy is weak" from "the policy is being fed the wrong thing":

    for every input channel, is the distribution the agent builds at evaluation
    inside the distribution the network was trained on?

Left column comes from the dataset (via the very same Dataset class training
used, so any preprocessing is included).  Right column comes from a trace.jsonl
written by the agent with `debug_dir` set.  Anything whose evaluation median
falls outside the dataset's 5-95% range is flagged, because that is the shape
the goal bug had.

Usage:
    python -m tools.compare_input_dist --dataset dataset --trace diag/trace.jsonl \
        --config configs/egca.yaml [--samples 4000]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch


def pct(v, q):
    return float(np.percentile(v, q)) if len(v) else float("nan")


def summarise(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)],
                   dtype=np.float64)
    if not len(v):
        return None
    return {"n": len(v), "p5": pct(v, 5), "med": pct(v, 50), "p95": pct(v, 95),
            "min": float(v.min()), "max": float(v.max())}


# ------------------------------------------------------------------- dataset
def dataset_stats(root, config, n_samples):
    from egca.config import load_config
    from egca.data.dataset import CarlaDrivingDataset

    cfg = load_config(config, [])
    if root:                      # let --dataset override the configured root
        cfg.data.root = root
    # Held-out routes of the training towns: the same split model selection used,
    # so these are frames the network never fitted.
    ds = CarlaDrivingDataset(cfg, cfg.data.towns_train, augment=False,
                             split="val")
    n = len(ds)
    if n == 0:
        sys.exit(f"no samples found under {root}")
    idx = np.linspace(0, n - 1, min(n_samples, n)).astype(int)
    acc = {k: [] for k in ("speed", "command", "goal_x", "goal_y", "wp0_x",
                           "wp0_y", "img_mean", "img_std", "n_pillars",
                           "n_lidar_pts")}
    for i in idx:
        s = ds[int(i)]
        acc["speed"].append(float(s["speed"][0]))
        acc["command"].append(int(s["command"]))
        acc["goal_x"].append(float(s["goal"][0]))
        acc["goal_y"].append(float(s["goal"][1]))
        wp = s["waypoints"]
        acc["wp0_x"].append(float(wp[0][0]))
        acc["wp0_y"].append(float(wp[0][1]))
        img = s["image"]
        acc["img_mean"].append(float(img.mean()))
        acc["img_std"].append(float(img.std()))
        acc["n_pillars"].append(int(s["pillar_mask"].shape[0]))
        acc["n_lidar_pts"].append(int(s["pillar_mask"].sum()))
    return acc


# --------------------------------------------------------------------- trace
def trace_stats(path):
    acc = {k: [] for k in ("speed", "command", "goal_x", "goal_y", "wp0_x",
                           "wp0_y", "img_mean", "img_std", "n_pillars",
                           "n_lidar_pts")}
    extra = {k: [] for k in ("gate", "steer", "throttle", "brake")}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            acc["speed"].append(r["speed"])
            acc["command"].append(r["command"])
            acc["goal_x"].append(r["goal_x"])
            acc["goal_y"].append(r["goal_y"])
            acc["wp0_x"].append(r["wp"][0][0])
            acc["wp0_y"].append(r["wp"][0][1])
            acc["img_mean"].append(r["img_mean"])
            acc["img_std"].append(r["img_std"])
            acc["n_pillars"].append(r["n_pillars"])
            acc["n_lidar_pts"].append(r["n_lidar_pts"])
            for k in extra:
                extra[k].append(r.get(k))
    return acc, extra


# -------------------------------------------------------------------- report
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--config", default="configs/egca.yaml")
    ap.add_argument("--samples", type=int, default=3000)
    a = ap.parse_args()

    tr, extra = trace_stats(a.trace)
    ds = dataset_stats(a.dataset, a.config, a.samples)

    print("input distribution: training set vs. what the agent built\n")
    print(f"{'channel':<13}{'train p5':>10}{'train med':>11}{'train p95':>11}"
          f"{'   ':>3}{'eval p5':>10}{'eval med':>10}{'eval p95':>10}  flag")
    print("-" * 92)
    flagged = []
    for k in ds:
        d, e = summarise(ds[k]), summarise(tr[k])
        if d is None or e is None:
            continue
        out_of_range = not (d["p5"] <= e["med"] <= d["p95"])
        mark = "  <-- OUT OF TRAINING RANGE" if out_of_range else ""
        if out_of_range:
            flagged.append(k)
        print(f"{k:<13}{d['p5']:>10.3f}{d['med']:>11.3f}{d['p95']:>11.3f}"
              f"{'   ':>3}{e['p5']:>10.3f}{e['med']:>10.3f}{e['p95']:>10.3f}"
              f"{mark}")

    print("\nagent-only signals (no dataset counterpart)")
    for k, v in extra.items():
        s = summarise(v)
        if s:
            print(f"  {k:<10} p5={s['p5']:.3f} med={s['med']:.3f} "
                  f"p95={s['p95']:.3f}  min={s['min']:.3f} max={s['max']:.3f}")

    # command is categorical: a histogram says more than percentiles
    print("\ncommand histogram (0 left, 1 right, 2 straight, 3 lanefollow)")
    for name, src in (("train", ds["command"]), ("eval", tr["command"])):
        tot = len(src) or 1
        h = {c: sum(1 for x in src if int(x) == c) for c in (0, 1, 2, 3)}
        print(f"  {name:<6}" + "  ".join(
            f"{c}:{100.0 * h[c] / tot:5.1f}%" for c in (0, 1, 2, 3)))

    print()
    if flagged:
        print(f"VERDICT: {len(flagged)} channel(s) outside the training "
              f"distribution -> {', '.join(flagged)}")
        print("These are input bugs, not model weakness. Fix before retraining.")
    else:
        print("VERDICT: every channel sits inside its training range.")
        print("The inputs are sound, so the closed-loop gap is the policy "
              "itself (capacity, data coverage or epochs) -- not plumbing.")


if __name__ == "__main__":
    main()
