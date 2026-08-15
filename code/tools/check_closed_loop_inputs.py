"""Does the network see the same world in CARLA that it saw in training?

The open-loop error is 0.051 m and the closed-loop driving score is 12. Those two
numbers cannot both describe a healthy system, and the same contradiction has
already been traced four times to the agent building an input the network was
never trained on -- a cloud displaced 2.6 m, a panorama torn into segments, a
goal at a third of its training distance, a map with no relation to the frame.

Each of those was found by comparing a *distribution*, not by reading code. This
does the comparison in one place: every quantity the agent feeds the network
during a real route, against the same quantity over the training set. A channel
that agrees rules itself out; a channel that does not is the next bug.

Run the agent with debug_dir set first, then:
    PYTHONPATH=. python tools/check_closed_loop_inputs.py \
        --trace ~/logs/trace_route0 --root ~/transfuser/data
"""
import argparse
import glob
import json
import os
import sys

import numpy as np


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def summarise(name, live, train, unit=""):
    """Print both distributions side by side and flag a real separation."""
    l_med, t_med = np.median(live), np.median(train)
    l_lo, l_hi = pct(live, 5), pct(live, 95)
    t_lo, t_hi = pct(train, 5), pct(train, 95)
    # overlap of the two 5-95% bands, as a fraction of the training band
    lo, hi = max(l_lo, t_lo), min(l_hi, t_hi)
    span = t_hi - t_lo
    overlap = max(0.0, hi - lo) / span if span > 1e-9 else 0.0
    flag = "" if overlap > 0.5 else "   <-- DIVERGES"
    print("  %-16s live %8.3f [%7.3f, %7.3f]   train %8.3f [%7.3f, %7.3f]"
          "   overlap %.2f%s"
          % (name + unit, l_med, l_lo, l_hi, t_med, t_lo, t_hi, overlap, flag))
    return overlap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, help="directory of JSONL traces")
    ap.add_argument("--config", default="configs/egca.yaml")
    ap.add_argument("--root", default="~/transfuser/data")
    ap.add_argument("--frames", type=int, default=1500)
    a = ap.parse_args()

    from egca.config import load_config
    from egca.data.transfuser_dataset import TransfuserDataset

    recs = []
    for f in sorted(glob.glob(os.path.join(os.path.expanduser(a.trace),
                                           "*.jsonl"))):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    if not recs:
        sys.exit("no trace records under %s" % a.trace)
    print("trace frames: %d" % len(recs))

    cfg = load_config(a.config, ["model.aux.bev_classes=3"])
    ds = TransfuserDataset(cfg, os.path.expanduser(a.root), ["Town05"],
                           augment=False, split="val")
    stride = max(len(ds) // a.frames, 1)
    idx = list(range(0, len(ds), stride))[:a.frames]

    tr_speed, tr_goal, tr_pil, tr_img_m, tr_img_s, tr_cmd = [], [], [], [], [], []
    for i in idx:
        s = ds[i]
        tr_speed.append(float(s["speed"][0]))
        tr_goal.append(float(np.hypot(*s["goal"].numpy())))
        tr_pil.append(int(s["pillar_feats"].shape[0]))
        tr_img_m.append(float(s["image"].mean()))
        tr_img_s.append(float(s["image"].std()))
        tr_cmd.append(int(s["command"]))

    live_speed = np.array([r["speed"] for r in recs])
    live_goal = np.array([np.hypot(r["goal_x"], r["goal_y"]) for r in recs])
    live_pil = np.array([r["n_pillars"] for r in recs])
    live_img_m = np.array([r["img_mean"] for r in recs])
    live_img_s = np.array([r["img_std"] for r in recs])

    print("\ninput channels, closed loop against training set")
    ov = []
    ov.append(summarise("speed", live_speed, np.array(tr_speed), " m/s"))
    ov.append(summarise("goal distance", live_goal, np.array(tr_goal), " m"))
    ov.append(summarise("pillars", live_pil, np.array(tr_pil)))
    ov.append(summarise("image mean", live_img_m, np.array(tr_img_m)))
    ov.append(summarise("image std", live_img_s, np.array(tr_img_s)))

    print("\ncommand mix")
    names = {0: "left", 1: "right", 2: "straight", 3: "follow"}
    lc = np.array([r["command"] for r in recs])
    tc = np.array(tr_cmd)
    for c in sorted(set(tc.tolist()) | set(lc.tolist())):
        print("  %-9s live %5.1f%%   train %5.1f%%"
              % (names.get(c, c), 100 * (lc == c).mean(), 100 * (tc == c).mean()))

    print("\nwhat the network then does")
    wp = np.array([r["wp"] for r in recs])            # frames x T x 2
    reach = wp[:, -1, 0]
    lateral = wp[:, -1, 1]
    print("  predicted forward reach of the last waypoint: "
          "median %.2f m, 5-95%% [%.2f, %.2f]"
          % (np.median(reach), pct(reach, 5), pct(reach, 95)))
    print("  predicted lateral offset: median %.2f m, 5-95%% [%.2f, %.2f]"
          % (np.median(lateral), pct(lateral, 5), pct(lateral, 95)))
    print("  implied target speed: median %.2f m/s"
          % (np.median(np.linalg.norm(wp[:, 1] - wp[:, 0], axis=-1))
             / cfg.model.decoder.wp_dt))
    thr = np.array([r["throttle"] for r in recs])
    brk = np.array([r["brake"] for r in recs])
    print("  throttle: mean %.2f, zero on %.0f%% of frames"
          % (thr.mean(), 100 * (thr < 0.01).mean()))
    print("  brake engaged on %.0f%% of frames" % (100 * (brk > 0.5).mean()))
    print("  gate: median %.3f" % np.median([r["gate"] for r in recs]))

    print()
    if min(ov) > 0.5:
        print("every input channel overlaps its training distribution: the "
              "inputs are not the explanation, and the failure is in the "
              "policy or in accumulated drift")
    else:
        print("at least one input channel diverges -- that channel is the "
              "next thing to fix, exactly as the previous four were")


if __name__ == "__main__":
    main()
