"""Does withholding the speedometer at deployment let a stopped car start again?

The policy trained with speed dropout still echoes the reading when it is there:
sweeping the input from 0 to 6 m/s swings the predicted target speed by 4.58 m/s,
a dependence of 0.76 against 0.85 without the treatment. So the training changed,
the deployment input did not, and the network takes the easy route it still has.

But the treatment left a second option that costs no retraining: feed the learned
"unknown" token instead of the real reading. The network then has no speedometer
to echo and has to decide from the image and the point cloud.

Whether that helps cannot be read off the reliance number, because that number is
measured on moving frames. The question here is the opposite one, and it is the
question the closed-loop failure actually turns on: on a frame where the car is
stopped, does the policy propose moving?

Stopped frames split into two kinds and they must not be pooled. Where the expert
stayed stopped -- a red light, a queue -- predicting a standstill is correct, and
a policy that drives off is worse, not better. Where the expert pulled away, a
standstill is the failure being chased. A treatment is only good news if it moves
the second group without moving the first.

Usage:
    PYTHONPATH=. python tools/check_absent_speed.py \
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
    ap.add_argument("--frames", type=int, default=4000,
                    help="frames to scan; stopped ones are a minority of them")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--stopped-below", type=float, default=0.5,
                    help="m/s under which the ego counts as stopped")
    ap.add_argument("--moves-above", type=float, default=1.0,
                    help="m/s of expert target speed that counts as pulling away")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from egca.config import Cfg
    from egca.data.transfuser_dataset import TransfuserDataset
    from egca.data.dataset import collate
    from egca.models import EGCAPolicy

    device = torch.device(a.device)
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    cfg = Cfg(ck["cfg"])
    model = EGCAPolicy(cfg, sensor_dropout=0.0).to(device).eval()
    model.load_state_dict(ck["model"], strict=True)
    drop = float(getattr(cfg.model.decoder, "speed_dropout", 0.0))
    print("checkpoint: %s (epoch %s, speed_dropout %.2f)"
          % (a.ckpt, ck.get("epoch"), drop))
    if drop == 0.0:
        # The token is a parameter that only ever receives gradient on samples
        # where the reading was withheld. Trained without dropout it is still at
        # its initial zero and means nothing, so the "absent" column below would
        # be measuring an untrained input rather than an alternative deployment.
        print("WARNING: this checkpoint was trained without speed dropout, so "
              "the absent-speed token was never trained; the absent column is "
              "not meaningful.")

    ds = TransfuserDataset(cfg, os.path.expanduser(a.root), ["Town05"],
                           augment=False, split="val")
    stride = max(len(ds) // a.frames, 1)
    idx = list(range(0, len(ds), stride))[:a.frames]
    dt = cfg.model.decoder.wp_dt

    # stopped frames, split by what the expert did next
    groups = {"expert pulls away": {"real": [], "absent": []},
              "expert stays stopped": {"real": [], "absent": []}}
    scanned = 0

    with torch.no_grad():
        for s in range(0, len(idx), a.batch):
            chunk = [ds[i] for i in idx[s:s + a.batch]]
            scanned += len(chunk)
            chunk = [c for c in chunk
                     if float(c["speed"][0]) < a.stopped_below]
            if not chunk:
                continue
            base = collate(chunk)
            # what the expert did: first step of the recorded trajectory
            gt = ((base["waypoints"][:, 1] - base["waypoints"][:, 0])
                  .norm(dim=-1) / dt).numpy()
            preds = {}
            for cond in ("real", "absent"):
                model.measure.force_absent = (cond == "absent")
                b = {k: t.to(device) for k, t in base.items()}
                wp = model(b)["waypoints"]
                preds[cond] = ((wp[:, 1] - wp[:, 0]).norm(dim=-1) / dt).cpu().numpy()
            model.measure.force_absent = False
            for j, g in enumerate(gt):
                key = ("expert pulls away" if g > a.moves_above
                       else "expert stays stopped")
                groups[key]["real"].append(float(preds["real"][j]))
                groups[key]["absent"].append(float(preds["absent"][j]))

    print("frames scanned: %d" % scanned)
    print("\npredicted target speed on frames where the ego is stopped\n")
    print("  %-22s %6s  %10s  %10s" % ("", "n", "real speed", "absent"))
    for key in ("expert pulls away", "expert stays stopped"):
        r, ab = groups[key]["real"], groups[key]["absent"]
        if not r:
            print("  %-22s %6d  %10s  %10s" % (key, 0, "-", "-"))
            continue
        print("  %-22s %6d  %8.2f m/s  %8.2f m/s"
              % (key, len(r), float(np.mean(r)), float(np.mean(ab))))

    go = groups["expert pulls away"]
    hold = groups["expert stays stopped"]
    print("\nThe treatment is worth deploying only if the first row rises and\n"
          "the second stays low: a policy that pulls away at a red light has\n"
          "traded a timeout for a collision, which scores worse, not better.")
    if go["real"]:
        gain = float(np.mean(go["absent"])) - float(np.mean(go["real"]))
        cost = float(np.mean(hold["absent"])) - float(np.mean(hold["real"])) \
            if hold["real"] else 0.0
        print("\n  pull-away frames  %+.2f m/s" % gain)
        print("  hold frames       %+.2f m/s" % cost)


if __name__ == "__main__":
    main()
