"""What does the validation waypoint error mean if the model does nothing clever?

A regression error in metres is uninterpretable on its own.  0.078 m sounds
small, but the only question that matters is whether it beats a predictor that
never looks at the scene -- and this project has already been burned once by a
number that looked healthy: the model trained on our own collection reached
0.133 m open-loop and then scored DS 2.7 in closed loop.

Two scene-blind baselines are scored here on exactly the frames the training
run validates on:

  standstill          predict that the vehicle stays where it is
  constant velocity   extrapolate the current speedometer reading straight ahead

The second is the honest one.  Beating standstill only proves the car moves;
beating constant velocity is what shows the network anticipates turns, stops
and speed changes rather than integrating the speedometer.

The metric is F.l1_loss, which is what training reports: the mean absolute
error over both coordinates, not the L1 distance per point.  Summing the two
coordinates instead doubles every number and makes the model look twice as good
as it is against a baseline computed the other way.

Usage:
    PYTHONPATH=. python tools/check_wp_baselines.py --root ~/transfuser/data
"""
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/egca.yaml")
    ap.add_argument("--root", default="~/transfuser/data")
    ap.add_argument("--towns", nargs="*", default=["Town05"])
    ap.add_argument("--frames", type=int, default=2000)
    ap.add_argument("--model-error", type=float, default=None,
                    help="the run's reported val wp error, printed for scale")
    a = ap.parse_args()

    from egca.config import load_config
    from egca.data.transfuser_dataset import TransfuserDataset

    cfg = load_config(a.config, ["model.aux.bev_classes=3"])
    ds = TransfuserDataset(cfg, os.path.expanduser(a.root), a.towns,
                           augment=False, split="val")
    dt = cfg.model.decoder.wp_dt
    horizon = cfg.model.decoder.horizon

    # Evenly spaced, so the sample spans every route of the town rather than
    # the first few, which are all one weather and one traffic density.
    stride = max(len(ds) // a.frames, 1)
    idx = range(0, len(ds), stride)

    still, const, speeds = [], [], []
    for i in idx:
        pose = ds._pose(i)
        if pose is None:
            continue
        _, wp, _, speed = pose
        still.append(np.abs(wp).mean())
        pred = np.stack([[speed * dt * (t + 1), 0.0] for t in range(horizon)])
        const.append(np.abs(wp - pred).mean())
        speeds.append(speed)

    print(f"towns {a.towns}, {len(still)} frames, "
          f"horizon {horizon} x {dt}s = {horizon * dt}s")
    print(f"  mean speed             {np.mean(speeds):5.2f} m/s")
    print(f"  standstill             {np.mean(still):5.3f} m")
    print(f"  constant velocity      {np.mean(const):5.3f} m")
    if a.model_error is not None:
        print(f"  model                  {a.model_error:5.3f} m   "
              f"({np.mean(const) / a.model_error:.1f}x better than "
              f"constant velocity)")


if __name__ == "__main__":
    main()
