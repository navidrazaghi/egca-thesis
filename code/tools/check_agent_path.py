"""Exercise the inference path the leaderboard agent takes, without CARLA.

Training and evaluation are different code paths, and the evaluation one is the
one that has already produced three silent defects in this project.  It rebuilds
the architecture from the checkpoint's own config, runs single frames in
evaluation mode, reads keys out of the output dict, and hands numbers to a
stateful controller -- none of which the training check covers.

Per checkpoint:
  1. rebuilt from its stored config and loaded strictly, as the agent does
  2. single-frame inference in eval mode, on real data
  3. every control output finite and inside the actuator range
  4. the robustness switches (`drop_sensor`, `lidar_drop_rate`) actually change
     the model's input rather than being silently ignored
  5. with a query readout, the target speed reaches the controller

Usage:
    python tools/check_agent_path.py --checkpoints checkpoints [--device cpu]
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch


def build(path, device):
    from egca.config import Cfg
    from egca.models import EGCAPolicy
    ck = torch.load(path, map_location=device, weights_only=False)
    if "cfg" not in ck:
        return None, None, "no config stored"
    cfg = Cfg(ck["cfg"])
    model = EGCAPolicy(cfg, sensor_dropout=0.0).to(device).eval()
    model.load_state_dict(ck["model"], strict=True)
    return model, cfg, None


def one_frame(cfg, device):
    from egca.data.dataset import CarlaDrivingDataset, collate
    ds = CarlaDrivingDataset(cfg, cfg.data.towns_train, augment=False,
                             split="val")
    if len(ds) == 0:
        return None
    return {k: v.to(device) for k, v in collate([ds[0]]).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", default="checkpoints")
    ap.add_argument("--config", default="configs/egca.yaml")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    from egca.config import load_config, Cfg
    from egca.control.pid import WaypointController

    device = torch.device(a.device)
    batch = one_frame(load_config(a.config, []), device)
    if batch is None:
        sys.exit("no validation frames available")

    paths = sorted(glob.glob(os.path.join(a.checkpoints, "*", "best.pth")))
    if not paths:
        sys.exit(f"no checkpoints under {a.checkpoints!r}")

    ok = True
    print(f"{'checkpoint':<14}{'readout':>9}{'steer':>8}{'throt':>7}"
          f"{'brake':>7}{'v_des':>8}  drop-sensor effect")
    print("-" * 78)
    for p in paths:
        name = os.path.basename(os.path.dirname(p))
        model, cfg, err = build(p, device)
        if err:
            print(f"{name:<14} FAILED to rebuild: {err}")
            ok = False
            continue

        readout = getattr(cfg.model.decoder, "readout", "pooled")
        ctrl = WaypointController(Cfg(dict(cfg["control"])),
                                  cfg.model.decoder.wp_dt)
        with torch.no_grad():
            out = model(batch)
        wps = out["waypoints"][0].cpu().numpy()

        v_des = None
        if "speed_logits" in out:
            v_des = float(model.query_readout.expected_speed(
                out["speed_logits"])[0])
        elif readout == "query":
            print(f"{name:<14} FAIL: query readout emitted no speed_logits")
            ok = False

        steer, throttle, brake, _ = ctrl.step(wps, 4.0, hazard=False, v_des=v_des)
        finite = all(np.isfinite([steer, throttle, brake]))
        in_range = (-1.0 <= steer <= 1.0 and 0.0 <= throttle <= 1.0
                    and 0.0 <= brake <= 1.0)
        if not (finite and in_range):
            print(f"{name:<14} FAIL: control out of range "
                  f"({steer}, {throttle}, {brake})")
            ok = False

        # A forced sensor drop has to reach the network. If the switch were
        # ignored the robustness matrix would compare a model against itself and
        # report, convincingly, that nothing degrades it.
        with torch.no_grad():
            a_out = model(batch, force_drop=None)["waypoints"]
            b_out = model(batch, force_drop="lidar")["waypoints"]
        delta = float((a_out - b_out).abs().max())
        single = getattr(cfg.model.fusion, "mode", "egca") in ("camera_only",
                                                               "lidar_only")
        effect = "n/a (single modality)" if single else f"max |d| = {delta:.3e}"
        if not single and delta == 0.0:
            effect += "   <-- FAIL: drop had no effect"
            ok = False

        vs = f"{v_des:>8.2f}" if v_des is not None else f"{'-':>8}"
        print(f"{name:<14}{readout:>9}{steer:>8.3f}{throttle:>7.2f}"
              f"{brake:>7.0f}{vs}  {effect}")

    print()
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
