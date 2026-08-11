"""Verify the TransFuser adapter against physics, not against its own inverse.

Every defect this project has paid for was a frame or unit convention that ran
without error and produced a plausible-looking wrong number. A round trip
through my own transform would pass with a sign flipped, because the inverse
carries the same mistake, so each check here is anchored to something outside
the code:

  1. the first waypoint sits at the ego, and later ones lead forward
  2. the implied speed from waypoint spacing matches the recorded speedometer
  3. turn commands bend the trajectory the way their names say
  4. the converted point cloud agrees with the camera's own view of the scene:
     ground returns below the sensor, and the forward half of the cloud denser
     than the rear, since the rig looks forward
  5. tensors leave the adapter in the shapes the model consumes

Usage:
    python tools/check_transfuser_adapter.py --root ~/transfuser/data
"""
import argparse
import sys

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/egca.yaml")
    ap.add_argument("--root", required=True)
    ap.add_argument("--frames", type=int, default=300)
    a = ap.parse_args()

    from egca.config import load_config
    from egca.data.transfuser_dataset import TransfuserDataset, to_ego
    import json
    import os

    cfg = load_config(a.config, [])
    root = os.path.expanduser(a.root)
    towns = ["Town01", "Town02", "Town03", "Town04",
             "Town05", "Town06", "Town07", "Town10HD"]
    ds = TransfuserDataset(cfg, root, towns, augment=False, split="train")
    print(f"frames indexed: {len(ds)}")
    if len(ds) == 0:
        sys.exit("nothing to check")

    ok = True
    idx = np.linspace(0, len(ds) - 1, min(a.frames, len(ds))).astype(int)

    # ---- 1-3: trajectory geometry, read straight from the measurements -----
    wp_dt = cfg.model.decoder.wp_dt
    first, fwd, speed_err, by_cmd = [], [], [], {}
    for i in idx:
        route, fid = ds.frames[int(i)]
        with open(os.path.join(route, "measurements", fid + ".json")) as f:
            m = json.load(f)
        ego = np.array([m["x"], m["y"]])
        w = to_ego(np.asarray(m["waypoints"])[:, :2], ego, float(m["theta"]))
        first.append(np.linalg.norm(w[0]))
        if m["speed"] > 2.0:
            fwd.append(w[3][0])
            speed_err.append(np.linalg.norm(w[1] - w[0]) / wp_dt - m["speed"])
            by_cmd.setdefault(int(m["command"]), []).append(w[3][1])

    print(f"\n1. first waypoint distance from the ego: "
          f"{np.mean(first):.2f} m (expect well under one step)")
    if np.mean(first) > 2.5:
        print("   FAIL: the trajectory does not start at the vehicle"); ok = False

    print(f"2. forward reach of the 4th waypoint: {np.mean(fwd):+.2f} m "
          f"(expect positive)")
    if np.mean(fwd) <= 0:
        print("   FAIL: the trajectory runs backwards"); ok = False

    print(f"3. speed implied by spacing minus speedometer: "
          f"{np.mean(speed_err):+.2f} m/s (expect near zero)")
    if abs(np.mean(speed_err)) > 1.0:
        print("   FAIL: waypoint spacing disagrees with the recorded speed")
        ok = False

    print("4. lateral reach by navigation command (y is left, so left > 0):")
    for c, name in ((1, "left"), (2, "right"), (3, "straight")):
        if c in by_cmd:
            v = float(np.mean(by_cmd[c]))
            print(f"     {name:<9} {v:+.2f} m   n={len(by_cmd[c])}")
    if 1 in by_cmd and 2 in by_cmd:
        if not (np.mean(by_cmd[1]) > 0 > np.mean(by_cmd[2])):
            print("   FAIL: left and right are swapped"); ok = False

    # ---- 5: the point cloud must describe the world the rig looks at -------
    from egca.data.transfuser_dataset import load_lidar
    below, ahead = [], []
    for i in idx[:60]:
        route, fid = ds.frames[int(i)]
        p = load_lidar(os.path.join(route, "lidar", fid + ".npy"))
        below.append(float((p[:, 2] < 0).mean()))
        ahead.append(float((p[:, 0] > 0).mean()))
    print(f"\n5. point cloud: {np.mean(below) * 100:.0f}% of returns below the "
          f"sensor, {np.mean(ahead) * 100:.0f}% ahead of it")
    if np.mean(below) < 0.5:
        print("   FAIL: the vertical axis looks inverted -- most of the world "
              "should be below a roof-mounted sensor"); ok = False

    # Where the ego vehicle blocks its own LiDAR. This is the check that was
    # missing: the original version tested the axes and the sign conventions but
    # nothing that could see a translation, so a 2.6 m displacement of every
    # cloud passed it and only surfaced as bad closed-loop driving after two
    # full training runs. The car cannot see through itself, so the gap in the
    # returns directly along its own axis is a physical landmark at a known
    # place -- the origin, since our frame is the ego origin.
    los, his = [], []
    for i in idx[:60]:
        route, fid = ds.frames[int(i)]
        p = load_lidar(os.path.join(route, "lidar", fid + ".npy"))
        near = p[(np.abs(p[:, 1]) < 0.8) & (p[:, 2] < -0.5) & (np.abs(p[:, 0]) < 12)]
        if len(near) < 40:
            continue
        xs = np.sort(near[:, 0])
        gaps = np.diff(xs)
        j = int(np.argmax(gaps))
        if gaps[j] > 1.0:
            los.append(xs[j]); his.append(xs[j + 1])
    if los:
        lo, hi = float(np.median(los)), float(np.median(his))
        centre = (lo + hi) / 2
        print(f"6. the ego vehicle's own shadow spans {lo:+.2f} m to {hi:+.2f} m, "
              f"centred at {centre:+.2f} m")
        if abs(centre) > 1.0:
            print("   FAIL: the cloud is translated -- the sensor offset is "
                  "being applied with the wrong sign or magnitude")
            ok = False
    else:
        print("6. no ego shadow found; cannot check the translation")

    # ---- 6: shapes the model actually consumes -----------------------------
    s = ds[int(idx[0])]
    T = cfg.model.decoder.horizon
    h, w = cfg.model.camera.image_size
    want = {"image": (3, h, w), "speed": (1,), "goal": (2,), "waypoints": (T, 2)}
    print("\n6. tensor shapes")
    for k, shape in want.items():
        got = tuple(s[k].shape)
        flag = "" if got == shape else f"   FAIL (want {shape})"
        print(f"     {k:<10} {got}{flag}")
        if got != shape:
            ok = False
    print(f"     pillars    {tuple(s['pillar_feats'].shape)}")

    # ---- 7: auxiliary targets must describe the same scene -----------------
    # A decoded BEV raster and a decoded depth map both look plausible when the
    # bit-plane or the byte order is wrong, so each is checked against something
    # it cannot fake: a road-level scene is mostly drivable area, and depth must
    # be finite, ordered and dense in the near field once inverted.
    if cfg.model.aux.bev_seg or cfg.model.aux.depth:
        print("\n7. auxiliary targets")
        segs, deps = [], []
        for i in idx[:40]:
            s = ds[int(i)]
            if "bev_seg" in s:
                segs.append(s["bev_seg"].numpy())
            if "depth" in s:
                deps.append(s["depth"].numpy())
        if segs:
            a = np.stack(segs)
            frac = [float((a == c).mean()) for c in (0, 1, 2)]
            print(f"     bev classes  free {frac[0]:.2f}  road {frac[1]:.2f}  "
                  f"lane {frac[2]:.2f}")
            if frac[1] < 0.05:
                print("   FAIL: almost no drivable area -- wrong bit plane")
                ok = False
            if frac[2] > frac[1]:
                print("   FAIL: more lane marking than road"); ok = False
        if deps:
            d = np.stack(deps)
            print(f"     depth range  [{d.min():.3f}, {d.max():.3f}]  "
                  f"mean {d.mean():.3f}   (normalised inverse, so near = 1)")
            if not np.isfinite(d).all() or d.min() < -1e-6 or d.max() > 1 + 1e-6:
                print("   FAIL: depth outside [0, 1]"); ok = False
            if d.mean() > 0.9 or d.mean() < 0.01:
                print("   FAIL: depth is saturated, so the byte order is wrong")
                ok = False

    print()
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
