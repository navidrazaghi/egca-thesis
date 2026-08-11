"""Does the agent hand the network the same thing training did?

Every input the model reads at inference is built twice by this project: once by
the dataset adapter for training, once by the agent inside CARLA. Nothing checks
that the two agree, and three separate mismatches survived every other test in
this repository because a model trained on a consistently wrong input simply
learns the wrong input and reports an excellent validation error:

  * the LiDAR clouds were displaced 2.6 m, because the sensor's forward offset
    was subtracted where it should be added
  * the camera strip was stitched from cameras at +-55 deg with the overlaps
    cropped, tearing two 13 deg holes out of a panorama the training images
    render seamlessly from +-60 deg
  * the navigation goal was a fixed 8 m look-ahead against a training
    distribution averaging 23 m and reaching 60 m

All three are invisible open-loop and fatal closed-loop. This compares the two
paths field by field on the same underlying frames, so the next one is caught
before a training run rather than after two.

Usage:
    PYTHONPATH=. python tools/check_train_deploy_parity.py --root ~/transfuser/data
"""
import argparse
import glob
import json
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/egca.yaml")
    ap.add_argument("--root", default="~/transfuser/data")
    ap.add_argument("--frames", type=int, default=40)
    a = ap.parse_args()

    from egca.config import load_config
    from egca.data.transfuser_dataset import (TransfuserDataset, load_lidar,
                                              to_ego)
    from egca.carla_sim.sensors import (CAM_W, CAM_H, CAM_FOV, FULL_W,
                                        STITCH_W, CAMERAS, stitch_cameras)

    cfg = load_config(a.config, ["model.aux.bev_classes=3"])
    root = os.path.expanduser(a.root)
    ok = True

    # ---- 1. camera geometry -------------------------------------------------
    # The published strip is three CAM_W-wide renders concatenated. Splitting it
    # back into three tiles and running them through the agent's stitcher must
    # reproduce the adapter's crop exactly -- byte for byte, since both are just
    # indexing into the same pixels.
    print("1. camera")
    yaws = [y for _, y in CAMERAS]
    print(f"     agent yaws {yaws}, fov {CAM_FOV}, tile {CAM_W}x{CAM_H}")
    if sorted(yaws) != [-60.0, 0.0, 60.0]:
        print("   FAIL: the published data was rendered at -60/0/+60; any other"
              " spacing changes which angle each column shows")
        ok = False
    if CAM_FOV * 3 != 180:
        print("   FAIL: three cameras must tile 180 deg without gap or overlap")
        ok = False

    import cv2
    ds = TransfuserDataset(cfg, root, ["Town01", "Town03"], augment=False,
                           split="val")
    idx = np.linspace(0, len(ds) - 1, a.frames).astype(int)
    worst = 0
    for i in idx[:8]:
        route, fid = ds.frames[int(i)]
        strip = cv2.imread(os.path.join(route, "rgb", fid + ".png"))
        if strip is None or strip.shape[1] != FULL_W:
            print(f"   FAIL: stored strip is {None if strip is None else strip.shape}"
                  f", expected width {FULL_W}")
            ok = False
            break
        tiles = {"cam_left": strip[:, :CAM_W],
                 "cam_front": strip[:, CAM_W:2 * CAM_W],
                 "cam_right": strip[:, 2 * CAM_W:]}
        agent_side = stitch_cameras(tiles)
        x0 = (FULL_W - STITCH_W) // 2
        train_side = strip[:, x0:x0 + STITCH_W]
        worst = max(worst, int(np.abs(agent_side.astype(int)
                                      - train_side.astype(int)).max()))
    print(f"     agent stitch vs training crop: max pixel difference {worst}")
    if worst != 0:
        print("   FAIL: the agent is not showing the network the same pixels")
        ok = False

    # ---- 2. LiDAR frame -----------------------------------------------------
    # The vehicle blocks its own returns, so the hole in the cloud is a landmark
    # at a place both paths must agree on: the ego origin.
    print("\n2. lidar")
    los, his = [], []
    for i in idx:
        route, fid = ds.frames[int(i)]
        p = load_lidar(os.path.join(route, "lidar", fid + ".npy"))
        near = p[(np.abs(p[:, 1]) < 0.8) & (p[:, 2] < -0.5)
                 & (np.abs(p[:, 0]) < 12)]
        if len(near) < 40:
            continue
        xs = np.sort(near[:, 0])
        gaps = np.diff(xs)
        j = int(np.argmax(gaps))
        if gaps[j] > 1.0:
            los.append(xs[j])
            his.append(xs[j + 1])
    if los:
        centre = (float(np.median(los)) + float(np.median(his))) / 2
        print(f"     ego shadow centred at {centre:+.2f} m "
              f"(the agent mounts its LiDAR at the ego origin, so 0.00)")
        if abs(centre) > 1.0:
            print("   FAIL: training clouds are translated relative to the "
                  "agent's")
            ok = False
    else:
        print("     no ego shadow found; translation not checked")

    # ---- 3. navigation goal -------------------------------------------------
    # The agent takes the next sparse route vertex; the data stores the same
    # quantity. Distances have to occupy the same range or the network is
    # reading a differently scaled instruction.
    print("\n3. navigation goal")
    d = []
    for i in idx:
        route, fid = ds.frames[int(i)]
        with open(os.path.join(route, "measurements", fid + ".json")) as f:
            m = json.load(f)
        g = to_ego(np.array([m["x_command"], m["y_command"]]),
                   np.array([m["x"], m["y"]]), float(m["theta"]))
        if np.isfinite(g).all():
            d.append(float(np.hypot(*g)))
    d = np.array(d)
    print(f"     training goal distance: mean {d.mean():.1f} m, "
          f"p10 {np.percentile(d, 10):.1f}, p90 {np.percentile(d, 90):.1f}")
    src = open("egca/carla_sim/leaderboard_agent.py").read()
    if "GOAL_DISTANCE" in src.split("_goal_and_command")[1][:900]:
        print("   FAIL: the agent still uses a fixed look-ahead for the goal")
        ok = False
    else:
        print("     agent takes the next sparse route vertex, as the data does")

    # ---- 4. scalar conventions ---------------------------------------------
    print("\n4. scalars")
    sp = []
    for i in idx:
        route, fid = ds.frames[int(i)]
        with open(os.path.join(route, "measurements", fid + ".json")) as f:
            sp.append(float(json.load(f)["speed"]))
    print(f"     training speed: mean {np.mean(sp):.2f} m/s, "
          f"max {np.max(sp):.2f} -- the agent reads input_data['speed'] in m/s")
    if np.max(sp) > 40:
        print("   FAIL: these look like km/h, not m/s")
        ok = False

    print()
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
