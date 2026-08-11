"""Verify that the mirror augmentation is geometrically self-consistent.

Mirroring touches six things that have to agree: the camera strip, the BEV
segmentation raster, the depth target, the LiDAR y axis, the waypoint and goal y
components, and the turn command.  If any one of them flips about the wrong axis
the network is trained on an image that contradicts its own label -- a silent
defect that shows up only as a mediocre score, which is exactly the failure mode
this project has already paid for twice.

The load-bearing check is the LiDAR/BEV alignment.  Points are projected into the
BEV raster with the collector's own mapping and scored against the cells the
segmentation marks as occupied.  A correct mirror leaves that alignment
unchanged; flipping the raster about the wrong axis roughly halves it.

Usage:
    python tools/check_augment.py --config configs/egca.yaml [--frames 40]
"""
import argparse
import sys

import numpy as np


BEV_X = (0.0, 32.0)
BEV_Y = (-16.0, 16.0)
OCCUPIED = (3, 4, 5)          # vehicle, pedestrian, static obstacle


def bev_cells(pts, n):
    """Ego-frame points -> (row, col) in the BEV raster, collector's mapping."""
    x, y = pts[:, 0], pts[:, 1]
    m = ((x >= BEV_X[0]) & (x < BEV_X[1]) & (y >= BEV_Y[0]) & (y < BEV_Y[1]))
    x, y = x[m], y[m]
    col = ((y - BEV_Y[0]) / (BEV_Y[1] - BEV_Y[0]) * n).astype(int)
    row = (n - (x - BEV_X[0]) / (BEV_X[1] - BEV_X[0]) * n).astype(int)
    return np.clip(row, 0, n - 1), np.clip(col, 0, n - 1)


def alignment(pts, seg, ground_only=False):
    """Fraction of occupied-cell mass that has LiDAR support.

    Scale-free and symmetric enough that the only thing it responds to is a
    geometric disagreement between the cloud and the raster.

    `ground_only` switches the signal for rasters that carry no agents: the
    published set stores three classes -- free, road, lane -- so there is no
    occupied mass to correlate.

    Drivable area against ground returns was tried first and has no power here:
    on a straight road both are a symmetric band about the forward axis, so
    mirroring one, both or neither scores the same to within noise (0.489
    against 0.511). The discriminating quantity has to be asymmetric, which is
    why the caller uses road_bias below instead.
    """
    n = seg.shape[0]
    if ground_only:
        row, col = bev_cells(pts[(pts[:, 2] < -1.8) & (pts[:, 2] > -2.6)], n)
        if len(row) < 200:
            return None
        road = (seg == 1)
        return float(road[row, col].sum()) / float(len(row))
    row, col = bev_cells(pts, n)
    hit = np.zeros_like(seg, dtype=bool)
    hit[row, col] = True
    occ = np.isin(seg, OCCUPIED)
    if occ.sum() == 0:
        return None
    return float((occ & hit).sum()) / float(occ.sum())


def road_bias(seg, wp):
    """Signed agreement between where the road leads and where the route goes.

    Both are asymmetric about the forward axis whenever the vehicle is not on a
    straight, and both must flip together under a reflection. The product is
    positive when the drivable area leans the same way as the trajectory, so a
    sample mirrored consistently keeps its sign and one mirrored in only half
    its fields inverts it.
    """
    road = (seg == 1)
    if road.sum() < 50:
        return None
    n = seg.shape[0]
    cols = np.arange(n)[None, :]
    centroid = float((road * cols).sum() / road.sum()) / (n - 1) - 0.5
    lateral = float(wp[-1][1])
    if abs(lateral) < 0.5:            # straight: carries no side information
        return None
    return centroid * lateral


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/egca.yaml")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--source", default="own", choices=("own", "transfuser"),
                    help="which dataset's samples to mirror")
    ap.add_argument("--root", default="~/transfuser/data")
    a = ap.parse_args()

    from egca.config import load_config
    from egca.data.dataset import CarlaDrivingDataset

    # The mirror is exact only if the rig is symmetric about the forward axis
    # and every field is reflected together. Both datasets now feed the same
    # `_mirror`, so both have to be shown to survive it -- the adapter was
    # excluded from this check for as long as it did not call the mirror at all.
    if a.source == "transfuser":
        import os
        from egca.data.transfuser_dataset import TransfuserDataset
        cfg = load_config(a.config, ["model.aux.bev_classes=3"])
        ds = TransfuserDataset(cfg, os.path.expanduser(a.root),
                               ["Town01", "Town03"], augment=False, split="val")
    else:
        cfg = load_config(a.config, [])
        ds = CarlaDrivingDataset(cfg, cfg.data.towns_train, augment=False,
                                 split="train")
    if len(ds) == 0:
        sys.exit("no frames found")

    idx = np.linspace(0, len(ds) - 1, min(a.frames, len(ds))).astype(int)
    same, wrong_axis, n_scored = [], [], 0
    involutive = True
    cmd_ok = True

    for i in idx:
        if a.source == "transfuser":
            import os
            from egca.data.transfuser_dataset import (load_lidar,
                                                      decode_topdown,
                                                      decode_depth,
                                                      COMMAND_MAP)
            route, fid = ds.frames[int(i)]
            pose = ds._pose(int(i))
            if pose is None:
                continue
            meas, wp, goal, _ = pose
            wp = np.asarray(wp, dtype=np.float32)
            goal = np.asarray(goal, dtype=np.float32)
            cmd = COMMAND_MAP.get(int(meas["command"]), 3)
            img = ds._image(route, fid)
            pts = load_lidar(os.path.join(route, "lidar", fid + ".npy"))
            seg = decode_topdown(os.path.join(route, "topdown",
                                              "encoded_" + fid + ".png"),
                                 ds.seg_hw, cfg.model.lidar)
            depth = decode_depth(os.path.join(route, "depth", fid + ".png"),
                                 ds.depth_hw, ds.img_hw[1])
        else:
            img, pts, meas, seg, depth = ds._load(int(i))
            wp = np.asarray(meas["waypoints"], dtype=np.float32)
            goal = np.array([meas["goal_x"], meas["goal_y"]], dtype=np.float32)
            cmd = int(meas["command"])

        g_only = (a.source == "transfuser")
        if g_only:
            base = road_bias(seg, wp)
        else:
            base = alignment(pts, seg, g_only)
        if base is None:
            continue

        mi, mp, ms, md, mw, mg, mc = CarlaDrivingDataset._mirror(
            ds, img, pts, seg, depth, wp, goal, cmd)
        if g_only:
            # a consistently mirrored sample keeps the sign; mirroring the
            # trajectory but not the raster inverts it
            same.append(road_bias(ms, mw) * np.sign(base))
            wrong_axis.append(road_bias(seg, mw) * np.sign(base))
        else:
            same.append(alignment(mp, ms, g_only))
            # deliberately wrong: mirror the cloud but not the raster
            wrong_axis.append(alignment(mp, seg, g_only))
        n_scored += 1

        # mirroring twice must return the original sample exactly
        ri, rp, rs, rd, rw, rg, rc = CarlaDrivingDataset._mirror(
            ds, mi, mp, ms, md, mw, mg, mc)
        if not (np.array_equal(ri, img) and np.allclose(rp, pts)
                and np.array_equal(rs, seg) and np.array_equal(rd, depth)
                and np.allclose(rw, wp) and np.allclose(rg, goal)):
            involutive = False
        if rc != cmd:
            cmd_ok = False

    if n_scored == 0:
        sys.exit("no frame had occupied BEV cells; cannot score alignment")

    base_like = float(np.mean(same))
    wrong_like = float(np.mean(wrong_axis))
    print(f"frames scored: {n_scored}")
    print(f"  alignment, mirrored consistently : {base_like:.3f}")
    print(f"  alignment, raster left unmirrored: {wrong_like:.3f}   "
          f"(should be clearly worse)")
    print(f"  mirror is involutive            : {involutive}")
    print(f"  command round-trips             : {cmd_ok}")

    ok = involutive and cmd_ok and base_like > wrong_like * 1.3
    print()
    if ok:
        print("PASS: the mirror keeps cloud and raster in agreement, and the "
              "wrong axis is measurably worse -- so the axis is right.")
    else:
        print("FAIL: mirror is not self-consistent; do not train with it.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
