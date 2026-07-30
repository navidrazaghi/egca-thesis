"""Check the camera-to-BEV geometry against the LiDAR, not against itself.

A round trip through my own derivation would pass even if a sign were wrong,
because the inverse carries the same mistake.  So this compares against
something external: the depth map and the point cloud of the same frame are two
measurements of one scene, so lifting the depth map with the transform must put
points where the LiDAR already sees surfaces.  A flipped axis produces a
mirrored world that still looks like a plausible BEV map on its own and only
reveals itself here.

Three deliberately corrupted variants are scored alongside the real one.  If the
real geometry does not beat all of them clearly, it is not trustworthy.

Usage:
    python tools/check_view_transform.py --config configs/egca.yaml [--frames 30]
"""
import argparse
import glob
import math
import os
import sys

import numpy as np


def lift(depth_m, dirs, offset, flip_y=False, flip_yaw=False, flip_z=False):
    """Depth map + per-pixel ray directions -> N x 3 points in the LiDAR frame."""
    d = dirs.copy()
    if flip_y:
        d[..., 1] *= -1.0
    if flip_z:
        d[..., 2] *= -1.0
    if flip_yaw:                      # mirror the whole strip left-right
        d = d[:, ::-1].copy()
        d[..., 1] *= -1.0
    pts = d * depth_m[..., None] + np.asarray(offset, dtype=np.float32)
    return pts.reshape(-1, 3)


def occupancy(pts, x_range, y_range, z_range, n):
    """Rasterise points to an n x n BEV occupancy grid.

    The height crop is not decoration: without it the grid ignores z entirely,
    and a transform with an inverted vertical axis scores exactly the same as
    the correct one -- which is what the first run of this check reported.
    Cropping to the range `pillarize` uses makes the test sensitive to the axis
    it is supposed to be testing.
    """
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    m = ((x >= x_range[0]) & (x < x_range[1])
         & (y >= y_range[0]) & (y < y_range[1])
         & (z >= z_range[0]) & (z < z_range[1]))
    if not m.any():
        return np.zeros((n, n), dtype=bool), 0
    ix = ((x[m] - x_range[0]) / (x_range[1] - x_range[0]) * n).astype(int)
    iy = ((y[m] - y_range[0]) / (y_range[1] - y_range[0]) * n).astype(int)
    g = np.zeros((n, n), dtype=bool)
    g[np.clip(ix, 0, n - 1), np.clip(iy, 0, n - 1)] = True
    return g, int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/egca.yaml")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--dataset", default="dataset")
    a = ap.parse_args()

    import torch
    from egca.config import load_config
    from egca.models.view_transform import ray_directions, CAM_OFFSET

    cfg = load_config(a.config, [])
    xr = tuple(cfg.model.lidar.x_range)
    yr = tuple(cfg.model.lidar.y_range)
    zr = tuple(cfg.model.lidar.z_range)
    N = 64                                   # the grid the LiDAR branch emits

    frames = sorted(glob.glob(os.path.join(a.dataset, "*", "*", "depth", "*.npy")))
    if not frames:
        sys.exit(f"no depth frames under {a.dataset!r}")
    step = max(1, len(frames) // a.frames)
    frames = frames[::step][:a.frames]

    variants = {"correct": {}, "y flipped": {"flip_y": True},
                "yaw mirrored": {"flip_yaw": True}, "z flipped": {"flip_z": True}}
    scores = {k: [] for k in variants}
    cells = {k: [] for k in variants}
    n_used = 0

    for dpath in frames:
        base = os.path.dirname(os.path.dirname(dpath))
        fid = os.path.basename(dpath)[:-4]
        lpath = os.path.join(base, "lidar", fid + ".npy")
        if not os.path.exists(lpath):
            continue
        norm_inv = np.load(dpath).astype(np.float32)
        # invert (1/d - 1/far) / (1/near - 1/far) back to metres
        near, far = 1.0, 100.0
        inv = norm_inv * (1.0 / near - 1.0 / far) + 1.0 / far
        depth_m = 1.0 / np.clip(inv, 1e-6, None)

        h, w = depth_m.shape
        dirs = ray_directions(h, w, torch.device("cpu")).numpy()

        lidar = np.load(lpath).astype(np.float32)
        lid_grid, n_lid = occupancy(lidar[:, :3], xr, yr, zr, N)
        if n_lid < 200:                       # too little evidence to judge
            continue

        for name, kw in variants.items():
            pts = lift(depth_m, dirs, CAM_OFFSET, **kw)
            cam_grid, _ = occupancy(pts, xr, yr, zr, N)
            inter = np.logical_and(cam_grid, lid_grid).sum()
            union = np.logical_or(cam_grid, lid_grid).sum()
            # Intersection over union, not over the camera's own footprint: a
            # variant that throws almost everything outside the height crop
            # leaves a handful of cells that happen to land well and scores
            # near 1 on the naive ratio, which is how the vertical flip first
            # came out ahead of the real geometry.
            scores[name].append(inter / union if union else 0.0)
            cells[name].append(int(cam_grid.sum()))
        n_used += 1

    if n_used == 0:
        sys.exit("no frame had both depth and LiDAR with enough points")

    print(f"frames scored: {n_used}")
    print("agreement with the point cloud, as BEV intersection over union\n")
    means = {k: float(np.mean(v)) for k, v in scores.items()}
    print(f"  {'variant':<14} {'IoU':>6} {'cells':>7}")
    for k in variants:
        mark = "   <- under test" if k == "correct" else ""
        print(f"  {k:<14} {means[k]:>6.3f} {np.mean(cells[k]):>7.0f}{mark}")

    best_wrong = max(v for k, v in means.items() if k != "correct")
    ok = means["correct"] > best_wrong * 1.25
    print()
    if ok:
        print("PASS: the real geometry agrees with the point cloud and every "
              "corrupted variant is clearly worse.")
    else:
        print("FAIL: a corrupted axis scores as well as the real one, so the "
              "transform is not pinned down. Do not train on it.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
