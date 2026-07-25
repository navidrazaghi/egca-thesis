"""Visual and statistical inspection of a collected dataset.

This is the single most useful sanity check before spending GPU-days on
training: a sign error in the ego frame, a broken BEV rasterization or an
all-zero depth map are invisible in the loss curves but obvious here.

For each sampled frame it writes one panel image containing
  * the stitched RGB strip,
  * the privileged BEV segmentation (colour-coded) with the waypoint labels
    drawn on it -- in the correct frame the waypoints must run *forward* (up)
    and bend towards the side the vehicle is actually turning,
  * the privileged inverse-depth map,
and it prints dataset-level statistics (label length, command balance, class
balance, fraction of stopped / perturbed frames).

Usage:
    python -m egca.data.inspect --root dataset --out inspect_out --samples 12
"""
import argparse
import json
import os
import random

import cv2
import numpy as np

# class ids written by collect_data.py
CLASS_COLORS = np.array([
    [30, 30, 30],        # 0 free
    [90, 90, 90],        # 1 road
    [255, 255, 255],     # 2 lane marking
    [0, 128, 255],       # 3 vehicle
    [255, 64, 64],       # 4 pedestrian
    [255, 200, 0],       # 5 static obstacle
], dtype=np.uint8)

BEV_GRID = 128
BEV_X = (0.0, 32.0)
BEV_Y = (-16.0, 16.0)


def bev_px(x, y):
    """Ego metres -> (row, col) of the BEV raster (same mapping as collection)."""
    col = (y - BEV_Y[0]) / (BEV_Y[1] - BEV_Y[0]) * BEV_GRID
    row = BEV_GRID - (x - BEV_X[0]) / (BEV_X[1] - BEV_X[0]) * BEV_GRID
    return int(round(row)), int(round(col))


def frames_of(route):
    ldir = os.path.join(route, "labels")
    if not os.path.isdir(ldir):
        return []
    return sorted(f[:-5] for f in os.listdir(ldir) if f.endswith(".json"))


def route_dirs(root):
    out = []
    for base, dirs, _ in os.walk(root):
        if "measurements" in dirs:
            out.append(base)
    return sorted(out)


def panel(route, fid):
    rgb = cv2.imread(os.path.join(route, "rgb", fid + ".jpg"))
    seg = cv2.imread(os.path.join(route, "bev_seg", fid + ".png"),
                     cv2.IMREAD_GRAYSCALE)
    depth = np.load(os.path.join(route, "depth", fid + ".npy")).astype(np.float32)
    with open(os.path.join(route, "measurements", fid + ".json")) as f:
        meas = json.load(f)
    with open(os.path.join(route, "labels", fid + ".json")) as f:
        wps = json.load(f)["waypoints"]

    seg_rgb = CLASS_COLORS[np.clip(seg, 0, len(CLASS_COLORS) - 1)][:, :, ::-1].copy()
    # ego vehicle marker at the bottom centre
    cv2.drawMarker(seg_rgb, (BEV_GRID // 2, BEV_GRID - 2), (0, 255, 0),
                   cv2.MARKER_TRIANGLE_UP, 8, 1)
    prev = bev_px(0.0, 0.0)
    for wx, wy in wps:
        p = bev_px(wx, wy)
        cv2.line(seg_rgb, (prev[1], prev[0]), (p[1], p[0]), (0, 255, 255), 1)
        cv2.circle(seg_rgb, (p[1], p[0]), 1, (0, 255, 255), -1)
        prev = p
    seg_rgb = cv2.resize(seg_rgb, (352, 352), interpolation=cv2.INTER_NEAREST)

    # the depth target is aligned with the RGB strip, so show it at the same width
    dep_rgb = cv2.applyColorMap((255 * np.clip(depth, 0, 1)).astype(np.uint8),
                                cv2.COLORMAP_MAGMA)
    dep_rgb = cv2.resize(dep_rgb, (704, 160), interpolation=cv2.INTER_NEAREST)

    left = np.zeros((352, 704, 3), dtype=np.uint8)
    left[:160] = rgb
    left[176:336] = dep_rgb
    canvas = np.concatenate([left, seg_rgb], axis=1)

    cmd = ["LEFT", "RIGHT", "STRAIGHT", "FOLLOW"][int(meas["command"])]
    txt = (f"{os.path.basename(route)}/{fid}  v={meas['speed']:.1f} "
           f"tgt={meas['target_speed']:.1f}  cmd={cmd}  "
           f"goal=({meas['goal_x']:.1f},{meas['goal_y']:.1f})  "
           f"red={int(meas['red_light'])} stop={int(meas['stop_sign'])} "
           f"lead={meas['lead_distance']:.0f} noise={int(meas['noise'])}")
    cv2.putText(canvas, txt, (6, 348), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--out", default="inspect_out")
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    routes = route_dirs(args.root)
    pool = [(r, f) for r in routes for f in frames_of(r)]
    if not pool:
        print(f"no labelled frames under {args.root} "
              "(run: python -m egca.data.build_labels --root <root>)")
        return
    for i, (r, f) in enumerate(random.sample(pool, min(args.samples, len(pool)))):
        cv2.imwrite(os.path.join(args.out, f"frame_{i:02d}.png"), panel(r, f))
    print(f"wrote {min(args.samples, len(pool))} panels to {args.out}/")

    # ---------------- dataset statistics
    cmd_hist = np.zeros(4, dtype=np.int64)
    cls_hist = np.zeros(len(CLASS_COLORS), dtype=np.int64)
    lens, speeds = [], []
    red = stop = noise = lidar_pts = 0
    depth_zero = 0
    n = 0
    # why was the expert slow?  split the frames by the reason it had to brake
    groups = {"red light": [], "stop sign": [], "lead vehicle": [],
              "lead walker": [], "free": []}
    leads, ahead = [], []
    for r, f in pool:
        with open(os.path.join(r, "measurements", f + ".json")) as fh:
            m = json.load(fh)
        with open(os.path.join(r, "labels", f + ".json")) as fh:
            wps = json.load(fh)["waypoints"]
        cmd_hist[int(m["command"])] += 1
        speeds.append(m["speed"])
        red += int(m["red_light"])
        stop += int(m["stop_sign"])
        noise += int(m["noise"])
        ahead.append(int(m.get("n_ahead", 0)))
        if m["red_light"]:
            groups["red light"].append(m["speed"])
        elif m["stop_sign"]:
            groups["stop sign"].append(m["speed"])
        elif m["lead_distance"] > 0:
            key = "lead walker" if m.get("lead_is_walker") else "lead vehicle"
            groups[key].append(m["speed"])
            leads.append(m["lead_distance"])
        else:
            groups["free"].append(m["speed"])
        lens.append(float(np.hypot(wps[-1][0], wps[-1][1])))
        n += 1
        if n % max(1, len(pool) // 20) == 0 or n <= 200:
            seg = cv2.imread(os.path.join(r, "bev_seg", f + ".png"),
                             cv2.IMREAD_GRAYSCALE)
            cls_hist += np.bincount(seg.ravel(), minlength=len(CLASS_COLORS))
            d = np.load(os.path.join(r, "depth", f + ".npy"))
            depth_zero += int(np.all(d == 0))
            lidar_pts += len(np.load(os.path.join(r, "lidar", f + ".npy")))

    print("\n================ dataset statistics ================")
    print(f"routes {len(routes)}   labelled frames {n}")
    print(f"speed        mean {np.mean(speeds):5.2f}  max {np.max(speeds):5.2f} m/s")
    print(f"|w_T| (2 s)  mean {np.mean(lens):5.2f}  p05 {np.percentile(lens,5):5.2f}"
          f"  p95 {np.percentile(lens,95):5.2f} m")
    print(f"commands     L {cmd_hist[0]}  R {cmd_hist[1]}  S {cmd_hist[2]}  "
          f"FOLLOW {cmd_hist[3]}   ({100*cmd_hist[3]/max(n,1):.0f}% lane-follow)")
    print(f"red light    {100*red/max(n,1):.1f}%   stop sign {100*stop/max(n,1):.1f}%"
          f"   perturbed {100*noise/max(n,1):.1f}%")
    print("\nwhy the expert was slow (share of frames / mean speed in each):")
    for name, vals in groups.items():
        if vals:
            print(f"  {name:14s} {100*len(vals)/max(n,1):5.1f}%   "
                  f"mean speed {np.mean(vals):4.2f} m/s   "
                  f"stopped {100*np.mean(np.array(vals) < 0.3):4.1f}%")
        else:
            print(f"  {name:14s}   0.0%")
    if leads:
        print(f"  lead distance when a blocker was seen: mean {np.mean(leads):.1f} m,"
              f" p10 {np.percentile(leads,10):.1f} m")
    if ahead:
        print(f"  vehicles loosely ahead (<20 m, |y|<5 m): "
              f"{100 * np.mean(np.array(ahead) > 0):.1f}% of frames, "
              f"mean {np.mean(ahead):.2f} per frame")
        print("  -> if this is clearly above 0% while 'lead vehicle' is 0%, the "
              "hazard corridor is too narrow; if both are ~0%, there simply was "
              "no traffic in front and the traffic density should be raised.")
    tot = max(cls_hist.sum(), 1)
    names = ["free", "road", "lane", "vehicle", "pedestrian", "static"]
    print("BEV classes  " + "  ".join(
        f"{nm} {100*c/tot:.2f}%" for nm, c in zip(names, cls_hist)))
    print(f"depth maps all-zero: {depth_zero} (must be 0)")
    print(f"lidar points per frame (sampled): {lidar_pts / max(1, n // max(1, len(pool)//20) + min(n,200)):.0f}")
    print("""
What to check
  * |w_T| mean should be ~2 s x cruise speed (about 8-12 m at 6 m/s); near 0
    means the expert never moved.
  * lane-follow should be roughly 70-90%; if it is ~100% the routes contain no
    junctions and the conditional policy cannot be learned.
  * "road" should be a large fraction of the BEV and "vehicle" non-zero with
    traffic enabled; all-"free" means the rasterization is broken.
  * depth all-zero must be 0, otherwise the depth cameras were not recorded.
  * red light > 0% is required -- it is the evidence that the expert stops.""")


if __name__ == "__main__":
    main()
