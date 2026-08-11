"""Find the transform that puts their top-down raster in our BEV frame.

Their renderer writes a 500x500 raster at 5 px/m -- 100 x 100 m with the ego at
the centre (PIXELS_AHEAD_VEHICLE is 0 when data_generation is set). Our BEV grid
is 32 m forward by 32 m wide with the ego on the bottom edge. Resizing one onto
the other, which is what the adapter did, trains the auxiliary head against a
map that has no geometric relationship to the frame the model reasons in.

The correct crop cannot be read off the code with any confidence: the renderer
adds pi/2 to the orientation, applies a shift its own comment calls weird, and
carries two FIXMEs about being inconsistent with the global path. So it is
measured instead, against something outside both: the LiDAR. Ground returns mark
where drivable surface is, so the crop that is right is the one whose road class
agrees with them, and it should win by a margin no near-miss can explain.

Usage:
    PYTHONPATH=. python tools/check_bev_alignment.py --root ~/transfuser/data
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np


PPM = 5          # their pixels per metre
RASTER = 500     # their raster is 500 x 500


def road_mask(path):
    """Their encoded PNG -> boolean drivable area, full 500 x 500."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    blue = img[:, :, 0]
    return ((blue & (1 << 7)) > 0)


def lidar_rasters(points, cfg_lidar, hw):
    """Two rasters from the same cloud: where road must be, and where it cannot.

    Ground returns alone do not discriminate -- kerbs, verges and pavement all
    lie flat, so every candidate crop scores about the same. Tall returns are
    the sharp signal: a wall or a building is never drivable, so a crop that
    puts them on road is wrong however well its road covers the ground.
    """
    x0, x1 = cfg_lidar.x_range
    y0, y1 = cfg_lidar.y_range
    h, w = hw
    def raster(sel):
        q = points[sel]
        q = q[(q[:, 0] >= x0) & (q[:, 0] < x1) & (q[:, 1] >= y0) & (q[:, 1] < y1)]
        if len(q) == 0:
            return np.zeros((h, w), dtype=bool)
        rr = ((x1 - q[:, 0]) / (x1 - x0) * h).astype(int).clip(0, h - 1)
        cc = ((q[:, 1] - y0) / (y1 - y0) * w).astype(int).clip(0, w - 1)
        out = np.zeros((h, w), dtype=bool)
        out[rr, cc] = True
        return out
    ground = raster((points[:, 2] < -1.8) & (points[:, 2] > -2.6))
    tall = raster(points[:, 2] > 0.0)
    if ground.sum() < 200:
        return None
    return ground, tall


def crop_candidate(mask, cfg_lidar, hw, rot, flip_y, ego_at):
    """Cut our grid's footprint out of their raster under one hypothesis."""
    m = np.rot90(mask, rot)
    if flip_y:
        m = m[:, ::-1]
    x0, x1 = cfg_lidar.x_range
    y0, y1 = cfg_lidar.y_range
    cy, cx = RASTER // 2, RASTER // 2
    # rows run from the far edge back to the ego, columns left to right
    top = cy - int(x1 * PPM) if ego_at == "bottom" else cy - int((x1 - x0) / 2 * PPM)
    left = cx + int(y0 * PPM)
    h_px, w_px = int((x1 - x0) * PPM), int((y1 - y0) * PPM)
    if top < 0 or left < 0 or top + h_px > RASTER or left + w_px > RASTER:
        return None
    sub = m[top:top + h_px, left:left + w_px].astype(np.uint8)
    h, w = hw
    return cv2.resize(sub, (w, h), interpolation=cv2.INTER_NEAREST) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/egca.yaml")
    ap.add_argument("--root", default="~/transfuser/data")
    ap.add_argument("--frames", type=int, default=60)
    a = ap.parse_args()

    from egca.config import load_config
    from egca.data.transfuser_dataset import TransfuserDataset, load_lidar

    cfg = load_config(a.config, ["model.aux.bev_classes=3"])
    ds = TransfuserDataset(cfg, os.path.expanduser(a.root), ["Town01", "Town03"],
                           augment=False, split="val")
    hw = tuple(cfg.data.seg_size)
    idx = np.linspace(0, len(ds) - 1, a.frames).astype(int)

    cands = [(rot, flip, ego)
             for rot in (0, 1, 2, 3) for flip in (False, True)
             for ego in ("bottom", "centre")]
    scores = {c: [] for c in cands}
    naive = []

    for i in idx:
        route, fid = ds.frames[int(i)]
        tp = os.path.join(route, "topdown", "encoded_" + fid + ".png")
        lp = os.path.join(route, "lidar", fid + ".npy")
        if not (os.path.exists(tp) and os.path.exists(lp)):
            continue
        rasters = lidar_rasters(load_lidar(lp), cfg.model.lidar, hw)
        if rasters is None:
            continue
        ground, tall = rasters
        full = road_mask(tp)
        for c in cands:
            sub = crop_candidate(full, cfg.model.lidar, hw, *c)
            if sub is None:
                continue
            hit = np.logical_and(sub, ground).sum() / max(ground.sum(), 1)
            miss = (np.logical_and(sub, tall).sum() / max(tall.sum(), 1)
                    if tall.sum() else 0.0)
            scores[c].append(hit - miss)
        whole = cv2.resize(full.astype(np.uint8), (hw[1], hw[0]),
                           interpolation=cv2.INTER_NEAREST) > 0
        naive.append(np.logical_and(whole, ground).sum() / max(ground.sum(), 1)
                     - (np.logical_and(whole, tall).sum() / max(tall.sum(), 1)
                        if tall.sum() else 0.0))

    print(f"frames scored: {len(naive)}")
    print("\nfraction of LiDAR ground returns that land on drivable area:")
    ranked = sorted(((float(np.mean(v)), k) for k, v in scores.items() if v),
                    reverse=True)
    for s, (rot, flip, ego) in ranked[:6]:
        print(f"   rot {rot * 90:>3} deg  flip_y {str(flip):<5}  ego {ego:<6}  "
              f"{s:.3f}")
    print(f"\n   whole raster resized (what the adapter does today): "
          f"{np.mean(naive):.3f}")

    best, second = ranked[0][0], ranked[1][0]
    print(f"\nbest {best:.3f} vs next {second:.3f}")
    ok = best > 0.35 and best - second > 0.05
    if not ok:
        print("FAIL: no candidate wins decisively -- the geometry is not "
              "established and the auxiliary target cannot be trusted")
    else:
        print("PASS: one arrangement agrees with the LiDAR and the rest do not")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
