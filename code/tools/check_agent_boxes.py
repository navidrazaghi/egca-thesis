"""Establish how label_raw stores agent boxes, by measurement rather than by reading.

The published dataset ships an auxiliary BEV target with three classes -- free,
drivable, lane marking -- and no agents at all, while the boxes for every vehicle
sit unused in `label_raw`. The policy trained against that target collides with
other vehicles 3.92 times per route over the 36 Longest6 routes, which is the
dominant failure by a wide margin. Rasterising those boxes into the target is the
obvious repair, and it needs the storage convention to be exactly right: a box
written with two axes swapped supervises the network to expect traffic where
there is none, which is worse than no supervision.

The convention cannot be settled by reading the numbers. `extent` medians of
(1.55, 4.51, 2.01) match a car's full height, length and width in that order,
but `position` is contradictory -- the ego's own entry is (-1.3, 0.0, -2.5),
which puts forward in column 0, while column 1 over other vehicles is strictly
positive in 0.5..48, which no lateral offset can be.

So it is measured against something outside the annotation: the LiDAR. A box in
the right place contains tall returns, and a box with its axes transposed
contains the empty road beside the car. The correct hypothesis should win by a
margin no near-miss can explain.

Usage:
    PYTHONPATH=. python tools/check_agent_boxes.py --root ~/transfuser/data
"""
import argparse
import glob
import json
import os
import sys

import numpy as np


# (forward column, lateral column, sign applied to lateral)
LAYOUTS = {
    "pos=[fwd, lat]":        (0, 1, +1.0),
    "pos=[fwd, -lat]":       (0, 1, -1.0),
    "pos=[lat, fwd]":        (1, 0, +1.0),
    "pos=[-lat, fwd]":       (1, 0, -1.0),
}
# (length index, width index) into extent
EXTENTS = {
    "extent=[h, len, wid]": (1, 2),
    "extent=[len, wid, h]": (0, 1),
    "extent=[wid, len, h]": (1, 0),
}


def box_corners(fwd, lat, yaw, length, width):
    """Four corners of one box in the ego frame (x forward, y left)."""
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy = length / 2.0, width / 2.0
    local = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([fwd, lat])


def inside(points, corners):
    """Boolean mask of the points lying inside a convex quad.

    The cross products keep a constant sign inside a convex polygon, but which
    sign depends on whether the corners were listed clockwise, which in turn
    depends on the handedness of the frame. Testing for one of them silently
    returns nothing when the guess is wrong -- the first version of this scored
    every hypothesis at exactly 0.0000, which reads as "the annotation is
    unusable" and was really "the polygon test rejects everything".
    """
    cross = np.empty((4, len(points)))
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        edge = b - a
        cross[i] = ((points[:, 0] - a[0]) * edge[1]
                    - (points[:, 1] - a[1]) * edge[0])
    return np.all(cross <= 0, axis=0) | np.all(cross >= 0, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/transfuser/data")
    ap.add_argument("--town", default="Town05")
    ap.add_argument("--frames", type=int, default=150)
    a = ap.parse_args()

    from egca.data.transfuser_dataset import load_lidar

    root = os.path.expanduser(a.root)
    labels = sorted(glob.glob(os.path.join(
        root, "*", "*%s*" % a.town, "*", "label_raw", "*.json")))
    if not labels:
        print("no label_raw under %s for %s" % (root, a.town))
        sys.exit(1)
    idx = np.linspace(0, len(labels) - 1, a.frames).astype(int)

    # Score against the annotation's own `num_points`, which records how many
    # LiDAR returns the dataset's authors found in each box. Counting points
    # inside the box and correlating with their count tests the convention
    # directly: the right one agrees per box, and a transposed one cannot,
    # because it is counting a different patch of road.
    #
    # The first metric here was the fraction of tall returns falling inside any
    # box, and it could not separate the candidates -- 0.0262 against 0.0236 --
    # because most tall returns belong to walls and buildings, which dilutes
    # every hypothesis equally.
    mine = {(p, e): [] for p in LAYOUTS for e in EXTENTS}
    theirs = []
    n = 0
    for i in idx:
        lf = labels[int(i)]
        route = os.path.dirname(os.path.dirname(lf))
        fid = os.path.basename(lf)[:-5]
        lp = os.path.join(route, "lidar", fid + ".npy")
        if not os.path.exists(lp):
            continue
        pts = load_lidar(lp)                       # x fwd, y left, z up
        # z is measured from the sensor, which sits 2.5 m up, so the road is at
        # about -2.5 and the roof of a 1.5 m car at about -1.0. The first
        # version took -1.5..0.5, which is the air above the traffic, and
        # scored every hypothesis at zero.
        tall = pts[(pts[:, 2] > -2.3) & (pts[:, 2] < -0.8)]
        if len(tall) < 300:
            continue
        try:
            ents = json.load(open(lf))[1:]         # entry 0 is the ego itself
        except Exception:
            continue
        if not ents:
            continue
        # Only boxes the authors actually counted; -1 means "not computed".
        ents = [e for e in ents if int(e.get("num_points", -1)) >= 0]
        if not ents:
            continue
        n += 1
        theirs.extend(int(e["num_points"]) for e in ents)
        for pname, (fi, li, sgn) in LAYOUTS.items():
            for ename, (Li, Wi) in EXTENTS.items():
                for e in ents:
                    p, ex = e["position"], e["extent"]
                    cs = box_corners(p[fi], sgn * p[li], float(e["yaw"]),
                                     ex[Li], ex[Wi])
                    mine[(pname, ename)].append(int(inside(tall[:, :2],
                                                           cs).sum()))

    t = np.array(theirs, dtype=float)
    print("frames scored: %d   boxes with a stored count: %d" % (n, len(t)))
    print("their num_points: mean %.1f  p50 %.0f  max %.0f\n"
          % (t.mean(), np.median(t), t.max()))
    print("agreement with the annotation's own point count, per box:")
    ranked = []
    for k, v in mine.items():
        m = np.array(v, dtype=float)
        if len(m) != len(t) or m.std() == 0:
            continue
        r = float(np.corrcoef(m, t)[0, 1])
        ranked.append((r, k, m.mean()))
    ranked.sort(reverse=True)
    for r, (p, e), mu in ranked:
        print("   %-22s %-22s  r=%+.3f   mean %5.1f pts/box" % (p, e, r, mu))
    if not ranked:
        print("nothing comparable")
        sys.exit(1)
    best, second = ranked[0][0], ranked[1][0]
    print("\nbest r=%+.3f vs next r=%+.3f" % (best, second))
    ok = best > 0.5 and best - second > 0.2
    print("\n%s" % ("PASS: one convention reproduces their own counts"
                    if ok else
                    "FAIL: no convention stands out -- do not rasterise on a guess"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
