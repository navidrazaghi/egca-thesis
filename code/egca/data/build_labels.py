"""Reconstruct waypoint labels from the recorded ego poses (Sec. 5-1).

`collect_data.py` logs the ego pose of every recorded frame; this pass turns
those poses into the supervision target of the decoder: for frame *i*, the
positions of the ego vehicle at the next T recorded frames, expressed in the
ego frame of frame *i* (x forward, y left).

Because frames are recorded at 2 Hz, the next T = 4 frames are exactly the
0.5 / 1.0 / 1.5 / 2.0 s horizon of Eq. (4.12).  Deriving the label from what the
vehicle *actually did*, instead of extrapolating the current speed along the
route, has three consequences that matter:

  * braking is supervised correctly -- in front of a red light the future
    positions bunch up and finally coincide with the current one, so the policy
    learns to stop rather than to keep a constant speed;
  * after an injected steering perturbation the label is the expert's real
    recovery manoeuvre, which is precisely the signal needed against covariate
    shift [14];
  * the label lives in the same frame as the sensors by construction, removing
    a whole class of sign/frame errors.

Usage:
    python -m egca.data.build_labels --root dataset --horizon 4
"""
import argparse
import json
import math
import os


def route_dirs(root):
    """Every directory that contains a `measurements` folder."""
    out = []
    for base, dirs, _ in os.walk(root):
        if "measurements" in dirs:
            out.append(base)
    return sorted(out)


def load_poses(route):
    """Frame ids, poses and navigation commands, ordered by frame id."""
    mdir = os.path.join(route, "measurements")
    fids = sorted(f[:-5] for f in os.listdir(mdir) if f.endswith(".json"))
    poses, cmds = [], []
    for fid in fids:
        with open(os.path.join(mdir, fid + ".json")) as f:
            m = json.load(f)
        poses.append((m["x"], m["y"], m["yaw"]))
        cmds.append(int(m.get("command", 3)))
    return fids, poses, cmds


def to_ego(x, y, ref):
    """World (x, y) -> (forward, left) in the ego frame of pose `ref`."""
    rx, ry, ryaw = ref
    yaw = math.radians(ryaw)
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = x - rx, y - ry
    return c * dx + s * dy, -(-s * dx + c * dy)


STOP_THRESHOLD = 0.5      # 2 s path below this counts as standing still (m)


def build_route(route, horizon, max_step=25.0, keep_edges=3):
    """Write `labels/*.json` for one route; returns (written, skipped, stats).

    Standstill resampling (Table 5-1): in a long stop only the beginning and the
    end carry information -- the decision to brake and the moment of moving off.
    The frames in the middle are identical to each other and, left in, would make
    ~40% of the dataset say "stay where you are"; a policy trained on that
    becomes timid and loses route completion.  `keep_edges` frames are kept at
    each end of every standstill and the rest are dropped by simply not writing a
    label (the loader takes `labels/` as the authoritative frame list).
    """
    fids, poses, cmds = load_poses(route)
    ldir = os.path.join(route, "labels")
    os.makedirs(ldir, exist_ok=True)
    for old in os.listdir(ldir):             # rebuild from scratch, idempotent
        if old.endswith(".json"):
            os.remove(os.path.join(ldir, old))

    # ---- pass 1: compute the label of every frame that has a future
    cand = {}
    skipped = 0
    for i in range(len(fids)):
        if i + horizon >= len(fids):
            skipped += 1                     # no future available at the tail
            continue
        wps = [to_ego(poses[i + k][0], poses[i + k][1], poses[i])
               for k in range(1, horizon + 1)]
        # A jump larger than max_step between consecutive frames means the
        # vehicle was teleported or the log is discontinuous: drop the frame
        # instead of teaching the policy an impossible motion.
        steps = [math.hypot(wps[0][0], wps[0][1])] + [
            math.hypot(wps[k][0] - wps[k - 1][0], wps[k][1] - wps[k - 1][1])
            for k in range(1, horizon)]
        if max(steps) > max_step:
            skipped += 1
            continue
        cand[i] = (wps, sum(steps))

    # ---- pass 2: thin out the interior of every standstill run.
    #
    # An earlier version exempted frames carrying a turn command, on the theory
    # that vehicles stand still at junctions and blind thinning would erase the
    # conditional signal.  That exemption backfired: waiting at a red light
    # before a turn produced a hundred identical frames all carrying the turn
    # command, none of which could be dropped, and 30% of the dataset became a
    # stationary car.  The conditional signal is in fact carried by the *moving*
    # frames that approach and cross the junction, which are never thinned here.
    idxs = sorted(cand)
    drop = set()
    run = []
    for i in idxs + [None]:
        is_stop = i is not None and cand[i][1] < STOP_THRESHOLD
        if is_stop:
            run.append(i)
            continue
        if len(run) > 2 * keep_edges:
            drop.update(run[keep_edges:len(run) - keep_edges])
        run = []

    written = stopped = 0
    total_len = 0.0
    for i in idxs:
        if i in drop:
            continue
        wps, path = cand[i]
        with open(os.path.join(ldir, fids[i] + ".json"), "w") as f:
            json.dump({"waypoints": [[float(a), float(b)] for a, b in wps]}, f)
        written += 1
        total_len += path
        if path < STOP_THRESHOLD:
            stopped += 1
    return written, skipped + len(drop), {
        "mean_path_m": total_len / max(written, 1),
        "stopped_frac": stopped / max(written, 1),
        "dropped_standstill": len(drop)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--keep-edges", type=int, default=3,
                    help="frames kept at each end of a standstill; use a large "
                         "value to disable standstill resampling")
    args = ap.parse_args()
    routes = route_dirs(args.root)
    if not routes:
        print(f"no routes found under {args.root}")
        return
    tot_w = tot_s = 0
    path_sum = stop_sum = 0.0
    for r in routes:
        w, s, st = build_route(r, args.horizon, keep_edges=args.keep_edges)
        tot_w += w
        tot_s += s
        path_sum += st["mean_path_m"] * w
        stop_sum += st["stopped_frac"] * w
        print(f"{os.path.relpath(r, args.root):46s} "
              f"labelled {w:6d}  dropped {s:4d} "
              f"(standstill {st['dropped_standstill']:4d})  "
              f"mean 2 s path {st['mean_path_m']:5.2f} m  "
              f"stopped {100 * st['stopped_frac']:4.1f}%")
    print("-" * 100)
    print(f"TOTAL labelled {tot_w}  dropped {tot_s}  "
          f"mean 2 s path {path_sum / max(tot_w, 1):.2f} m  "
          f"stopped frames {100 * stop_sum / max(tot_w, 1):.1f}%")
    print("\nSanity guide: with a 6 m/s cruise speed the mean 2 s path should be "
          "roughly 6-11 m; a value near 0 means the expert never moved, a value "
          "above ~25 m means the recording rate is not 2 Hz.")


if __name__ == "__main__":
    main()
