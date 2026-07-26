"""Integrity check for a collected dataset.

Run before training.  It looks for the failure modes that do not raise an error
anywhere else and would silently poison the training set:

  * **seams** -- two different routes written into the same directory.  A
    resumed collection that picks an already-used route index overwrites the
    first frames of an existing route, so the directory holds the tail of one
    drive glued to the head of another.  Every individual frame stays
    self-consistent, which is why nothing crashes; only the ego trajectory
    reveals it, as a jump of tens of metres between consecutive frames.
  * missing modality files for a recorded frame;
  * routes too short to be useful;
  * label counts that do not match the frame counts.

Usage:
    python -m tools.check_dataset --root dataset
    python -m tools.check_dataset --root dataset --quarantine dataset_bad
"""
import argparse
import json
import math
import os
import shutil

MODALITIES = ("rgb/{}.jpg", "lidar/{}.npy", "bev_seg/{}.png", "depth/{}.npy")
SEAM_JUMP_M = 25.0        # 2 Hz at 6 m/s is ~3 m; 25 m cannot be real motion
MIN_FRAMES = 30


def route_dirs(root):
    out = []
    for base, dirs, _ in os.walk(root):
        if "measurements" in dirs:
            out.append(base)
    return sorted(out)


def check_route(route):
    """Return (ok, list of problems, stats)."""
    mdir = os.path.join(route, "measurements")
    fids = sorted(f[:-5] for f in os.listdir(mdir) if f.endswith(".json"))
    problems = []
    if len(fids) < MIN_FRAMES:
        problems.append(f"only {len(fids)} frames")

    poses, missing = [], 0
    for fid in fids:
        with open(os.path.join(mdir, fid + ".json")) as f:
            m = json.load(f)
        poses.append((m["x"], m["y"]))
        for pat in MODALITIES:
            if not os.path.exists(os.path.join(route, pat.format(fid))):
                missing += 1
    if missing:
        problems.append(f"{missing} missing modality files")

    jumps = [math.dist(poses[i], poses[i + 1]) for i in range(len(poses) - 1)]
    seams = [i for i, d in enumerate(jumps) if d > SEAM_JUMP_M]
    if seams:
        problems.append(f"{len(seams)} trajectory seam(s) at frame(s) "
                        + ", ".join(str(s) for s in seams[:5]))

    ldir = os.path.join(route, "labels")
    n_labels = (len([f for f in os.listdir(ldir) if f.endswith(".json")])
                if os.path.isdir(ldir) else 0)

    stats = {"frames": len(fids), "labels": n_labels,
             "max_jump_m": max(jumps) if jumps else 0.0}
    return not problems, problems, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--quarantine", default=None,
                    help="move damaged routes here instead of only reporting "
                         "them; nothing is ever deleted")
    args = ap.parse_args()

    routes = route_dirs(args.root)
    if not routes:
        print(f"no routes under {args.root}")
        return
    bad, total_frames, total_labels = [], 0, 0
    for r in routes:
        ok, problems, st = check_route(r)
        total_frames += st["frames"]
        total_labels += st["labels"]
        if not ok:
            bad.append((r, problems))
            print(f"BAD  {os.path.relpath(r, args.root):40s} "
                  f"frames={st['frames']:4d} max_jump={st['max_jump_m']:6.1f} m"
                  f"  -> {'; '.join(problems)}")
    print(f"\n{len(routes)} routes, {total_frames} frames, {total_labels} labels")
    print(f"{len(bad)} damaged route(s)")

    if bad and args.quarantine:
        os.makedirs(args.quarantine, exist_ok=True)
        for r, _ in bad:
            dest = os.path.join(args.quarantine, os.path.basename(r))
            i = 1
            while os.path.exists(dest):
                dest = os.path.join(args.quarantine,
                                    f"{os.path.basename(r)}_{i}")
                i += 1
            shutil.move(r, dest)
            print(f"moved {r} -> {dest}")
        print(f"\n{len(bad)} route(s) moved to {args.quarantine} "
              "(review them there; nothing was deleted)")
    elif bad:
        print("re-run with --quarantine <dir> to move them out of the dataset")


if __name__ == "__main__":
    main()
