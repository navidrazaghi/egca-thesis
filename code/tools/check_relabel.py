"""Did relabelling actually fix the frames it was meant to fix?

The stored TransFuser waypoint block agrees with the pose the car reached to
0.080 m while it is moving and disagrees by a factor of 5.6 on the frames where
it pulls away from a standstill. `data.tf_relabel` rebuilds the target from the
realized poses. This checks the result rather than assuming it, and it checks
both directions, because a relabelling that quietly changed the moving frames
would be trading a known failure for an unmeasured one.

Three things have to hold:
  1. on pull-away frames the target now shows the travel that really happened;
  2. on moving frames it barely moves, since the stored block was already right
     there to 0.080 m;
  3. no frame is served with a padded, standstill target.

Usage:
    PYTHONPATH=. python tools/check_relabel.py --root ~/transfuser/data
"""
import argparse
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/egca.yaml")
    ap.add_argument("--root", default="~/transfuser/data")
    ap.add_argument("--town", default="Town05")
    ap.add_argument("--stride", type=int, default=7)
    a = ap.parse_args()

    from egca.config import load_config
    from egca.data.transfuser_dataset import TransfuserDataset

    root = os.path.expanduser(a.root)
    old = TransfuserDataset(load_config(a.config, ["data.tf_relabel=false"]),
                            root, [a.town], augment=False, split="val")
    new = TransfuserDataset(load_config(a.config, ["data.tf_relabel=true"]),
                            root, [a.town], augment=False, split="val")
    assert old.frames == new.frames, "the two configs must index the same frames"

    rows = {"moving": [], "pull-away": [], "held": []}
    refused = 0
    for i in range(0, len(old.frames), a.stride):
        po = old._pose(i)
        pn = new._pose(i)
        if po is None:
            continue
        if pn is None:
            refused += 1
            continue
        speed = po[3]
        reach_o = float(np.linalg.norm(po[1][-1]))
        reach_n = float(np.linalg.norm(pn[1][-1]))
        if abs(speed) >= 0.5:
            key = "moving"
        else:
            key = "pull-away" if reach_n > 1.0 else "held"
        rows[key].append((reach_o, reach_n))

    print("town %s, every %dth frame\n" % (a.town, a.stride))
    print("  %-11s %7s   %14s  %14s" % ("", "n", "stored 2 s", "relabelled 2 s"))
    for key in ("moving", "pull-away", "held"):
        v = np.array(rows[key]) if rows[key] else None
        if v is None:
            print("  %-11s %7d   %14s  %14s" % (key, 0, "-", "-"))
            continue
        print("  %-11s %7d   %12.2f m  %12.2f m"
              % (key, len(v), v[:, 0].mean(), v[:, 1].mean()))
    print("\n  frames refused for having no future in their route: %d" % refused)

    # train.py caps validation by replacing dataset.frames after construction.
    # Anything that resolves a frame by its position in that list silently
    # points at the wrong future once it is subsampled, and the result looks
    # like a number rather than like an error: validation sat at a constant
    # 159.56 m for eleven hours while the training curve fell normally. So the
    # subsampling is reproduced here rather than trusted.
    sub = TransfuserDataset(load_config(a.config, ["data.tf_relabel=true"]),
                            root, [a.town], augment=False, split="val")
    stride = len(sub.frames) / 800.0
    keep = [sub.frames[int(i * stride)] for i in range(800)]
    orig = {f: new._pose(i) for i, f in enumerate(new.frames)
            if f in set(keep)}
    sub.frames = keep
    moved = bad = 0
    for i, f in enumerate(sub.frames):
        before, after = orig.get(f), sub._pose(i)
        if before is None or after is None:
            continue
        moved += 1
        if not np.allclose(before[1], after[1], atol=1e-6):
            bad += 1
    print("\n  after subsampling, targets that changed: %d of %d" % (bad, moved))

    mv = np.array(rows["moving"]) if rows["moving"] else np.zeros((0, 2))
    pa = np.array(rows["pull-away"]) if rows["pull-away"] else np.zeros((0, 2))
    ok = True
    if len(pa) and pa[:, 1].mean() < 2.0 * max(pa[:, 0].mean(), 1e-6):
        print("\nFAIL: pull-away targets did not grow -- relabelling changed "
              "nothing where it was supposed to")
        ok = False
    if len(mv) and abs(mv[:, 1].mean() - mv[:, 0].mean()) > 0.5:
        print("\nFAIL: moving targets moved by more than 0.5 m -- the stored "
              "block was already right there, so this is a new error, not a fix")
        ok = False
    if bad:
        print("\nFAIL: %d targets changed when the frame list was subsampled -- "
              "frames are being resolved by position, and validation will score "
              "against the future of some other frame" % bad)
        ok = False
    if ok:
        print("\nPASS: the standstill frames carry the travel that happened and "
              "the moving frames are unchanged")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
