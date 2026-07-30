"""Turn the robustness matrix into the comparison the thesis actually claims.

The gate and the modality dropout are robustness mechanisms, so the number that
matters is not the driving score in any single condition but how much of it
survives when the sensors degrade.  A configuration that scores well in clear
weather and collapses in rain is worse, for the claim being made, than one that
starts lower and holds.

Reported per configuration:
  * the score in each condition
  * retention, the degraded score as a fraction of that configuration's own
    nominal score -- this is the robustness claim, and it is deliberately
    self-normalised so that a weaker model is not credited for having less to
    lose in absolute terms

Every cell is the same fixed route set with the same seed, so the comparison
between configurations is paired; the per-route standard error is reported
because the absolute spread between training seeds is far larger than the
differences being read here.

Usage:
    python tools/summarise_robustness.py --results results/robustness
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys

NAME = re.compile(r"^(?P<cfg>.+?)__(?P<cond>.+?)__r(?P<route>\d+)\.json$")
CONDITIONS = ["nominal", "lidardrop", "rain"]
LABEL = {"nominal": "nominal", "lidardrop": "lidar 50% lost",
         "rain": "hard rain, night"}


def load(path):
    """Return the per-route driving scores recorded in one checkpoint file."""
    try:
        with open(path) as f:
            recs = json.load(f)["_checkpoint"].get("records", [])
    except Exception:
        return []
    return [r["scores"]["score_composed"] for r in recs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/robustness")
    a = ap.parse_args()

    cells = {}
    for path in sorted(glob.glob(os.path.join(a.results, "*.json"))):
        m = NAME.match(os.path.basename(path))
        if not m:
            continue
        cells.setdefault((m["cfg"], m["cond"]), []).extend(load(path))

    if not cells:
        sys.exit(f"no result files under {a.results!r}")

    configs = sorted({c for c, _ in cells})
    width = max(len(c) for c in configs) + 2

    print("driving score by condition (mean +- standard error over routes)\n")
    head = f"{'config':<{width}}" + "".join(f"{LABEL[c]:>20}" for c in CONDITIONS)
    print(head)
    print("-" * len(head))
    for cfg in configs:
        row = f"{cfg:<{width}}"
        for cond in CONDITIONS:
            v = cells.get((cfg, cond), [])
            if not v:
                row += f"{'-':>20}"
            elif len(v) > 1:
                se = statistics.stdev(v) / (len(v) ** 0.5)
                row += f"{statistics.mean(v):>13.1f} ±{se:>4.1f}"
            else:
                row += f"{v[0]:>20.1f}"
        print(row)

    print("\nretention: degraded score / that configuration's own nominal\n")
    head = f"{'config':<{width}}" + "".join(
        f"{LABEL[c]:>20}" for c in CONDITIONS[1:])
    print(head)
    print("-" * len(head))
    for cfg in configs:
        base = cells.get((cfg, "nominal"), [])
        row = f"{cfg:<{width}}"
        if not base or statistics.mean(base) <= 0:
            print(row + f"{'no nominal score':>20}")
            continue
        b = statistics.mean(base)
        for cond in CONDITIONS[1:]:
            v = cells.get((cfg, cond), [])
            row += f"{100.0 * statistics.mean(v) / b:>19.0f}%" if v else f"{'-':>20}"
        print(row)

    n = {k: len(v) for k, v in cells.items()}
    print(f"\nroutes per cell: min {min(n.values())}, max {max(n.values())}")
    if len(set(n.values())) > 1:
        print("WARNING: cells hold different route counts, so the comparison "
              "is no longer paired. Finish the missing runs before reading it.")


if __name__ == "__main__":
    main()
