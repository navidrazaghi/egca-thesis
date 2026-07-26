"""Turn evaluation runs into the numbers of Chapter 5.

Every table and figure of the results chapter is produced here from the raw
result files, so no number in the thesis is ever typed by hand.  The script also
performs the statistical tests, because the honest statement about a 3-point
margin depends entirely on the test behind it.

Input: a directory of result files named

    {config}__{weather}__s{seed}.json

for example

    egca__ClearNoon__s0.json
    egca__HardRainNight__s0.json
    transfuser__ClearNoon__s0.json
    egca_no_gate__mixed__s0.json

Two file formats are accepted:
  * the CARLA leaderboard 1.0 checkpoint format (`_checkpoint.records`), which
    is what `leaderboard_evaluator.py` writes;
  * the list-of-routes format written by `egca.carla_sim.evaluate`.

Output: a markdown report and `numbers.json`, the latter consumed by the figure
scripts so that plots and tables can never drift apart.

Usage:
    python -m egca.eval.aggregate_results --results results --out report
"""
import argparse
import json
import math
import os
import re
from collections import defaultdict

# The infraction penalties of the official leaderboard, repeated here only so
# that the home-made evaluator and the official one score identically.
PENALTIES = {
    "collisions_pedestrian": 0.50,
    "collisions_vehicle": 0.60,
    "collisions_layout": 0.65,
    "red_light": 0.70,
    "stop_infraction": 0.80,
}

FNAME = re.compile(r"^(?P<config>.+?)__(?P<weather>.+?)__s(?P<seed>\d+)\.json$")


# --------------------------------------------------------------------- input
def parse_file(path):
    """Return a list of per-route dicts: town, route_id, RC, IS, DS, km, infractions."""
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "_checkpoint" in raw:
        return _parse_leaderboard(raw)
    if isinstance(raw, list):
        return _parse_own(raw)
    raise ValueError(f"unrecognised result format: {path}")


def _parse_leaderboard(raw):
    out = []
    for rec in raw["_checkpoint"]["records"]:
        s = rec["scores"]
        inf = {k: len(v) for k, v in rec.get("infractions", {}).items()}
        out.append({
            "town": rec.get("town_name", "?"),
            "route_id": rec.get("route_id", "?"),
            "RC": float(s["score_route"]),
            "IS": float(s["score_penalty"]),
            "DS": float(s["score_composed"]),
            "km": float(rec.get("meta", {}).get("route_length", 0.0)) / 1000.0,
            "infractions": inf,
        })
    return out


def _parse_own(raw):
    out = []
    for r in raw:
        inf = r.get("infractions", {})
        is_ = 1.0
        for name, count in inf.items():
            is_ *= PENALTIES.get(name, 1.0) ** count
        rc = float(r["completion"])
        out.append({
            "town": r.get("town", "?"),
            "route_id": r.get("start", "?"),
            "RC": rc, "IS": is_, "DS": rc * is_,
            "km": float(r.get("distance_km", 0.0)),
            "infractions": inf,
        })
    return out


def load_all(results_dir):
    """runs[(config, weather, seed)] = [route dicts]"""
    runs = {}
    for name in sorted(os.listdir(results_dir)):
        m = FNAME.match(name)
        if not m:
            continue
        key = (m["config"], m["weather"], int(m["seed"]))
        runs[key] = parse_file(os.path.join(results_dir, name))
    return runs


# ----------------------------------------------------------------- statistics
def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs):
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def aggregate_run(routes):
    """Benchmark-level RC / IS / DS for one run (Eqs. 5.1-5.3).

    DS is the mean of the per-route products, which is *not* the product of the
    two means; the two differ whenever completion and infractions correlate, and
    reporting them as if they were equal is a common error.
    """
    n = max(len(routes), 1)
    km = sum(r["km"] for r in routes)
    inf = sum(sum(r["infractions"].values()) for r in routes)
    return {
        "DS": mean(r["DS"] for r in routes),
        "RC": mean(r["RC"] for r in routes),
        "IS": mean(r["IS"] for r in routes),
        "n_routes": n,
        "infractions_per_10km": 10.0 * inf / km if km > 0 else float("nan"),
    }


def t_test_paired(a, b):
    """Paired t-test.  Returns (mean difference, t, dof, p) with p from a
    Student-t survival function; falls back to a critical-value verdict if
    scipy is not installed."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    if n < 2:
        return mean(d), float("nan"), 0, float("nan")
    md, sd = mean(d), stdev(d)
    t = md / (sd / math.sqrt(n)) if sd > 0 else float("inf")
    dof = n - 1
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), dof))
    except ImportError:
        p = float("nan")
    return md, t, dof, p


# 95% two-sided critical values, for the report when scipy is unavailable
T_CRIT_95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36,
             8: 2.31, 9: 2.26, 10: 2.23, 15: 2.13, 20: 2.09, 30: 2.04}


def significance_note(t, dof, p):
    if not math.isnan(p):
        return f"p = {p:.4f}" + ("  (significant)" if p < 0.05
                                 else "  (NOT significant)")
    crit = T_CRIT_95.get(dof, 2.0)
    return (f"|t| = {abs(t):.2f} vs t_crit(95%, df={dof}) = {crit:.2f}"
            + ("  (significant)" if abs(t) > crit else "  (NOT significant)"))


# --------------------------------------------------------------------- report
def build(runs, baseline, proposed, ref_weather):
    """Assemble every number the results chapter needs."""
    out = {"per_run": {}, "by_config": {}, "weather": {}, "per_town": {},
           "tests": {}, "ref_weather": ref_weather}

    for (cfg, weather, seed), routes in runs.items():
        agg = aggregate_run(routes)
        out["per_run"][f"{cfg}__{weather}__s{seed}"] = agg

    # ---- headline table: one row per config, averaged over seeds, ref weather
    by_cfg = defaultdict(list)
    for (cfg, weather, seed), routes in runs.items():
        if weather == ref_weather:
            by_cfg[cfg].append(aggregate_run(routes))
    for cfg, aggs in by_cfg.items():
        out["by_config"][cfg] = {
            "DS": mean(a["DS"] for a in aggs), "DS_sd": stdev(a["DS"] for a in aggs),
            "RC": mean(a["RC"] for a in aggs), "RC_sd": stdev(a["RC"] for a in aggs),
            "IS": mean(a["IS"] for a in aggs), "IS_sd": stdev(a["IS"] for a in aggs),
            "infractions_per_10km": mean(a["infractions_per_10km"] for a in aggs),
            "seeds": len(aggs),
        }

    # ---- weather table with the relative degradation of Eq. (5.4)
    by_cw = defaultdict(list)
    for (cfg, weather, seed), routes in runs.items():
        by_cw[(cfg, weather)].append(aggregate_run(routes)["DS"])
    for (cfg, weather), ds in by_cw.items():
        out["weather"].setdefault(cfg, {})[weather] = {
            "DS": mean(ds), "DS_sd": stdev(ds), "seeds": len(ds)}
    for cfg, per_w in out["weather"].items():
        clear = per_w.get("ClearNoon", {}).get("DS")
        for weather, d in per_w.items():
            d["delta_rob"] = (100.0 * (clear - d["DS"]) / clear
                              if clear else float("nan"))

    # ---- per-town breakdown (Appendix C) and the paired test over towns
    for (cfg, weather, seed), routes in runs.items():
        if weather != ref_weather:
            continue
        towns = defaultdict(list)
        for r in routes:
            towns[r["town"]].append(r["DS"])
        for town, ds in towns.items():
            out["per_town"].setdefault(cfg, {}).setdefault(town, []).extend(ds)
    town_means = {cfg: {t: mean(v) for t, v in towns.items()}
                  for cfg, towns in out["per_town"].items()}
    out["per_town_mean"] = town_means

    if proposed in town_means and baseline in town_means:
        common = sorted(set(town_means[proposed]) & set(town_means[baseline]))
        a = [town_means[proposed][t] for t in common]
        b = [town_means[baseline][t] for t in common]
        md, t, dof, p = t_test_paired(a, b)
        out["tests"]["paired_towns"] = {
            "proposed": proposed, "baseline": baseline, "towns": common,
            "mean_diff": md, "sd_diff": stdev([x - y for x, y in zip(a, b)]),
            "t": t, "dof": dof, "p": p, "verdict": significance_note(t, dof, p),
        }

    # ---- least significant difference for the ablation table
    sds = [v["DS_sd"] for v in out["by_config"].values() if v["seeds"] > 1]
    if sds:
        sd = max(sds)
        n = max(v["seeds"] for v in out["by_config"].values())
        crit = T_CRIT_95.get(2 * (n - 1), 2.78)
        out["tests"]["least_significant_difference"] = {
            "sd": sd, "seeds": n,
            "lsd": crit * sd * math.sqrt(2.0 / n),
            "note": "unpaired comparison of two configurations at 95%",
        }
    return out


def markdown(res):
    L = []
    A = L.append
    A("# EGCA evaluation summary\n")
    A("All numbers are produced from the raw result files; do not edit by hand.\n")

    A("\n## Comparison with the state of the art (Table 5-3)\n")
    A("| config | DS | RC (%) | IS | infractions / 10 km | seeds |")
    A("|---|---|---|---|---|---|")
    for cfg, v in sorted(res["by_config"].items(), key=lambda kv: -kv[1]["DS"]):
        A(f"| {cfg} | {v['DS']:.1f} ± {v['DS_sd']:.1f} | "
          f"{v['RC']:.1f} ± {v['RC_sd']:.1f} | {v['IS']:.2f} | "
          f"{v['infractions_per_10km']:.1f} | {v['seeds']} |")

    weathers = sorted({w for c in res["weather"].values() for w in c}
                      - {res.get("ref_weather")})
    if weathers:
        A("\n## Weather robustness (Table 5-4)\n")
        A("| config | " + " | ".join(weathers) + " |")
        A("|---" * (len(weathers) + 1) + "|")
        for cfg, per_w in sorted(res["weather"].items()):
            if not any(w in per_w for w in weathers):
                continue                      # config was not swept over weather
            cells = []
            for w in weathers:
                d = per_w.get(w)
                if d is None:
                    cells.append("-")
                elif w == "ClearNoon" or math.isnan(d["delta_rob"]):
                    cells.append(f"{d['DS']:.1f}")
                else:
                    cells.append(f"{d['DS']:.1f} ({d['delta_rob']:+.1f}%)")
            A(f"| {cfg} | " + " | ".join(cells) + " |")
        A("\nThe percentage is the relative degradation of Eq. (5.4) with respect "
          "to ClearNoon.  It is a *relative* measure: a model that is bad "
          "everywhere degrades little, so it must always be read together with "
          "the absolute worst-case score.")

    if res.get("per_town_mean"):
        A("\n## Per-town breakdown (Appendix C)\n")
        towns = sorted({t for c in res["per_town_mean"].values() for t in c})
        A("| config | " + " | ".join(towns) + " |")
        A("|---" * (len(towns) + 1) + "|")
        for cfg, per_t in sorted(res["per_town_mean"].items()):
            A(f"| {cfg} | " + " | ".join(
                f"{per_t[t]:.1f}" if t in per_t else "-" for t in towns) + " |")

    A("\n## Statistics\n")
    pt = res["tests"].get("paired_towns")
    if pt:
        A(f"Paired t-test over towns, **{pt['proposed']}** vs **{pt['baseline']}**:")
        A(f"- towns: {', '.join(pt['towns'])}")
        A(f"- mean difference {pt['mean_diff']:+.2f} DS "
          f"(sd {pt['sd_diff']:.2f}), t = {pt['t']:.2f}, df = {pt['dof']}")
        A(f"- **{pt['verdict']}**")
        A("\nThe paired test is the right one here: the towns differ far more "
          "from each other than the two methods differ on any single town, and "
          "pairing removes exactly that nuisance variation.")
    lsd = res["tests"].get("least_significant_difference")
    if lsd:
        A(f"\nLeast significant difference between two configurations "
          f"(unpaired, {lsd['seeds']} seeds, sd {lsd['sd']:.2f}): "
          f"**{lsd['lsd']:.1f} DS**.")
        A("Any ablation gap smaller than this must be reported as statistical "
          "parity, not as an improvement.")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="report")
    ap.add_argument("--baseline", default="reasonnet",
                    help="config name used as the reference in the paired test")
    ap.add_argument("--proposed", default="egca")
    ap.add_argument("--ref-weather", default="mixed",
                    help="weather tag of the headline benchmark runs")
    args = ap.parse_args()

    runs = load_all(args.results)
    if not runs:
        print(f"no result files matching '<config>__<weather>__s<seed>.json' "
              f"under {args.results}")
        return
    print(f"loaded {len(runs)} runs, "
          f"{sum(len(v) for v in runs.values())} routes in total")
    res = build(runs, args.baseline, args.proposed, args.ref_weather)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "numbers.json"), "w") as f:
        json.dump(res, f, indent=2)
    md = markdown(res)
    with open(os.path.join(args.out, "report.md"), "w") as f:
        f.write(md)
    print(md)
    print(f"wrote {args.out}/report.md and {args.out}/numbers.json")


if __name__ == "__main__":
    main()
