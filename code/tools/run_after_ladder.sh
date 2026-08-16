#!/usr/bin/env bash
# Wait for the training ladder, then evaluate both rungs on Longest6.
#
# The ladder is about 36 hours and each evaluation about 11, so without this the
# card sits idle from whenever training finishes until somebody notices. The
# ordering is deliberate: tf_query first, because it is the headline number and
# an interrupted queue should leave the more important half done.
#
# It refuses to evaluate a run that did not finish. A checkpoint exists from the
# first epoch onward, so "the file is there" says nothing -- the completed epoch
# count in the log is what says the schedule ran to the end, and evaluating a
# half-trained model for eleven hours produces a number that means nothing and
# looks like it means something.
#
#   tmux new-session -d -s chain 'bash tools/run_after_ladder.sh'
set -u

cd "$(dirname "$(readlink -f "$0")")/.." || exit 1

EPOCHS=25
: "${SLOTS:=4}"

# ---- wait for training to release the card ---------------------------------
while pgrep -f "python -u -m egca.training.train" >/dev/null; do
    sleep 300
done
echo "$(date +%F\ %T) training finished"

trained () {   # 1 if the run completed every epoch of *its own* schedule
    local log="$HOME/logs/train_$1.log"
    [ -f "$log" ] || return 1
    # The ablations ran a 20-epoch budget and the TransFuser rungs a 25-epoch
    # one, so a single threshold either rejects four finished runs or accepts a
    # truncated one. The point of this check is "did it run to the end of what
    # it was asked to do", which is per-run.
    local want=${2:-$EPOCHS}
    local n
    n=$(grep -c "^epoch " "$log" 2>/dev/null || echo 0)
    [ "$n" -ge "$want" ]
}

evaluate () {
    local name=$1
    # Which checkpoint to drive. Selection is normally on the validation
    # waypoint error, so best.pth is right -- but a run whose validation metric
    # was broken has a best.pth chosen on nonsense. tf_base_rl is exactly that:
    # its first thirteen epochs scored every frame against another frame's
    # future and reported a constant 159.56 m, which froze best.pth at epoch 3.
    # Its training targets were never affected, so last.pth is a fully trained
    # model, and the cosine schedule anneals to zero anyway -- the previous run's
    # best and last differed by one millimetre.
    local ckpt=${2:-best.pth}
    local want=${3:-$EPOCHS}
    local out="$HOME/thesis/code/results/eval_$name"
    if ! trained "$name" "$want"; then
        echo "$(date +%F\ %T) $name did not finish $want epochs; not evaluating"
        return 1
    fi
    if [ ! -f "$HOME/thesis/code/checkpoints/$name/$ckpt" ]; then
        echo "$(date +%F\ %T) $name has no $ckpt; not evaluating"
        return 1
    fi
    echo "$(date +%F\ %T) $name will be driven from $ckpt"
    python - "$name" "$ckpt" <<'PYEOF'
import json, os, sys
root = os.path.abspath(".")
name, ckpt = sys.argv[1], sys.argv[2]
os.makedirs("configs/agent", exist_ok=True)
# Controller settings are written explicitly and identically for every run,
# rather than taken from each checkpoint's stored config, for two reasons that
# both bite here.
#
# They are not all present: the ablations were trained before creeping existed,
# so their stored control block has no control_hz or creep_* at all, and the
# controller reads those. Inside a leaderboard agent that failure is not a
# crash anybody sees -- the evaluator records 36 unscored routes, which looks
# like a driving result.
#
# And they are not all the same: tf_base_rl carries creep settings the three
# ablations do not, so taking each from its own checkpoint would drive one model
# with a standstill-escape heuristic and three without it, then attribute the
# difference to the fusion mechanism. The internal comparison is the only place
# the central claim is testable, and that confound would quietly destroy it.
#
# Creeping is off. It was measured against the standstill and lost -- 21 paired
# routes gave RC 18.6 against 20.7, because the hazard interlock correctly
# refuses to creep into dense traffic -- and leaving it on would score the
# heuristic rather than the policy, which is what this comparison is about.
CONTROL = {"control_hz": 20.0, "creep_after_s": 1e9, "creep_for_s": 0.0,
           "creep_speed": 0.0, "max_throttle": 0.75, "brake_speed_ratio": 1.05,
           "clip_delta": 0.25,
           "lateral": {"kp": 1.25, "ki": 0.75, "kd": 0.30, "window": 20},
           "longitudinal": {"kp": 5.0, "ki": 0.5, "kd": 1.0, "window": 20}}
json.dump({"config": os.path.join(root, "configs/egca.yaml"),
           "checkpoint": os.path.join(root, "checkpoints", name, ckpt),
           "control": CONTROL,
           "drop_sensor": None, "lidar_drop_rate": 0.0,
           "seed": 0, "debug_dir": None},
          open("configs/agent/%s.json" % name, "w"), indent=2)
PYEOF
    echo "$(date +%F\ %T) === evaluating $name ==="
    OUT="$out" \
    AGENT="$HOME/thesis/code/egca/carla_sim/leaderboard_agent.py" \
    AGENT_CONFIG="$HOME/thesis/code/configs/agent/$name.json" \
    TRACK=SENSORS SLOTS="$SLOTS" \
        bash tools/run_autopilot_val.sh > "$HOME/logs/eval_$name.log" 2>&1
    echo "$(date +%F\ %T) $name evaluation exit=$?"
}

# Ordered by what the thesis cannot be written without. tf_base_rl is the
# relabelling result and answers whether the standstill is fixed at all; the
# three after it are the internal comparison the central claim rests on, since
# open-loop error does not separate the fusion strategies -- egca 0.135, concat
# 0.137, late 0.135, all inside a 0.003 seed spread. An interrupted queue should
# leave the earlier ones done.
evaluate tf_base_rl last.pth 25
evaluate concat  best.pth 20
evaluate late    best.pth 20
evaluate egca_s0 best.pth 20

echo "CHAIN_DONE"
for n in tf_base_rl concat late egca_s0; do
    d="$HOME/thesis/code/results/eval_$n"
    [ -d "$d" ] || continue
    echo "--- $n"
    python - "$d" <<'PYEOF'
import glob, json, statistics, sys
ds, rc, pen = [], [], []
for f in sorted(glob.glob(sys.argv[1] + "/route_*.json")):
    try:
        recs = json.load(open(f))["_checkpoint"].get("records", [])
    except Exception:
        continue
    for r in recs:
        s = r.get("scores", {})
        if "score_composed" not in s:
            continue
        ds.append(s["score_composed"]); rc.append(s["score_route"])
        pen.append(s["score_penalty"])
if ds:
    m = statistics.mean
    print("  routes %d   DS %5.2f   RC %5.2f   IS %.3f" %
          (len(ds), m(ds), m(rc), m(pen)))
    print("  expert in this same chain: DS 64.03, RC 73.38, IS 0.884")
else:
    print("  nothing scored")
PYEOF
done
