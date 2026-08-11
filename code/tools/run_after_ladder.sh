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

trained () {   # 1 if the run completed every epoch of the schedule
    local log="$HOME/logs/train_$1.log"
    [ -f "$log" ] || return 1
    local n
    n=$(grep -c "^epoch " "$log" 2>/dev/null || echo 0)
    [ "$n" -ge "$EPOCHS" ]
}

evaluate () {
    local name=$1
    local out="$HOME/thesis/code/results/eval_$name"
    if ! trained "$name"; then
        echo "$(date +%F\ %T) $name did not finish $EPOCHS epochs; not evaluating"
        return 1
    fi
    # Selection is on the validation waypoint error now, so best.pth is the
    # right file to drive; last.pth was only ever the workaround for the old
    # criterion keeping a worse model.
    python - "$name" <<'PYEOF'
import json, os, sys
root = os.path.abspath(".")
name = sys.argv[1]
os.makedirs("configs/agent", exist_ok=True)
json.dump({"config": os.path.join(root, "configs/egca.yaml"),
           "checkpoint": os.path.join(root, "checkpoints", name, "best.pth"),
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

evaluate tf_query
evaluate tf_base

echo "CHAIN_DONE"
for n in tf_query tf_base; do
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
