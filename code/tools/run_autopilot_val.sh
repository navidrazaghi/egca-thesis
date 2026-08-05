#!/usr/bin/env bash
# Validate our evaluation harness against TransFuser's own published number.
#
# They ship results/autopilot_longest6.json: DS 74.49, RC 82.71 over all 36
# Longest6 routes, produced by team_code_autopilot/autopilot.py.  Running that
# same agent through our stack and landing on the same figure is the strongest
# end-to-end check available -- it validates the route definitions, the scenario
# triggers, the traffic, the infraction detectors and the DS/RC arithmetic
# against an external reference rather than against our own expectations.
#
# The agent is privileged and needs none of the mmcv/mmdet stack that blocked
# the sensor-based baselines, so it runs in the ordinary evaluation environment.
#
# Waits for the training run to release the GPU before starting.
# Paths are resolved from this script's own location so a fresh
# checkout works; the copies that produced the recorded results lived in
# ~/scripts on the evaluation machine and differed only in that line.
set -u

cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
source tools/eval_env.sh
ICD=/usr/share/vulkan/icd.d/nvidia_icd.json
SPLIT="$ROUTES6/longest6_split"
OUT="$HOME/thesis/code/results/autopilot_val"
AGENT="$HOME/transfuser/team_code_autopilot/autopilot.py"
# Routes are handed to slots by `route % SLOTS`, so the divisor also decides how
# a partial re-run spreads.  The six routes lost to slot 0's dead simulator are
# all multiples of 4 and would queue up behind each other again at SLOTS=4;
# SLOTS=3 puts two in each slot instead.  Overridable for exactly that reason.
: "${SLOTS:=4}"
mkdir -p "$OUT"

# DRY_RUN=1 walks the whole control flow without a simulator or a GPU: every
# loop, every variable expansion and every route assignment runs for real, only
# the two expensive calls are stubbed.  This exists because the first attempt
# died on an unbound variable inside a function and `bash -n` had reported the
# file as fine -- syntax checking cannot see an expansion that only happens at
# run time, so there was no way to find it short of burning the GPU window.
: "${DRY_RUN:=0}"

# ---- wait for the GPU ------------------------------------------------------
while [ "$DRY_RUN" = 0 ] && tmux has-session -t aug 2>/dev/null; do
    echo "$(date +%T) waiting for training to finish ..."
    sleep 300
done
[ "$DRY_RUN" = 0 ] && echo "$(date +%T) GPU free, starting"

boot () {
    local port=$1
    if [ "$DRY_RUN" = 1 ]; then
        echo "[dry] boot carla on port $port"
        return 0
    fi
    docker rm -f "carla-ap-$port" >/dev/null 2>&1
    docker run -d --name "carla-ap-$port" --gpus all --net=host \
        -e NVIDIA_DRIVER_CAPABILITIES=all -v $ICD:$ICD:ro \
        carlasim/carla:0.9.14 /bin/bash CarlaUE4.sh -RenderOffScreen -nosound \
        -quality-level=Epic -carla-rpc-port=$port >/dev/null
    local i
    for i in $(seq 1 60); do
        if python -c "import carla; c=carla.Client('localhost',$port); c.set_timeout(10.0); c.get_server_version()" >/dev/null 2>&1; then
            echo "$(date +%T) carla up on $port"; return 0
        fi
        sleep 10
    done
    echo "$(date +%T) carla FAILED on $port"; return 1
}

# A route counts as done only when its checkpoint holds a scored record.  The
# evaluator writes the file early and exits 0 on a simulator it can no longer
# talk to, so neither the exit code nor the file's existence is sufficient on
# its own -- the first run of this script lost six routes because slot 0's
# simulator died after three and every later route was skipped in silence.
scored () {
    [ -s "$1" ] || return 1
    python - "$1" <<'PYEOF' >/dev/null 2>&1
import json, sys
recs = json.load(open(sys.argv[1]))["_checkpoint"].get("records", [])
sys.exit(0 if recs and "score_composed" in recs[0].get("scores", {}) else 1)
PYEOF
}

run_slot () {
    # One assignment per line, deliberately.  Bash expands every argument of
    # `local` before the builtin runs, so `local slot=$1 port=$((slot*10))`
    # evaluates the arithmetic while slot is still unset -- which under `set -u`
    # kills the function on its first line.  That is what emptied all four slots
    # on the first attempt and cost a three-hour GPU window, and `bash -n`
    # cannot see it because the expansion only happens at run time.
    local slot=$1
    local port=$((2000 + slot * 10))
    local tm=$((8100 + slot * 10))
    local r
    local try
    local rc
    boot "$port" || return 1
    for r in $(seq 0 35); do
        [ $((r % SLOTS)) -eq "$slot" ] || continue
        if scored "$OUT/route_$r.json"; then
            echo "$(date +%T) [s$slot] route $r done"; continue
        fi
        if [ "$DRY_RUN" = 1 ]; then
            echo "[dry] slot=$slot route=$r port=$port tm=$tm"
            continue
        fi
        # Two attempts, the second on a freshly booted simulator.  A crashed
        # CARLA poisons every route that follows it in this slot, so the retry
        # is what keeps one dead simulator from costing a quarter of the run.
        for try in 1 2; do
            echo "$(date +%T) [s$slot] route $r attempt $try ..."
            python -u "$LEADERBOARD_ROOT/leaderboard/leaderboard_evaluator.py" \
                --routes="$SPLIT/longest_weathers_$r.xml" \
                --scenarios="$ROUTES6/eval_scenarios.json" \
                --agent="$AGENT" \
                --checkpoint="$OUT/route_$r.json" \
                --track=MAP --port="$port" --trafficManagerPort="$tm" \
                > "$HOME/logs/ap_route_$r.log" 2>&1
            rc=$?
            scored "$OUT/route_$r.json" && break
            echo "$(date +%T) [s$slot] route $r UNSCORED (exit $rc)"
            rm -f "$OUT/route_$r.json"
            [ "$try" = 1 ] && { boot "$port" || return 1; }
        done
    done
    [ "$DRY_RUN" = 0 ] && docker rm -f "carla-ap-$port" >/dev/null 2>&1
    echo "$(date +%T) [s$slot] finished"
}

for s in $(seq 0 $((SLOTS - 1))); do
    run_slot "$s" &
    sleep 45          # stagger: four simulators compiling shaders at once is slow
done
wait

echo "ALLDONE"
python - <<'PYEOF'
import glob, json, statistics
ds, rc, pen, n, bad = [], [], [], 0, 0
for f in sorted(glob.glob("results/autopilot_val/route_*.json")):
    try:
        recs = json.load(open(f))["_checkpoint"].get("records", [])
    except Exception:
        bad += 1; continue
    for r in recs:
        s = r["scores"]
        ds.append(s["score_composed"]); rc.append(s["score_route"])
        pen.append(s["score_penalty"]); n += 1
if n:
    print("routes scored: %d  (unreadable files: %d)" % (n, bad))
    print("  DS      = %5.2f     published: 74.49" % statistics.mean(ds))
    print("  RC      = %5.2f     published: 82.71" % statistics.mean(rc))
    print("  penalty = %5.2f     published:  0.89" % statistics.mean(pen))
else:
    print("no routes scored (unreadable files: %d)" % bad)
PYEOF
