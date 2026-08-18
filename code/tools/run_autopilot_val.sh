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
# It must run through $EVALUATOR, which is the Longest6 `*_local.py` copy -- see
# tools/eval_env.sh.  A first pass of this script used the stock evaluator and
# produced RC 87.94 against their published 82.71, which looked like a pass and
# was not: the stock evaluator runs lighter traffic, so the benchmark was easier,
# and it also penalises stop-sign infractions, which Longest6 does not.  Results
# produced that way are not comparable with anything and were discarded.
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
# Defaults drive the reference expert.  The same sharding, retry and
# scored-record logic is what a policy evaluation needs, and run_eval.sh cannot
# provide it: it hands all 36 routes to one evaluator call, which at the ~73
# minutes per route measured here is 44 hours in a single slot against 11 across
# four.  Overriding these three variables points the identical machinery at our
# own agent.
: "${OUT:=$HOME/thesis/code/results/autopilot_val}"
: "${AGENT:=$HOME/transfuser/team_code_autopilot/autopilot.py}"
: "${AGENT_CONFIG:=}"
: "${TRACK:=MAP}"
# Container names are derived from the output directory so a policy run and the
# expert run can be told apart, and so a stale container from one never gets
# removed by the other.
TAG=$(basename "$OUT" | tr -c "a-zA-Z0-9" "-")
# Routes are handed to slots by `route % SLOTS`, so the divisor also decides how
# a partial re-run spreads.  The six routes lost to slot 0's dead simulator are
# all multiples of 4 and would queue up behind each other again at SLOTS=4;
# SLOTS=3 puts two in each slot instead.  Overridable for exactly that reason.
: "${SLOTS:=4}"
# Port bases, overridable because this machine is shared and another evaluation
# may already hold the default range. A simulator that cannot bind its port
# fails the whole slot, and the slot's routes are then skipped -- which is how
# six routes were lost silently once before.
# Which routes to drive. The full 36 is the only set comparable with published
# Longest6 numbers and is what the reported result must use. A subset is for
# screening: 11 h per evaluation is too expensive to ask whether an idea helps
# at all, and the measured cost of a smaller sample is known -- over the 36
# scored routes of one run the route-to-route spread is 6.25 DS, so 12 routes
# estimate the mean to about +-2.4 at 90%. Large effects survive that; small
# ones were never going to be visible in a thesis-sized budget anyway.
#
# Pairing does not rescue it. The route-level DS correlation between two
# different runs is -0.09, because a route's score is decided by where the
# policy happens to fail rather than by the route's difficulty, so scoring the
# same routes for both arms buys nothing. Only the count helps.
: "${ROUTE_IDS:=$(seq 0 35)}"
: "${PORT_BASE:=2000}"
: "${TM_BASE:=8100}"
mkdir -p "$OUT"

# DRY_RUN=1 walks the whole control flow without a simulator or a GPU: every
# loop, every variable expansion and every route assignment runs for real, only
# the two expensive calls are stubbed.  This exists because the first attempt
# died on an unbound variable inside a function and `bash -n` had reported the
# file as fine -- syntax checking cannot see an expansion that only happens at
# run time, so there was no way to find it short of burning the GPU window.
: "${DRY_RUN:=0}"

# ---- wait for the GPU ------------------------------------------------------
# Watching one hard-coded session name was wrong the moment the training session
# was called something else: this waited on `aug`, the ladder runs as `ladder`,
# so the check passed instantly and four simulators would have been started on
# top of a training run that already holds most of the card. Look for the
# process, not for what someone named its window.
while [ "$DRY_RUN" = 0 ] && pgrep -f "python -u -m egca.training.train" >/dev/null; do
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
    # Refuse a port somebody else is already serving. Docker will happily start
    # a container that cannot bind, and the slot then spends its whole run
    # talking to a simulator that is not ours -- or to nothing at all.
    if python -c "import socket,sys; s=socket.socket(); sys.exit(0 if s.connect_ex(('127.0.0.1',$port))==0 else 1)" 2>/dev/null; then
        echo "$(date +%T) port $port is already in use; set PORT_BASE to a free range"
        return 1
    fi
    docker rm -f "carla-$TAG-$port" >/dev/null 2>&1
    docker run -d --name "carla-$TAG-$port" --gpus all --net=host \
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
if not recs or "score_composed" not in recs[0].get("scores", {}):
    sys.exit(1)
# A route the simulator dropped is not a driving result. On a shared card,
# "A sensor took too long to send their data" is recorded as a completed route
# with DS 0.00 and no infractions, which is indistinguishable from a policy that
# never moved -- and it drags the mean down by a full route each time. Reject it
# so the slot retries on a fresh simulator.
if "crash" in recs[0].get("status", "").lower():
    sys.exit(1)
sys.exit(0)
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
    local port=$((PORT_BASE + slot * 10))
    local tm=$((TM_BASE + slot * 10))
    local r
    local try
    local rc
    boot "$port" || return 1
    for r in $ROUTE_IDS; do
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
            python -u "$EVALUATOR" \
                --routes="$SPLIT/longest_weathers_$r.xml" \
                --scenarios="$ROUTES6/eval_scenarios.json" \
                --agent="$AGENT" \
                ${AGENT_CONFIG:+--agent-config="$AGENT_CONFIG"} \
                --checkpoint="$OUT/route_$r.json" \
                --track="$TRACK" --port="$port" --trafficManagerPort="$tm" \
                > "$HOME/logs/$(basename "$OUT")_route_$r.log" 2>&1
            rc=$?
            scored "$OUT/route_$r.json" && break
            echo "$(date +%T) [s$slot] route $r UNSCORED (exit $rc)"
            rm -f "$OUT/route_$r.json"
            [ "$try" = 1 ] && { boot "$port" || return 1; }
        done
    done
    [ "$DRY_RUN" = 0 ] && docker rm -f "carla-$TAG-$port" >/dev/null 2>&1
    echo "$(date +%T) [s$slot] finished"
}

for s in $(seq 0 $((SLOTS - 1))); do
    run_slot "$s" &
    sleep 45          # stagger: four simulators compiling shaders at once is slow
done
wait

echo "ALLDONE"
python - "$OUT" <<'PYEOF'
import glob, json, statistics, sys
ds, rc, pen, n, bad = [], [], [], 0, 0
for f in sorted(glob.glob(sys.argv[1] + "/route_*.json")):
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
