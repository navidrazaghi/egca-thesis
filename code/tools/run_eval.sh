#!/usr/bin/env bash
# One closed-loop Longest6 evaluation, supervised and resumable.
#
# Same reasoning as tools/collect_town.sh: a 36-route evaluation is many hours
# of simulator time and CARLA does not survive it in one piece.  The leaderboard
# writes its checkpoint file after every route and can restart from it, so the
# supervisor here simply recycles the container and re-enters with --resume=1
# until the checkpoint reports all routes finished.  An interrupted run -- a
# crashed simulator, a killed ssh session, a machine that went to sleep --
# therefore costs at most the route that was in flight.
#
# Usage:
#   bash tools/run_eval.sh NAME SEED WEATHER CKPT_DIR
#
#   NAME      config name in the result file, e.g. egca, egca_no_gate
#   SEED      training seed the checkpoint came from
#   WEATHER   "mixed" for the native per-route Longest6 conditions (the headline
#             benchmark), or a preset name from egca.carla_sim.weather for the
#             robustness sweep, in which case the route file is regenerated with
#             that condition on all 36 routes
#   CKPT_DIR  checkpoints/<dir> holding best.pth
#
# Optional, via the environment:
#   DROP_SENSOR=cam|lidar     permanent modality failure
#   LIDAR_DROP_RATE=0.5       per-frame LiDAR loss
#   PORT=2000 TMPORT=8000     simulator ports
#   RESULTS=results           output directory
#   DEBUG_DIR=agent_debug     dump a diagnostic panel every 20 frames
#
# Output: $RESULTS/${NAME}__${WEATHER}__s${SEED}.json, which is exactly the name
# egca.eval.aggregate_results expects.
set -u

NAME=${1:?config name}
SEED=${2:?seed}
WEATHER=${3:?weather or "mixed"}
CKPT_DIR=${4:?checkpoint dir}

: "${PORT:=2000}"
: "${TMPORT:=8000}"
: "${RESULTS:=results}"
: "${DROP_SENSOR:=}"
: "${LIDAR_DROP_RATE:=0.0}"
: "${DEBUG_DIR:=}"
: "${MAX_ATTEMPTS:=12}"
: "${SERVER_TIMEOUT:=600}"

: "${LEADERBOARD_ROOT:?source tools/eval_env.sh first}"
: "${ROUTES6:?source tools/eval_env.sh first}"

TAG="${NAME}__${WEATHER}__s${SEED}"
OUT="$RESULTS/${TAG}.json"
CONF="$RESULTS/.agent_${TAG}.json"
LOG="${LOGDIR:-$HOME/logs}/eval_${TAG}.log"
CONTAINER="carla-eval-${PORT}"
ICD=/usr/share/vulkan/icd.d/nvidia_icd.json

mkdir -p "$RESULTS" "$(dirname "$LOG")"

# ---- routes: native Longest6 weathers, or one fixed condition ---------------
if [ "$WEATHER" = "mixed" ]; then
    ROUTES="$ROUTES6/longest6.xml"
else
    ROUTES="routes_weather/longest6_${WEATHER}.xml"
    if [ ! -f "$ROUTES" ]; then
        python tools/make_weather_routes.py --routes "$ROUTES6/longest6.xml" \
            --weather "$WEATHER" --out "$ROUTES" || exit 1
    fi
fi

# ---- agent config ----------------------------------------------------------
python - "$CONF" "$CKPT_DIR" "$SEED" "$DROP_SENSOR" "$LIDAR_DROP_RATE" \
         "$DEBUG_DIR" <<'PY'
import json, sys
conf, ckpt, seed, drop, rate, debug = sys.argv[1:7]
json.dump({"config": "configs/egca.yaml",
           "checkpoint": f"checkpoints/{ckpt}/best.pth",
           "drop_sensor": drop or None,
           "lidar_drop_rate": float(rate),
           "seed": int(seed),
           "debug_dir": debug or None}, open(conf, "w"), indent=2)
PY
[ $? -eq 0 ] || exit 1

# ---- how far did we get? ---------------------------------------------------
progress() {   # "<done> <total>", or "0 0" if there is no checkpoint yet
    python - "$OUT" <<'PY'
import json, sys
try:
    p = json.load(open(sys.argv[1]))["_checkpoint"].get("progress") or [0, 0]
except Exception:
    p = [0, 0]
print(p[0], p[1])
PY
}

# Wait until the simulator actually answers, rather than for a fixed interval.
# Several simulators compiling shaders on the same GPU take far longer to come
# up than one does, and a fixed sleep that is generous for one slot is not
# enough for four: the evaluator then dies on its own 60 s connect timeout and
# the whole attempt is wasted booting a server that was nearly ready.
wait_for_server() {
    local deadline=$(( SECONDS + SERVER_TIMEOUT ))
    while [ "$SECONDS" -lt "$deadline" ]; do
        if python - "$PORT" 2>/dev/null <<'PY'
import sys, carla
c = carla.Client("localhost", int(sys.argv[1]))
c.set_timeout(10.0)
c.get_server_version()
PY
        then
            echo "$(date +%H:%M:%S) [$TAG] simulator ready on $PORT"
            return 0
        fi
        sleep 10
    done
    echo "$(date +%H:%M:%S) [$TAG] simulator did not come up within ${SERVER_TIMEOUT}s"
    return 1
}

start_server() {
    docker rm -f "$CONTAINER" >/dev/null 2>&1
    docker run -d --name "$CONTAINER" --gpus all --net=host \
        -e NVIDIA_DRIVER_CAPABILITIES=all \
        -v "$ICD":"$ICD":ro \
        carlasim/carla:0.9.14 /bin/bash CarlaUE4.sh \
        -RenderOffScreen -nosound -quality-level=Epic \
        -carla-rpc-port="$PORT" >/dev/null
    echo "$(date +%H:%M:%S) [$TAG] started $CONTAINER, waiting for shaders"
    wait_for_server
}

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    read -r done_n total_n <<<"$(progress)"
    if [ "$total_n" -gt 0 ] && [ "$done_n" -ge "$total_n" ]; then
        echo "$(date +%F\ %T) [$TAG] complete: $done_n/$total_n routes"
        break
    fi
    RESUME=0
    [ -f "$OUT" ] && RESUME=1

    if ! start_server; then
        continue                     # count it as an attempt and try again
    fi
    echo "$(date +%F\ %T) [$TAG] attempt $attempt, resume=$RESUME, at $done_n routes"

    # -u so the log is live rather than sitting in the pipe buffer
    python -u "$EVALUATOR" \
        --routes="$ROUTES" \
        --scenarios="$ROUTES6/eval_scenarios.json" \
        --agent=egca/carla_sim/leaderboard_agent.py \
        --agent-config="$CONF" \
        --checkpoint="$OUT" \
        --track=SENSORS \
        --port="$PORT" --trafficManagerPort="$TMPORT" \
        --resume="$RESUME" 2>&1 | tee -a "$LOG"
    echo "$(date +%F\ %T) [$TAG] evaluator exited ($?)"
    sleep 5
done

read -r done_n total_n <<<"$(progress)"
docker rm -f "$CONTAINER" >/dev/null 2>&1
echo "$(date +%F\ %T) [$TAG] finished at $done_n/$total_n routes -> $OUT"
[ "$total_n" -gt 0 ] && [ "$done_n" -ge "$total_n" ]
