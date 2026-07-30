#!/usr/bin/env bash
# The robustness matrix: the experiments the thesis actually claims.
#
# The gate and the modality dropout are robustness mechanisms.  Everything we
# have measured so far was nominal -- clear weather, both sensors healthy --
# which is the one regime where they are not supposed to matter, and sure enough
# `no_gate` and `no_dropout` came out indistinguishable from the full model.
# Neither claim has been refuted; neither has been tested.
#
# Five configurations under three conditions:
#
#   configs     egca_aug     the model, gate and dropout on
#               no_gate      gate removed          -> tests the gate
#               no_dropout   dropout removed       -> tests sensor dropout
#               camera_only  no LiDAR branch       -> is fusion worth anything
#               concat       cross-attention out   -> is the mechanism worth it
#
#   conditions  nominal            reference point
#               lidar_drop_0.5     half the LiDAR frames lost (Fig. 5-4)
#               HardRainNight      out-of-distribution weather (Table 5-4)
#
# Every cell runs the SAME routes with the SAME seed, so the comparisons are
# paired.  That matters: TransFuser reports +-7 DS of variance between training
# seeds on Longest6, which would swamp any of these differences if the cells
# were independent samples.  Paired on identical routes, a much smaller gap is
# still readable -- but it also means the route subset must never change between
# cells, which is why it is fixed here rather than sampled.
#
# 18 routes rather than 36: this is a relative comparison, not the headline
# number, and halving the routes halves a two-day run.  Every other route keeps
# the town coverage.
# Paths are resolved from this script's own location so a fresh
# checkout works; the copies that produced the recorded results lived in
# ~/scripts on the evaluation machine and differed only in that line.
set -u

cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
source tools/eval_env.sh
ICD=/usr/share/vulkan/icd.d/nvidia_icd.json
SPLIT="$ROUTES6/longest6_split"
OUT="$HOME/thesis/code/results/robustness"
WEATHER_DIR="$HOME/thesis/code/routes_weather"
SLOTS=4
mkdir -p "$OUT" "$WEATHER_DIR"

CONFIGS="egca_aug no_gate no_dropout camera_only concat"
ROUTES=$(seq 0 2 35)          # 18 of the 36, spread across all six towns

# Weather variants of the per-route split files.  Longest6 bakes a different
# weather into each route, which is right for the headline number and wrong
# here: the condition has to be the independent variable, so every route is
# rewritten to the same one.
mkdir -p "$WEATHER_DIR/split"
for r in $ROUTES; do
    src="$SPLIT/longest_weathers_$r.xml"
    dst="$WEATHER_DIR/split/longest_weathers_$r.xml"
    [ -s "$dst" ] && continue
    python tools/make_weather_routes.py --routes "$src" \
        --weather HardRainNight --out "$dst" || exit 1
done
echo "$(date +%T) weather route files ready"

boot () {
    local port=$1
    docker rm -f "carla-rb-$port" >/dev/null 2>&1
    docker run -d --name "carla-rb-$port" --gpus all --net=host \
        -e NVIDIA_DRIVER_CAPABILITIES=all -v $ICD:$ICD:ro \
        carlasim/carla:0.9.14 /bin/bash CarlaUE4.sh -RenderOffScreen -nosound \
        -quality-level=Epic -carla-rpc-port=$port >/dev/null
    local i
    for i in $(seq 1 60); do
        if python -c "import carla; c=carla.Client('localhost',$port); c.set_timeout(10.0); c.get_server_version()" >/dev/null 2>&1; then
            return 0
        fi
        sleep 10
    done
    echo "$(date +%T) carla FAILED on $port"; return 1
}

agent_config () {                       # name ckpt drop rate -> path
    local f="$OUT/.agent_$1.json"
    cat > "$f" <<JSON
{"config": "configs/egca.yaml",
 "checkpoint": "checkpoints/$2/best.pth",
 "drop_sensor": $3,
 "lidar_drop_rate": $4,
 "seed": 0,
 "debug_dir": null}
JSON
    echo "$f"
}

run_slot () {
    local slot=$1 port=$((2000 + slot * 10)) tm=$((8100 + slot * 10))
    boot "$port" || return 1
    local i=0 cfg cond r conf routes tag
    for cfg in $CONFIGS; do
        for cond in nominal lidardrop rain; do
            case $cond in
                nominal)   conf=$(agent_config "${cfg}_$cond" "$cfg" null 0.0)
                           routes="$SPLIT" ;;
                lidardrop) conf=$(agent_config "${cfg}_$cond" "$cfg" null 0.5)
                           routes="$SPLIT" ;;
                rain)      conf=$(agent_config "${cfg}_$cond" "$cfg" null 0.0)
                           routes="$WEATHER_DIR/split" ;;
            esac
            for r in $ROUTES; do
                i=$((i + 1))
                [ $((i % SLOTS)) -eq "$slot" ] || continue
                tag="${cfg}__${cond}__r${r}"
                [ -s "$OUT/$tag.json" ] && continue
                python -u "$LEADERBOARD_ROOT/leaderboard/leaderboard_evaluator.py" \
                    --routes="$routes/longest_weathers_$r.xml" \
                    --scenarios="$ROUTES6/eval_scenarios.json" \
                    --agent=egca/carla_sim/leaderboard_agent.py \
                    --agent-config="$conf" \
                    --checkpoint="$OUT/$tag.json" \
                    --track=SENSORS --port="$port" --trafficManagerPort="$tm" \
                    > "$HOME/logs/rb_$tag.log" 2>&1
                echo "$(date +%T) [s$slot] $tag"
            done
        done
    done
    docker rm -f "carla-rb-$port" >/dev/null 2>&1
    echo "$(date +%T) [s$slot] finished"
}

for s in $(seq 0 $((SLOTS - 1))); do
    run_slot "$s" &
    sleep 45
done
wait
echo "ALLDONE"
python tools/summarise_robustness.py --results "$OUT"
