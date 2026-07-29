#!/usr/bin/env bash
# The whole closed-loop evaluation matrix of Chapter 5, in priority order.
#
# A single 36-route Longest6 evaluation is roughly a day of simulator time, and
# the matrix below is around thirty of them, so two things matter more than
# anything else here: it must survive interruption, and it must use the machine.
#
#   * Resumability comes from tools/run_eval.sh, which re-enters the evaluator
#     with --resume=1 until the checkpoint file reports all routes done.  A job
#     already finished is detected and skipped, so re-running this script after
#     a crash, a reboot or a sleeping laptop costs nothing.
#   * The A100 fits several simulators at once (one CARLA is ~3.6 GB and the
#     policy ~2 GB), so SLOTS jobs run concurrently on disjoint port pairs.
#     Wall time falls almost linearly; 4 slots is comfortable on 40 GB / 32
#     cores, which is what the default assumes.
#
# The order is deliberate: the headline number first, then the ablations that
# carry the thesis' actual claims, then the robustness sweeps.  Stopping the
# script early therefore always leaves a coherent subset of Chapter 5 finished
# rather than a scattering of half-tables.
#
# Usage:
#   cd ~/thesis/code
#   source tools/eval_env.sh
#   bash tools/run_eval_queue.sh            # everything
#   SLOTS=2 bash tools/run_eval_queue.sh    # fewer parallel simulators
#   EVAL_GROUPS="headline ablation" bash tools/run_eval_queue.sh
#
# (EVAL_GROUPS, not GROUPS: bash owns GROUPS as a read-only array of the
# caller's group ids, so assigning to it silently selects no jobs at all.)
set -u

: "${SLOTS:=4}"
: "${EVAL_GROUPS:=headline ablation weather sensor}"
: "${BASE_PORT:=2000}"
: "${BASE_TMPORT:=8100}"
: "${RESULTS:=results}"
: "${LOGDIR:=$HOME/logs}"
export RESULTS LOGDIR

: "${LEADERBOARD_ROOT:?source tools/eval_env.sh first}"

# ---- the matrix ------------------------------------------------------------
# one job per line:  group | name | seed | weather | ckpt_dir | env-overrides
JOBS=$(cat <<'EOF'
headline | egca | 0 | mixed | egca_s0 |
headline | egca | 1 | mixed | egca_s1 |
headline | egca | 2 | mixed | egca_s2 |
ablation | egca_full_attn  | 0 | mixed | full_attn  |
ablation | egca_no_gate    | 0 | mixed | no_gate    |
ablation | egca_no_dropout | 0 | mixed | no_dropout |
ablation | egca_camera_only| 0 | mixed | camera_only|
ablation | egca_lidar_only | 0 | mixed | lidar_only |
ablation | egca_concat     | 0 | mixed | concat     |
ablation | egca_no_aux     | 0 | mixed | no_aux     |
ablation | egca_late       | 0 | mixed | late       |
weather  | egca | 0 | ClearNoon     | egca_s0 |
weather  | egca | 0 | WetNoon       | egca_s0 |
weather  | egca | 0 | HardRainNoon  | egca_s0 |
weather  | egca | 0 | FogMorning    | egca_s0 |
weather  | egca | 0 | ClearNight    | egca_s0 |
weather  | egca | 0 | HardRainNight | egca_s0 |
sensor   | egca_drop_cam    | 0 | mixed | egca_s0 | DROP_SENSOR=cam
sensor   | egca_drop_lidar  | 0 | mixed | egca_s0 | DROP_SENSOR=lidar
sensor   | egca_ldrop25     | 0 | mixed | egca_s0 | LIDAR_DROP_RATE=0.25
sensor   | egca_ldrop50     | 0 | mixed | egca_s0 | LIDAR_DROP_RATE=0.50
sensor   | egca_ldrop75     | 0 | mixed | egca_s0 | LIDAR_DROP_RATE=0.75
EOF
)

# ---- select the requested groups, keeping the order above ------------------
SELECTED=()
while IFS='|' read -r group name seed weather ckpt envs; do
    group=$(echo "$group" | xargs)
    [ -z "$group" ] && continue
    case " $EVAL_GROUPS " in *" $group "*) ;; *) continue ;; esac
    SELECTED+=("$(echo "$name" | xargs)|$(echo "$seed" | xargs)|$(echo "$weather" | xargs)|$(echo "$ckpt" | xargs)|$(echo "$envs" | xargs)")
done <<<"$JOBS"

echo "$(date +%F\ %T) queue: ${#SELECTED[@]} jobs over $SLOTS slots (groups: $EVAL_GROUPS)"

# ---- one worker per slot, jobs assigned round-robin -------------------------
worker() {
    local slot=$1
    local port=$(( BASE_PORT + slot * 10 ))
    local tmport=$(( BASE_TMPORT + slot * 10 ))
    local i=0
    for job in "${SELECTED[@]}"; do
        if [ $(( i % SLOTS )) -eq "$slot" ]; then
            IFS='|' read -r name seed weather ckpt envs <<<"$job"
            local out="$RESULTS/${name}__${weather}__s${seed}.json"
            echo "$(date +%F\ %T) [slot $slot] -> $name $weather s$seed (port $port)"
            # shellcheck disable=SC2086
            env $envs PORT="$port" TMPORT="$tmport" \
                bash tools/run_eval.sh "$name" "$seed" "$weather" "$ckpt" \
                >>"$LOGDIR/eval_queue_slot${slot}.log" 2>&1
            echo "$(date +%F\ %T) [slot $slot] done $name $weather s$seed (rc=$?)"
        fi
        i=$(( i + 1 ))
    done
}

mkdir -p "$LOGDIR" "$RESULTS"
for slot in $(seq 0 $(( SLOTS - 1 ))); do
    worker "$slot" &
done
wait

echo "$(date +%F\ %T) ALL EVALUATION DONE"
echo "aggregate with:  python -m egca.eval.aggregate_results --results $RESULTS --out report"
