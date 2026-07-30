#!/usr/bin/env bash
# Ablation ladder on our own dataset, ordered by expected value.
#
# Control is egca_aug -- ResNet-34, mean pooling into a GRU, goal fed to the GRU
# at every step, mirror augmentation on.  Each run is otherwise identical
# (40 epochs, seed 0, same 137k frames).
#
#   egca_query   Learned planning queries read the fused tokens; the GRU is gone
#                and a dedicated head predicts target speed.
#                Removes three bottlenecks at once: the mean pool that reduced
#                4536 spatial tokens to one 256-d vector, the late goal
#                conditioning LEAD attributes +6.7 DS to fixing, and the
#                lateral/longitudinal coupling that made a standstill
#                self-sustaining -- a collapsed trajectory used to set the
#                desired speed to zero, and zero speed reproduced the collapse.
#
#                Note this rung is not a single change: with the GRU removed the
#                goal has no route except the fusion token, so the goal fix
#                comes along whether or not we want it isolated. egca_goal below
#                exists to say how much of any gain was the goal alone.
#
#   egca_regnet  RegNetY-3.2GF instead of ResNet-34.
#                Both papers found the backbone the most impactful architecture
#                choice; TransFuser measured ResNet34+ResNet18 at 42.0 DS
#                against 49.5-56.7 for RegNetY. Orthogonal to the readout, so it
#                stacks with whatever wins above.
#
#   egca_goal    Goal as a fusion token, GRU kept but no longer fed the goal.
#                Attribution only: it decomposes egca_query. Run it last, and
#                only if egca_query moved the number enough to be worth
#                explaining.
#
# Held for the TransFuser data (3500 routes vs our 50): egca_bev and egca_pool.
# Both are our own hypotheses, and both are most distorted by the overfitting
# regime this dataset is in -- validation flattened at epoch 12 while training
# error kept falling.
#
#   run egca_bev  model.fusion.camera_space=bev
#   run egca_pool model.fusion.pooling=attention model.fusion.gate_form=vector
#
# Not here: creeping. Controller only, no retraining, measurable today.
#
# Launch when the GPU is free:  bash ~/scripts/train_ladder.sh
# Paths are resolved from this script's own location so a fresh
# checkout works; the copies that produced the recorded results lived in
# ~/scripts on the evaluation machine and differed only in that line.
set -u

cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
source tools/eval_env.sh

run () {
    local name=$1; shift
    if [ -f "checkpoints/$name/best.pth" ]; then
        echo "$(date +%T) $name already trained, skipping"
        return 0
    fi
    echo "$(date +%T) === training $name ==="
    python -u -m egca.training.train \
        --config configs/egca.yaml --seed 0 \
        --set train.epochs=40 "train.ckpt_dir=checkpoints/$name" "$@" \
        > "$HOME/logs/train_$name.log" 2>&1
    echo "$(date +%T) $name exit=$?"
}

run egca_query  model.decoder.readout=query model.fusion.goal_injection=fusion
run egca_regnet model.camera.backbone=regnet_y_3_2gf
run egca_goal   model.fusion.goal_injection=fusion

echo "LADDER_DONE"
printf '%-13s %9s %11s\n' run best_val final_train
for n in egca_aug egca_query egca_regnet egca_goal; do
    f="$HOME/logs/train_$n.log"
    [ -f "$f" ] || continue
    b=$(grep -o "val -\?[0-9.]* (wp [0-9.]* m)" "$f" | grep -o "wp [0-9.]*" \
        | cut -d" " -f2 | sort -n | head -1)
    t=$(grep -o "train -\?[0-9.]* (wp [0-9.]* m)" "$f" | grep -o "wp [0-9.]*" \
        | cut -d" " -f2 | tail -1)
    printf '%-13s %9s %11s\n' "$n" "$b" "$t"
done
