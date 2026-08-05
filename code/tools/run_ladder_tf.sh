#!/usr/bin/env bash
# Ablation ladder on the published TransFuser dataset.
#
# The earlier ladder ran on our own collection and the control finished first:
# mirror augmentation moved validation from 0.137 m to 0.133 m and the
# train/val gap reopened, so regularisation was not the constraint. 50 routes
# over 5 towns is. This dataset has 488 routes over 8 towns, and holding one
# town out entirely measures generalisation to unseen layout rather than to
# unseen frames of layouts already learned:
#
#     train 203,129 frames over 7 towns      val 55,737 frames of Town05
#
# Runs, ordered by expected value:
#
#   tf_base     the architecture as the thesis describes it. Nothing here is
#               comparable to anything without it, since the control from our
#               own data was trained on different frames.
#
#   tf_query    learned planning queries read the fused tokens; the GRU is gone
#               and a dedicated head predicts target speed. Removes the mean
#               pool that reduced 4536 spatial tokens to one vector, the late
#               goal conditioning LEAD attributes +6.7 DS to fixing, and the
#               coupling that made a standstill self-sustaining.
#
#   tf_regnet   RegNetY-3.2GF instead of ResNet-34. Both TransFuser and LEAD
#               report the backbone as the most impactful architecture choice.
#               Orthogonal to the readout, so it stacks with whichever wins.
#
# bev_classes=3 on every run: their HD map carries drivable area and lane
# marking, so a six-class head would train three outputs against labels that do
# not exist. Agents are in label_raw as boxes and could be rasterised later.
#
# 25 epochs rather than 40. The set is 1.5x larger, so this is close to the same
# number of gradient steps the 40-epoch runs took on our data, and every run
# here shares the schedule so the comparison stays internally consistent.
#
# Launch when the GPU is free:  bash tools/run_ladder_tf.sh
set -u

cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
# One thread per process before the pools are sized: the 24 loader workers each
# inherit this, and the CPU work happens inside them, not in the parent.
THREADS_PER_SLOT=1
source tools/eval_env.sh

# Keep glibc from serving 465 MB batch tensors out of its heap.  Growing the
# heap for allocations that size asks the kernel for high-order pages, and this
# machine fails that request 87% of the time it tries: kcompactd sat at 25% of a
# core, 3 GB of swap filled, and throughput decayed from 67 to 43 samples/s over
# five epochs while 67 GB stayed available -- fragmentation, not exhaustion.
# Routing anything over 1 MB straight to mmap sidesteps the heap entirely and
# measured 90 samples/s against 52 on the identical step of an identical seed.
export MALLOC_MMAP_THRESHOLD_=1048576
export MALLOC_TRIM_THRESHOLD_=1048576
export MALLOC_ARENA_MAX=2

EPOCHS=25
COMMON="data.source=transfuser model.aux.bev_classes=3 train.epochs=$EPOCHS"

run () {
    local name=$1; shift
    local done_epoch
    # "best.pth exists" used to mean "already trained", which is wrong the
    # moment a run is interrupted: best.pth is written from the first epoch
    # onwards, so a run killed at epoch 5 of 25 looked complete and was skipped.
    # Completion is now read from the log, and anything short of it resumes.
    done_epoch=$(grep -c "^epoch " "$HOME/logs/train_$name.log" 2>/dev/null || echo 0)
    if [ "$done_epoch" -ge "$EPOCHS" ]; then
        echo "$(date +%T) $name already trained ($done_epoch epochs), skipping"
        return 0
    fi
    if [ "$done_epoch" -gt 0 ]; then
        echo "$(date +%T) === resuming $name from epoch $done_epoch ==="
    else
        echo "$(date +%T) === training $name ==="
    fi
    python -u -m egca.training.train \
        --config configs/egca.yaml --seed 0 --resume \
        --set $COMMON "train.ckpt_dir=checkpoints/$name" "$@" \
        >> "$HOME/logs/train_$name.log" 2>&1
    echo "$(date +%T) $name exit=$?"
}

run tf_base
run tf_query  model.decoder.readout=query model.fusion.goal_injection=fusion

# tf_regnet is deliberately not run here.  Each rung is about 22 h -- the loader
# is bound by random reads over 234 GB against 88 GB of page cache, not by the
# GPU -- and the two rungs above are the ones Chapter 5 cannot be written
# without: one number comparable to the published baselines, and one ablation
# that attributes part of it to the mechanism the thesis proposes.  The backbone
# swap is orthogonal to both, so it can be run afterwards without invalidating
# anything already measured:
#
#   bash tools/run_ladder_tf.sh   # after adding: run tf_regnet model.camera.backbone=regnet_y_3_2gf

echo "LADDER_TF_DONE"
printf '%-12s %9s %11s\n' run best_val final_train
for n in tf_base tf_query; do
    f="$HOME/logs/train_$n.log"
    [ -f "$f" ] || continue
    b=$(grep -o "val -\?[0-9.]* (wp [0-9.]* m)" "$f" | grep -o "wp [0-9.]*" \
        | cut -d" " -f2 | sort -n | head -1)
    t=$(grep -o "train -\?[0-9.]* (wp [0-9.]* m)" "$f" | grep -o "wp [0-9.]*" \
        | cut -d" " -f2 | tail -1)
    printf '%-12s %9s %11s\n' "$n" "$b" "$t"
done
