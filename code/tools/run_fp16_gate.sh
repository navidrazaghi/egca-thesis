#!/usr/bin/env bash
# Run the pre-queue check under real mixed precision, as soon as there is room.
#
# The CPU pass cleared the logic but says nothing about fp16: autocast on CPU is
# either off or bfloat16, and bfloat16 keeps the fp32 exponent range, so the one
# failure mode being looked for -- an accumulator running past 65504 and turning
# the loss into NaN -- cannot occur there.  The linear attention sums thousands
# of strictly positive terms, which is exactly the shape of that failure, and
# the query readout is new code on that path.
#
# It waits for training to release the card, then for enough free memory to fit
# alongside the evaluation rather than displacing it: four CARLA instances leave
# roughly 24 GB, and this needs about 4.
# Paths are resolved from this script's own location so a fresh
# checkout works; the copies that produced the recorded results lived in
# ~/scripts on the evaluation machine and differed only in that line.
set -u

cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
LOG="$HOME/logs/fp16_gate.log"
: > "$LOG"

say () { echo "$(date -u '+%H:%M:%S') $*" | tee -a "$LOG"; }

say "waiting for the training run to finish"
while tmux has-session -t aug 2>/dev/null; do sleep 120; done
say "training done"

free_mib () {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1
}

for attempt in 1 2 3; do
    say "attempt $attempt: waiting for 6 GB of free GPU memory"
    for _ in $(seq 1 60); do            # up to 2 hours
        f=$(free_mib)
        [ "${f:-0}" -ge 6000 ] && break
        sleep 120
    done
    f=$(free_mib)
    if [ "${f:-0}" -lt 6000 ]; then
        say "only ${f} MiB free, giving up on this attempt"
        continue
    fi
    say "${f} MiB free, running the check on cuda"
    source tools/eval_env.sh
    if OMP_NUM_THREADS=4 python tools/check_train_step.py \
            --config configs/egca.yaml --device cuda >> "$LOG" 2>&1; then
        say "FP16 GATE PASSED -- the queue is clear to start"
        exit 0
    fi
    say "attempt $attempt failed; see the traceback above"
    sleep 600
done

say "FP16 GATE FAILED -- do not start the queue until this is understood"
exit 1
