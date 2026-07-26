#!/usr/bin/env bash
# Supervised, resumable collection for one town.
#
# The CARLA server is not stable over a multi-hour run: spawning and destroying
# tens of thousands of actors leaks memory until the process dies with SIGSEGV
# (observed: three of four simulators died after 3-6 hours, exit code 139).
# No amount of client-side care prevents that, so the simulator is treated as a
# disposable resource:
#
#   * the container is recycled every CHUNK routes, before the leak grows large
#     enough to crash it;
#   * if it died anyway, it is recreated;
#   * collection resumes at the next unused route index, so nothing is lost and
#     nothing is overwritten;
#   * the weather rotation follows the route index, so resuming keeps the
#     14-condition cycle intact.
#
# Usage:
#   bash tools/collect_town.sh Town02 2004 8104 75
#
#   town  rpc-port  tm-port  target-routes
set -u

TOWN=${1:?town}
PORT=${2:?rpc port}
TMPORT=${3:?traffic manager port}
TARGET=${4:-75}
CHUNK=${5:-15}

NAME="carla-${PORT}"
OUT="dataset/${TOWN}"
ICD=/usr/share/vulkan/icd.d/nvidia_icd.json

start_server() {
    docker rm -f "$NAME" >/dev/null 2>&1
    docker run -d --name "$NAME" --gpus all --net=host \
        -e NVIDIA_DRIVER_CAPABILITIES=all \
        -v "$ICD":"$ICD":ro \
        carlasim/carla:0.9.14 /bin/bash CarlaUE4.sh \
        -RenderOffScreen -nosound -quality-level=Epic \
        -carla-rpc-port="$PORT" >/dev/null
    echo "$(date +%H:%M:%S) [$TOWN] started $NAME, waiting for shaders"
    sleep 180
}

next_index() {
    # Highest existing route id + 1.  Two traps here, both of which produced a
    # resumed run that pointed at an already-occupied directory:
    #   * bash arithmetic reads a leading zero as octal, so $((013 + 1)) is 12,
    #     not 14 -- hence the explicit 10# base prefix;
    #   * a digit in the parent path would be picked up by a greedy match, so
    #     the id is extracted from the basename only.
    local last=-1 n
    for d in "$OUT"/route_*; do
        [ -d "$d" ] || continue
        n=$(basename "$d" | sed -n 's/^route_\([0-9]\{1,\}\)_.*/\1/p')
        if [ -n "$n" ]; then
            n=$((10#$n))
            [ "$n" -gt "$last" ] && last=$n
        fi
    done
    echo $(( last + 1 ))
}

route_count() {
    ls -d "$OUT"/route_* 2>/dev/null | wc -l
}

while true; do
    done_n=$(route_count)
    if [ "$done_n" -ge "$TARGET" ]; then
        echo "$(date +%H:%M:%S) [$TOWN] complete: $done_n routes"
        break
    fi
    next=$(next_index)
    want=$(( TARGET - done_n ))
    [ "$want" -gt "$CHUNK" ] && want=$CHUNK

    # a fresh simulator for every chunk
    start_server

    echo "$(date +%H:%M:%S) [$TOWN] $done_n/$TARGET routes; collecting $want from index $next"
    PYTHONPATH=. python -m egca.carla_sim.collect_data \
        --town "$TOWN" --port "$PORT" --tm-port "$TMPORT" \
        --routes "$want" --first-route "$next" \
        --max-seconds 300 --scenario-every-m 100 \
        --seed "$next" --output "$OUT"
    rc=$?
    echo "$(date +%H:%M:%S) [$TOWN] chunk finished (exit $rc)"
    sleep 5
done

docker rm -f "$NAME" >/dev/null 2>&1
echo "$(date +%H:%M:%S) [$TOWN] done, simulator removed"
