#!/usr/bin/env bash
# Make the leaderboard-1.0 stack run on CARLA 0.9.14.
#
# The evaluation uses the `leaderboard` and `scenario_runner` copies vendored in
# the TransFuser repository, because those are the versions the published
# Longest6 numbers were produced with -- using a different scorer would make the
# comparison meaningless.  They were written for CARLA 0.9.10, so two things
# need bridging.  Both are recorded here as a script rather than as hand-edits,
# so the whole evaluation stack can be rebuilt from the published sources.
#
#   1. `agents.navigation.global_route_planner_dao` was removed in 0.9.12; the
#      shim next to this file restores it (installed into the simulator's own
#      agents package).
#   2. `carla.Map.name` returned "Town01" in 0.9.10 and returns
#      "Carla/Maps/Town01" from 0.9.11 on, so the evaluator's sanity check that
#      the right town is loaded always fails.  Comparing the basename fixes it
#      without changing what is being checked.
#
# Usage:  bash tools/compat/patch_leaderboard.sh
# Requires CARLA_ROOT, LEADERBOARD_ROOT and SCENARIO_RUNNER_ROOT to be set.
set -eu

: "${CARLA_ROOT:?set CARLA_ROOT}"
: "${LEADERBOARD_ROOT:?set LEADERBOARD_ROOT}"
: "${SCENARIO_RUNNER_ROOT:?set SCENARIO_RUNNER_ROOT}"

HERE="$(cd "$(dirname "$0")" && pwd)"

# ---- 1. route-planner shim -------------------------------------------------
cp "$HERE/global_route_planner_dao.py" \
   "$CARLA_ROOT/carla/agents/navigation/global_route_planner_dao.py"
echo "installed GlobalRoutePlannerDAO shim"

# ---- 2. map-name comparison ------------------------------------------------
patch_name () {
    local f=$1
    [ -f "$f" ] || return 0
    if grep -q "os.path.basename(CarlaDataProvider.get_map().name)" "$f" \
       || grep -q "os.path.basename(world.get_map().name)" "$f"; then
        echo "already patched: $f"; return 0
    fi
    sed -i \
      -e 's#CarlaDataProvider.get_map().name != town#os.path.basename(CarlaDataProvider.get_map().name) != town#' \
      -e 's#world.get_map().name != self.town#os.path.basename(world.get_map().name) != self.town#' \
      "$f"
    grep -q '^import os' "$f" || sed -i '0,/^import /s//import os\nimport /' "$f"
    echo "patched map-name check: $f"
}

patch_name "$LEADERBOARD_ROOT/leaderboard/leaderboard_evaluator.py"
patch_name "$LEADERBOARD_ROOT/leaderboard/leaderboard_evaluator_local.py"
patch_name "$SCENARIO_RUNNER_ROOT/srunner/scenarioconfigs/openscenario_configuration.py"

echo "leaderboard stack patched for CARLA 0.9.14"
