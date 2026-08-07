# Environment for the official CARLA leaderboard evaluation.  Source, do not run:
#
#     source tools/eval_env.sh
#
# The `leaderboard` and `scenario_runner` copies are the ones vendored in the
# TransFuser repository: they are the exact versions the published Longest6
# numbers were produced with, so the scorer matches and the comparison in
# Table 5-3 is meaningful.  They target CARLA 0.9.10 and need
# tools/compat/patch_leaderboard.sh applied once to run on 0.9.14.
#
# Every path can be overridden from the caller's environment; the defaults are
# the layout on the training machine.
: "${CONDA_SH:=$HOME/miniconda3/etc/profile.d/conda.sh}"
: "${CONDA_ENV:=egca}"
[ -f "$CONDA_SH" ] && . "$CONDA_SH" && conda activate "$CONDA_ENV"

: "${CARLA_ROOT:=$HOME/carla_api}"
: "${LEADERBOARD_ROOT:=$HOME/transfuser/leaderboard}"
: "${SCENARIO_RUNNER_ROOT:=$HOME/transfuser/scenario_runner}"
: "${EGCA_ROOT:=$HOME/thesis/code}"
export CARLA_ROOT LEADERBOARD_ROOT SCENARIO_RUNNER_ROOT EGCA_ROOT

export ROUTES6=$LEADERBOARD_ROOT/data/longest6

# Longest6 is not the stock leaderboard protocol, and using the stock evaluator
# for it silently produces numbers that cannot be compared with any published
# Longest6 result.  TransFuser's own README says the benchmark runs through the
# `*_local.py` copies, which differ in two ways that push the score in opposite
# directions:
#
#   * dense traffic (route_scenario_local.py) -- harder, lowers route completion
#   * no penalty for stop-sign infractions (statistics_manager_local.py)
#
# Measured on this project's own autopilot validation, the two are worth a great
# deal.  Scored the stock way: RC 87.94, IS 0.757, DS 67.52.  Removing only the
# stop penalty: IS 0.916, DS 82.82, against their published 0.89 and 74.49.  The
# remaining gap is the traffic density -- our RC came out above theirs because
# the benchmark was running lighter, not because the agent drove better.
: "${EVALUATOR:=$LEADERBOARD_ROOT/leaderboard/leaderboard_evaluator_local.py}"
export EVALUATOR

# `leaderboard_agent.py` is loaded by the evaluator as a standalone module from
# its path, so it has no parent package and must import `egca.*` absolutely --
# hence $EGCA_ROOT on the path.  $CARLA_ROOT/carla/agents is needed because the
# route planner is imported as `agents.navigation.*`.
export PYTHONPATH=$CARLA_ROOT/carla:$CARLA_ROOT/carla/agents:$LEADERBOARD_ROOT:$SCENARIO_RUNNER_ROOT:$EGCA_ROOT:${PYTHONPATH:-}

export CHALLENGE_TRACK_CODENAME=SENSORS
export REPETITIONS=1
export DEBUG_CHALLENGE=0

# Cap the CPU thread pools.  Torch sizes its intra-op pool from the core count,
# so an unconfigured agent opens ~130 threads; four of them next to four
# simulators put 500+ threads on 32 cores and the load average sat near 60 while
# the GPU idled at 0-15%.  The policy runs on the GPU -- these threads are for
# pillarisation and image normalisation -- so a small pool per slot is enough,
# and leaving cores for CarlaUE4 is what actually determines throughput.
: "${THREADS_PER_SLOT:=4}"
export OMP_NUM_THREADS=$THREADS_PER_SLOT
export MKL_NUM_THREADS=$THREADS_PER_SLOT
export OPENBLAS_NUM_THREADS=$THREADS_PER_SLOT
export NUMEXPR_NUM_THREADS=$THREADS_PER_SLOT
