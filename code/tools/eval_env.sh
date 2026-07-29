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

# `leaderboard_agent.py` is loaded by the evaluator as a standalone module from
# its path, so it has no parent package and must import `egca.*` absolutely --
# hence $EGCA_ROOT on the path.  $CARLA_ROOT/carla/agents is needed because the
# route planner is imported as `agents.navigation.*`.
export PYTHONPATH=$CARLA_ROOT/carla:$CARLA_ROOT/carla/agents:$LEADERBOARD_ROOT:$SCENARIO_RUNNER_ROOT:$EGCA_ROOT:$PYTHONPATH

export CHALLENGE_TRACK_CODENAME=SENSORS
export REPETITIONS=1
export DEBUG_CHALLENGE=0
