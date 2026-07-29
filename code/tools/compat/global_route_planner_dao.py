"""Compatibility shim for running the leaderboard 1.0 stack on CARLA 0.9.14.

The vendored `leaderboard` and `scenario_runner` used to produce the published
Longest6 numbers were written against CARLA 0.9.10, where the route planner had
a two-step API:

    dao = GlobalRoutePlannerDAO(world.get_map(), resolution)
    grp = GlobalRoutePlanner(dao)
    grp.setup()

CARLA 0.9.12 folded the data-access object into the planner itself, so the
module simply no longer exists and every import of it fails.  Rather than
editing four files inside two third-party repositories -- which would make the
evaluation stack impossible to reproduce from their published sources -- this
file restores the old surface on top of the new implementation.

Install it into the simulator's own agents package:

    cp tools/compat/global_route_planner_dao.py \
       $CARLA_ROOT/carla/agents/navigation/

It is inert for any code that does not import it.
"""
from agents.navigation.global_route_planner import GlobalRoutePlanner


class GlobalRoutePlannerDAO(object):
    """Carries the two arguments the modern planner takes directly."""

    def __init__(self, wmap, sampling_resolution=1.0):
        self._wmap = wmap
        self._sampling_resolution = sampling_resolution

    def get_topology(self):
        return self._wmap.get_topology()

    def get_waypoint(self, location):
        return self._wmap.get_waypoint(location)

    def get_resolution(self):
        return self._sampling_resolution


_orig_init = GlobalRoutePlanner.__init__


def _init(self, wmap, sampling_resolution=None):
    """Accept either the new (map, resolution) call or a legacy DAO."""
    if isinstance(wmap, GlobalRoutePlannerDAO):
        wmap, sampling_resolution = wmap._wmap, wmap._sampling_resolution
    if sampling_resolution is None:
        sampling_resolution = 1.0
    _orig_init(self, wmap, sampling_resolution)


GlobalRoutePlanner.__init__ = _init

if not hasattr(GlobalRoutePlanner, "setup"):
    # the modern planner builds its graph in __init__, so setup() is a no-op
    GlobalRoutePlanner.setup = lambda self: None
