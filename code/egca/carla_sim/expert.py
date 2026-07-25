"""Privileged rule-based expert used to generate the imitation dataset
(Sec. 5-1, "learning by cheating" [15]).

The expert has full access to the simulator state and drives by rules, so the
quality of the whole imitation dataset -- and therefore the ceiling of the
learned policy -- is set here.  It handles:

  * a dense global route produced by CARLA's GlobalRoutePlanner, with the
    RoadOption of each point providing the discrete navigation command;
  * traffic lights and stop signs (a policy trained from an expert that runs
    red lights can never obtain a good infraction score);
  * posted speed limits and curvature-dependent speed reduction;
  * privileged hazard detection for vehicles and pedestrians in a forward
    corridor whose length grows with the current speed.

Frame convention (used consistently by the sensors, the labels and the
controller): ego frame with x forward and **y to the left** (right-handed).
CARLA's world frame is left-handed, so every lateral quantity taken from the
simulator is negated exactly once.  A positive CARLA `steer` turns right, hence
a target on the left (y > 0) maps to a negative steering command.

NOTE: waypoint labels are *not* produced here.  They are reconstructed after
collection from the recorded ego poses (see `build_labels.py`), which makes them
the true future trajectory -- including deceleration in front of a red light and
the recovery manoeuvre after an injected steering perturbation.
"""
import math

import numpy as np

try:
    import carla
except ImportError:                                   # pragma: no cover
    carla = None

try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    from agents.navigation.local_planner import RoadOption
    _HAS_AGENTS = True
except ImportError:                                   # pragma: no cover
    _HAS_AGENTS = False

GOAL_DISTANCE = 8.0        # look-ahead of the sparse navigation goal (m)
PLAN_RESOLUTION = 1.0      # spacing of the dense global plan (m)
COMMAND_HORIZON = 15.0     # distance over which a turn command is announced (m)

# RoadOption -> command index expected by MeasurementEncoder
# (0 left, 1 right, 2 straight, 3 lane-follow)
COMMAND_MAP = {"LEFT": 0, "RIGHT": 1, "STRAIGHT": 2, "LANEFOLLOW": 3,
               "CHANGELANELEFT": 3, "CHANGELANERIGHT": 3, "VOID": 3}


def _yaw_diff(a, b):
    """Signed smallest difference between two headings, in degrees."""
    return (b - a + 180.0) % 360.0 - 180.0


class PrivilegedExpert:
    """Rule-based expert driving along a global route.

    Parameters
    ----------
    world : carla.World
    vehicle : carla.Vehicle          ego vehicle (already spawned)
    targets : list of carla.Location destinations, visited in order
    base_speed : float               desired cruise speed in m/s
    """

    # longitudinal
    KP_SPEED = 0.55
    MAX_THROTTLE = 0.75
    BRAKE_MARGIN = 1.0         # brake if v exceeds target by this much (m/s)
    STOP_SPEED = 0.3           # below this the vehicle counts as stopped (m/s)
    MAX_DECEL = 4.0            # assumed comfortable deceleration (m/s^2)
    # lateral (pure pursuit)
    FULL_LOCK_ANGLE = 45.0     # heading error that saturates the steering (deg)
    STEER_SMOOTH = 0.6         # weight of the new command in the low-pass
    # hazards
    CORRIDOR_HALF_WIDTH = 1.6  # lateral gate for vehicles (m)
    WALKER_HALF_WIDTH = 2.2    # wider gate for pedestrians (m)
    STOP_SIGN_MARGIN = 2.0     # slack added to the stop-sign trigger volume (m)

    def __init__(self, world, vehicle, targets, base_speed=6.0):
        if not _HAS_AGENTS:
            raise ImportError(
                "CARLA's `agents` package is required for the global route "
                "planner. Copy it out of the simulator, e.g.\n"
                "  docker cp carla-2000:/home/carla/PythonAPI/carla/agents .")
        self.world = world
        self.vehicle = vehicle
        self.map = world.get_map()
        self.base_speed = base_speed
        self.plan = self._build_plan(targets)
        self.idx = 0
        self.prev_idx = 0
        self.prev_steer = 0.0
        self.cleared_stops = set()
        self.stop_signs = list(world.get_actors().filter("*traffic.stop*"))

    # ------------------------------------------------------------ route plan
    def _build_plan(self, targets):
        """Dense plan as a list of (x, y, yaw_deg, command).  Consecutive
        target locations are chained, so a route can be arbitrarily long."""
        grp = GlobalRoutePlanner(self.map, PLAN_RESOLUTION)
        start = self.vehicle.get_location()
        plan = []
        for tgt in targets:
            for wp, opt in grp.trace_route(start, tgt):
                tf = wp.transform
                plan.append((tf.location.x, tf.location.y, tf.rotation.yaw,
                             COMMAND_MAP.get(str(opt).split(".")[-1], 3)))
            start = tgt
        return plan

    @property
    def route_length(self):
        return sum(math.dist(self.plan[i][:2], self.plan[i + 1][:2])
                   for i in range(len(self.plan) - 1))

    def done(self):
        return self.idx >= len(self.plan) - 2

    def progress(self):
        return self.idx / max(len(self.plan) - 1, 1)

    # ------------------------------------------------------------- geometry
    def _to_ego(self, x, y, tf):
        """World (x, y) -> (forward, left) in the ego frame."""
        yaw = math.radians(tf.rotation.yaw)
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = x - tf.location.x, y - tf.location.y
        return c * dx + s * dy, -(-s * dx + c * dy)

    def _advance(self, tf):
        """Move the plan cursor to the closest point within a forward window."""
        best, best_d = self.idx, float("inf")
        for i in range(self.idx, min(self.idx + 80, len(self.plan))):
            d = math.dist((self.plan[i][0], self.plan[i][1]),
                          (tf.location.x, tf.location.y))
            if d < best_d:
                best, best_d = i, d
        self.idx = best

    def _point_at(self, distance, tf):
        """Ego-frame point `distance` metres ahead along the plan."""
        acc, i = 0.0, self.idx
        while i + 1 < len(self.plan) and acc < distance:
            acc += math.dist(self.plan[i][:2], self.plan[i + 1][:2])
            i += 1
        return self._to_ego(self.plan[i][0], self.plan[i][1], tf)

    # ------------------------------------------------- policy-facing outputs
    def nav_command(self):
        """Discrete high-level command, announced COMMAND_HORIZON ahead."""
        acc, i = 0.0, self.idx
        while i + 1 < len(self.plan) and acc < COMMAND_HORIZON:
            if self.plan[i][3] != 3:            # a turn is coming up
                return self.plan[i][3]
            acc += math.dist(self.plan[i][:2], self.plan[i + 1][:2])
            i += 1
        return self.plan[min(i, len(self.plan) - 1)][3]

    def sparse_goal(self):
        """Sparse navigation target ~GOAL_DISTANCE ahead along the global route,
        in the ego frame.  Deliberately taken from the route and never from the
        expert trajectory: feeding a future ego position back as an input would
        leak the label the decoder has to regress."""
        return self._point_at(GOAL_DISTANCE, self.vehicle.get_transform())

    # ---------------------------------------------------------- speed limits
    def _posted_speed(self):
        limit = self.vehicle.get_speed_limit()          # km/h, 0 right at spawn
        return (limit / 3.6) if limit and limit > 1.0 else 8.33   # 30 km/h

    def _curvature_speed(self):
        """Slow down for the upcoming curve: the sharper the heading change over
        the next 20 m, the lower the allowed speed."""
        i0 = self.idx
        acc, i = 0.0, i0
        while i + 1 < len(self.plan) and acc < 20.0:
            acc += math.dist(self.plan[i][:2], self.plan[i + 1][:2])
            i += 1
        turn = abs(_yaw_diff(self.plan[i0][2], self.plan[i][2]))
        if turn < 10.0:
            return self.base_speed
        if turn < 45.0:
            return 0.7 * self.base_speed
        return 0.45 * self.base_speed                   # junction turn

    # --------------------------------------------------------------- hazards
    def _light_hazard(self):
        """True while a red or yellow light applies to the ego vehicle."""
        if carla is None or not self.vehicle.is_at_traffic_light():
            return False
        light = self.vehicle.get_traffic_light()
        if light is None:
            return False
        return light.get_state() != carla.TrafficLightState.Green

    def _stop_sign_hazard(self, tf, speed):
        """True until the vehicle has come to a full stop for the stop sign
        whose trigger volume it is currently inside."""
        for sign in self.stop_signs:
            if sign.id in self.cleared_stops:
                continue
            try:
                vol = sign.trigger_volume
                centre = sign.get_transform().transform(vol.location)
                radius = max(vol.extent.x, vol.extent.y) + self.STOP_SIGN_MARGIN
            except (AttributeError, RuntimeError):
                continue
            fwd, lat = self._to_ego(centre.x, centre.y, tf)
            if -2.0 < fwd < radius + 4.0 and abs(lat) < radius:
                if speed < self.STOP_SPEED:
                    self.cleared_stops.add(sign.id)     # obligation fulfilled
                    return False
                return True
        return False

    def _actor_hazard(self, tf, speed):
        """Closest vehicle/pedestrian inside a speed-dependent forward corridor.
        Returns (distance, speed) of the blocker, or (None, None)."""
        reach = float(np.clip(4.0 + 1.2 * speed
                              + speed ** 2 / (2.0 * self.MAX_DECEL), 6.0, 30.0))
        best_d, best_v = None, None
        for act in self.world.get_actors():
            if act.id == self.vehicle.id:
                continue
            is_walker = act.type_id.startswith("walker.")
            if not (is_walker or act.type_id.startswith("vehicle.")):
                continue
            loc = act.get_location()
            fwd, lat = self._to_ego(loc.x, loc.y, tf)
            try:
                margin = max(act.bounding_box.extent.x,
                             act.bounding_box.extent.y)
            except (AttributeError, RuntimeError):
                margin = 1.0
            gate = (self.WALKER_HALF_WIDTH if is_walker
                    else self.CORRIDOR_HALF_WIDTH) + margin
            if 0.5 < fwd < reach and abs(lat) < gate:
                if best_d is None or fwd < best_d:
                    v = act.get_velocity()
                    best_d, best_v = fwd, math.hypot(v.x, v.y)
        return best_d, best_v

    # ------------------------------------------------------------------ step
    def step(self):
        """Advance the expert by one control step.

        Returns (throttle, steer, brake, info) where `info` carries everything
        the collector has to log: the discrete command, the sparse goal, the
        target speed and the individual hazard flags.
        """
        tf = self.vehicle.get_transform()
        vel = self.vehicle.get_velocity()
        speed = math.hypot(vel.x, vel.y)
        self.prev_idx = self.idx
        self._advance(tf)

        # ---- desired speed
        target = min(self.base_speed, 0.9 * self._posted_speed(),
                     self._curvature_speed())
        red_light = self._light_hazard()
        stop_sign = self._stop_sign_hazard(tf, speed)
        lead_d, lead_v = self._actor_hazard(tf, speed)
        if red_light or stop_sign:
            target = 0.0
        elif lead_d is not None:
            # keep a two-second gap, and match the leader's speed
            safe = max(0.0, (lead_d - 5.0) / 2.0)
            target = min(target, max(0.0, (lead_v or 0.0) - 0.5), safe)

        # ---- longitudinal control
        err = target - speed
        if target < self.STOP_SPEED:
            throttle, brake = 0.0, 1.0
        elif err < -self.BRAKE_MARGIN:
            throttle, brake = 0.0, 1.0
        else:
            throttle = float(np.clip(self.KP_SPEED * err, 0.0, self.MAX_THROTTLE))
            brake = 0.0

        # ---- lateral control (pure pursuit with a speed-dependent look-ahead)
        ld = float(np.clip(2.5 + 0.6 * speed, 3.0, 12.0))
        ax, ay = self._point_at(ld, tf)
        angle = math.degrees(math.atan2(ay, max(ax, 0.1)))
        raw = float(np.clip(-angle / self.FULL_LOCK_ANGLE, -1.0, 1.0))
        steer = self.STEER_SMOOTH * raw + (1.0 - self.STEER_SMOOTH) * self.prev_steer
        self.prev_steer = steer

        info = {
            "command": self.nav_command(),
            "goal": self.sparse_goal(),
            "target_speed": target,
            "red_light": bool(red_light),
            "stop_sign": bool(stop_sign),
            "lead_distance": -1.0 if lead_d is None else float(lead_d),
            "progress": self.progress(),
            # advance along the plan since the previous step, used by the
            # collector to detect a permanent deadlock
            "progress_delta": (self.idx - self.prev_idx) / max(len(self.plan) - 1, 1),
        }
        return throttle, float(steer), brake, info
