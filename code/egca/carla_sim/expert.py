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
    KP_BRAKE = 0.25            # proportional brake gain (per m/s of overspeed)
    MAX_THROTTLE = 0.75
    BRAKE_MARGIN = 0.5         # brake if v exceeds target by this much (m/s)
    STOP_SPEED = 0.3           # below this the vehicle counts as stopped (m/s)
    MAX_DECEL = 4.0            # emergency deceleration used for hazard reach
    COMFORT_DECEL = 2.0        # deceleration used to plan stops (m/s^2)
    STOP_LINE_MARGIN = 4.0     # stop this far before a stop line (m)
    # cornering
    A_LAT = 3.0                # lateral acceleration budget (m/s^2)
    CURVE_LOOKAHEAD = 15.0     # distance over which curvature is anticipated (m)
    # deadlock escape
    CREEP_AFTER_STEPS = 80     # 8 s at 10 Hz standing still with no obligation
    CREEP_SPEED = 1.5          # m/s
    CREEP_MIN_GAP = 4.0        # only creep if there is at least this much space
    OFF_ROUTE_DIST = 6.0       # beyond this the plan cursor is re-localized (m)
    # lateral (pure pursuit)
    FULL_LOCK_ANGLE = 45.0     # heading error that saturates the steering (deg)
    STEER_SMOOTH = 0.6         # weight of the new command in the low-pass
    # hazards
    CORRIDOR_HALF_WIDTH = 1.1  # lateral gate for vehicles, plus their half-width
    WALKER_HALF_WIDTH = 1.8    # wider gate for pedestrians (m)
    STOP_SIGN_MARGIN = 2.0     # slack added to the stop-sign trigger volume (m)
    # car following
    STANDSTILL_GAP = 5.0       # gap kept at zero speed (m)
    HEADWAY = 1.5              # time headway (s)
    GAP_GAIN = 0.5             # how strongly a gap error changes the target speed

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
        self.cum = self._arc_lengths()
        self.idx = 0
        self.prev_idx = 0
        self.prev_steer = 0.0
        self.cleared_stops = set()
        self.stopped_steps = 0
        self.off_route = 0.0
        self.stop_signs = list(world.get_actors().filter("*traffic.stop*"))
        self.lights_on_route = self._index_traffic_lights()

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

    def _arc_lengths(self):
        """Cumulative arc length of the plan, so distances along the route are
        O(1) lookups instead of repeated summations."""
        cum = [0.0]
        for i in range(len(self.plan) - 1):
            cum.append(cum[-1] + math.dist(self.plan[i][:2], self.plan[i + 1][:2]))
        return cum

    def _index_traffic_lights(self):
        """Attach every traffic light to the plan index of its stop line.

        Reacting only inside a light's trigger volume (what `is_at_traffic_light`
        reports) means the expert discovers a red light at the last moment and
        stops abruptly.  Pre-indexing the lights along the route lets it brake
        smoothly from a distance, which is both realistic and what the imitation
        labels should contain.
        """
        out = []
        for tl in self.world.get_actors().filter("*traffic_light*"):
            try:
                stops = tl.get_affected_lane_waypoints()
            except (AttributeError, RuntimeError):
                continue
            for wp in stops:
                lx, ly = wp.transform.location.x, wp.transform.location.y
                best, best_d = None, 4.0
                for i, p in enumerate(self.plan):
                    d = math.dist((p[0], p[1]), (lx, ly))
                    if d < best_d:
                        best, best_d = i, d
                if best is not None:
                    out.append((best, tl))
        return sorted(out, key=lambda t: t[0])

    @property
    def route_length(self):
        return self.cum[-1] if self.cum else 0.0

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
        """Move the plan cursor to the closest point within a forward window.

        If the vehicle ends up far from the plan -- it overshot a junction, was
        pushed off the lane by traffic or by the injected steering noise -- the
        cursor is re-localized over the whole plan.  Without this the cursor
        lags behind, the look-ahead point falls *behind* the vehicle, the
        pure-pursuit angle saturates at full lock and the vehicle stalls against
        a kerb: the observed symptom was a route stopping dead at exactly the
        same 57% of its length on every attempt.
        """
        ego = (tf.location.x, tf.location.y)
        best, best_d = self.idx, float("inf")
        for i in range(self.idx, min(self.idx + 80, len(self.plan))):
            d = math.dist((self.plan[i][0], self.plan[i][1]), ego)
            if d < best_d:
                best, best_d = i, d
        if best_d > self.OFF_ROUTE_DIST:
            for i in range(len(self.plan)):
                d = math.dist((self.plan[i][0], self.plan[i][1]), ego)
                if d < best_d:
                    best, best_d = i, d
        self.idx = best
        self.off_route = best_d

    def _point_at(self, distance, tf, min_forward=0.5):
        """Ego-frame point `distance` metres ahead along the plan.

        The returned point is guaranteed to lie in front of the vehicle: after
        walking the requested arc length, the cursor keeps advancing while the
        candidate is still behind.  A target behind the car would otherwise
        saturate the steering command instead of steering towards the route.
        """
        acc, i = 0.0, self.idx
        while i + 1 < len(self.plan) and acc < distance:
            acc += math.dist(self.plan[i][:2], self.plan[i + 1][:2])
            i += 1
        for _ in range(200):
            x, y = self._to_ego(self.plan[i][0], self.plan[i][1], tf)
            if x > min_forward or i + 1 >= len(self.plan):
                return x, y
            i += 1
        return self._to_ego(self.plan[i][0], self.plan[i][1], tf)

    # ------------------------------------------------- policy-facing outputs
    def nav_command(self, speed=None):
        """Discrete high-level command for the next junction.

        The announcement window scales with speed (a fixed 15 m window means a
        slow vehicle spends most of its frames announcing a turn, which skews the
        command distribution of the dataset), with a floor so the command is
        always given early enough to be actionable.
        """
        horizon = COMMAND_HORIZON if speed is None else max(8.0, 2.0 * speed)
        i = self.idx
        while (i + 1 < len(self.plan)
               and self.cum[i] - self.cum[self.idx] < horizon):
            if self.plan[i][3] != 3:            # a turn is coming up
                return self.plan[i][3]
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
        """Physical cornering limit instead of hand-picked factors.

        The heading change per unit arc length is the path curvature k = 1/R, and
        a lateral acceleration budget A_LAT gives v_max = sqrt(A_LAT / k).  The
        maximum curvature over the next CURVE_LOOKAHEAD metres is used, so the
        vehicle is already slow when it reaches the corner.
        """
        i0 = self.idx
        kappa_max = 0.0
        i = i0
        while (i + 1 < len(self.plan)
               and self.cum[i] - self.cum[i0] < self.CURVE_LOOKAHEAD):
            ds = math.dist(self.plan[i][:2], self.plan[i + 1][:2])
            if ds > 1e-3:
                dpsi = abs(_yaw_diff(self.plan[i][2], self.plan[i + 1][2]))
                kappa_max = max(kappa_max, math.radians(dpsi) / ds)
            i += 1
        if kappa_max < 1e-3:                            # essentially straight
            return self.base_speed
        return min(self.base_speed, math.sqrt(self.A_LAT / kappa_max))

    # --------------------------------------------------------------- hazards
    def _light_distance(self, speed):
        """Distance along the route to the stop line of the next red/yellow
        light, or None if the way is clear.

        If the closest light on the route is green, no constraint is returned:
        the vehicle will have passed it before reaching any light behind it.
        """
        if carla is None:
            return None
        # a light whose trigger volume we are already inside
        if self.vehicle.is_at_traffic_light():
            tl = self.vehicle.get_traffic_light()
            if tl is not None and tl.get_state() != carla.TrafficLightState.Green:
                return 0.0
        horizon = max(15.0, speed ** 2 / (2.0 * self.COMFORT_DECEL) + 2.0 * speed)
        for pi, tl in self.lights_on_route:
            if pi <= self.idx:
                continue
            d = self.cum[pi] - self.cum[self.idx]
            if d > horizon:
                return None
            try:
                green = tl.get_state() == carla.TrafficLightState.Green
            except RuntimeError:
                return None
            return None if green else d
        return None

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

    def _is_oncoming(self, act, tf):
        """True if the actor's heading is roughly opposite to ours."""
        try:
            return abs(_yaw_diff(tf.rotation.yaw,
                                 act.get_transform().rotation.yaw)) > 120.0
        except RuntimeError:
            return False

    def _actor_hazard(self, tf, speed):
        """Closest vehicle/pedestrian inside a speed-dependent forward corridor.

        The reach must be clearly larger than the desired following gap,
        otherwise a leader is first noticed when it is already at the target
        distance and the car-following law never gets room to act (which is
        exactly why the first version of this expert never followed anybody).

        Returns (distance, speed) of the blocker and the number of vehicles
        loosely ahead, the latter purely as a diagnostic.
        """
        gap_desired = self.STANDSTILL_GAP + self.HEADWAY * speed
        reach = float(np.clip(2.0 * gap_desired + 5.0, 12.0, 40.0))
        n_ahead = 0
        best_d, best_v = None, None
        for act in self.world.get_actors():
            if act.id == self.vehicle.id:
                continue
            is_walker = act.type_id.startswith("walker.")
            if not (is_walker or act.type_id.startswith("vehicle.")):
                continue
            loc = act.get_location()
            fwd, lat = self._to_ego(loc.x, loc.y, tf)
            if not is_walker and 0.5 < fwd < 20.0 and abs(lat) < 5.0:
                n_ahead += 1              # diagnostic: anything loosely ahead
            # Only the actor's half-*width* widens the lateral gate.  Using its
            # half-length (extent.x, ~2.5 m for a car) would inflate the gate to
            # ~4 m and flag oncoming and adjacent-lane traffic as blockers, which
            # makes the expert brake almost continuously on a two-lane road.
            try:
                margin = act.bounding_box.extent.y
            except (AttributeError, RuntimeError):
                margin = 0.9
            gate = (self.WALKER_HALF_WIDTH if is_walker
                    else self.CORRIDOR_HALF_WIDTH) + margin
            # An oncoming vehicle in the opposite lane is not a blocker; only
            # treat it as one if it is nearly head-on inside our own lane.
            if not is_walker and self._is_oncoming(act, tf) and abs(lat) > 1.2:
                continue
            if 0.5 < fwd < reach and abs(lat) < gate:
                if best_d is None or fwd < best_d:
                    v = act.get_velocity()
                    best_d, best_v = fwd, math.hypot(v.x, v.y)
        return best_d, best_v, n_ahead

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
        light_d = self._light_distance(speed)
        stop_sign = self._stop_sign_hazard(tf, speed)
        lead_d, lead_v, n_ahead = self._actor_hazard(tf, speed)
        if light_d is not None:
            # brake profile that reaches zero STOP_LINE_MARGIN before the line
            target = min(target, math.sqrt(
                2.0 * self.COMFORT_DECEL
                * max(0.0, light_d - self.STOP_LINE_MARGIN)))
        red_light = light_d is not None and light_d < 2.0 * self.STOP_LINE_MARGIN
        if stop_sign:
            target = 0.0
        elif lead_d is not None:
            # Car-following law: aim for a 1.5 s time headway on top of a 5 m
            # standstill gap.  At the desired gap the target equals the leader's
            # speed; a larger gap allows closing in, a smaller one brakes below
            # the leader.  (The previous form capped the speed at (d-5)/2
            # regardless of the leader, so following a car 10 m ahead crawled at
            # 2.5 m/s and the whole dataset became a traffic jam.)
            gap_desired = self.STANDSTILL_GAP + self.HEADWAY * speed
            gap_err = lead_d - gap_desired
            follow = (lead_v or 0.0) + self.GAP_GAIN * gap_err
            target = min(target, max(0.0, follow))

        # ---- deadlock escape.  A purely reactive expert has no way out of a
        # standoff (an unprotected left turn with continuous oncoming traffic, or
        # a queue whose leader is itself stuck): it waits forever and the route
        # is thrown away.  After CREEP_AFTER seconds of standing still with no
        # legal obligation to stop and some free space ahead, a slow creep is
        # allowed, which resolves the great majority of these standoffs.
        self.stopped_steps = self.stopped_steps + 1 if speed < 0.3 else 0
        creeping = False
        if (self.stopped_steps > self.CREEP_AFTER_STEPS
                and light_d is None and not stop_sign
                and (lead_d is None or lead_d > self.CREEP_MIN_GAP)):
            target = max(target, self.CREEP_SPEED)
            creeping = True

        # ---- longitudinal control.  Braking is proportional: an on/off brake
        # produced a saw-tooth of hard stops and re-accelerations that dragged
        # the average speed of the whole dataset down.
        err = target - speed
        if target < self.STOP_SPEED and speed < 0.5:
            throttle, brake = 0.0, 1.0                   # hold at a stop
        elif err < -self.BRAKE_MARGIN:
            throttle = 0.0
            brake = float(np.clip(self.KP_BRAKE * (-err), 0.15, 1.0))
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
            "command": self.nav_command(speed),
            "goal": self.sparse_goal(),
            "target_speed": target,
            "red_light": bool(red_light),
            "stop_sign": bool(stop_sign),
            "lead_distance": -1.0 if lead_d is None else float(lead_d),
            "n_ahead": int(n_ahead),
            "creeping": bool(creeping),
            "off_route_m": float(self.off_route),
            "progress": self.progress(),
            # advance along the plan since the previous step, used by the
            # collector to detect a permanent deadlock
            "progress_delta": (self.idx - self.prev_idx) / max(len(self.plan) - 1, 1),
        }
        return throttle, float(steer), brake, info
