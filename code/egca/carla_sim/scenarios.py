"""Scripted safety-critical scenarios injected along a collection route (Sec. 5-1).

Pedestrians and traffic driven by CARLA's own AI produce critical situations only
by accident, so their frequency cannot be controlled or reported.  This module
places *scripted* events at chosen points of the route, each armed when the ego
vehicle comes close and fired at a fixed distance, which makes the share of
safety-critical frames a measured quantity instead of an estimate.

Three scenario types, all of them standard in the CARLA scenario suite:

  pedestrian_crossing  a pedestrian waiting at the kerb steps onto the road when
                       the ego is TRIGGER_DIST away -- the canonical "sudden
                       pedestrian" event;
  lead_brake           a vehicle driving ahead in the ego lane brakes hard;
  junction_violation   a vehicle on a crossing lane enters the junction against
                       the right of way as the ego approaches it.

Every scenario reports its name while it is running, so `collect_data.py` can
label each recorded frame with the scenario that was active.

The design rule throughout: a scenario that cannot be placed (no navigable
pavement, no junction nearby, spawn point occupied) is skipped silently.  A data
collection that runs for hours must never die because one event could not be
staged.
"""
import math
import random

try:
    import carla
except ImportError:                                   # pragma: no cover
    carla = None

KINDS = ("pedestrian_crossing", "lead_brake", "junction_violation")

ARM_DIST = 60.0            # start staging the scenario at this distance (m)
TRIGGER_DIST = 14.0        # fire it when the ego is this close (m)
LEAD_TRIGGER_DIST = 13.0   # ... or this close to the lead vehicle itself
STAGE_TIMEOUT_S = 45.0     # time *without approach* before a staged scenario is
                           # cancelled (see _Scenario.hold)
DURATION_S = 8.0           # how long the scripted action lasts
WALKER_SPEED = 1.6         # m/s of the crossing pedestrian
LEAD_CRUISE = 0.45         # throttle of the lead vehicle before it brakes
BRAKE_STRENGTH = 1.0


class _Scenario:
    """One staged event: spawn actors, wait for the ego, act, then clean up.

    `carla_map` and `blueprints` are passed in rather than fetched: every call to
    `world.get_map()` re-downloads and re-parses the OpenDRIVE map, and doing it
    once per staged scenario stalled the simulator until the RPC timed out.
    """

    def __init__(self, kind, idx, world, plan, carla_map, blueprints):
        self.kind = kind
        self.idx = idx                      # plan index of the trigger point
        self.world = world
        self.map = carla_map
        self.bl = blueprints
        self.plan = plan
        self.actors = []
        self.state = "pending"              # pending -> staged -> running -> done
        self.timer = 0.0
        self.staged_for = 0.0
        self.closest_gap = float("inf")
        self.vehicle = None
        self.walker = None
        self.reason = ""

    # ------------------------------------------------------------- staging
    def stage(self):
        """Create the actors.  Returns False if the scenario cannot be placed.

        The failure reason is recorded rather than swallowed: a silently failing
        stage looks exactly like a scenario that was never scheduled, and that
        ambiguity cost two debugging rounds.
        """
        try:
            ok = getattr(self, f"_stage_{self.kind}")()
            self.reason = "" if ok else "no valid placement"
        except Exception as exc:              # never kill a long collection
            ok = False
            self.reason = f"{type(exc).__name__}: {exc}"
        self.state = "staged" if ok else "done"
        return ok

    def _trigger_waypoint(self):
        x, y, _, _ = self.plan[self.idx]
        return self.map.get_waypoint(carla.Location(x=x, y=y))

    def _stage_pedestrian_crossing(self):
        wp = self._trigger_waypoint()
        right = wp.transform.get_right_vector()
        # start on the pavement, roughly one lane width to the right
        side = wp.lane_width * 0.5 + 1.5
        loc = wp.transform.location + carla.Location(x=right.x * side,
                                                     y=right.y * side, z=1.0)
        bp = random.choice(self.bl.filter("walker.pedestrian.*"))
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        walker = self.world.try_spawn_actor(bp, carla.Transform(loc))
        if walker is None:
            return False
        self.actors.append(walker)
        self.walker = walker
        # cross towards the far side of the road
        self.direction = carla.Vector3D(x=-right.x, y=-right.y, z=0.0)
        return True

    def _stage_lead_brake(self):
        # a vehicle a few metres ahead of the trigger point, in our lane
        wp = self._trigger_waypoint()
        ahead = wp.next(12.0)
        if not ahead:
            return False
        tf = ahead[0].transform
        tf.location.z += 0.5
        bp = random.choice(self.bl.filter("vehicle.*"))
        veh = self.world.try_spawn_actor(bp, tf)
        if veh is None:
            return False
        # NOTE: do *not* call set_autopilot(False) here.  Even with False, CARLA
        # instantiates a Traffic Manager on its default port (8000) and raises
        # if that port is taken -- which silently killed every vehicle-based
        # scenario on this machine.  A freshly spawned vehicle is not on
        # autopilot anyway; the explicit port is only needed in retire().
        self.actors.append(veh)
        self.vehicle = veh
        return True

    def _stage_junction_violation(self):
        wp = self._trigger_waypoint()
        # find the junction the route is about to enter
        probe, junction = wp, None
        for _ in range(40):
            nxt = probe.next(2.0)
            if not nxt:
                break
            probe = nxt[0]
            if probe.is_junction:
                junction = probe.get_junction()
                break
        if junction is None:
            return False
        # pick a lane through the junction that crosses ours at a large angle
        ours = wp.transform.rotation.yaw
        best = None
        for start, _end in junction.get_waypoints(carla.LaneType.Driving):
            d = abs((start.transform.rotation.yaw - ours + 180) % 360 - 180)
            if 60.0 < d < 150.0:
                best = start
                break
        if best is None:
            return False
        back = best.previous(15.0)
        if not back:
            return False
        tf = back[0].transform
        tf.location.z += 0.5
        bp = random.choice(self.bl.filter("vehicle.*"))
        veh = self.world.try_spawn_actor(bp, tf)
        if veh is None:
            return False
        # NOTE: do *not* call set_autopilot(False) here.  Even with False, CARLA
        # instantiates a Traffic Manager on its default port (8000) and raises
        # if that port is taken -- which silently killed every vehicle-based
        # scenario on this machine.  A freshly spawned vehicle is not on
        # autopilot anyway; the explicit port is only needed in retire().
        self.actors.append(veh)
        self.vehicle = veh
        return True

    # -------------------------------------------------------------- waiting
    def hold(self, dt, gap=None):
        """Called on every step while the scenario is staged but not yet fired.

        The lead vehicle has to keep rolling.  Staged motionless it is simply a
        wall in the ego lane: the ego stops behind it, therefore never gets close
        enough to trigger anything, and the two wait for each other until the
        route is aborted -- which is precisely what the first version did (three
        of three routes lost, zero scenarios fired).

        The staleness timer counts time *without approach*, not wall-clock time:
        an ego that queues at a red light on its way to the trigger is still
        coming, and a fixed 20 s budget expired most scenarios before the ego
        ever arrived.
        """
        if gap is not None and gap < self.closest_gap - 1.0:
            self.closest_gap = gap
            self.staged_for = 0.0
        self.staged_for += dt
        if self.kind == "lead_brake" and self.vehicle is not None:
            try:
                self.vehicle.apply_control(
                    carla.VehicleControl(throttle=LEAD_CRUISE))
            except RuntimeError:
                self.state = "done"

    def ready(self, ego_loc, gap_along_route):
        """Whether the trigger condition is met.  The lead-brake event triggers
        on the distance to the vehicle itself, since that vehicle is moving and
        its position no longer matches the planned trigger point."""
        if self.kind == "lead_brake" and self.vehicle is not None:
            try:
                loc = self.vehicle.get_location()
            except RuntimeError:
                return False
            return math.hypot(loc.x - ego_loc.x,
                              loc.y - ego_loc.y) < LEAD_TRIGGER_DIST
        return gap_along_route < TRIGGER_DIST

    def expired(self):
        return self.staged_for > STAGE_TIMEOUT_S

    # -------------------------------------------------------------- acting
    def fire(self):
        self.state = "running"
        self.timer = 0.0

    def act(self, dt):
        """One control step of the scripted action."""
        self.timer += dt
        try:
            if self.kind == "pedestrian_crossing":
                self.walker.apply_control(carla.WalkerControl(
                    direction=self.direction, speed=WALKER_SPEED))
            elif self.kind == "lead_brake":
                # drive off, then brake hard after two seconds
                if self.timer < 2.0:
                    self.vehicle.apply_control(
                        carla.VehicleControl(throttle=0.5))
                else:
                    self.vehicle.apply_control(
                        carla.VehicleControl(throttle=0.0, brake=BRAKE_STRENGTH))
            elif self.kind == "junction_violation":
                self.vehicle.apply_control(carla.VehicleControl(throttle=0.65))
        except RuntimeError:
            self.state = "done"
        if self.timer > DURATION_S:
            self.state = "done"

    def retire(self, tm_port):
        """Hand the scenario vehicle back to the autopilot instead of deleting it
        in front of the camera: a car vanishing mid-frame is an artefact that
        would end up in the training images."""
        if self.vehicle is not None:
            try:
                self.vehicle.set_autopilot(True, tm_port)
                return True
            except RuntimeError:
                pass
        return False

    def cleanup(self):
        for a in self.actors:
            try:
                a.destroy()
            except RuntimeError:
                pass
        self.actors = []


class ScriptedScenarios:
    """Places scenarios along a route and drives them as the ego passes by."""

    def __init__(self, world, vehicle, plan, cum, every_m=120.0, rng=None,
                 kinds=KINDS, carla_map=None, tm_port=8100):
        self.world = world
        self.vehicle = vehicle
        self.plan = plan
        self.cum = cum
        self.rng = rng or random
        self.tm_port = tm_port
        self.retired = []
        self.staged_ok = 0
        self.failures = []          # (kind, reason) of every failed staging
        self.expired_count = 0
        # fetched once; see the note in _Scenario
        self.map = carla_map or world.get_map()
        self.bl = world.get_blueprint_library()
        self.pending = []
        self.active = None
        self.fired = []                      # (kind, plan index) actually fired
        if not kinds or every_m <= 0:
            return
        # trigger points spaced along the route, first one after a warm-up
        d = 60.0
        while d < cum[-1] - 40.0:
            idx = self._index_at(d)
            self.pending.append(_Scenario(self.rng.choice(list(kinds)), idx,
                                          world, plan, self.map, self.bl))
            d += every_m * self.rng.uniform(0.7, 1.3)

    def _index_at(self, distance):
        lo, hi = 0, len(self.cum) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self.cum[mid] < distance:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def tick(self, ego_idx, dt):
        """Advance the scenario state machine; returns the active kind or None."""
        try:
            ego_loc = self.vehicle.get_location()
        except RuntimeError:
            return None
        if self.active is not None:
            self.active.act(dt)
            if self.active.state == "done":
                if self.active.retire(self.tm_port):
                    self.retired.append(self.active)   # destroyed at route end
                else:
                    self.active.cleanup()
                self.active = None
            else:
                return self.active.kind
        here = self.cum[min(ego_idx, len(self.cum) - 1)]
        for sc in list(self.pending):
            gap = self.cum[min(sc.idx, len(self.cum) - 1)] - here
            if gap < -10.0:                        # already passed it
                sc.cleanup()
                self.pending.remove(sc)
            elif sc.state == "pending" and gap < ARM_DIST:
                if sc.stage():
                    self.staged_ok += 1
                else:
                    self.failures.append((sc.kind, sc.reason))
                    self.pending.remove(sc)
            elif sc.state == "staged":
                sc.hold(dt, gap)
                if sc.ready(ego_loc, gap):
                    sc.fire()
                    self.pending.remove(sc)
                    self.active = sc
                    self.fired.append((sc.kind, sc.idx))
                    return sc.kind
                if sc.expired() or sc.state == "done":
                    # it never became reachable; remove it rather than let it
                    # stand in the road and deadlock the route
                    self.expired_count += 1
                    sc.cleanup()
                    self.pending.remove(sc)
        return None

    def summary(self):
        """One-line account of what happened to the scheduled scenarios."""
        why = {}
        for kind, reason in self.failures:
            why[f"{kind}: {reason}"] = why.get(f"{kind}: {reason}", 0) + 1
        parts = [f"fired {len(self.fired)}", f"staged {self.staged_ok}",
                 f"expired {self.expired_count}",
                 f"still pending {len(self.pending)}"]
        if why:
            parts.append("failed to stage -> "
                         + "; ".join(f"{k} x{v}" for k, v in why.items()))
        return "  scenarios: " + ", ".join(parts)

    def cleanup(self):
        if self.active is not None:
            self.active.cleanup()
        for sc in self.pending + self.retired:
            sc.cleanup()
        self.pending, self.active, self.retired = [], None, []
