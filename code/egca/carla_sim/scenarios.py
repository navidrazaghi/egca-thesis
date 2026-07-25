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

ARM_DIST = 60.0        # start staging the scenario at this distance (m)
TRIGGER_DIST = 14.0    # fire it when the ego is this close (m)
DURATION_S = 8.0       # how long the scripted action lasts
WALKER_SPEED = 1.6     # m/s of the crossing pedestrian
BRAKE_STRENGTH = 1.0


class _Scenario:
    """One staged event: spawn actors, wait for the ego, act, then clean up."""

    def __init__(self, kind, idx, world, plan):
        self.kind = kind
        self.idx = idx                      # plan index of the trigger point
        self.world = world
        self.plan = plan
        self.actors = []
        self.state = "pending"              # pending -> staged -> running -> done
        self.timer = 0.0

    # ------------------------------------------------------------- staging
    def stage(self):
        """Create the actors.  Returns False if the scenario cannot be placed."""
        try:
            ok = getattr(self, f"_stage_{self.kind}")()
        except (RuntimeError, AttributeError, IndexError):
            ok = False
        self.state = "staged" if ok else "done"
        return ok

    def _trigger_waypoint(self):
        x, y, _, _ = self.plan[self.idx]
        return self.world.get_map().get_waypoint(carla.Location(x=x, y=y))

    def _stage_pedestrian_crossing(self):
        wp = self._trigger_waypoint()
        right = wp.transform.get_right_vector()
        # start on the pavement, roughly one lane width to the right
        side = wp.lane_width * 0.5 + 1.5
        loc = wp.transform.location + carla.Location(x=right.x * side,
                                                     y=right.y * side, z=1.0)
        bps = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        bp = random.choice(bps)
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
        bp = random.choice(
            self.world.get_blueprint_library().filter("vehicle.*"))
        veh = self.world.try_spawn_actor(bp, tf)
        if veh is None:
            return False
        veh.set_autopilot(False)
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
        bp = random.choice(
            self.world.get_blueprint_library().filter("vehicle.*"))
        veh = self.world.try_spawn_actor(bp, tf)
        if veh is None:
            return False
        veh.set_autopilot(False)
        self.actors.append(veh)
        self.vehicle = veh
        return True

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
                 kinds=KINDS):
        self.world = world
        self.vehicle = vehicle
        self.plan = plan
        self.cum = cum
        self.rng = rng or random
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
                                          world, plan))
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
        if self.active is not None:
            self.active.act(dt)
            if self.active.state == "done":
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
                if not sc.stage():
                    self.pending.remove(sc)
            elif sc.state == "staged" and gap < TRIGGER_DIST:
                sc.fire()
                self.pending.remove(sc)
                self.active = sc
                self.fired.append((sc.kind, sc.idx))
                return sc.kind
        return None

    def cleanup(self):
        if self.active is not None:
            self.active.cleanup()
        for sc in self.pending:
            sc.cleanup()
        self.pending, self.active = [], None
