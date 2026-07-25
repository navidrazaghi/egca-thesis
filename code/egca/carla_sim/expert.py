"""Expert driver: privileged rule-based agent using full simulator state for
data collection (Sec. 5-1, "learning by cheating" [15]).

Based on TransFuser privileged planner: global waypoint path + local reactive
steering with privileged detection of dynamic obstacles via BEV occupancy.

Frame convention (used consistently by the sensors, the labels and the
controller): ego frame with x forward and **y to the left** (right-handed).
CARLA's world frame is left-handed, so every lateral quantity taken from the
simulator is negated once.  A positive CARLA `steer` command turns right, hence
a target on the left (y > 0) maps to a negative steering command.
"""
import math
import numpy as np

GOAL_DISTANCE = 8.0      # look-ahead of the sparse navigation goal (m)


class PrivilegedExpert:
    def __init__(self, vehicle, route_waypoints):
        self.vehicle = vehicle
        self.route = route_waypoints      # list of carla.Transform
        self.idx = 0
        self.target_speed = 4.5           # m/s (16 km/h in dense traffic)

    def _to_ego(self, loc, ego):
        """World location -> (x forward, y left) in the ego frame."""
        yaw = math.radians(ego.rotation.yaw)
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = loc.x - ego.location.x, loc.y - ego.location.y
        return c * dx + s * dy, -(-s * dx + c * dy)

    def nav_command(self):
        """Discrete high-level command from the geometry of the global route:
        0 left, 1 right, 2 straight (through an intersection), 3 lane-follow."""
        i = min(self.idx, len(self.route) - 1)
        if i + 2 >= len(self.route):
            return 3
        a, b, c = (self.route[i].location, self.route[i + 1].location,
                   self.route[i + 2].location)
        h1 = math.atan2(b.y - a.y, b.x - a.x)
        h2 = math.atan2(c.y - b.y, c.x - b.x)
        dpsi = math.degrees((h2 - h1 + math.pi) % (2 * math.pi) - math.pi)
        dpsi = -dpsi                       # left-handed world -> y-left frame
        if dpsi > 15.0:
            return 0                       # left
        if dpsi < -15.0:
            return 1                       # right
        return 2 if abs(dpsi) > 3.0 else 3

    def sparse_goal(self):
        """Sparse navigation target ~GOAL_DISTANCE ahead *along the global
        route*, in the ego frame.  It is deliberately taken from the route and
        not from the expert trajectory: feeding a predicted waypoint back as an
        input would leak the label the decoder has to regress."""
        ego = self.vehicle.get_transform()
        loc = self._interp_route(GOAL_DISTANCE)
        if loc is None:
            wpt = self.next_waypoint()
            if wpt is None:
                return 0.0, 0.0
            loc = wpt.location
        return self._to_ego(loc, ego)

    def next_waypoint(self):
        if self.idx >= len(self.route):
            return None
        wpt = self.route[self.idx]
        ego = self.vehicle.get_transform()
        dx = wpt.location.x - ego.location.x
        dy = wpt.location.y - ego.location.y
        dist = math.hypot(dx, dy)
        if dist < 2.0:
            self.idx += 1
            return self.next_waypoint()
        return wpt

    def detect_hazard(self, world):
        """Returns (dist, speed) of closest dynamic obstacle in front,
        privileged access to all actors."""
        ego = self.vehicle.get_transform()
        ego_loc = ego.location
        fwd = ego.get_forward_vector()
        closest_dist, closest_speed = 100.0, 0.0
        for act in world.get_actors():
            if act.id == self.vehicle.id:
                continue
            if act.type_id.startswith("vehicle.") or act.type_id.startswith("walker."):
                loc = act.get_location()
                dx, dy = loc.x - ego_loc.x, loc.y - ego_loc.y
                fwd_dist = dx * fwd.x + dy * fwd.y
                if 0.5 < fwd_dist < closest_dist:
                    lat_dist = abs(-dx * fwd.y + dy * fwd.x)
                    if lat_dist < 2.5:
                        closest_dist = fwd_dist
                        v = act.get_velocity()
                        closest_speed = math.sqrt(v.x**2 + v.y**2)
        return closest_dist, closest_speed

    def step(self, world):
        """Returns (throttle, steer, brake, target_waypoint_ego).
        target_waypoint_ego: 4 x 2 array of waypoints in ego frame (for label)."""
        wpt = self.next_waypoint()
        if wpt is None:
            return 0.0, 0.0, 1.0, np.zeros((4, 2), dtype=np.float32)
        ego = self.vehicle.get_transform()
        x_ego, y_ego = self._to_ego(wpt.location, ego)   # y > 0 = left
        angle = math.atan2(y_ego, x_ego)
        steer = np.clip(-2.0 * angle, -1.0, 1.0)         # left target -> steer < 0
        dist_haz, speed_haz = self.detect_hazard(world)
        v = self.vehicle.get_velocity()
        speed = math.sqrt(v.x**2 + v.y**2)
        target_v = self.target_speed
        if dist_haz < 12.0:
            target_v = min(target_v, speed_haz - 0.5)
        brake = False
        if speed > 1.2 * target_v or target_v < 0.2:
            brake = True
            throttle = 0.0
        else:
            throttle = np.clip(0.6 * (target_v - speed), 0.0, 0.75)
        # future waypoints for label (uniformly spaced 0.5 s apart)
        wps = []
        for dt in [0.5, 1.0, 1.5, 2.0]:
            wloc = self._interp_route(dt * max(speed, 1.0))
            if wloc is None:
                wloc = wpt.location
            wps.append(list(self._to_ego(wloc, ego)))
        return throttle, steer, float(brake), np.array(wps, dtype=np.float32)

    def _interp_route(self, dist_ahead):
        """Privileged look-ahead along global route."""
        ego = self.vehicle.get_location()
        acc = 0.0
        for i in range(self.idx, len(self.route) - 1):
            loc1 = self.route[i].location
            loc2 = self.route[i + 1].location
            seg = math.hypot(loc2.x - loc1.x, loc2.y - loc1.y)
            if acc + seg >= dist_ahead:
                f = (dist_ahead - acc) / max(seg, 1e-3)
                return type(loc1)(x=loc1.x + f * (loc2.x - loc1.x),
                                  y=loc1.y + f * (loc2.y - loc1.y),
                                  z=loc1.z)
            acc += seg
        return None
