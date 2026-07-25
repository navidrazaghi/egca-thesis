"""Data collection script (Sec. 5-1): spawn privileged expert in CARLA, drive
in diverse conditions and record (image, lidar, measurements, BEV seg, depth).

Usage (requires CARLA 0.9.14 server running):
    python -m egca.carla_sim.collect_data --town Town01 --routes 10 \
        --output dataset/Town01 --traffic-density 0.2
"""
import argparse
import json
import math
import os
import random
import time

import cv2
import numpy as np

# carla module is optional; this script fails gracefully if not available
try:
    import carla
except ImportError:
    carla = None

from .expert import PrivilegedExpert
from .sensors import (CAMERAS, spawn_rig, carla_image_to_array,
                      carla_lidar_to_array, carla_depth_to_array,
                      depth_to_normalized_inverse, stitch_cameras,
                      transform_to_ego)
from .weather import TRAIN_WEATHERS, apply_weather

CONTROL_HZ = 10.0            # simulator / expert control rate
RECORD_EVERY = 5             # -> 2 Hz recording rate (Table 5-1)
MAX_ROUTE_SECONDS = 300.0
NOISE_PROB = 0.05            # fraction of steps with injected steering noise
NOISE_STD = 0.10             # std of the injected steering perturbation


class DataCollector:
    def __init__(self, world, vehicle, out_dir):
        self.world, self.vehicle, self.out_dir = world, vehicle, out_dir
        for d in ["rgb", "lidar", "bev_seg", "depth", "measurements"]:
            os.makedirs(os.path.join(out_dir, d), exist_ok=True)
        self.frame_id = 0
        self.sensor_data = {}
        self.actors = spawn_rig(world, vehicle, self._sensor_cb, with_depth=True)
        self.n_sensors = 3 + 3 + 3   # 3 RGB + 3 depth + lidar/imu/gnss
        self._map_wps = None

    def _sensor_cb(self, name, data):
        self.sensor_data[name] = data

    def tick(self, expert_out, record=True):
        """expert_out: (throttle, steer, brake, waypoints_ego, command, goal).
        The simulator advances every call (CONTROL_HZ), but a frame is written
        only when `record` is set, which yields the 2 Hz sampling rate of
        Table 5-1: consecutive 10 Hz frames are almost identical and would
        merely inflate the dataset with correlated samples."""
        frame = self.world.tick()
        while len(self.sensor_data) < self.n_sensors:   # wait for all sensors
            time.sleep(0.01)
        if not record:
            self.sensor_data.clear()
            return
        imgs = {n: carla_image_to_array(self.sensor_data[n])
                for n in ["cam_left", "cam_front", "cam_right"]
                if n in self.sensor_data}
        strip = stitch_cameras(imgs)
        lidar = carla_lidar_to_array(self.sensor_data["lidar"])
        v = self.vehicle.get_velocity()
        speed = np.linalg.norm([v.x, v.y, v.z])
        # privileged ground truth for the auxiliary heads
        bev_seg = self._render_bev_seg()
        depth = self._render_depth()
        fid = f"{self.frame_id:06d}"
        cv2.imwrite(os.path.join(self.out_dir, "rgb", fid + ".jpg"),
                    cv2.cvtColor(strip, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        np.save(os.path.join(self.out_dir, "lidar", fid + ".npy"), lidar)
        cv2.imwrite(os.path.join(self.out_dir, "bev_seg", fid + ".png"), bev_seg)
        np.save(os.path.join(self.out_dir, "depth", fid + ".npy"),
                depth.astype(np.float16))
        _, _, _, wps, command, goal = expert_out
        meas = {"speed": float(speed), "command": int(command),
                "goal_x": float(goal[0]), "goal_y": float(goal[1]),
                "waypoints": wps.tolist()}
        with open(os.path.join(self.out_dir, "measurements", fid + ".json"), "w") as f:
            json.dump(meas, f)
        self.frame_id += 1
        self.sensor_data.clear()

    # ---- privileged ground truth (uses full simulator state) -------------
    #   class ids: 0 free, 1 road, 2 lane marking, 3 vehicle, 4 pedestrian,
    #              5 static obstacle   (cfg.model.aux.bev_classes = 6)
    BEV_GRID = 128
    BEV_X = (0.0, 32.0)        # metres forward
    BEV_Y = (-16.0, 16.0)      # metres, positive to the left

    def _bev_px(self, x, y):
        """Ego-frame metres -> (row, col) of the BEV raster.  Rows increase
        forward-to-backward so that the image is a top-down view with the ego
        vehicle at the bottom centre, matching the LiDAR pseudo-image."""
        n = self.BEV_GRID
        col = (y - self.BEV_Y[0]) / (self.BEV_Y[1] - self.BEV_Y[0]) * n
        row = n - (x - self.BEV_X[0]) / (self.BEV_X[1] - self.BEV_X[0]) * n
        return int(round(row)), int(round(col))

    def _render_bev_seg(self):
        """Privileged 128 x 128 top-down semantic map in the ego frame."""
        seg = np.zeros((self.BEV_GRID, self.BEV_GRID), dtype=np.uint8)
        ego = self.vehicle.get_transform()
        step = (self.BEV_X[1] - self.BEV_X[0]) / self.BEV_GRID
        if self._map_wps is None:      # town geometry is static: sample once
            self._map_wps = self.world.get_map().generate_waypoints(step * 2)
        # road surface + lane markings from the map's lane geometry
        for wp in self._map_wps:
            x, y = transform_to_ego(
                np.array([[wp.transform.location.x, wp.transform.location.y]]),
                ego)[0]
            if not (self.BEV_X[0] <= x < self.BEV_X[1]
                    and self.BEV_Y[0] <= y < self.BEV_Y[1]):
                continue
            half = wp.lane_width / 2.0
            yaw = math.radians(wp.transform.rotation.yaw) - math.radians(ego.rotation.yaw)
            nx, ny = -math.sin(-yaw), math.cos(-yaw)     # lane normal, y-left frame
            p1 = self._bev_px(x - half * nx, y - half * ny)
            p2 = self._bev_px(x + half * nx, y + half * ny)
            cv2.line(seg, (p1[1], p1[0]), (p2[1], p2[0]), 1, thickness=2)
            c = self._bev_px(x + half * nx, y + half * ny)
            if 0 <= c[0] < self.BEV_GRID and 0 <= c[1] < self.BEV_GRID:
                seg[c[0], c[1]] = 2                       # lane boundary
        # dynamic actors
        for act in self.world.get_actors():
            if act.id == self.vehicle.id:
                continue
            if act.type_id.startswith("vehicle."):
                cls = 3
            elif act.type_id.startswith("walker."):
                cls = 4
            elif act.type_id.startswith("static."):
                cls = 5
            else:
                continue
            box = act.bounding_box
            tf = act.get_transform()
            corners = []
            for sx in (-1, 1):
                for sy in (-1, 1):
                    lx = box.extent.x * sx
                    ly = box.extent.y * sy
                    yaw = math.radians(tf.rotation.yaw)
                    wx = tf.location.x + lx * math.cos(yaw) - ly * math.sin(yaw)
                    wy = tf.location.y + lx * math.sin(yaw) + ly * math.cos(yaw)
                    corners.append([wx, wy])
            pts = transform_to_ego(np.array(corners), ego)
            px = np.array([self._bev_px(x, y) for x, y in pts])
            if px[:, 0].max() < 0 or px[:, 1].max() < 0 \
                    or px[:, 0].min() >= self.BEV_GRID or px[:, 1].min() >= self.BEV_GRID:
                continue
            hull = cv2.convexHull(px[:, ::-1].astype(np.int32))
            cv2.fillConvexPoly(seg, hull, int(cls))
        return seg

    def _render_depth(self):
        """Privileged normalized inverse depth aligned with the RGB strip and
        down-sampled to cfg.data.depth_size (80 x 352)."""
        needed = [n + "_depth" for n, _ in CAMERAS]
        if not all(n in self.sensor_data for n in needed):
            return np.zeros((80, 352), dtype=np.float32)
        deps = {n: carla_depth_to_array(self.sensor_data[n + "_depth"])
                for n, _ in CAMERAS}
        strip = stitch_cameras({n: np.repeat(deps[n][:, :, None], 3, axis=2)
                                for n, _ in CAMERAS})[:, :, 0]
        inv = depth_to_normalized_inverse(strip)
        return cv2.resize(inv, (352, 80), interpolation=cv2.INTER_AREA)

    def cleanup(self):
        for a in self.actors:
            a.destroy()


def collect_route(client, town, out_base, route_id, weather, traffic_density=0.2):
    world = client.get_world()
    if world.get_map().name.split("/")[-1] != town:
        world = client.load_world(town)
    settings = world.get_settings()          # deterministic, reproducible ticks
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / CONTROL_HZ
    world.apply_settings(settings)
    apply_weather(world, weather)
    world.set_weather(world.get_weather())
    bp = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    vehicle = world.spawn_actor(bp, spawn_points[0])
    time.sleep(0.5)
    # spawn traffic
    traffic = []
    for sp in spawn_points[1: int(len(spawn_points) * traffic_density)]:
        vbp = random.choice(world.get_blueprint_library().filter("vehicle.*"))
        veh = world.try_spawn_actor(vbp, sp)
        if veh:
            veh.set_autopilot(True)
            traffic.append(veh)
    route_pts = spawn_points[: max(20, len(spawn_points) // 4)]
    random.shuffle(route_pts)
    out_dir = os.path.join(out_base, f"route_{route_id:03d}_{weather}")
    os.makedirs(out_dir, exist_ok=True)
    collector = DataCollector(world, vehicle, out_dir)
    expert = PrivilegedExpert(vehicle, route_pts)
    print(f"collecting {town} route {route_id} weather={weather} ...")
    steps = 0
    max_steps = int(MAX_ROUTE_SECONDS * CONTROL_HZ)
    while steps < max_steps and expert.next_waypoint() is not None:
        throttle, steer, brake, wps = expert.step(world)
        # Noise injection (Sec. 5-1): the *applied* steering is perturbed so
        # that the dataset also covers slightly off-lane states and how the
        # expert recovers from them, which mitigates covariate shift [14].
        # The recorded label stays the unperturbed expert trajectory.
        applied_steer = steer
        if random.random() < NOISE_PROB:
            applied_steer = float(np.clip(steer + random.gauss(0.0, NOISE_STD),
                                          -1.0, 1.0))
        vehicle.apply_control(carla.VehicleControl(
            throttle=throttle, steer=applied_steer, brake=brake))
        collector.tick((throttle, steer, brake, wps, expert.nav_command(),
                        expert.sparse_goal()),
                       record=(steps % RECORD_EVERY == 0))
        steps += 1
        if steps % 100 == 0:
            print(f"  step {steps}")
    collector.cleanup()
    vehicle.destroy()
    for t in traffic:
        t.destroy()
    print(f"  collected {collector.frame_id} frames")


def main():
    if carla is None:
        print("CARLA Python API not available; install with: pip install carla==0.9.14")
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--town", default="Town01")
    ap.add_argument("--routes", type=int, default=5)
    ap.add_argument("--output", required=True)
    ap.add_argument("--traffic-density", type=float, default=0.2)
    args = ap.parse_args()
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    weathers = list(TRAIN_WEATHERS.keys())
    for route_id in range(args.routes):
        w = weathers[route_id % len(weathers)]
        collect_route(client, args.town, args.output, route_id, w,
                      args.traffic_density)


if __name__ == "__main__":
    main()
