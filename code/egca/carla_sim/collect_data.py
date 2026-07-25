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
import shutil
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
from .scenarios import ScriptedScenarios
from .weather import TRAIN_WEATHERS, apply_weather

CONTROL_HZ = 10.0            # simulator / expert control rate
RECORD_EVERY = 5             # -> 2 Hz recording rate (Table 5-1)
MAX_ROUTE_SECONDS = 300.0
MIN_ROUTE_METERS = 800.0     # plan long enough to contain several junctions
TRAFFIC_WARMUP_STEPS = 30    # let the spawned traffic disperse before recording
STUCK_SECONDS = 25.0         # abandon a gridlocked route instead of recording it
# Traffic density is drawn per route from this range.  A single fixed value is
# both unrealistic and risky: 0.4 gridlocks a small town such as Town01 (one dry
# run produced 8 usable frames out of 120), while a fixed low value never
# exercises car following.
DENSITY_RANGE = (0.10, 0.30)
MIN_USEFUL_FRAMES = 40       # *moving* frames below which a route is retried
MAX_ROUTE_ATTEMPTS = 3
NOISE_PROB = 0.01            # probability of starting a perturbation burst
NOISE_STEPS = 5              # length of a burst (0.5 s at 10 Hz)
NOISE_STD = 0.15             # std of the injected steering perturbation


class DataCollector:
    def __init__(self, world, vehicle, out_dir):
        self.world, self.vehicle, self.out_dir = world, vehicle, out_dir
        for d in ["rgb", "lidar", "bev_seg", "depth", "measurements", "labels"]:
            os.makedirs(os.path.join(out_dir, d), exist_ok=True)
        self.frame_id = 0
        self.moving_frames = 0
        self.sensor_data = {}
        self.actors = spawn_rig(world, vehicle, self._sensor_cb, with_depth=True)
        self.n_sensors = 3 + 3 + 3   # 3 RGB + 3 depth + lidar/imu/gnss
        self._map_wps = None

    def _sensor_cb(self, name, data):
        self.sensor_data[name] = data

    def tick(self, control, info, record=True, noise=False):
        """Advance the simulator one step and optionally write a frame.

        `control` is the (throttle, steer, brake) actually applied and `info` is
        the expert's side information.  The simulator advances on every call
        (CONTROL_HZ), but a frame is written only when `record` is set, which
        yields the 2 Hz sampling rate of Table 5-1: consecutive 10 Hz frames are
        almost identical and would only inflate the dataset with correlated
        samples.

        The ego pose is logged with every recorded frame; the waypoint labels are
        reconstructed from those poses afterwards by `build_labels.py`.
        """
        self.world.tick()
        while len(self.sensor_data) < self.n_sensors:   # wait for all sensors
            time.sleep(0.01)
        if not record:
            self.sensor_data.clear()
            return
        imgs = {n: carla_image_to_array(self.sensor_data[n])
                for n in ["cam_left", "cam_front", "cam_right"]
                if n in self.sensor_data}
        strip = stitch_cameras(imgs)
        lidar = self._compress_lidar(carla_lidar_to_array(self.sensor_data["lidar"]))
        tf = self.vehicle.get_transform()
        v = self.vehicle.get_velocity()
        speed = float(np.linalg.norm([v.x, v.y, v.z]))
        # privileged ground truth for the auxiliary heads
        bev_seg = self._render_bev_seg()
        depth = self._render_depth()
        fid = f"{self.frame_id:06d}"
        cv2.imwrite(os.path.join(self.out_dir, "rgb", fid + ".jpg"),
                    cv2.cvtColor(strip, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        np.save(os.path.join(self.out_dir, "lidar", fid + ".npy"), lidar)
        cv2.imwrite(os.path.join(self.out_dir, "bev_seg", fid + ".png"), bev_seg)
        np.save(os.path.join(self.out_dir, "depth", fid + ".npy"),
                depth.astype(np.float16))
        throttle, steer, brake = control
        meas = {
            "speed": speed,
            "command": int(info["command"]),
            "goal_x": float(info["goal"][0]), "goal_y": float(info["goal"][1]),
            # ego pose in world coordinates -> used by build_labels.py
            "x": float(tf.location.x), "y": float(tf.location.y),
            "yaw": float(tf.rotation.yaw),
            # expert action and reasons, kept for analysis and sanity checks
            "throttle": float(throttle), "steer": float(steer),
            "brake": float(brake),
            "target_speed": float(info["target_speed"]),
            "red_light": bool(info["red_light"]),
            "stop_sign": bool(info["stop_sign"]),
            "lead_distance": float(info["lead_distance"]),
            "n_ahead": int(info.get("n_ahead", 0)),
            "lead_is_walker": bool(info.get("lead_is_walker", False)),
            "creeping": bool(info.get("creeping", False)),
            "scenario": str(info.get("scenario", "")),
            "noise": bool(noise),
        }
        with open(os.path.join(self.out_dir, "measurements", fid + ".json"), "w") as f:
            json.dump(meas, f)
        self.frame_id += 1
        if speed > 0.5:
            self.moving_frames += 1
        self.sensor_data.clear()

    def _compress_lidar(self, pts):
        """Crop to the BEV region of interest and store as float16.

        Points outside the region are never used by the pillar encoder, and
        float16 has ~3 mm resolution over a 32 m range -- far below the LiDAR's
        own noise.  Together this cuts the dataset from ~750 kB to ~150 kB per
        frame, which keeps 200 k frames inside the page cache of the machine and
        makes the training loop insensitive to disk latency.
        """
        x0, x1 = self.BEV_X
        y0, y1 = self.BEV_Y
        m = ((pts[:, 0] >= x0) & (pts[:, 0] < x1)
             & (pts[:, 1] >= y0) & (pts[:, 1] < y1)
             & (pts[:, 2] >= -2.5) & (pts[:, 2] < 1.5))
        return pts[m].astype(np.float16)

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
        """Stop the sensors before destroying them.

        Destroying a listening sensor lets its callback fire on an actor that no
        longer exists, which aborts the process with a C++ `std::runtime_error`
        ("trying to operate on a destroyed actor").  Stopping first, then
        ticking once so queued callbacks drain, avoids it.
        """
        for a in self.actors:
            try:
                if a.is_listening:
                    a.stop()
            except RuntimeError:
                pass
        try:
            self.world.tick()
        except RuntimeError:
            pass
        for a in self.actors:
            try:
                a.destroy()
            except RuntimeError:
                pass
        self.actors = []


def spawn_walkers(client, world, n, cross_factor=0.35, near=None, radius=45.0):
    """Spawn `n` pedestrians with CARLA's navigation AI.

    Without pedestrians the dataset contains no vulnerable road users at all:
    the "pedestrian" class of the BEV target stays empty, the policy never sees
    one during training, and it then collides with the pedestrians of the
    safety-critical evaluation scenarios.  `cross_factor` is the share of
    pedestrians allowed to cross outside crossings, which is what creates the
    genuinely critical events.

    `near` is a list of (x, y) route points; pedestrians are then only accepted
    within `radius` of the route.  Spreading them uniformly over a whole town
    puts almost none inside the 32 m x 32 m perception region -- measured 0.01% of
    the BEV target, i.e. the pedestrian class stays effectively untrained.
    """
    wbps = world.get_blueprint_library().filter("walker.pedestrian.*")
    world.set_pedestrians_cross_factor(cross_factor)
    walkers, controllers = [], []
    attempts = 0
    while len(walkers) < n and attempts < 30 * n:
        attempts += 1
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        if near is not None and not any(
                (loc.x - px) ** 2 + (loc.y - py) ** 2 < radius ** 2
                for px, py in near):
            continue
        bp = random.choice(wbps)
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        w = world.try_spawn_actor(bp, carla.Transform(loc))
        if w is None:
            continue
        walkers.append(w)
    world.tick()
    cbp = world.get_blueprint_library().find("controller.ai.walker")
    for w in walkers:
        c = world.try_spawn_actor(cbp, carla.Transform(), attach_to=w)
        if c is None:
            continue
        controllers.append(c)
    world.tick()
    for c in controllers:
        c.start()
        target = world.get_random_location_from_navigation()
        if target is not None:
            c.go_to_location(target)
        c.set_max_speed(random.uniform(0.9, 1.8))
    return walkers, controllers


def destroy_walkers(walkers, controllers):
    for c in controllers:
        try:
            c.stop()
            c.destroy()
        except RuntimeError:
            pass
    for w in walkers:
        try:
            w.destroy()
        except RuntimeError:
            pass


def clear_world(world):
    """Destroy vehicles, walkers and sensors left behind by an interrupted run.

    Without this, a crashed collection leaves its ego vehicle parked on a spawn
    point (so the next run fails with "collision at spawn position") and its
    sensors still rendering, silently eating GPU time.
    """
    n = 0
    for a in list(world.get_actors()):
        if not a.type_id.startswith(("vehicle.", "walker.", "sensor.")):
            continue
        try:
            if a.type_id.startswith("sensor.") and a.is_listening:
                a.stop()
            a.destroy()
            n += 1
        except RuntimeError:
            pass
    if n:
        print(f"  cleared {n} leftover actors")
    return n


def spawn_ego(world, bp, spawn_points):
    """Spawn the ego vehicle at the first free point; returns (actor, index)."""
    for i, sp in enumerate(spawn_points):
        actor = world.try_spawn_actor(bp, sp)
        if actor is not None:
            return actor, i
    raise RuntimeError("no free spawn point in this town")


def collect_route(client, town, out_base, route_id, weather, traffic_density=0.2,
                  max_seconds=MAX_ROUTE_SECONDS, min_route_m=MIN_ROUTE_METERS,
                  tm_port=8000, n_walkers=60, scenario_every_m=0.0):
    """Collect one route.  Always leaves the simulator in asynchronous mode, even
    on failure: a server abandoned in synchronous mode waits forever for a tick
    that nobody sends, and every later client would hang."""
    world = client.get_world()
    if world.get_map().name.split("/")[-1] != town:
        world = client.load_world(town)
    settings = world.get_settings()          # deterministic, reproducible ticks
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / CONTROL_HZ
    world.apply_settings(settings)
    apply_weather(world, weather)
    clear_world(world)
    world.tick()
    bp = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    vehicle, ego_i = spawn_ego(world, bp, spawn_points)
    others = [sp for i, sp in enumerate(spawn_points) if i != ego_i]
    world.tick()
    traffic, peds = [], ([], [])
    try:
        return _drive_route(client, world, vehicle, others, town, out_base,
                            route_id, weather, traffic_density, max_seconds,
                            min_route_m, tm_port, traffic, n_walkers, peds,
                            scenario_every_m)
    finally:
        destroy_walkers(*peds)
        # Order matters: release the Traffic Manager before its vehicles are
        # destroyed, then destroy the actors in one batch, and only then put the
        # server back into asynchronous mode.
        if traffic_density > 0:
            try:
                client.get_trafficmanager(tm_port).set_synchronous_mode(False)
            except RuntimeError:
                pass
        try:
            client.apply_batch_sync(
                [carla.command.DestroyActor(a) for a in traffic + [vehicle]],
                True)
        except RuntimeError:
            for a in traffic + [vehicle]:
                try:
                    a.destroy()
                except RuntimeError:
                    pass
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)


def _drive_route(client, world, vehicle, spawn_points, town, out_base, route_id,
                 weather, traffic_density, max_seconds, min_route_m, tm_port,
                 traffic, n_walkers=0, peds=None, scenario_every_m=0.0):
    # background traffic under CARLA's own autopilot.  Each simulator instance
    # needs its own Traffic Manager port, otherwise a second instance fails to
    # bind (the TM RPC server lives on the host network).
    n_traffic = (max(2, int(len(spawn_points) * traffic_density))
                 if traffic_density > 0 else 0)
    if n_traffic:
        tm = client.get_trafficmanager(tm_port)
        tm.set_synchronous_mode(True)
        for sp in spawn_points[:n_traffic]:
            vbp = random.choice(world.get_blueprint_library().filter("vehicle.*"))
            veh = world.try_spawn_actor(vbp, sp)
            if veh:
                veh.set_autopilot(True, tm.get_port())
                traffic.append(veh)
        # Traffic spawns in one instant and starts out clumped at the spawn
        # points; a short warm-up lets it disperse, so the ego does not begin
        # every route inside an artificial queue.
        for _ in range(TRAFFIC_WARMUP_STEPS):
            world.tick()
    # Route destinations are taken from spawn points that are *not* used by the
    # background traffic, so the route does not end on top of a parked car.
    target_pts = spawn_points[n_traffic:n_traffic + 8] or spawn_points[-8:]
    # a route long enough to contain several junctions: chain destinations until
    # the planned length exceeds min_route_m
    targets, expert = [], None
    for sp in target_pts:
        targets.append(sp.location)
        try:
            expert = PrivilegedExpert(world, vehicle, targets)
        except (RuntimeError, ValueError, IndexError) as e:
            print(f"  route planning failed for one target ({e}), retrying")
            targets.pop()
            continue
        if expert.route_length > min_route_m:
            break
    if expert is None or expert.route_length < 50.0:
        print("  could not plan a route, skipping")
        return 0, None
    # Pedestrians are spawned only now, so they can be placed along the planned
    # route instead of uniformly over the town.
    if n_walkers and peds is not None:
        route_pts = [(p[0], p[1]) for p in expert.plan[::20]]
        w, c = spawn_walkers(client, world, n_walkers, near=route_pts)
        peds[0].extend(w)
        peds[1].extend(c)
        print(f"  {len(w)} pedestrians along the route")
    out_dir = os.path.join(out_base, f"route_{route_id:03d}_{weather}")
    os.makedirs(out_dir, exist_ok=True)
    collector = DataCollector(world, vehicle, out_dir)
    scenarios = ScriptedScenarios(world, vehicle, expert.plan, expert.cum,
                                  every_m=scenario_every_m, rng=random,
                                  carla_map=expert.map, tm_port=tm_port)
    print(f"collecting {town} route {route_id} weather={weather} "
          f"({expert.route_length:.0f} m, "
          f"{len(scenarios.pending)} scripted scenarios) ...")
    steps, noise_left, stuck = 0, 0, 0
    max_stuck = int(STUCK_SECONDS * CONTROL_HZ)
    max_steps = int(max_seconds * CONTROL_HZ)
    while steps < max_steps and not expert.done():
        throttle, steer, brake, info = expert.step()
        # A permanent traffic deadlock would otherwise burn the whole route
        # budget; abort and let the next route start.
        stuck = stuck + 1 if info["progress_delta"] < 1e-4 else 0
        if stuck > max_stuck:
            print(f"  aborting: no progress for {STUCK_SECONDS:.0f} s")
            break
        # Noise injection (Sec. 5-1): the *applied* steering is perturbed for a
        # short burst so that the dataset also covers slightly off-lane states
        # and the recovery from them, which mitigates covariate shift [14].
        # The label is reconstructed from the real future trajectory, so the
        # recovery manoeuvre itself becomes the supervision signal.
        if noise_left == 0 and random.random() < NOISE_PROB:
            noise_left = NOISE_STEPS
            noise_mag = random.gauss(0.0, NOISE_STD)
        applied_steer = steer
        if noise_left > 0:
            applied_steer = float(np.clip(steer + noise_mag, -1.0, 1.0))
            noise_left -= 1
        vehicle.apply_control(carla.VehicleControl(
            throttle=throttle, steer=applied_steer, brake=brake))
        info["scenario"] = scenarios.tick(expert.idx, 1.0 / CONTROL_HZ) or ""
        collector.tick((throttle, applied_steer, brake), info,
                       record=(steps % RECORD_EVERY == 0),
                       noise=(noise_left > 0))
        steps += 1
        if steps % 200 == 0:
            print(f"  step {steps}  progress {100 * info['progress']:.0f}%  "
                  f"frames {collector.frame_id}")
    meta = {"town": town, "weather": weather, "control_hz": CONTROL_HZ,
            "record_every": RECORD_EVERY, "frames": collector.frame_id,
            "route_length_m": expert.route_length,
            "completed": bool(expert.done()),
            "scenarios_fired": [k for k, _ in scenarios.fired],
            "scenario_failures": scenarios.failures,
            "moving_frames": collector.moving_frames}
    with open(os.path.join(out_dir, "route.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(scenarios.summary())
    scenarios.cleanup()
    collector.cleanup()
    print(f"  collected {collector.frame_id} frames "
          f"({'completed' if expert.done() else 'timed out'})")
    return collector.moving_frames, out_dir


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
    ap.add_argument("--traffic-density", type=float, default=-1.0,
                    help="fraction of spawn points filled with traffic; the "
                         f"default draws it per route from {DENSITY_RANGE}, "
                         "which both varies the scene and avoids gridlock")
    ap.add_argument("--max-seconds", type=float, default=MAX_ROUTE_SECONDS,
                    help="per-route time budget; use ~60 for a dry run")
    ap.add_argument("--min-route-m", type=float, default=MIN_ROUTE_METERS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--first-route", type=int, default=0,
                    help="id of the first route (to resume an interrupted run)")
    ap.add_argument("--scenario-every-m", type=float, default=0.0,
                    help="mean spacing of scripted safety-critical scenarios "
                         "along a route in metres; 0 disables them")
    ap.add_argument("--walkers", type=int, default=60,
                    help="pedestrians spawned per route (0 disables them)")
    ap.add_argument("--tm-port", type=int, default=8100,
                    help="Traffic Manager RPC port; must differ per simulator "
                         "instance when several run in parallel (CARLA's own "
                         "default of 8000 is often already taken on a server)")
    args = ap.parse_args()
    random.seed(args.seed)
    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)          # loading a town can take a while
    weathers = list(TRAIN_WEATHERS.keys())
    for route_id in range(args.first_route, args.first_route + args.routes):
        w = weathers[route_id % len(weathers)]
        density = (random.uniform(*DENSITY_RANGE) if args.traffic_density < 0
                   else args.traffic_density)
        # Outcome-based retry: some spawn points start the ego in a place it
        # cannot leave (blocked lane, route beginning behind a divider).  Rather
        # than diagnosing every such geometry, a route that yields almost no
        # usable frames is discarded and retried from a different spawn point.
        for attempt in range(MAX_ROUTE_ATTEMPTS):
            try:
                frames, out_dir = collect_route(
                    client, args.town, args.output, route_id, w, density,
                    args.max_seconds, args.min_route_m, args.tm_port,
                    args.walkers, args.scenario_every_m)
            except (RuntimeError, IndexError) as e:
                # One bad route must not kill a multi-day collection.
                print(f"route {route_id} failed: {e}")
                frames, out_dir = 0, None
            if frames >= MIN_USEFUL_FRAMES:
                break
            if out_dir and os.path.isdir(out_dir):
                shutil.rmtree(out_dir, ignore_errors=True)
            if attempt < MAX_ROUTE_ATTEMPTS - 1:
                print(f"  only {frames} moving frames, retrying route {route_id} "
                      f"from a different spawn point")


if __name__ == "__main__":
    main()
