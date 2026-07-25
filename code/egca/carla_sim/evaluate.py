"""Closed-loop evaluation on CARLA Longest6 benchmark (Sec. 5-2 / 5-3).

Usage (CARLA 0.9.14 server must be running):
    python -m egca.carla_sim.evaluate --config configs/egca.yaml \
        --checkpoint checkpoints/egca/best.pth --weather ClearNoon \
        --output results/egca_clear.json

For ablations (hard sensor failure, Sec. 5-4-2):
    python -m egca.carla_sim.evaluate ... --drop-sensor cam
    python -m egca.carla_sim.evaluate ... --lidar-drop-rate 0.5

The closed-loop control step is CONTROL_HZ (10 Hz), matched to the LiDAR
rotation frequency and to the simulator's fixed time step; the 2 Hz figure in
Table 5-1 is the sub-sampling rate used when *recording* the training set.
"""
import argparse
import json
import math
import os
import time

import cv2
import numpy as np
import torch

try:
    import carla
except ImportError:
    carla = None

from ..config import load_config
from ..models import EGCAPolicy
from ..control.pid import WaypointController
from .sensors import (spawn_rig, carla_image_to_array, carla_lidar_to_array,
                      stitch_cameras)
from .weather import apply_weather
from ..eval.metrics import RouteResult

LONGEST6_ROUTES = [
    ("Town01", [(36, 40), (39, 35), (110, 114), (7, 3), (0, 4), (68, 50)]),
    ("Town02", [(38, 76), (47, 62), (17, 44), (70, 66), (79, 14), (61, 18)]),
    ("Town03", [(107, 33), (44, 14), (71, 82), (153, 152), (65, 218), (48, 80)]),
    ("Town04", [(33, 103), (207, 133), (102, 145), (57, 84), (106, 153), (152, 146)]),
    ("Town05", [(145, 92), (65, 168), (53, 30), (27, 34), (67, 126), (24, 44)]),
    ("Town06", [(77, 68), (79, 84), (19, 602), (251, 165), (76, 38), (124, 45)]),
]


CONTROL_HZ = 10.0            # closed-loop policy rate (= simulator tick rate)
MAX_ROUTE_SECONDS = 300.0    # per-route time budget
STUCK_SECONDS = 30.0         # abort a route after this long without motion
GOAL_DISTANCE = 8.0          # look-ahead of the sparse navigation goal (m)

# RoadOption -> the command indices expected by MeasurementEncoder
# (0 left, 1 right, 2 straight, 3 lane-follow)
COMMAND_MAP = {"LEFT": 0, "RIGHT": 1, "STRAIGHT": 2, "LANEFOLLOW": 3,
               "CHANGELANELEFT": 3, "CHANGELANERIGHT": 3, "VOID": 3}


def plan_route(world, start_loc, end_loc, resolution=2.0):
    """Dense global plan as a list of (x, y, command).  The route length of this
    plan — not the straight-line distance — is the denominator of RC (Eq. 5.1)."""
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    grp = GlobalRoutePlanner(world.get_map(), resolution)
    plan = grp.trace_route(start_loc, end_loc)
    pts = [(wp.transform.location.x, wp.transform.location.y,
            COMMAND_MAP.get(str(opt).split(".")[-1], 3)) for wp, opt in plan]
    length = sum(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
                 for i in range(len(pts) - 1))
    return pts, max(length, 1e-6)


def next_goal(plan, idx, loc, yaw):
    """Advance the plan cursor past the ego position and return the look-ahead
    goal in the ego frame together with the active navigation command."""
    while idx + 1 < len(plan) and \
            math.hypot(plan[idx][0] - loc.x, plan[idx][1] - loc.y) < 2.0:
        idx += 1
    j = idx
    while j + 1 < len(plan) and \
            math.hypot(plan[j][0] - loc.x, plan[j][1] - loc.y) < GOAL_DISTANCE:
        j += 1
    dx, dy = plan[j][0] - loc.x, plan[j][1] - loc.y
    c, s = math.cos(yaw), math.sin(yaw)
    gx = c * dx + s * dy                 # forward
    gy = -(-s * dx + c * dy)             # left
    return idx, (gx, gy), plan[j][2]


def run_route(world, model, ctrl, route_id, start_idx, end_idx, drop_sensor,
              lidar_drop_rate=0.0, rng=None):
    spawn_points = world.get_map().get_spawn_points()
    bp = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
    vehicle = world.spawn_actor(bp, spawn_points[start_idx])
    time.sleep(0.5)
    sensor_data = {}
    actors = spawn_rig(world, vehicle, lambda n, d: sensor_data.update({n: d}))
    target = spawn_points[end_idx].location
    start = vehicle.get_location()
    plan, total_dist = plan_route(world, start, target)
    plan_idx = 0
    infractions = {"collision_pedestrian": 0, "collision_vehicle": 0,
                   "collision_static": 0}
    ctrl.reset()
    travelled = 0.0
    stuck_ticks = 0
    prev_loc = start
    rng = rng or np.random.default_rng(0)
    for step in range(int(MAX_ROUTE_SECONDS * CONTROL_HZ)):
        world.tick()
        while len(sensor_data) < 4:
            time.sleep(0.01)
        loc = vehicle.get_location()
        travelled += math.hypot(loc.x - prev_loc.x, loc.y - prev_loc.y)
        prev_loc = loc
        if math.hypot(loc.x - target.x, loc.y - target.y) < 3.0:
            completion = 100.0
            break
        v = vehicle.get_velocity()
        speed = math.sqrt(v.x**2 + v.y**2)
        if speed < 0.1:
            stuck_ticks += 1
            if stuck_ticks > int(STUCK_SECONDS * CONTROL_HZ):
                completion = 100.0 * travelled / total_dist
                break
        else:
            stuck_ticks = 0
        yaw = math.radians(vehicle.get_transform().rotation.yaw)
        plan_idx, goal, command = next_goal(plan, plan_idx, loc, yaw)
        imgs = {n: carla_image_to_array(sensor_data[n])
                for n in ["cam_left", "cam_front", "cam_right"]}
        strip = stitch_cameras(imgs)
        strip = (strip.astype(np.float32) / 255.0
                 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        lidar = carla_lidar_to_array(sensor_data["lidar"])
        from ..models.lidar_encoder import pillarize
        feats, coords, mask = pillarize(lidar, model.cfg.model.lidar)
        batch = {
            "image": torch.from_numpy(strip.transpose(2, 0, 1)).unsqueeze(0).float(),
            "pillar_feats": feats.unsqueeze(0),
            "pillar_coords": coords.unsqueeze(0),
            "pillar_mask": mask.unsqueeze(0),
            "pillar_batch": torch.zeros(feats.shape[0], dtype=torch.long),
            "speed": torch.tensor([[speed]], dtype=torch.float32),
            "command": torch.tensor([command], dtype=torch.long),
            "goal": torch.tensor([list(goal)], dtype=torch.float32),
        }
        batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}
        # per-frame LiDAR loss (Sec. 5-4-2): the dropped frame is replaced by
        # the learnable absent-modality token, exactly as during training.
        force = drop_sensor
        if force is None and lidar_drop_rate > 0 and rng.random() < lidar_drop_rate:
            force = "lidar"
        with torch.no_grad():
            out = model(batch, force_drop=force)
        wps = out["waypoints"][0].cpu().numpy()
        steer, throttle, brake = ctrl.step(wps, speed)
        vehicle.apply_control(carla.VehicleControl(
            throttle=throttle, steer=steer, brake=brake))
        sensor_data.clear()
    else:
        completion = 100.0 * travelled / total_dist
    for a in actors:
        a.destroy()
    vehicle.destroy()
    # collision listener stub (real impl uses carla.CollisionSensor callback)
    return RouteResult(completion=completion, infractions=infractions,
                       distance_km=travelled / 1000.0)


def main():
    if carla is None:
        print("CARLA Python API not available")
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--weather", default="ClearNoon")
    ap.add_argument("--output", required=True)
    ap.add_argument("--drop-sensor", choices=["cam", "lidar"], default=None,
                    help="hard, permanent failure of one modality")
    ap.add_argument("--lidar-drop-rate", type=float, default=0.0,
                    help="probability of losing each individual LiDAR frame")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    args = ap.parse_args()
    cfg = load_config(args.config, [])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = EGCAPolicy(cfg, sensor_dropout=0.0).to(device).eval()
    model.load_state_dict(ckpt["model"])
    ctrl = WaypointController(cfg.control, cfg.model.decoder.wp_dt)
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    rng = np.random.default_rng(args.seed)
    results = []
    for town, routes in LONGEST6_ROUTES:
        world = client.load_world(town)
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / CONTROL_HZ
        world.apply_settings(settings)
        apply_weather(world, args.weather)
        for start, end in routes:
            print(f"{town} {start}->{end}")
            res = run_route(world, model, ctrl, len(results), start, end,
                            args.drop_sensor, args.lidar_drop_rate, rng)
            results.append({"town": town, "start": start, "end": end,
                            "completion": res.completion, "infractions": res.infractions,
                            "distance_km": res.distance_km})
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    from ..eval.metrics import aggregate
    keep = ("completion", "infractions", "distance_km")
    agg = aggregate([RouteResult(**{k: r[k] for k in keep}) for r in results])
    print(f"DS={agg['DS']:.1f}  RC={agg['RC']:.1f}  IS={agg['IS']:.2f}  "
          f"inf/10km={agg['infractions_per_10km']:.1f}")


if __name__ == "__main__":
    main()
