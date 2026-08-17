"""EGCA policy wrapped as an agent for the official CARLA leaderboard evaluator.

Why the official evaluator and not our own loop: the driving score of Table 5-3
is only comparable with published numbers if it is produced by the same scorer.
The leaderboard owns the route definitions, the scenario triggers, the collision
and traffic-light detectors and the DS/RC/IS arithmetic, so using it removes the
single largest source of "your numbers are not comparable" criticism.

Run it with (leaderboard 1.0, CARLA 0.9.14):

    python ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
        --routes=${ROUTES}/longest6.xml \
        --scenarios=${ROUTES}/eval_scenarios.json \
        --agent=egca/carla_sim/leaderboard_agent.py \
        --agent-config=configs/agent_egca.json \
        --checkpoint=results/egca_clear.json \
        --track=SENSORS --port=2000 --trafficManagerPort=8100

`--agent-config` points at a small JSON file:

    {"config": "configs/egca.yaml",
     "checkpoint": "checkpoints/egca/best.pth",
     "drop_sensor": null,          // "cam" | "lidar" for a hard failure
     "lidar_drop_rate": 0.0,       // per-frame LiDAR loss (Sec. 5-4-2)
     "seed": 0}

which is what makes the robustness ablations runnable through the very same
scorer as the main result, rather than through a second, home-made one.

Sensor rig, frames and preprocessing are identical to `collect_data.py`; any
divergence here would be a train/test mismatch that no amount of training can
recover from.
"""
import json
import math
import os

import cv2
import numpy as np
import torch

try:
    import carla
except ImportError:                                   # pragma: no cover
    carla = None

try:
    from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
except ImportError:                                   # pragma: no cover
    AutonomousAgent, Track = object, None

# Absolute imports on purpose: the leaderboard loads this file as a standalone
# module from its path, so it has no parent package and relative imports fail.
# `<repo>/code` must therefore be on PYTHONPATH (see eval_env.sh).
from egca.config import Cfg, load_config
from egca.control.pid import WaypointController
from egca.models import EGCAPolicy
from egca.models.lidar_encoder import pillarize
from egca.carla_sim.sensors import CAM_FOV, CAM_H, CAM_W, stitch_cameras

EARTH_RADIUS = 6378137.0
GOAL_DISTANCE = 8.0        # must match PrivilegedExpert.GOAL_DISTANCE
COMMAND_HORIZON = 15.0
PLAN_RESOLUTION = 1.0      # must match PrivilegedExpert.PLAN_RESOLUTION

# RoadOption -> the command indices the network was trained with
COMMAND_MAP = {"LEFT": 0, "RIGHT": 1, "STRAIGHT": 2, "LANEFOLLOW": 3,
               "CHANGELANELEFT": 3, "CHANGELANERIGHT": 3, "VOID": 3}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_entry_point():
    return "EGCAAgent"


class EGCAAgent(AutonomousAgent):
    # ------------------------------------------------------------------ setup
    def setup(self, path_to_conf_file):
        self.track = Track.SENSORS
        with open(path_to_conf_file) as f:
            conf = json.load(f)
        base = os.path.dirname(os.path.abspath(path_to_conf_file))

        def _rel(p):
            return p if os.path.isabs(p) else os.path.join(base, "..", p)

        # Belt and braces with the OMP_* caps in eval_env.sh: torch reads those
        # only at import time, and the evaluator imports it before the agent.
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(_rel(conf["checkpoint"]), map_location=self.device)
        # The architecture comes from the checkpoint, not from the yaml.  Every
        # ablation was trained by overriding the same base config -- full
        # attention, no gate, concat fusion, no auxiliary heads -- so rebuilding
        # from the yaml gives a model whose shapes do not match the weights, and
        # the evaluator dutifully scores 36 failed routes.  train.py stores the
        # config it actually trained with, which cannot drift from the weights.
        self.cfg = (Cfg(ckpt["cfg"]) if "cfg" in ckpt
                    else load_config(_rel(conf["config"]),
                                     conf.get("overrides") or []))
        self.model = EGCAPolicy(self.cfg, sensor_dropout=0.0).to(self.device).eval()
        # Checkpoints predating speed dropout have a 5-input measurement encoder
        # against today's 6. That is every ablation, which are precisely the runs
        # the internal comparison needs, and without the migration each one dies
        # here -- inside an agent the evaluator will happily score as 36 failed
        # routes rather than as a crash. strict=True so a genuine mismatch still
        # stops rather than driving with half the weights initialised at random.
        self.model.load_state_dict(
            EGCAPolicy.upgrade_state_dict(ckpt["model"]), strict=True)
        # Only worth running the auxiliary head at inference when it was
        # trained with an agent class; on a 3-class checkpoint it predicts road
        # and lane, which the controller cannot brake on.
        self.brake_on_bev = (bool(conf.get("brake_on_bev", True))
                             and self.cfg.model.aux.bev_seg
                             and self.cfg.model.aux.bev_classes >= 5)
        self.model.aux_at_inference = self.brake_on_bev
        # Controller settings come from the checkpoint like everything else, but
        # unlike the architecture they are not fixed by the weights: creeping and
        # the PID gains can be changed without retraining, and the whole point of
        # measuring creeping is to run the same checkpoint with and without it.
        # Note the assignment goes through the underlying dict -- Cfg.__getattr__
        # returns a fresh wrapper on every access, so updating `self.cfg.control`
        # would write to a temporary and vanish.
        ctrl_cfg = dict(self.cfg["control"])
        ctrl_cfg.update(conf.get("control") or {})
        self.ctrl = WaypointController(Cfg(ctrl_cfg),
                                       self.cfg.model.decoder.wp_dt)
        # perturbation settings for the robustness experiments
        self.drop_sensor = conf.get("drop_sensor")
        self.lidar_drop_rate = float(conf.get("lidar_drop_rate", 0.0))
        self.rng = np.random.default_rng(int(conf.get("seed", 0)))
        self.debug_dir = conf.get("debug_dir")
        # route state
        self.origin = None
        self.plan_xy = []          # [(x_east, y_north, command)]
        self.plan_cum = []
        self.idx = 0
        self.step_i = 0
        self._heading_check = []

    def destroy(self):
        self.model = None
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- sensors
    def sensors(self):
        """Exactly the rig of Appendix B: three 60-degree cameras stitched to a
        704 x 160 strip, one LiDAR, plus GNSS/IMU/speedometer for localization."""
        cams = [("rgb_left", -55.0), ("rgb_front", 0.0), ("rgb_right", 55.0)]
        out = [{"type": "sensor.camera.rgb", "x": 1.3, "y": 0.0, "z": 2.3,
                "roll": 0.0, "pitch": 0.0, "yaw": yaw,
                "width": CAM_W, "height": CAM_H, "fov": CAM_FOV, "id": name}
               for name, yaw in cams]
        out += [
            {"type": "sensor.lidar.ray_cast", "x": 0.0, "y": 0.0, "z": 2.5,
             "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "id": "lidar"},
            {"type": "sensor.other.imu", "x": 0.0, "y": 0.0, "z": 0.0,
             "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
             "sensor_tick": 0.05, "id": "imu"},
            {"type": "sensor.other.gnss", "x": 0.0, "y": 0.0, "z": 0.0,
             "sensor_tick": 0.01, "id": "gps"},
            {"type": "sensor.speedometer", "reading_frequency": 20,
             "id": "speed"},
        ]
        return out

    # ------------------------------------------------------------ route plan
    def _init_plan(self):
        """Project the GPS route onto a local metric frame (east, north).

        An equirectangular projection around the first plan point is accurate to
        well under a centimetre over the few kilometres of a leaderboard route,
        and avoids depending on the evaluator's internal GPS calibration.

        The plan is then resampled to PLAN_RESOLUTION.  This is not cosmetic:
        the leaderboard hands the agent a *downsampled* route whose vertices are
        tens of metres apart, while the expert that generated the training data
        walked a 1 m plan straight out of the GlobalRoutePlanner.  Every
        arc-length computation below -- the goal look-ahead above all -- would
        otherwise snap to the next distant vertex and feed the network a goal
        far outside the range it was trained on.
        """
        plan = self._global_plan            # [({lat, lon, z}, RoadOption)]
        lat0 = plan[0][0]["lat"]
        lon0 = plan[0][0]["lon"]
        self.origin = (lat0, lon0)
        sparse = []
        for gnss, opt in plan:
            x, y = self._latlon_to_xy(gnss["lat"], gnss["lon"])
            sparse.append((x, y, COMMAND_MAP.get(str(opt).split(".")[-1], 3)))
        # The densified plan drives the local geometry; the sparse vertices are
        # kept because the navigation goal is defined on them, not on a fixed
        # arc length along the dense one.
        self.sparse_xy = sparse
        self.sparse_idx = 0
        self.plan_xy = self._densify(sparse)
        self.plan_cum = [0.0]
        for i in range(len(self.plan_xy) - 1):
            self.plan_cum.append(self.plan_cum[-1] + math.dist(
                self.plan_xy[i][:2], self.plan_xy[i + 1][:2]))

    @staticmethod
    def _densify(sparse, step=PLAN_RESOLUTION):
        """Resample a polyline to `step` spacing, keeping every original vertex.

        Interpolated points inherit the command of the segment they start from,
        which is where the evaluator announces it.  The chord between two
        vertices cuts the corner of a curved lane slightly; over the 8 m
        look-ahead used here that error is far below the metre-scale accuracy
        the goal input needs, and it is what keeps the junction commands aligned
        with the vertices that carry them.
        """
        if len(sparse) < 2:
            return list(sparse)
        out = []
        for (x0, y0, cmd), (x1, y1, _) in zip(sparse, sparse[1:]):
            seg = math.dist((x0, y0), (x1, y1))
            n = max(1, int(seg / step))
            for k in range(n):                       # excludes the end vertex,
                t = k / n                            # which the next segment adds
                out.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0), cmd))
        out.append(sparse[-1])
        return out

    def _latlon_to_xy(self, lat, lon):
        lat0, lon0 = self.origin
        x = EARTH_RADIUS * math.radians(lon - lon0) * math.cos(math.radians(lat0))
        y = EARTH_RADIUS * math.radians(lat - lat0)
        return x, y

    def _to_ego(self, x, y, ego_xy, compass):
        """(east, north) -> (forward, left).

        CARLA's IMU compass is the heading measured clockwise from north, so the
        heading unit vector in the (east, north) frame is (sin, cos) and its left
        normal is (-cos, sin).
        """
        de, dn = x - ego_xy[0], y - ego_xy[1]
        s, c = math.sin(compass), math.cos(compass)
        return de * s + dn * c, -de * c + dn * s

    def _advance(self, ego_xy):
        best, best_d = self.idx, float("inf")
        for i in range(self.idx, min(self.idx + 80, len(self.plan_xy))):
            d = math.dist(self.plan_xy[i][:2], ego_xy)
            if d < best_d:
                best, best_d = i, d
        if best_d > 15.0:                    # off-route: re-localize globally
            for i in range(len(self.plan_xy)):
                d = math.dist(self.plan_xy[i][:2], ego_xy)
                if d < best_d:
                    best, best_d = i, d
        self.idx = best

    def _point_at(self, distance, ego_xy, compass, min_forward=0.5):
        """Ego-frame point `distance` metres ahead along the plan.

        Mirrors PrivilegedExpert._point_at, including the guarantee that the
        returned point lies in front of the vehicle: a target behind the car
        saturates the steering command instead of steering towards the route,
        and the training data was generated with that guarantee in force.
        """
        acc, i = 0.0, self.idx
        while i + 1 < len(self.plan_xy) and acc < distance:
            acc += math.dist(self.plan_xy[i][:2], self.plan_xy[i + 1][:2])
            i += 1
        for _ in range(200):
            x, y = self._to_ego(self.plan_xy[i][0], self.plan_xy[i][1],
                                ego_xy, compass)
            if x > min_forward or i + 1 >= len(self.plan_xy):
                return x, y
            i += 1
        return self._to_ego(self.plan_xy[i][0], self.plan_xy[i][1],
                            ego_xy, compass)

    def _goal_and_command(self, ego_xy, compass, speed):
        """The next vertex of the *sparse* route, in the ego frame, with its
        navigation command -- which is what `x_command`/`y_command` mean in the
        dataset this model is trained on.

        Their expert takes `command_route[1]`, the next node of the undensified
        route, so the goal sits wherever the planner happened to put that
        vertex: a measured mean of 23.1 m over their data, ranging from 9.6 to
        59.5 m. Feeding a fixed 8 m look-ahead instead -- which is what the
        expert behind our own collection used, and what this method used to
        mirror -- hands the network a goal three times nearer than anything it
        was trained to read, on every frame of every route.
        """
        i = self.sparse_idx
        while i + 1 < len(self.sparse_xy):
            x, y = self._to_ego(self.sparse_xy[i][0], self.sparse_xy[i][1],
                                ego_xy, compass)
            if x > 1.0:                      # the first vertex still ahead
                break
            i += 1
        self.sparse_idx = i
        goal = self._to_ego(self.sparse_xy[i][0], self.sparse_xy[i][1],
                            ego_xy, compass)
        return goal, self.sparse_xy[i][2]

    # ------------------------------------------------------------------- step
    def run_step(self, input_data, timestamp):
        if self.origin is None:
            self._init_plan()
        speed = float(input_data["speed"][1]["speed"])
        compass = float(input_data["imu"][1][-1])
        if not np.isfinite(compass):         # the IMU occasionally returns NaN
            compass = 0.0
        lat, lon = input_data["gps"][1][0], input_data["gps"][1][1]
        ego_xy = self._latlon_to_xy(lat, lon)
        self._advance(ego_xy)
        goal, command = self._goal_and_command(ego_xy, compass, speed)

        strip = self._build_image(input_data)
        lidar = self._build_lidar(input_data)
        batch = self._to_batch(strip, lidar, speed, command, goal)

        force = self.drop_sensor
        if force is None and self.lidar_drop_rate > 0 \
                and self.rng.random() < self.lidar_drop_rate:
            force = "lidar"
        with torch.no_grad():
            out = self.model(batch, force_drop=force)
        wps = out["waypoints"][0].cpu().numpy()
        v_des = None
        if "speed_logits" in out:
            v_des = float(self.model.query_readout.expected_speed(
                out["speed_logits"])[0])
        hazard = self._hazard_ahead(lidar)
        # Predicted traffic in the corridor overrides the trajectory. The
        # waypoints are what the policy intends; this is what it sees, and when
        # the two disagree about an occupied cell straight ahead the safe
        # reading wins.
        d_veh = (self._bev_vehicle_distance(out) if self.brake_on_bev
                 else None)
        veh_ahead = d_veh is not None and d_veh < self.BRAKE_DIST_M
        if veh_ahead:
            hazard = True                    # also holds the creep nudge back
        steer, throttle, brake = self.ctrl.step(
            wps, speed, hazard=hazard, v_des=0.0 if veh_ahead else v_des)
        if self.debug_dir:
            self._trace(batch, wps, speed, command, goal, out,
                        steer, throttle, brake, hazard)
            if self.step_i % 20 == 0:
                self._dump_debug(strip, wps, speed, command, goal, out)
        self.step_i += 1
        return carla.VehicleControl(steer=float(steer), throttle=float(throttle),
                                    brake=float(brake))

    # --------------------------------------------------------------- helpers
    def _build_image(self, input_data):
        imgs = {}
        for key, name in (("rgb_left", "cam_left"), ("rgb_front", "cam_front"),
                          ("rgb_right", "cam_right")):
            bgra = input_data[key][1]
            imgs[name] = bgra[:, :, :3][:, :, ::-1]        # BGRA -> RGB
        return stitch_cameras(imgs)

    def _build_lidar(self, input_data):
        pts = np.array(input_data["lidar"][1], dtype=np.float32).reshape(-1, 4)
        pts = pts.copy()
        pts[:, 1] *= -1.0            # UE4 left-handed -> our right-handed frame
        return pts

    def _to_batch(self, strip, lidar, speed, command, goal):
        img = (strip.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        feats, coords, mask = pillarize(lidar, self.cfg.model.lidar)
        batch = {
            "image": torch.from_numpy(img.transpose(2, 0, 1).copy()).unsqueeze(0),
            "pillar_feats": feats,
            "pillar_coords": coords,
            "pillar_mask": mask,
            "pillar_batch": torch.zeros(feats.shape[0], dtype=torch.long),
            "speed": torch.tensor([[speed]], dtype=torch.float32),
            "command": torch.tensor([command], dtype=torch.long),
            "goal": torch.tensor([list(goal)], dtype=torch.float32),
        }
        return {k: v.to(self.device) for k, v in batch.items()}

    # Lane-width corridor searched in the predicted BEV, and the following
    # distance under which the car gives way.
    #
    # The signal is the distance to the nearest predicted vehicle, not whether
    # one is present. Presence does not discriminate: Longest6 runs 256 vehicles
    # and something is in the corridor most of the time, so a binary rule brakes
    # on 38% of the frames the expert drives straight through. Measured against
    # the expert over 800 held-out frames, distance does:
    #
    #     nearer than  5 m -> fires on 36% of its stops,  1% of its driving
    #     nearer than  7 m -> fires on 45% of its stops,  4% of its driving
    #     nearer than 10 m -> fires on 50% of its stops, 29% of its driving
    #
    # 7 m is the knee. It catches under half of the expert's stops, which is
    # expected -- the rest are red lights, which this cannot see -- so it is an
    # additional layer under the policy, not a replacement for it.
    BRAKE_CORRIDOR_M = (-1.4, 1.4)     # lateral half-width
    BRAKE_SKIP_M = 4.0                 # the ego's own footprint in the raster
    BRAKE_DIST_M = 7.0
    BRAKE_ROW_CELLS = 2                # cells per row before a row counts
    VEHICLE_CLASS = 3          # decode_topdown: 0 free 1 road 2 lane 3 vehicle

    # Box swept by the car during one creep nudge, plus a margin: 4 m/s for
    # 1.5 s is 6 m, and the ego is about 2 m wide.
    HAZARD_BOX = (1.0, 7.0, -1.2, 1.2, -1.8, 1.0)   # x0, x1, y0, y1, z0, z1

    def _bev_vehicle_distance(self, out):
        """Metres to the nearest predicted vehicle ahead, or None if no head.

        The policy collided with other vehicles 3.92 times per route, and the
        controller had nothing to brake on: its only hazard signal was a raw
        LiDAR occupancy box, which fires on kerbs, walls and parked cars alike
        and so had to be kept small enough to be nearly useless in traffic.
        Once the auxiliary head is trained with an agent class it predicts the
        one thing that matters here, and it comes free with the forward pass
        the policy already runs.
        """
        seg = out.get("bev_seg")
        if seg is None:
            return None
        cls = seg[0].argmax(0).cpu().numpy()          # H x W class map
        h, w = cls.shape
        x0, x1 = self.cfg.model.lidar.x_range
        y0, y1 = self.cfg.model.lidar.y_range
        b0, b1 = self.BRAKE_CORRIDOR_M
        # Rows run from the far edge down to the ego, columns left to right --
        # the arrangement decode_topdown produces and check_bev_alignment
        # measured against the LiDAR.
        c0 = int(round((b0 - y0) / (y1 - y0) * w))
        c1 = int(round((b1 - y0) / (y1 - y0) * w))
        c0, c1 = max(0, min(c0, w)), max(0, min(c1, w))
        if c1 <= c0:
            return None
        rows = np.arange(h)
        dist = x1 - (rows + 0.5) * (x1 - x0) / h      # metres ahead per row
        occupied = ((cls[:, c0:c1] == self.VEHICLE_CLASS).sum(1)
                    >= self.BRAKE_ROW_CELLS)
        # Skip the ego's own footprint: the target paints agents over whatever
        # they stand on, and the car itself reaches about 3 m past the sensor.
        ahead = occupied & (dist > self.BRAKE_SKIP_M)
        return float(dist[ahead].min()) if ahead.any() else float("inf")

    def _hazard_ahead(self, lidar, min_points=8):
        """Is something solid directly in front of the car?

        Creeping exists to break a self-sustaining standstill, but the car is
        often stationary precisely because an obstacle is there, and nudging
        into it would turn a blocked route into a collision.  A handful of
        returns inside the swept box is enough to hold position; a single stray
        point is not, hence the count threshold.
        """
        if lidar is None or len(lidar) == 0:
            return False
        x0, x1, y0, y1, z0, z1 = self.HAZARD_BOX
        m = ((lidar[:, 0] > x0) & (lidar[:, 0] < x1)
             & (lidar[:, 1] > y0) & (lidar[:, 1] < y1)
             & (lidar[:, 2] > z0) & (lidar[:, 2] < z1))
        return bool(m.sum() >= min_points)

    def _trace(self, batch, wps, speed, command, goal, out, steer, throttle,
               brake, hazard=False):
        """One JSONL record per frame, holding every quantity the network is fed.

        Two of the three defects found so far were train/eval input mismatches
        that open-loop validation cannot see -- the look-ahead goal was 3-6x
        beyond its training range while the waypoint error stayed at 0.137 m.
        The fix for that class of bug is to compare the distribution of what the
        agent actually builds against the dataset it was trained on, which needs
        the inputs themselves rather than a picture of them.
        """
        os.makedirs(self.debug_dir, exist_ok=True)
        img = batch["image"]
        rec = {
            "step": self.step_i,
            "speed": round(speed, 3),
            "command": int(command),
            "goal_x": round(float(goal[0]), 3),
            "goal_y": round(float(goal[1]), 3),
            "wp": [[round(float(a), 3), round(float(b), 3)] for a, b in wps],
            "gate": round(float(out["gate"][0]), 4),
            "steer": round(float(steer), 4),
            "throttle": round(float(throttle), 4),
            "brake": round(float(brake), 4),
            # normalised-image moments: catch a colour-order or scaling error
            "img_mean": round(float(img.mean()), 4),
            "img_std": round(float(img.std()), 4),
            # how much LiDAR actually survives the BEV crop
            "n_pillars": int(batch["pillar_mask"].shape[0]),
            "n_lidar_pts": int(batch["pillar_mask"].sum()),
            "route_idx": self.idx,
            # Whether the creep interlock is holding the car. Without this the
            # only way to tell a policy that will not start from a creep that is
            # forbidden from firing is to run the route twice with creeping on
            # and off and compare -- which was done, and gave byte-identical
            # scores on two routes and a large swing on a third, explaining
            # nothing. These two flags separate the cases in one run.
            "hazard": bool(hazard),
            "creeping": bool(self.ctrl.creep_steps > 0),
            "stuck_steps": int(self.ctrl.stuck_steps),
        }
        with open(os.path.join(self.debug_dir, "trace.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _dump_debug(self, strip, wps, speed, command, goal, out):
        """Occasional panel so a frame/sign error is visible immediately instead
        of showing up as an unexplained low score."""
        os.makedirs(self.debug_dir, exist_ok=True)
        canvas = cv2.cvtColor(strip, cv2.COLOR_RGB2BGR).copy()
        txt = (f"v={speed:.1f} cmd={command} goal=({goal[0]:.1f},{goal[1]:.1f}) "
               f"wp0=({wps[0][0]:.1f},{wps[0][1]:.1f}) "
               f"gate={float(out['gate'][0]):.2f}")
        cv2.putText(canvas, txt, (6, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(self.debug_dir, f"{self.step_i:06d}.jpg"), canvas)
