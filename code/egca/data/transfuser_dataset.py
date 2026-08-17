"""Read the published TransFuser dataset with our own model's conventions.

Why this exists
---------------
Our own collection covers about 50 routes over 5 towns and 137k frames, and the
model overfits it: validation flattened at epoch 12 while training error kept
falling, and mirror augmentation moved the final number by 0.004 m. Data is the
binding constraint. Training on the dataset the published baselines used --
258,866 frames over 488 routes and 8 towns -- removes that constraint and, more
importantly, makes the comparison in Chapter 5 a measurement under identical
conditions rather than a citation.

Conventions, each verified against the data rather than inferred from their code
-------------------------------------------------------------------------------
Frames are stored per route under

    <scenario>/Routes_..._<Town>_..._Seed<n>/<route>/{rgb,lidar,measurements,...}

`measurements[*].waypoints` holds 8 future ego poses as (x, y, theta) in world
coordinates. Measured spacing is 0.497 s, so the first four are exactly the
0.5 s horizon this model predicts and no resampling is needed.

The local frame is the part that cannot be guessed. Rotating a world offset by
their `pi/2 + theta` matrix yields components whose meaning was established by
grouping the fourth waypoint by navigation command:

    command 1 (left)      component 0 = -1.20
    command 2 (right)     component 0 = +1.57
    command 3 (straight)  component 0 = -0.01

so component 0 grows to the right, and component 1 is negative forward. This
model uses x forward and y left, hence the sign flips in `to_ego`. Getting this
wrong trains the network on a mirrored world, and nothing in the loss would
report it.
"""
import bisect
import glob
import json
import os
import re

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..models.lidar_encoder import pillarize

TOWN = re.compile(r"(Town\d+HD|Town\d+)")

# CARLA RoadOption -> the four commands this model was trained with
# (0 left, 1 right, 2 straight, 3 lane-follow). Values above 4 are lane changes,
# which the expert drives as lane-following.
COMMAND_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 3, 6: 3}


def to_ego(points_world, ego_xy, theta):
    """World (x, y) -> this model's ego frame: x forward, y left."""
    c, s = np.cos(np.pi / 2 + theta), np.sin(np.pi / 2 + theta)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    local = (np.asarray(points_world, dtype=np.float64) - ego_xy) @ rot
    return np.stack([-local[..., 1], -local[..., 0]], axis=-1)


TOPDOWN_PPM = 5        # their renderer's pixels per metre
TOPDOWN_PX = 500       # and the size of the raster it writes


def decode_topdown(path, out_hw, cfg_lidar, with_agents=True):
    """Their encoded BEV PNG -> our class raster.

    0 free, 1 road, 2 lane, and with `with_agents` also 3 vehicle, 4 pedestrian.

    The PNG packs fifteen binary layers into five bit-planes of three colour
    channels, and their own loader keeps only two of them. This read that as
    "the rest is not here" and took the road and lane pair alone, leaving the
    auxiliary head with a map of where the tarmac is and no notion that other
    traffic exists. The policy trained on it collided with other vehicles 3.92
    times per route over the 36 Longest6 routes -- the dominant failure by a
    wide margin, ahead of layout collisions at 1.03.

    The agents were in the same file the whole time. Their encoder documents the
    packing exactly:

        channels  0-4  -> plane 0   road, lane, lights
        channels  5-9  -> plane 1   vehicle, pedestrian
        channels 10-14 -> plane 2   future vehicles

    and cv2 writes plane 0 to blue and plane 1 to green, so vehicles are bit 7
    of the green channel. Their loader looks like it keeps planes 10-11 only
    because it reads the PNG as RGB, where red and blue are swapped relative to
    this one -- both end up with road and lane.

    Occupancy over 120 frames confirms the reading before it is relied on: road
    0.364, lane 0.045, lights 0.0001, vehicle 0.0038, pedestrian 0.0001. The
    vehicle figure matches their 22x9 px vehicle template at 5 px/m over the
    handful of cars inside a 100 x 100 m window.

    Taking the agents from here rather than rasterising `label_raw` boxes also
    avoids re-deriving a coordinate convention: this raster's geometry is
    already established against the LiDAR, and three separate attempts to settle
    the box convention statistically could not separate the candidates, because
    the vehicles carrying enough returns to measure sit almost entirely straight
    ahead where a lateral sign flip changes nothing.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"unreadable topdown: {path}")
    blue = img[:, :, 0]                      # cv2 loads BGR; blue is index 0
    road = (blue & (1 << 7)) > 0
    lane = (blue & (1 << 6)) > 0
    seg = np.zeros(blue.shape, dtype=np.int64)
    seg[road] = 1
    seg[lane] = 2                            # lane wins where both are set
    if with_agents:
        green = img[:, :, 1]
        # Agents are painted over the surface they stand on: a car occupying a
        # lane cell is a car, and labelling it lane would tell the network the
        # cell is drivable.
        seg[(green & (1 << 7)) > 0] = 3      # vehicle
        seg[(green & (1 << 6)) > 0] = 4      # pedestrian

    # Their raster is 100 x 100 m at 5 px/m with the ego at the centre and
    # heading down the image, while this model's BEV grid is 32 m forward by
    # 32 m wide with the ego on the bottom edge and heading up. Resizing the
    # whole thing onto that grid, which is what this did, hands the auxiliary
    # head a map bearing no geometric relation to the frame the network reasons
    # in. Measured against LiDAR -- ground returns must land on drivable area,
    # tall returns must not -- the arrangement below scores 0.434 over 398
    # frames where the plain resize scores 0.081, and the runner-up 0.361.
    seg = seg[::-1]                          # their forward is down the image
    x0, x1 = cfg_lidar.x_range
    y0, y1 = cfg_lidar.y_range
    cy = cx = TOPDOWN_PX // 2
    top = cy - int(x1 * TOPDOWN_PPM)         # ego sits on the bottom edge
    left = cx + int(y0 * TOPDOWN_PPM)
    seg = seg[top:top + int((x1 - x0) * TOPDOWN_PPM),
              left:left + int((y1 - y0) * TOPDOWN_PPM)]

    h, w = out_hw
    if seg.shape != (h, w):
        seg = cv2.resize(seg.astype(np.uint8), (w, h),
                         interpolation=cv2.INTER_NEAREST).astype(np.int64)
    return seg


def decode_depth(path, out_hw, crop_w, near=1.0, far=100.0):
    """Their depth PNG -> the normalised inverse depth this model is trained on.

    CARLA packs metric depth into three bytes; their loader rescales it linearly
    and clips at 50 m, ours uses inverse depth so that resolution concentrates
    in the near field where driving decisions are made.  Converting to metres
    first and then applying our own transform keeps the target identical in
    meaning to the one the collected dataset stores, which is what lets the same
    loss and the same head work unchanged across both sources.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"unreadable depth: {path}")
    b, g, r = img[:, :, 0].astype(np.float32), img[:, :, 1].astype(np.float32), \
        img[:, :, 2].astype(np.float32)
    # Byte order determined by measurement, not by the documented CARLA formula.
    # That formula (R + G*256 + B*65536) puts the median pixel of these images
    # at 644 m with only 4% of the scene inside 50 m, which no street can be;
    # reversing the ends gives a median of 16-63 m and roughly two thirds of the
    # scene within 50 m.  Their own loader also reads the channels this way.
    metres = (b + g * 256.0 + r * 65536.0) / (256 ** 3 - 1) * 1000.0
    if metres.shape[1] != crop_w:            # match the RGB crop exactly
        x0 = (metres.shape[1] - crop_w) // 2
        metres = metres[:, x0:x0 + crop_w]
    d = np.clip(metres, near, far)
    inv = (1.0 / d - 1.0 / far) / (1.0 / near - 1.0 / far)
    h, w = out_hw
    if inv.shape != (h, w):
        inv = cv2.resize(inv, (w, h), interpolation=cv2.INTER_AREA)
    return inv.astype(np.float32)


def load_lidar(path):
    """Their .npy is an object array; the point cloud is its second element."""
    raw = np.load(path, allow_pickle=True)
    pts = np.asarray(raw[1] if raw.dtype == object else raw, dtype=np.float32)
    # Their sensor sits at x=1.3 with a -90 deg yaw; ours is mounted at the ego
    # origin with x forward and y left, which is the frame `pillarize` crops in.
    #
    # The offset is added, not subtracted. A point 8.7 m ahead of their sensor is
    # 10.0 m ahead of the ego origin, so moving into our frame means +1.3.
    # Subtracting instead displaced every cloud by 2.6 m, which nothing in
    # training could reveal -- inputs and targets were shifted together, so the
    # model simply learned the shifted world and reached 0.051 m open-loop --
    # and which broke the moment the agent fed it a real, unshifted cloud in
    # CARLA. The measurement that settles it is the hole the ego vehicle carves
    # in its own returns: at -1.3 it sits at -2.57 m, at +1.3 at 0.03 m.
    out = np.empty_like(pts)
    out[:, 0] = -pts[:, 1] + 1.3
    out[:, 1] = -pts[:, 0]
    out[:, 2] = pts[:, 2]
    out[:, 3] = pts[:, 3]
    return out


class TransfuserDataset(Dataset):
    """Frames of the published TransFuser dataset, served in our model's format.

    The split is by town, not by frame or route: consecutive frames of one route
    are nearly identical, so a frame-level split leaks, and this dataset already
    spans eight towns, which makes a held-out town the honest choice.
    """

    # CarlaDrivingDataset._mirror is borrowed unbound, and it reads this off
    # `self`, so the adapter has to carry it too.
    FLIP_COMMAND = {0: 1, 1: 0, 2: 2, 3: 3}

    def __init__(self, cfg, root, towns, augment=False, split="train"):
        self.cfg = cfg
        self.augment = augment
        self.horizon = cfg.model.decoder.horizon
        self.img_hw = tuple(cfg.model.camera.image_size)
        self.seg_hw = tuple(cfg.data.seg_size)
        self.depth_hw = tuple(cfg.data.depth_size)
        self.frames = []

        wanted = set(towns)
        for meas in glob.glob(os.path.join(root, "*", "*", "*", "measurements",
                                           "*.json")):
            route = os.path.dirname(os.path.dirname(meas))
            town = TOWN.search(route)
            if not town or town.group(1) not in wanted:
                continue
            self.frames.append((route, os.path.basename(meas)[:-5]))
        self.frames.sort()
        if not self.frames:
            raise RuntimeError(
                f"no frames for towns {sorted(wanted)} under {root!r}")

        # Rebuild the trajectory target from the pose the car actually reached
        # instead of the one stored with the frame.  The stored block agrees
        # with reality to 0.080 m while the car is moving and disagrees by a
        # factor of 5.6 on the frames where it pulls away from a standstill: on
        # 126 measured pull-aways the label says 0.82 m of travel over two
        # seconds and the car covered 4.56 m.  Across Town05, 14.1% of stopped
        # frames are pull-aways and only 10% of those carry a label that shows
        # any motion at all.
        #
        # A network fitted to that learns "when stopped, creep", reproduces it
        # to 0.056 m open-loop, and then sits still on the road -- which is the
        # 0.30 m/s it predicts at zero speed and the DS 11.56.  Open loop looked
        # healthy because it was scored against the same wrong target.
        #
        # Relabelling is all frames or none.  Switching definition by speed
        # would put a discontinuity at the threshold and leave two different
        # meanings of "waypoint" in one training set; the realized pose is one
        # definition and it is the one the closed-loop metric rewards.
        self.relabel = bool(getattr(cfg.data, "tf_relabel", True))
        self._pose_xy = None
        if self.relabel:
            self._build_pose_index(root, towns)

    def _build_pose_index(self, root, towns):
        """Per-frame (x, y) and the last index of each frame's route.

        Read once here rather than four extra JSON opens per sample: the loader
        already runs at 80 samples/s and five reads where there was one would
        show up as throughput.  Cached to disk because this is a fixed function
        of the dataset, and a 258k-frame scan is minutes that every run would
        otherwise repeat.
        """
        # Keep the full ordering. Callers subsample `self.frames` after the
        # dataset is built -- train.py caps validation at val_max_frames by
        # replacing the list -- and a position in the subsampled list is not a
        # position in this index. Looked up positionally, the target became the
        # offset between two unrelated frames and validation reported a constant
        # 159.56 m while training error fell normally. Frames are resolved by
        # (route, fid) through this list instead, which no amount of
        # subsampling can invalidate.
        self._all_frames = list(self.frames)
        key = "%s_%d" % ("-".join(sorted(towns)), len(self.frames))
        cache = os.path.join(root, ".egca_poses_%s.npz" % key)
        if os.path.exists(cache):
            try:
                z = np.load(cache)
                if len(z["xy"]) == len(self.frames):
                    self._pose_xy, self._route_end = z["xy"], z["end"]
                    return
            except Exception:
                pass                      # a corrupt cache is rebuilt, not fatal

        xy = np.full((len(self.frames), 2), np.nan, dtype=np.float64)
        for i, (route, fid) in enumerate(self.frames):
            try:
                with open(os.path.join(route, "measurements",
                                       fid + ".json")) as f:
                    m = json.load(f)
                xy[i] = (m["x"], m["y"])
            except Exception:
                pass                      # stays NaN and is rejected in _pose

        # frames.sort() puts one route's frames in one contiguous block, so the
        # route boundary is where the directory changes
        end = np.empty(len(self.frames), dtype=np.int64)
        start = 0
        for i in range(1, len(self.frames) + 1):
            if i == len(self.frames) or self.frames[i][0] != self.frames[start][0]:
                end[start:i] = i - 1
                start = i
        self._pose_xy, self._route_end = xy, end
        try:
            np.savez(cache, xy=xy, end=end)
        except Exception:
            pass                          # read-only dataset root is fine

    def __len__(self):
        return len(self.frames)

    def _image(self, route, fid):
        """Their 960 x 160 strip cropped to the 704 x 160 this model expects."""
        img = cv2.imread(os.path.join(route, "rgb", fid + ".png"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        h, w = self.img_hw
        if img.shape[1] != w:                    # centre crop, as their loader does
            x0 = (img.shape[1] - w) // 2
            img = img[:, x0:x0 + w]
        if img.shape[0] != h:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        return img

    def _pose(self, idx):
        """Ego-frame targets for one frame, or None if its pose is unusable."""
        route, fid = self.frames[idx]
        with open(os.path.join(route, "measurements", fid + ".json")) as f:
            meas = json.load(f)

        ego = np.array([meas["x"], meas["y"]], dtype=np.float64)
        theta = float(meas["theta"])
        if not np.isfinite(theta):               # the IMU occasionally returns NaN
            theta = 0.0

        if self.relabel:
            # The next `horizon` frames of this route are its future, at the
            # measured 0.5 s stride -- the same spacing the stored block uses.
            j = bisect.bisect_left(self._all_frames, (route, fid))
            if (j >= len(self._all_frames)
                    or self._all_frames[j] != (route, fid)):
                return None
            if j + self.horizon > int(self._route_end[j]):
                # No future to read. Padding by holding the last pose is what
                # produced a standstill target here, so the frame is refused and
                # __getitem__ walks back to one that has a real future.
                return None
            wp_world = self._pose_xy[j + 1:j + 1 + self.horizon]
            if not np.isfinite(wp_world).all():
                return None
            wp = to_ego(wp_world, ego, theta)
            # 25 m in 0.5 s is 180 km/h, which this expert never drives, so a
            # step that large means the target is not this frame's future. The
            # positional-index bug produced exactly that and nothing rejected
            # it: validation reported a constant 159.56 m for eleven hours
            # while the training curve looked perfectly healthy. Refusing the
            # frame turns any future variant of the same mistake into missing
            # data, which is visible, instead of a plausible-looking number.
            if float(np.abs(np.diff(np.vstack([[0.0, 0.0], wp]), axis=0)
                            ).sum(axis=1).max()) > 25.0:
                return None
        else:
            wp_world = np.asarray(meas["waypoints"], dtype=np.float64)
            wp_world = wp_world[:, :2] if wp_world.size else np.zeros((0, 2))
            wp = to_ego(wp_world, ego, theta)[:self.horizon]
            if len(wp) == 0:                     # no future recorded at all
                return None
            if len(wp) < self.horizon:           # end of route: hold last pose
                wp = np.vstack([wp,
                                np.repeat(wp[-1:], self.horizon - len(wp), 0)])

        goal = to_ego(np.array([meas["x_command"], meas["y_command"]]),
                      ego, theta)
        speed = float(meas["speed"])
        if not (np.isfinite(wp).all() and np.isfinite(goal).all()
                and np.isfinite(speed)):
            return None
        return meas, wp, goal, speed

    def __getitem__(self, idx):
        # Roughly one frame in 1500 stores its entire waypoint block as NaN --
        # the expert's own recorder, not this conversion. At batch 128 over
        # 203k frames that is about 135 poisoned batches per epoch, which is
        # what turned the training loss into NaN from epoch 0 while validation
        # started healthy and then decayed as the weights absorbed it; a
        # batch-8 smoke test touches 296 samples and sails straight past it.
        # Substituting a standstill would be worse than the NaN, since it
        # teaches the model to stop at a junction for no visible reason, so the
        # sample is served from a neighbouring frame instead. Walking backwards
        # deterministically keeps the dataset length fixed and the run
        # reproducible from its seed.
        pose = None
        for k in range(8):
            pose = self._pose((idx - k) % len(self.frames))
            if pose is not None:
                idx = (idx - k) % len(self.frames)
                break
        if pose is None:
            raise RuntimeError(f"eight consecutive unusable frames near {idx}")
        meas, wp, goal, speed = pose
        route, fid = self.frames[idx]

        img = self._image(route, fid)
        pts = load_lidar(os.path.join(route, "lidar", fid + ".npy"))
        command = COMMAND_MAP.get(int(meas["command"]), 3)

        # The auxiliary rasters are decoded before augmentation, not after,
        # because the mirror has to reflect every field of the sample at once.
        # Reflecting the image and the cloud while leaving the map and the
        # trajectory alone would be worse than not augmenting at all.
        # The agent classes are only emitted when the head has outputs for them.
        # Writing class 3 into a target a 3-class head is trained against would
        # be an out-of-range label, and the loss would either crash or, with
        # ignore_index set, silently drop every vehicle cell -- which is the
        # state this was already in, by a different route.
        seg = decode_topdown(
            os.path.join(route, "topdown", "encoded_" + fid + ".png"),
            self.seg_hw, self.cfg.model.lidar,
            with_agents=self.cfg.model.aux.bev_classes >= 5) \
            if self.cfg.model.aux.bev_seg else None
        dep = decode_depth(os.path.join(route, "depth", fid + ".png"),
                           self.depth_hw, self.img_hw[1]) \
            if self.cfg.model.aux.depth else None

        if self.augment:
            from .dataset import CarlaDrivingDataset
            img = CarlaDrivingDataset._augment_img(self, img)
            pts = CarlaDrivingDataset._augment_lidar(self, pts)
            # This rig is symmetric about the ego's forward axis -- equal-FOV
            # cameras at -60/0/+60 tiling without overlap -- so the reflection
            # is an exact sample rather than an approximation, and it doubles
            # the set that was the binding constraint in the first place.
            if np.random.rand() < float(getattr(self.cfg.train, "flip_prob", 0.0)):
                blank_seg = np.zeros(self.seg_hw, dtype=np.int64)
                blank_dep = np.zeros(self.depth_hw, dtype=np.float32)
                (img, pts, s2, d2, wp, goal,
                 command) = CarlaDrivingDataset._mirror(
                    self, img, pts,
                    blank_seg if seg is None else seg,
                    blank_dep if dep is None else dep,
                    wp, goal, command)
                seg = None if seg is None else s2
                dep = None if dep is None else d2

        img = (img / 255.0 - np.array([0.485, 0.456, 0.406])) \
            / np.array([0.229, 0.224, 0.225])
        feats, coords, mask = pillarize(pts, self.cfg.model.lidar)
        out = {
            "image": torch.from_numpy(img.transpose(2, 0, 1).copy()).float(),
            "pillar_feats": feats, "pillar_coords": coords, "pillar_mask": mask,
            "speed": torch.tensor([speed], dtype=torch.float32),
            "command": torch.tensor(command, dtype=torch.long),
            "goal": torch.from_numpy(np.asarray(goal, dtype=np.float32)),
            "waypoints": torch.from_numpy(np.asarray(wp, dtype=np.float32)),
        }
        if seg is not None:
            out["bev_seg"] = torch.from_numpy(np.ascontiguousarray(seg))
        if dep is not None:
            out["depth"] = torch.from_numpy(
                np.ascontiguousarray(dep)).unsqueeze(0)
        return out
