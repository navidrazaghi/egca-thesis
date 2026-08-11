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


def decode_topdown(path, out_hw, cfg_lidar):
    """Their encoded BEV PNG -> our class raster (0 free, 1 road, 2 lane).

    The PNG packs fifteen binary layers into five bit-planes of three colour
    channels; their own loader keeps channels 10 and 11, which are bit 7 and
    bit 6 of the blue channel and carry drivable area and lane marking.  The
    remaining layers describe agents and are read from `label_raw` instead, so
    the raster here is the same three-class HD-map target their paper uses
    rather than a partly-filled version of ours.
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

        wp_world = np.asarray(meas["waypoints"], dtype=np.float64)
        wp_world = wp_world[:, :2] if wp_world.size else np.zeros((0, 2))
        wp = to_ego(wp_world, ego, theta)[:self.horizon]
        if len(wp) == 0:                         # no future recorded at all
            return None
        if len(wp) < self.horizon:               # end of route: hold the last pose
            wp = np.vstack([wp, np.repeat(wp[-1:], self.horizon - len(wp), 0)])

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
        seg = decode_topdown(
            os.path.join(route, "topdown", "encoded_" + fid + ".png"),
            self.seg_hw, self.cfg.model.lidar) \
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
