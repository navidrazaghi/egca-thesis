"""Dataset for expert demonstrations collected in CARLA (Sec. 5-1).

Each frame directory layout (written by carla_sim/collect_data.py):
    rgb/000123.jpg          stitched 3-camera image 704 x 160
    lidar/000123.npy        N x 4 point cloud, float16, cropped to the ROI
    bev_seg/000123.png      128 x 128 uint8 class map
    depth/000123.npy        80 x 352 float16 normalized inverse depth
    measurements/000123.json  {speed, command, goal_x, goal_y, x, y, yaw, ...}
    labels/000123.json      {waypoints: [[x, y] * T]}   <- egca/data/build_labels.py

Only frames that have a label file are used: `build_labels.py` deliberately
leaves the last T frames of every route unlabelled (no future is available) and
drops frames whose pose log is discontinuous.
"""
import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..models.lidar_encoder import pillarize


class CarlaDrivingDataset(Dataset):
    """Frames of the collected routes.

    The validation split is taken from held-out *routes* of the training towns,
    never from a held-out town: Town05 is reserved for the closed-loop
    generalization measurement of Appendix C and must not influence training or
    model selection in any way -- not even as a validation signal.  Routes are
    split by index, so the split is identical across seeds and reruns.
    """

    def __init__(self, cfg, towns, augment=False, synthetic_len=0, split="train"):
        self.cfg = cfg
        self.augment = augment
        self.synthetic_len = synthetic_len
        self.frames = []
        if synthetic_len:            # random data for smoke tests / CI
            return
        root = cfg.data.root
        val_every = int(getattr(cfg.data, "val_every_n_routes", 20))
        for town in towns:
            tdir = os.path.join(root, town)
            if not os.path.isdir(tdir):
                continue
            routes = sorted(r for r in os.listdir(tdir)
                            if os.path.isdir(os.path.join(tdir, r)))
            for i, route in enumerate(routes):
                if (split == "val") != (i % val_every == val_every - 1):
                    continue
                # labels/ is the authoritative frame list (see build_labels.py)
                ldir = os.path.join(tdir, route, "labels")
                if not os.path.isdir(ldir):
                    continue
                for f in sorted(os.listdir(ldir)):
                    if f.endswith(".json"):
                        self.frames.append((os.path.join(tdir, route),
                                            f[:-5]))
        if not synthetic_len and not self.frames:
            raise RuntimeError(
                f"no labelled {split} frames under {root} for towns {towns}. "
                "Did you run: python -m egca.data.build_labels --root <root> ?")

    def __len__(self):
        return self.synthetic_len or len(self.frames)

    def _augment_img(self, img):
        if np.random.rand() < 0.5:   # color jitter
            img = np.clip(img * np.random.uniform(0.8, 1.2, size=(1, 1, 3))
                          + np.random.uniform(-12, 12), 0, 255)
        return img

    def _load(self, idx):
        h, w = self.cfg.model.camera.image_size
        sh, sw = self.cfg.data.seg_size
        dh, dw = self.cfg.data.depth_size
        T = self.cfg.model.decoder.horizon
        if self.synthetic_len:
            rng = np.random.default_rng(idx)
            img = rng.uniform(0, 255, (h, w, 3)).astype(np.float32)
            pts = rng.uniform([0, -16, -2, 0], [32, 16, 1, 1],
                              (2048, 4)).astype(np.float32)
            meas = {"speed": float(rng.uniform(0, 10)), "command": int(rng.integers(4)),
                    "goal_x": 8.0, "goal_y": 0.5,
                    "waypoints": rng.uniform(-1, 6, (T, 2)).tolist()}
            seg = rng.integers(0, self.cfg.model.aux.bev_classes,
                               (sh, sw)).astype(np.int64)
            depth = rng.uniform(0, 1, (dh, dw)).astype(np.float32)
            return img, pts, meas, seg, depth
        base, fid = self.frames[idx]
        img = cv2.cvtColor(cv2.imread(os.path.join(base, "rgb", fid + ".jpg")),
                           cv2.COLOR_BGR2RGB).astype(np.float32)
        pts = np.load(os.path.join(base, "lidar", fid + ".npy")).astype(np.float32)
        with open(os.path.join(base, "measurements", fid + ".json")) as f:
            meas = json.load(f)
        with open(os.path.join(base, "labels", fid + ".json")) as f:
            meas["waypoints"] = json.load(f)["waypoints"]
        seg = cv2.imread(os.path.join(base, "bev_seg", fid + ".png"),
                         cv2.IMREAD_GRAYSCALE).astype(np.int64)
        depth = np.load(os.path.join(base, "depth", fid + ".npy")).astype(np.float32)
        return img, pts, meas, seg, depth

    def __getitem__(self, idx):
        img, pts, meas, seg, depth = self._load(idx)
        if self.augment:
            img = self._augment_img(img)
        img = (img / 255.0 - np.array([0.485, 0.456, 0.406])) \
            / np.array([0.229, 0.224, 0.225])
        feats, coords, mask = pillarize(pts, self.cfg.model.lidar)
        return {
            "image": torch.from_numpy(img.transpose(2, 0, 1).copy()).float(),
            "pillar_feats": feats, "pillar_coords": coords, "pillar_mask": mask,
            "speed": torch.tensor([meas["speed"]], dtype=torch.float32),
            "command": torch.tensor(meas["command"], dtype=torch.long),
            "goal": torch.tensor([meas["goal_x"], meas["goal_y"]],
                                 dtype=torch.float32),
            "waypoints": torch.tensor(meas["waypoints"], dtype=torch.float32),
            "bev_seg": torch.from_numpy(seg),
            "depth": torch.from_numpy(depth).unsqueeze(0),
        }


def collate(samples):
    """Concatenate variable-length pillar tensors with a batch index."""
    out = {}
    for k in samples[0]:
        if k.startswith("pillar_"):
            continue
        out[k] = torch.stack([s[k] for s in samples])
    feats, coords, masks, bidx = [], [], [], []
    for i, s in enumerate(samples):
        feats.append(s["pillar_feats"])
        coords.append(s["pillar_coords"])
        masks.append(s["pillar_mask"])
        bidx.append(torch.full((s["pillar_feats"].shape[0],), i, dtype=torch.long))
    out["pillar_feats"] = torch.cat(feats)
    out["pillar_coords"] = torch.cat(coords)
    out["pillar_mask"] = torch.cat(masks)
    out["pillar_batch"] = torch.cat(bidx)
    return out
