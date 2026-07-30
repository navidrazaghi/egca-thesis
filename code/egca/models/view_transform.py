"""Lift the camera features into the LiDAR's bird's-eye grid (Sec. 4-3, revised).

Why this exists
---------------
The fusion blocks currently attend between camera tokens, which live in the
image plane, and LiDAR tokens, which live in a top-down grid.  Nothing tells the
network how those two frames relate, so the correspondence has to be discovered
from 137k frames of driving.  The measured outcome is that it was not
discovered: the reliability gate saturated at 0.995, dropping LiDAR entirely
left the driving score unchanged, and `camera_only` matched the full model.

Projecting both branches into one metric grid first -- the approach BEVFusion
takes -- removes the problem instead of asking the network to solve it.  After
the transform, a camera cell and a LiDAR cell at the same index refer to the
same square metre of road, so cross-attention between them is comparing things
that are actually comparable.

Geometry
--------
The three cameras share an optical centre and differ only in yaw, and the strip
the network sees is a crop-and-concatenate of their images, so a strip column
maps to a known bearing.  With CAM_W = 320, CAM_FOV = 60 the focal length is

    f = (CAM_W / 2) / tan(CAM_FOV / 2) = 277.13 px

and the stitch (crop = 64) lays the segments out as

    strip col   0..255  -> cam_left  col u,        yaw -55 deg
    strip col 256..511  -> cam_front col u - 224,  yaw   0 deg
    strip col 512..703  -> cam_right col u - 448,  yaw +55 deg

CARLA yaw is left-handed and positive to the right, while the point cloud is
converted to a right-handed frame with +y to the left, so a camera mounted at
CARLA yaw psi has its boresight at theta0 = -psi in the frame used here.  For a
pixel offset (dy, dz) = ((u_cam - cx) / f, (v - cy) / f), pointing right and
down in the image, the ray direction in the LiDAR frame is

    x =  cos(theta0) + dy * sin(theta0)
    y =  sin(theta0) - dy * cos(theta0)
    z = -dz

scaled so the component along the optical axis is 1, which is the convention
CARLA's depth buffer uses.  A point at depth d is then

    P = cam_offset + d * (x, y, z)

with cam_offset the camera position expressed relative to the LiDAR mount:
the camera sits at (1.3, 0, 2.3) and the LiDAR at (0, 0, 2.5), so the offset is
(1.3, 0, -0.2).  Getting any sign in that block wrong produces a plausible-
looking BEV map that disagrees with the point cloud, which is why
`tools/check_view_transform.py` round-trips known 3-D points through it.
"""
import math

import torch
import torch.nn as nn

from ..carla_sim.sensors import CAM_FOV, CAM_H, CAM_W, STITCH_W, CAMERAS

# Camera position minus LiDAR position: both are given in the vehicle frame in
# sensors.py, and every geometric quantity here lives in the LiDAR frame.
CAM_OFFSET = (1.3, 0.0, 2.3 - 2.5)


def _strip_column_to_camera(u):
    """Strip column -> (column within its source image, that camera's yaw)."""
    crop = (3 * CAM_W - STITCH_W) // 4          # 64, see sensors.stitch_cameras
    seg = CAM_W - crop                          # 256 columns survive per camera
    yaws = [y for _, y in CAMERAS]
    if u < seg:
        return u, yaws[0]                       # left: cols 0..255
    if u < 2 * seg:
        return (u - seg) + crop // 2, yaws[1]   # front: cols 32..287
    return (u - 2 * seg) + crop, yaws[2]        # right: cols 64..


def ray_directions(feat_h, feat_w, device, dtype=torch.float32):
    """Direction in the LiDAR frame for each cell of the camera feature map.

    Returns feat_h x feat_w x 3, scaled so the component along each camera's
    optical axis is 1, so multiplying by CARLA's planar depth gives the point.
    """
    f = (CAM_W / 2.0) / math.tan(math.radians(CAM_FOV / 2.0))
    cx, cy = CAM_W / 2.0, CAM_H / 2.0
    sy, sx = CAM_H / feat_h, STITCH_W / feat_w   # stride actually realised

    dirs = torch.zeros(feat_h, feat_w, 3, dtype=dtype)
    for j in range(feat_w):
        u_strip = (j + 0.5) * sx
        u_cam, yaw_deg = _strip_column_to_camera(u_strip)
        theta0 = math.radians(-yaw_deg)
        c0, s0 = math.cos(theta0), math.sin(theta0)
        dy = (u_cam - cx) / f                    # right in image
        for i in range(feat_h):
            v = (i + 0.5) * sy
            dz = (v - cy) / f                    # down in image
            dirs[i, j, 0] = c0 + dy * s0
            dirs[i, j, 1] = s0 - dy * c0
            dirs[i, j, 2] = -dz
    return dirs.to(device)


class CameraToBEV(nn.Module):
    """Lift-splat of the camera features onto the LiDAR's BEV grid.

    A depth distribution is predicted per feature cell and the cell's feature is
    splatted into every bin weighted by its probability, rather than committing
    to one depth.  A wrong hard depth puts the feature in the wrong square metre
    with full confidence; a distribution degrades gracefully and, because the
    dataset already stores a metric depth map, it can be supervised directly
    instead of being left to emerge from the driving loss alone.
    """

    def __init__(self, dim, grid_hw, x_range, y_range,
                 depth_bins=32, depth_min=1.0, depth_max=35.0):
        super().__init__()
        self.dim = dim
        self.grid_h, self.grid_w = grid_hw          # x cells, y cells
        self.x0, self.x1 = x_range
        self.y0, self.y1 = y_range
        self.depth_min, self.depth_max = depth_min, depth_max
        self.depth_bins = depth_bins
        self.depth_head = nn.Conv2d(dim, depth_bins, 1)
        self.register_buffer(
            "bin_depths",
            torch.linspace(depth_min, depth_max, depth_bins).view(1, -1, 1, 1),
            persistent=False)
        self._dirs = {}

    def _directions(self, h, w, device, dtype):
        key = (h, w, device, dtype)
        if key not in self._dirs:
            self._dirs[key] = ray_directions(h, w, device, dtype)
        return self._dirs[key]

    def forward(self, feat, return_depth=False):
        """feat: B x d x h x w  ->  B x d x grid_h x grid_w."""
        b, d, h, w = feat.shape
        dirs = self._directions(h, w, feat.device, feat.dtype)      # h x w x 3
        logits = self.depth_head(feat)                              # B x D x h x w
        prob = logits.softmax(dim=1)

        # Points: B x D x h x w x 3, from direction * depth + camera offset.
        depths = self.bin_depths.to(feat.dtype)                     # 1 x D x 1 x 1
        pts = dirs.view(1, 1, h, w, 3) * depths.view(1, -1, 1, 1, 1)
        off = torch.tensor(CAM_OFFSET, device=feat.device, dtype=feat.dtype)
        pts = pts + off.view(1, 1, 1, 1, 3)

        cell_x = (self.x1 - self.x0) / self.grid_h
        cell_y = (self.y1 - self.y0) / self.grid_w
        ix = ((pts[..., 0] - self.x0) / cell_x).floor().long()
        iy = ((pts[..., 1] - self.y0) / cell_y).floor().long()
        # ix, iy and valid are 1 x D x h x w: the geometry is the same for every
        # sample in the batch, only the depth probabilities differ.
        valid = ((ix >= 0) & (ix < self.grid_h)
                 & (iy >= 0) & (iy < self.grid_w)).to(feat.dtype)

        flat_idx = (ix.clamp(0, self.grid_h - 1) * self.grid_w
                    + iy.clamp(0, self.grid_w - 1))                 # 1 x D x h x w
        flat_idx = flat_idx.reshape(1, -1).expand(b, -1)            # B x N
        weight = (prob * valid).reshape(b, 1, -1)                   # B x 1 x N

        # feature repeated over depth bins, weighted by that bin's probability
        contrib = feat.reshape(b, d, 1, h * w).expand(b, d, self.depth_bins,
                                                      h * w)
        contrib = contrib.reshape(b, d, -1) * weight                # B x d x N

        bev = feat.new_zeros(b, d, self.grid_h * self.grid_w)
        bev.scatter_add_(2, flat_idx.unsqueeze(1).expand(b, d, -1), contrib)
        bev = bev.reshape(b, d, self.grid_h, self.grid_w)
        if return_depth:
            return bev, logits
        return bev

    def expected_depth(self, logits):
        """Bin distribution -> one metric depth per cell, for supervision."""
        return (logits.softmax(dim=1) * self.bin_depths).sum(dim=1)
