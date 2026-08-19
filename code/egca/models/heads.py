"""Auxiliary heads (Sec. 4-3-3): BEV semantic segmentation from mid-block
LiDAR tokens and depth estimation from mid-block camera tokens.
Both are active only during training.
"""
import torch
import torch.nn as nn


def _up(ci, co):
    return nn.Sequential(
        nn.ConvTranspose2d(ci, co, 4, stride=2, padding=1),
        nn.BatchNorm2d(co), nn.ReLU(inplace=True))


class BEVSegHead(nn.Module):
    """64x64 lidar tokens -> num_classes x 128 x 128 logits.

    The token grid and the target run in opposite row orders. The pillar
    canvas indexes rows by x forward -- row 0 is the cell the ego stands on --
    while both BEV targets are written as top-down pictures with the ego on
    the bottom edge, so their row 0 is 32 m ahead (collect_data._bev_px and
    transfuser_dataset.decode_topdown, the latter measured against LiDAR in
    check_bev_alignment). A deconvolution is translation-equivariant and
    cannot express a global flip, and the LiDAR branch has no intra-modal
    attention to route information across the grid, so without the flip below
    the head is asked to predict, at each token, the class of the cell
    mirrored fore-aft about the grid centre -- supervision that is unlearnable
    locally and, worse, teaches whatever does leak through cross-attention to
    describe the wrong end of the road. That is consistent with the measured
    no_aux ablation sitting inside the seed spread of the full model: the
    auxiliary target as trained carried almost no usable signal.

    Flipping the token rows here aligns input and target cell-for-cell and
    keeps the head's output in the target's own convention (row 0 far, ego on
    the bottom edge) -- which is what the agent's _bev_vehicle_distance
    already assumes when it converts rows back to metres.
    """

    def __init__(self, dim=256, num_classes=6):
        super().__init__()
        self.net = nn.Sequential(_up(dim, 128),
                                 nn.Conv2d(128, num_classes, 1))

    def forward(self, tokens, hw):
        b, n, d = tokens.shape
        x = tokens.transpose(1, 2).reshape(b, d, *hw)
        x = torch.flip(x, dims=[2])      # canvas rows (ego first) -> target rows
        return self.net(x)


class DepthHead(nn.Module):
    """10x44 camera tokens -> 1 x 80 x 352 normalized inverse depth."""

    def __init__(self, dim=256):
        super().__init__()
        self.net = nn.Sequential(_up(dim, 128), _up(128, 64), _up(64, 32),
                                 nn.Conv2d(32, 1, 1), nn.Sigmoid())

    def forward(self, tokens, hw):
        b, n, d = tokens.shape
        x = tokens.transpose(1, 2).reshape(b, d, *hw)
        return self.net(x)
