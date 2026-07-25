"""Auxiliary heads (Sec. 4-3-3): BEV semantic segmentation from mid-block
LiDAR tokens and depth estimation from mid-block camera tokens.
Both are active only during training.
"""
import torch.nn as nn


def _up(ci, co):
    return nn.Sequential(
        nn.ConvTranspose2d(ci, co, 4, stride=2, padding=1),
        nn.BatchNorm2d(co), nn.ReLU(inplace=True))


class BEVSegHead(nn.Module):
    """64x64 lidar tokens -> num_classes x 128 x 128 logits."""

    def __init__(self, dim=256, num_classes=6):
        super().__init__()
        self.net = nn.Sequential(_up(dim, 128),
                                 nn.Conv2d(128, num_classes, 1))

    def forward(self, tokens, hw):
        b, n, d = tokens.shape
        x = tokens.transpose(1, 2).reshape(b, d, *hw)
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
