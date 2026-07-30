"""Camera branch: ResNet-34 backbone -> flattened tokens + 2-D sinusoidal PE.

Implements Eq. (4.1) of the thesis: F_c = Flatten(Conv1x1(phi_c(I))) + E_pos.
Input  : stitched 3-camera image, 3 x 160 x 704
Output : stride-16 tokens, N_c = 10 x 44 = 440, of dimension d.

Tokens are emitted at stride 16 (stage 3) and enriched with the stride-32
stage-4 map through a single-level top-down (FPN-style) merge.  The stride-16
resolution is what makes the quadratic cost of full cross-attention the
dominant term of the fusion module (Sec. 4-3-1) and the linear formulation of
Sec. 3-4-4 worthwhile, while the top-down path keeps the receptive field and
the semantic level of the complete backbone.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def sinusoidal_pe_2d(h, w, dim, device=None):
    """2-D sinusoidal positional encoding (dim/2 for rows, dim/2 for cols)."""
    assert dim % 4 == 0, "embed dim must be divisible by 4 for 2-D PE"
    d = dim // 2
    pe = torch.zeros(h, w, dim, device=device)
    div = torch.exp(torch.arange(0, d, 2, device=device).float()
                    * (-math.log(10000.0) / d))
    ys = torch.arange(h, device=device).float().unsqueeze(1) * div  # h x d/2
    xs = torch.arange(w, device=device).float().unsqueeze(1) * div  # w x d/2
    pe[..., 0:d:2] = ys.sin().unsqueeze(1).expand(h, w, -1)
    pe[..., 1:d:2] = ys.cos().unsqueeze(1).expand(h, w, -1)
    pe[..., d::2] = xs.sin().unsqueeze(0).expand(h, w, -1)
    pe[..., d + 1::2] = xs.cos().unsqueeze(0).expand(h, w, -1)
    return pe.reshape(h * w, dim)


class CameraEncoder(nn.Module):
    STRIDE = 16          # stride of the emitted tokens

    def __init__(self, embed_dim=256, backbone="resnet34", pretrained=True):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        net = getattr(torchvision.models, backbone)(weights=weights)
        if backbone.startswith("regnet"):
            # RegNet names its stages differently but lays them out identically:
            # stem at stride 2, then four blocks ending at strides 4/8/16/32, so
            # the stride-16 stage still emits the same 10 x 44 token grid.  Both
            # TransFuser and LEAD found the backbone to be the single most
            # impactful architecture choice, with RegNetY well ahead of a
            # ResNet-34, which is what this branch exists to allow.
            self.stem = net.stem
            t = net.trunk_output
            self.layer1, self.layer2 = t.block1, t.block2
            self.layer3, self.layer4 = t.block3, t.block4
        else:
            self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
            self.layer1, self.layer2 = net.layer1, net.layer2
            self.layer3, self.layer4 = net.layer3, net.layer4
        c3, c4 = self._stage_channels()
        self.lat3 = nn.Conv2d(c3, embed_dim, kernel_size=1)
        self.lat4 = nn.Conv2d(c4, embed_dim, kernel_size=1)
        self.smooth = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True))
        self.embed_dim = embed_dim
        self._pe_cache = {}

    @torch.no_grad()
    def _stage_channels(self):
        """Channel widths of the stride-16 and stride-32 stages.

        Measured with a dummy forward rather than tabulated per backbone: the
        previous hard-coded pair was correct only for the ResNet family, and a
        silently wrong width here becomes a shape error deep in the FPN merge.
        Run in eval mode so the probe leaves no trace in the BatchNorm
        statistics.
        """
        was_training = self.training
        self.eval()
        x = torch.zeros(1, 3, 64, 64)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        f3 = self.layer3(x)
        f4 = self.layer4(f3)
        if was_training:
            self.train()
        return f3.shape[1], f4.shape[1]

    def forward(self, img, with_pe=True):
        """img: B x 3 x H x W  ->  tokens: B x (H/16 * W/16) x d, the stride-8
        feature map, and the token grid shape.

        `with_pe=False` returns the tokens without the image-plane positional
        code.  The bird's-eye variant projects these features into a metric grid
        and adds that grid's own code afterwards; carrying the image-plane code
        through the projection would smear a position in the picture across the
        square metres it lands on, which is not information about the road.
        """
        x = self.stem(img)
        x = self.layer1(x)
        x = self.layer2(x)          # stride 8
        feat_s8 = x
        f3 = self.layer3(x)         # stride 16
        f4 = self.layer4(f3)        # stride 32
        x = self.lat3(f3) + F.interpolate(self.lat4(f4), size=f3.shape[-2:],
                                          mode="nearest")
        x = self.smooth(x)          # B x d x h x w  (stride 16)
        b, d, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)          # B x N x d
        if not with_pe:
            return tokens, feat_s8, (h, w)
        key = (h, w, x.device)
        if key not in self._pe_cache:
            self._pe_cache[key] = sinusoidal_pe_2d(h, w, d, x.device)
        return tokens + self._pe_cache[key], feat_s8, (h, w)
