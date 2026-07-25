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
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2 = net.layer1, net.layer2
        self.layer3, self.layer4 = net.layer3, net.layer4
        narrow = backbone in ("resnet18", "resnet34")
        c3, c4 = (256, 512) if narrow else (1024, 2048)
        self.lat3 = nn.Conv2d(c3, embed_dim, kernel_size=1)
        self.lat4 = nn.Conv2d(c4, embed_dim, kernel_size=1)
        self.smooth = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True))
        self.embed_dim = embed_dim
        self._pe_cache = {}

    def forward(self, img):
        """img: B x 3 x H x W  ->  tokens: B x (H/16 * W/16) x d, the stride-8
        feature map, and the token grid shape."""
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
        key = (h, w, x.device)
        if key not in self._pe_cache:
            self._pe_cache[key] = sinusoidal_pe_2d(h, w, d, x.device)
        return tokens + self._pe_cache[key], feat_s8, (h, w)
