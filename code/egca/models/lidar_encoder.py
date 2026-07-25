"""LiDAR branch: PointPillars-style encoder -> BEV tokens (Sec. 3-3 / 4-2-2).

Pillarization implements Eq. (3.8): each point is augmented with offsets to
the pillar mean and pillar center; a shared MLP + max pooling (Eq. 3.9)
yields pillar features which are scattered into a 128 x 128 pseudo-image.
A light conv backbone reduces it to a 64 x 64 grid -> N_l = 4096 tokens
(0.5 m per BEV token over the 32 m x 32 m region of interest).
"""
import torch
import torch.nn as nn

from .camera_encoder import sinusoidal_pe_2d


def pillarize(points, cfg):
    """points: N x 4 (x, y, z, intensity) numpy/torch, ego frame.
    Returns (pillar_feats P x M x 9, coords P x 2, mask P x M) tensors."""
    if not torch.is_tensor(points):
        points = torch.as_tensor(points, dtype=torch.float32)
    x0, x1 = cfg.x_range
    y0, y1 = cfg.y_range
    z0, z1 = cfg.z_range
    ps = cfg.pillar_size
    nx = int(round((x1 - x0) / ps))
    ny = int(round((y1 - y0) / ps))

    m = ((points[:, 0] >= x0) & (points[:, 0] < x1)
         & (points[:, 1] >= y0) & (points[:, 1] < y1)
         & (points[:, 2] >= z0) & (points[:, 2] < z1))
    pts = points[m]
    if pts.numel() == 0:
        return (torch.zeros(1, cfg.max_points_per_pillar, 9),
                torch.zeros(1, 2, dtype=torch.long), torch.zeros(1, cfg.max_points_per_pillar))

    ix = ((pts[:, 0] - x0) / ps).long().clamp(0, nx - 1)
    iy = ((pts[:, 1] - y0) / ps).long().clamp(0, ny - 1)
    flat = ix * ny + iy
    uniq, inv = torch.unique(flat, return_inverse=True)
    P = min(len(uniq), cfg.max_pillars)
    M = cfg.max_points_per_pillar

    feats = torch.zeros(P, M, 9)
    mask = torch.zeros(P, M)
    coords = torch.zeros(P, 2, dtype=torch.long)
    counts = torch.zeros(len(uniq), dtype=torch.long)
    order = torch.randperm(len(pts))            # random point subsampling
    for i in order.tolist():
        p = inv[i].item()
        if p >= P or counts[p] >= M:
            continue
        j = counts[p].item()
        xc = x0 + (ix[i].float() + 0.5) * ps
        yc = y0 + (iy[i].float() + 0.5) * ps
        feats[p, j, :4] = pts[i]
        feats[p, j, 7] = pts[i, 0] - xc          # offset to pillar center
        feats[p, j, 8] = pts[i, 1] - yc
        mask[p, j] = 1.0
        coords[p, 0], coords[p, 1] = ix[i], iy[i]
        counts[p] += 1
    # offsets to pillar mean (Eq. 3.8, terms x - x_bar etc.)
    mean = (feats[..., :3] * mask.unsqueeze(-1)).sum(1, keepdim=True) \
        / mask.sum(1, keepdim=True).clamp(min=1).unsqueeze(-1)
    feats[..., 4:7] = (feats[..., :3] - mean) * mask.unsqueeze(-1)
    return feats, coords, mask


class PillarFeatureNet(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(9, out_dim), nn.BatchNorm1d(out_dim),
                                 nn.ReLU(inplace=True))
        self.out_dim = out_dim

    def forward(self, feats, mask):
        """feats: B* x M x 9 -> pillar embedding B* x C via masked max (Eq. 3.9)."""
        P, M, _ = feats.shape
        x = self.mlp(feats.reshape(P * M, 9)).reshape(P, M, -1)
        x = x.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
        x = x.max(dim=1).values
        return torch.nan_to_num(x, neginf=0.0)


class LidarEncoder(nn.Module):
    """PointPillars encoder producing N_l = 4096 BEV tokens (64 x 64 grid)."""

    def __init__(self, cfg, embed_dim=256):
        super().__init__()
        self.cfg = cfg
        self.pfn = PillarFeatureNet(cfg.pillar_feat_dim)
        c = cfg.pillar_feat_dim
        self.grid = (int(round((cfg.x_range[1] - cfg.x_range[0]) / cfg.pillar_size)),
                     int(round((cfg.y_range[1] - cfg.y_range[0]) / cfg.pillar_size)))
        # 128 -> 64 (one strided stage), then one same-resolution stage
        def block(ci, co, stride=1):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, stride=stride, padding=1), nn.BatchNorm2d(co),
                nn.ReLU(inplace=True),
                nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co),
                nn.ReLU(inplace=True))
        self.backbone = nn.Sequential(block(c, 128, stride=2),
                                      block(128, embed_dim))
        self.embed_dim = embed_dim
        self._pe_cache = {}

    def scatter(self, pillar_emb, coords, batch_size, batch_idx):
        nx, ny = self.grid
        canvas = pillar_emb.new_zeros(batch_size, self.pfn.out_dim, nx, ny)
        canvas[batch_idx, :, coords[:, 0], coords[:, 1]] = pillar_emb
        return canvas

    def forward(self, feats, coords, mask, batch_idx, batch_size):
        """Batched pillars (concatenated over the batch with `batch_idx`)."""
        emb = self.pfn(feats, mask)                       # P_total x C
        canvas = self.scatter(emb, coords, batch_size, batch_idx)
        # The scattered pseudo-image is very sparse; strided convs over it are
        # numerically unstable in fp16, so the backbone always runs in fp32.
        with torch.autocast(device_type=canvas.device.type, enabled=False):
            x = self.backbone(canvas.float())             # B x d x 64 x 64
        b, d, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        key = (h, w, x.device)
        if key not in self._pe_cache:
            self._pe_cache[key] = sinusoidal_pe_2d(h, w, d, x.device)
        return tokens + self._pe_cache[key], (h, w)
