"""Full EGCA driving policy (Fig. 4-1): encoders + EGCA fusion + GRU decoder
+ auxiliary heads. ~29 M parameters with the default config (Table 4-1);
the two auxiliary heads (~1.3 M) are dropped at inference time.
"""
import torch
import torch.nn as nn

from .camera_encoder import CameraEncoder
from .lidar_encoder import LidarEncoder
from .fusion import EGCAFusion
from .decoder import WaypointDecoder, MeasurementEncoder
from .heads import BEVSegHead, DepthHead


class EGCAPolicy(nn.Module):
    def __init__(self, cfg, sensor_dropout=0.15):
        super().__init__()
        self.cfg = cfg
        m = cfg.model
        d = m.embed_dim
        self.camera = CameraEncoder(d, m.camera.backbone, m.camera.pretrained)
        self.lidar = LidarEncoder(m.lidar, d)
        self.fusion = EGCAFusion(
            dim=d, num_blocks=m.fusion.num_blocks, num_heads=m.fusion.num_heads,
            ffn_dim=m.fusion.ffn_dim, attention=m.fusion.attention,
            gate=m.fusion.gate, gate_hidden=m.fusion.gate_hidden,
            sensor_dropout=sensor_dropout)
        self.measure = MeasurementEncoder(d)
        self.decoder = WaypointDecoder(d, m.decoder.hidden_dim, m.decoder.horizon)
        self.seg_head = BEVSegHead(d, m.aux.bev_classes) if m.aux.bev_seg else None
        self.depth_head = DepthHead(d) if m.aux.depth else None

    def forward(self, batch, force_drop=None):
        """batch keys: image B3HW, pillar_feats, pillar_coords, pillar_mask,
        pillar_batch, speed B x 1, command B, goal B x 2."""
        b = batch["image"].shape[0]
        fc, _, _ = self.camera(batch["image"])
        fl, lidar_hw = self.lidar(batch["pillar_feats"], batch["pillar_coords"],
                                  batch["pillar_mask"], batch["pillar_batch"], b)
        z, gate, aux_tokens, _ = self.fusion(fc, fl, force_drop=force_drop)
        m = self.measure(batch["speed"], batch["command"])
        waypoints = self.decoder(z, m, batch["goal"])
        out = {"waypoints": waypoints, "gate": gate}
        if self.training and aux_tokens is not None:
            aux_c, aux_l = aux_tokens
            if self.seg_head is not None:
                out["bev_seg"] = self.seg_head(aux_l, lidar_hw)
            if self.depth_head is not None:
                s = self.camera.STRIDE
                cam_hw = (batch["image"].shape[2] // s,
                          batch["image"].shape[3] // s)
                out["depth"] = self.depth_head(aux_c, cam_hw)
        return out

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
