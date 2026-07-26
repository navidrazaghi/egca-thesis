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
        self.mode = getattr(m.fusion, "mode", "egca")
        # A single-modality ablation does not build the unused branch at all:
        # keeping it would inflate the parameter count of a baseline that is
        # supposed to answer "what does this modality alone achieve".
        self.use_cam = self.mode != "lidar_only"
        self.use_lidar = self.mode != "camera_only"
        self.camera = (CameraEncoder(d, m.camera.backbone, m.camera.pretrained)
                       if self.use_cam else None)
        self.lidar = LidarEncoder(m.lidar, d) if self.use_lidar else None
        self.fusion = EGCAFusion(
            dim=d, num_blocks=m.fusion.num_blocks, num_heads=m.fusion.num_heads,
            ffn_dim=m.fusion.ffn_dim, attention=m.fusion.attention,
            gate=m.fusion.gate, gate_hidden=m.fusion.gate_hidden,
            sensor_dropout=sensor_dropout, mode=self.mode)
        self.measure = MeasurementEncoder(d)
        self.decoder = WaypointDecoder(d, m.decoder.hidden_dim, m.decoder.horizon)
        # True decision-level fusion shares nothing between the branches, so the
        # second modality gets its own decoder rather than reusing the first.
        self.decoder_l = (WaypointDecoder(d, m.decoder.hidden_dim,
                                          m.decoder.horizon)
                          if self.mode == "late" else None)
        # The BEV head reads LiDAR tokens and the depth head camera tokens, so an
        # auxiliary task whose branch does not exist is dropped with it.
        self.seg_head = (BEVSegHead(d, m.aux.bev_classes)
                         if m.aux.bev_seg and self.use_lidar else None)
        self.depth_head = (DepthHead(d)
                           if m.aux.depth and self.use_cam else None)

    def forward(self, batch, force_drop=None):
        """batch keys: image B3HW, pillar_feats, pillar_coords, pillar_mask,
        pillar_batch, speed B x 1, command B, goal B x 2."""
        b = batch["image"].shape[0]
        fc = self.camera(batch["image"])[0] if self.use_cam else None
        lidar_hw = None
        if self.use_lidar:
            fl, lidar_hw = self.lidar(
                batch["pillar_feats"], batch["pillar_coords"],
                batch["pillar_mask"], batch["pillar_batch"], b)
        else:
            fl = None
        z, gate, aux_tokens, _, (zc, zl) = self.fusion(fc, fl,
                                                       force_drop=force_drop)
        m = self.measure(batch["speed"], batch["command"])
        if self.mode == "late":
            # decision-level fusion: decode each modality on its own and average
            # the two trajectories, so no inter-modal information is exchanged
            # anywhere in the network
            wc = self.decoder(zc, m, batch["goal"])
            wl = self.decoder_l(zl, m, batch["goal"])
            waypoints = 0.5 * (wc + wl)
        else:
            waypoints = self.decoder(z, m, batch["goal"])
        out = {"waypoints": waypoints, "gate": gate}
        if self.training and aux_tokens is not None:
            aux_c, aux_l = aux_tokens
            if self.seg_head is not None and aux_l is not None:
                out["bev_seg"] = self.seg_head(aux_l, lidar_hw)
            if self.depth_head is not None and aux_c is not None:
                s = self.camera.STRIDE
                cam_hw = (batch["image"].shape[2] // s,
                          batch["image"].shape[3] // s)
                out["depth"] = self.depth_head(aux_c, cam_hw)
        return out

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
