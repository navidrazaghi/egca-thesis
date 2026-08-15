"""Full EGCA driving policy (Fig. 4-1): encoders + EGCA fusion + GRU decoder
+ auxiliary heads. ~29 M parameters with the default config (Table 4-1);
the two auxiliary heads (~1.3 M) are dropped at inference time.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .camera_encoder import CameraEncoder, sinusoidal_pe_2d
from .lidar_encoder import LidarEncoder
from .fusion import EGCAFusion
from .view_transform import CameraToBEV
from .query_readout import QueryReadout
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
        # "bev" projects the camera features into the LiDAR's metric grid before
        # fusion, so a camera token and a LiDAR token at the same index describe
        # the same square metre.  In "image" the two branches are in different
        # frames and the correspondence has to be learned.
        self.camera_space = getattr(m.fusion, "camera_space", "image")
        if self.camera_space not in ("image", "bev"):
            raise ValueError(f"unknown camera_space {self.camera_space!r}")
        self.cam_to_bev = None
        if self.camera_space == "bev" and self.use_cam:
            grid = (int(round((m.lidar.x_range[1] - m.lidar.x_range[0])
                              / m.lidar.pillar_size)) // 2,
                    int(round((m.lidar.y_range[1] - m.lidar.y_range[0])
                              / m.lidar.pillar_size)) // 2)
            self.cam_to_bev = CameraToBEV(
                d, grid, tuple(m.lidar.x_range), tuple(m.lidar.y_range),
                depth_bins=getattr(m.fusion, "depth_bins", 32))
            self._bev_pe = {}
        self.fusion = EGCAFusion(
            dim=d, num_blocks=m.fusion.num_blocks, num_heads=m.fusion.num_heads,
            ffn_dim=m.fusion.ffn_dim, attention=m.fusion.attention,
            gate=m.fusion.gate, gate_hidden=m.fusion.gate_hidden,
            sensor_dropout=sensor_dropout, mode=self.mode,
            # getattr, not attribute access: checkpoints trained before these
            # options existed carry a config without them and must still build.
            pooling=getattr(m.fusion, "pooling", "mean"),
            gate_form=getattr(m.fusion, "gate_form", "scalar"),
            pool_queries=getattr(m.fusion, "pool_queries", 4),
            ego_cond=getattr(m.fusion, "ego_cond", "none"), ego_dim=d,
            goal_token=getattr(m.fusion, "goal_injection", "decoder") != "decoder",
            token_output=getattr(m.decoder, "readout", "pooled") == "query")
        # "decoder" keeps the goal only in the GRU (original), "both" adds the
        # fusion token, "fusion" removes it from the GRU so the goal reaches the
        # policy through perception alone.
        self.goal_injection = getattr(m.fusion, "goal_injection", "decoder")
        self.measure = MeasurementEncoder(
            d, speed_dropout=float(getattr(m.decoder, "speed_dropout", 0.0)))
        # "query" reads the fused tokens with learned planning queries; "pooled"
        # is the original mean-pool into a GRU.  Decision-level fusion decodes
        # each modality separately and so has no single fused token set to read,
        # which is why it stays on the pooled path.
        self.readout = getattr(m.decoder, "readout", "pooled")
        if self.readout not in ("pooled", "query"):
            raise ValueError(f"unknown readout {self.readout!r}")
        if self.readout == "query" and self.mode == "late":
            raise ValueError("readout=query is incompatible with late fusion")
        if self.readout == "query" and self.goal_injection == "decoder":
            # There is no decoder to inject into, so the goal would reach the
            # network through no path at all and the policy would drive with no
            # idea where it is going -- silently, since nothing else would fail.
            raise ValueError(
                "readout=query needs goal_injection=fusion (or both): with the "
                "GRU gone, the goal token is the only route the goal has")
        self.query_readout = None
        if self.readout == "query":
            self.query_readout = QueryReadout(
                d, horizon=m.decoder.horizon,
                num_layers=getattr(m.decoder, "query_layers", 3),
                num_heads=m.fusion.num_heads, ffn_dim=m.fusion.ffn_dim,
                attention=m.fusion.attention,
                speed_bins=tuple(getattr(m.decoder, "speed_bins",
                                         (0.0, 2.0, 4.0, 6.0, 8.0))))
        use_goal_in_gru = self.goal_injection != "fusion"
        self.decoder = WaypointDecoder(d, m.decoder.hidden_dim, m.decoder.horizon,
                                       use_goal=use_goal_in_gru) \
            if self.readout == "pooled" else None
        # True decision-level fusion shares nothing between the branches, so the
        # second modality gets its own decoder rather than reusing the first.
        self.decoder_l = (WaypointDecoder(d, m.decoder.hidden_dim,
                                          m.decoder.horizon,
                                          use_goal=use_goal_in_gru)
                          if self.mode == "late" else None)
        # The BEV head reads LiDAR tokens and the depth head camera tokens, so an
        # auxiliary task whose branch does not exist is dropped with it.
        self.seg_head = (BEVSegHead(d, m.aux.bev_classes)
                         if m.aux.bev_seg and self.use_lidar else None)
        # In bird's-eye mode the view transform's own depth distribution carries
        # the depth supervision, so this head would be 1.2 M parameters that are
        # allocated, counted and never called.
        self.depth_head = (DepthHead(d)
                           if m.aux.depth and self.use_cam
                           and self.camera_space == "image" else None)

    def forward(self, batch, force_drop=None):
        """batch keys: image B3HW, pillar_feats, pillar_coords, pillar_mask,
        pillar_batch, speed B x 1, command B, goal B x 2."""
        b = batch["image"].shape[0]
        cam_hw, depth_logits = None, None
        if self.use_cam:
            fc, _, cam_hw = self.camera(batch["image"],
                                        with_pe=self.camera_space == "image")
            if self.camera_space == "bev":
                d_ = fc.shape[-1]
                cam_map = fc.transpose(1, 2).reshape(b, d_, *cam_hw)
                bev, depth_logits = self.cam_to_bev(cam_map, return_depth=True)
                gh, gw = bev.shape[-2:]
                fc = bev.flatten(2).transpose(1, 2)
                key = (gh, gw, fc.device)
                if key not in self._bev_pe:
                    self._bev_pe[key] = sinusoidal_pe_2d(gh, gw, d_, fc.device)
                # The same code the LiDAR branch uses, so a camera token and a
                # LiDAR token standing on one cell carry one position.
                fc = fc + self._bev_pe[key]
        else:
            fc = None
        lidar_hw = None
        if self.use_lidar:
            fl, lidar_hw = self.lidar(
                batch["pillar_feats"], batch["pillar_coords"],
                batch["pillar_mask"], batch["pillar_batch"], b)
        else:
            fl = None
        # The measurement embedding is built before fusion, not after: with
        # `ego_cond` enabled the fusion blocks are conditioned on it, so speed
        # and command reach perception instead of only the decoder.
        m = self.measure(batch["speed"], batch["command"])
        z, gate, aux_tokens, fused, (zc, zl) = self.fusion(fc, fl,
                                                           force_drop=force_drop,
                                                           ego=m,
                                                           goal=batch["goal"])
        if self.readout == "query":
            tc, tl = fused
            tokens = torch.cat([t for t in (tc, tl) if t is not None], dim=1)
            waypoints, speed_logits = self.query_readout(tokens, m)
            out = {"waypoints": waypoints, "gate": gate,
                   "speed_logits": speed_logits}
            self._add_aux(out, aux_tokens, depth_logits, lidar_hw, batch)
            return out
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
        self._add_aux(out, aux_tokens, depth_logits, lidar_hw, batch)
        return out

    def _add_aux(self, out, aux_tokens, depth_logits, lidar_hw, batch):
        """Attach the training-only auxiliary predictions.

        Shared by both readouts: the auxiliary heads hang off the fusion blocks,
        not off whatever decodes them, so switching the readout must not quietly
        drop the supervision that TransFuser measured as worth 14 points of
        route completion when removed entirely.
        """
        if not self.training or aux_tokens is None:
            return
        aux_c, aux_l = aux_tokens
        if self.seg_head is not None and aux_l is not None:
            out["bev_seg"] = self.seg_head(aux_l, lidar_hw)
        if depth_logits is not None:
            # In bird's-eye mode the depth that matters is the one driving the
            # projection, so the existing depth target supervises it directly
            # instead of a second, unrelated head.  Reported in the same
            # normalised-inverse form the dataset stores, so the loss needs no
            # change.
            out["depth"] = self._depth_as_target(depth_logits,
                                                 self.cfg.data.depth_size)
        elif self.depth_head is not None and aux_c is not None:
            s = self.camera.STRIDE
            hw = (batch["image"].shape[2] // s, batch["image"].shape[3] // s)
            out["depth"] = self.depth_head(aux_c, hw)

    def _depth_as_target(self, logits, size, near=1.0, far=100.0):
        """Expected metric depth -> the normalised inverse depth the dataset stores.

        Mirrors sensors.depth_to_normalized_inverse so the prediction and the
        stored target are the same quantity; doing the conversion here keeps the
        loss identical between the two camera spaces.
        """
        d = self.cam_to_bev.expected_depth(logits).unsqueeze(1)
        d = d.clamp(near, far)
        inv = (1.0 / d - 1.0 / far) / (1.0 / near - 1.0 / far)
        return F.interpolate(inv, size=tuple(size), mode="bilinear",
                             align_corners=False)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
