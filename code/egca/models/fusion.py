"""EGCA fusion module (Sec. 4-3): bidirectional linear cross-attention blocks,
sensor-reliability gate (Eq. 4.7-4.9) and modality-level sensor dropout
(Eq. 4.15) with learnable absent-modality tokens.

`mode` selects the fusion strategy compared in Tables 5-4 and 5-5:

    egca         bidirectional cross-attention + reliability gate (proposed)
    concat       identical stack with the cross-attention removed; the pooled
                 summaries are concatenated and linearly merged
    late         same as concat, but the two summaries are decoded separately
                 and the predicted trajectories are averaged (decision level)
    camera_only  only the image branch is built and processed
    lidar_only   only the BEV branch is built and processed

`concat` is deliberately *not* a smaller network: it keeps the same number of
blocks, the same feed-forward stack and nearly the same parameter count, and
removes only the cross-attention. An ablation that also shrinks the model would
confound the effect of the mechanism with the effect of capacity.

Early fusion is intentionally absent: fusing raw camera and LiDAR before feature
extraction requires projecting one modality into the other's frame, which is a
different architecture rather than an ablation of this one. Section 5-5 states
this instead of reporting a straw-man row.
"""
import torch
import torch.nn as nn

from .attention import make_attention


class EGCABlock(nn.Module):
    """One bidirectional fusion block, Eqs. (4.4)-(4.6)."""

    def __init__(self, dim, num_heads, ffn_dim, attention="linear",
                 cross=True, use_cam=True, use_lidar=True):
        super().__init__()
        # Only the parts a mode actually uses are allocated, so the parameter
        # count reported for each ablation is the parameter count it really has.
        self.cross = cross
        if cross:
            self.attn_c = make_attention(attention, dim, num_heads)  # cam<-lidar
            self.attn_l = make_attention(attention, dim, num_heads)  # lidar<-cam
            self.ln_c1, self.ln_l1 = nn.LayerNorm(dim), nn.LayerNorm(dim)

        def ffn():
            return nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(),
                                 nn.Linear(ffn_dim, dim))
        if use_cam:
            self.ln_c2, self.ffn_c = nn.LayerNorm(dim), ffn()
        if use_lidar:
            self.ln_l2, self.ffn_l = nn.LayerNorm(dim), ffn()

    def forward(self, fc, fl):
        """Without `cross` only the inter-modal attention is dropped; the rest of
        the block is untouched, which is what the `concat` and `late` ablations
        need in order to change exactly one thing."""
        if self.cross:
            # Both directions read the *same* block input (Eqs. 4.4-4.5), so the
            # update is symmetric and the two attentions can run in parallel.
            zc = self.attn_c(fc, fl)
            zl = self.attn_l(fl, fc)
            fc = self.ln_c1(fc + zc)
            fl = self.ln_l1(fl + zl)
        if fc is not None:
            fc = self.ln_c2(fc + self.ffn_c(fc))
        if fl is not None:
            fl = self.ln_l2(fl + self.ffn_l(fl))
        return fc, fl


class EGCAFusion(nn.Module):
    MODES = ("egca", "concat", "late", "camera_only", "lidar_only")

    def __init__(self, dim=256, num_blocks=4, num_heads=4, ffn_dim=512,
                 attention="linear", gate=True, gate_hidden=128,
                 sensor_dropout=0.15, mode="egca"):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown fusion mode {mode!r}, expected {self.MODES}")
        self.mode = mode
        self.single = mode in ("camera_only", "lidar_only")
        self.cross = mode == "egca"
        self.blocks = nn.ModuleList(
            EGCABlock(dim, num_heads, ffn_dim, attention, cross=self.cross,
                      use_cam=mode != "lidar_only", use_lidar=mode != "camera_only")
            for _ in range(num_blocks))
        if mode == "concat":
            self.merge = nn.Linear(2 * dim, dim)
        self.use_gate = gate and mode == "egca"
        if gate:
            self.gate_mlp = nn.Sequential(
                nn.Linear(2 * dim, gate_hidden), nn.ReLU(inplace=True),
                nn.Linear(gate_hidden, 1))
        self.absent_cam = nn.Parameter(torch.zeros(1, 1, dim))
        self.absent_lidar = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.absent_cam, std=0.02)
        nn.init.normal_(self.absent_lidar, std=0.02)
        self.rho = sensor_dropout
        self.aux_block = 1          # tokens of this block feed the aux heads

    def drop_modality(self, fc, fl, force=None):
        """Sensor dropout (Eq. 4.15): with probability rho exactly one of the
        two modalities is replaced by its learnable absent-modality token, the
        modality being chosen uniformly.  Both modalities are therefore never
        dropped simultaneously, and each is dropped with probability rho/2.
        `force` in {None, 'cam', 'lidar'} lets the evaluation code simulate a
        hard sensor failure deterministically."""
        if force is None:
            if not (self.training and self.rho > 0):
                return fc, fl
            n = fc.shape[0]
            dev = fc.device
            drop = torch.rand(n, device=dev) < self.rho          # b_m ~ Bern(rho)
            pick_cam = torch.rand(n, device=dev) < 0.5           # uniform choice
            mc = (drop & pick_cam).view(n, 1, 1)
            ml = (drop & ~pick_cam).view(n, 1, 1)
            fc = torch.where(mc, self.absent_cam.expand_as(fc), fc)
            fl = torch.where(ml, self.absent_lidar.expand_as(fl), fl)
            return fc, fl
        if force == "cam":
            fc = self.absent_cam.expand_as(fc).clone()
        elif force == "lidar":
            fl = self.absent_lidar.expand_as(fl).clone()
        return fc, fl

    def forward(self, fc, fl, force_drop=None):
        """Returns (context, gate, aux tokens, final tokens, per-modality
        summaries).  The last item is only used by the `late` ablation, which
        decodes each modality separately."""
        if not self.single:
            fc, fl = self.drop_modality(fc, fl, force=force_drop)
        aux = None
        for i, blk in enumerate(self.blocks):
            fc, fl = blk(fc, fl)
            if i == self.aux_block:
                aux = (fc, fl)      # mid-level tokens for aux supervision
        zc = fc.mean(dim=1) if fc is not None else None   # Eq. (4.7)
        zl = fl.mean(dim=1) if fl is not None else None

        if self.mode == "camera_only":
            z, g = zc, torch.ones(zc.shape[0], 1, device=zc.device)
        elif self.mode == "lidar_only":
            z, g = zl, torch.zeros(zl.shape[0], 1, device=zl.device)
        elif self.mode == "concat":
            z = self.merge(torch.cat([zc, zl], dim=-1))
            g = torch.full_like(zc[:, :1], 0.5)
        elif self.mode == "late":
            z = None                      # each modality is decoded separately
            g = torch.full_like(zc[:, :1], 0.5)
        else:
            if self.use_gate:
                g = torch.sigmoid(self.gate_mlp(torch.cat([zc, zl], dim=-1)))  # (4.8)
            else:
                g = torch.full_like(zc[:, :1], 0.5)
            z = g * zc + (1.0 - g) * zl                   # Eq. (4.9)
        return z, g.squeeze(-1), aux, (fc, fl), (zc, zl)
