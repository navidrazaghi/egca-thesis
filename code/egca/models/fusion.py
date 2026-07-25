"""EGCA fusion module (Sec. 4-3): bidirectional linear cross-attention blocks,
sensor-reliability gate (Eq. 4.7-4.9) and modality-level sensor dropout
(Eq. 4.15) with learnable absent-modality tokens.
"""
import torch
import torch.nn as nn

from .attention import make_attention


class EGCABlock(nn.Module):
    """One bidirectional fusion block, Eqs. (4.4)-(4.6)."""

    def __init__(self, dim, num_heads, ffn_dim, attention="linear"):
        super().__init__()
        self.attn_c = make_attention(attention, dim, num_heads)  # cam <- lidar
        self.attn_l = make_attention(attention, dim, num_heads)  # lidar <- cam
        self.ln_c1, self.ln_c2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ln_l1, self.ln_l2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        def ffn():
            return nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(),
                                 nn.Linear(ffn_dim, dim))
        self.ffn_c, self.ffn_l = ffn(), ffn()

    def forward(self, fc, fl):
        # Both directions read the *same* block input (Eqs. 4.4-4.5), so the
        # update is symmetric and the two attentions can run in parallel.
        zc = self.attn_c(fc, fl)
        zl = self.attn_l(fl, fc)
        fc = self.ln_c1(fc + zc)
        fl = self.ln_l1(fl + zl)
        fc = self.ln_c2(fc + self.ffn_c(fc))
        fl = self.ln_l2(fl + self.ffn_l(fl))
        return fc, fl


class EGCAFusion(nn.Module):
    def __init__(self, dim=256, num_blocks=4, num_heads=4, ffn_dim=512,
                 attention="linear", gate=True, gate_hidden=128,
                 sensor_dropout=0.15):
        super().__init__()
        self.blocks = nn.ModuleList(
            EGCABlock(dim, num_heads, ffn_dim, attention)
            for _ in range(num_blocks))
        self.use_gate = gate
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
        fc, fl = self.drop_modality(fc, fl, force=force_drop)
        aux = None
        for i, blk in enumerate(self.blocks):
            fc, fl = blk(fc, fl)
            if i == self.aux_block:
                aux = (fc, fl)      # mid-level tokens for aux supervision
        zc, zl = fc.mean(dim=1), fl.mean(dim=1)          # Eq. (4.7)
        if self.use_gate:
            g = torch.sigmoid(self.gate_mlp(torch.cat([zc, zl], dim=-1)))  # (4.8)
        else:
            g = torch.full_like(zc[:, :1], 0.5)
        z = g * zc + (1.0 - g) * zl                       # Eq. (4.9)
        return z, g.squeeze(-1), aux, (fc, fl)
