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


class FiLM(nn.Module):
    """Feature-wise modulation of a token set by the ego state.

    Zero-initialised on purpose: at step 0 the scale is 1 and the shift is 0, so
    a conditioned model starts out numerically identical to an unconditioned one
    and any difference in the final score is something the network chose to
    learn, not a different initialisation.
    """

    def __init__(self, ego_dim, dim):
        super().__init__()
        self.to_scale_shift = nn.Linear(ego_dim, 2 * dim)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x, ego):
        scale, shift = self.to_scale_shift(ego).chunk(2, dim=-1)
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class EGCABlock(nn.Module):
    """One bidirectional fusion block, Eqs. (4.4)-(4.6)."""

    def __init__(self, dim, num_heads, ffn_dim, attention="linear",
                 cross=True, use_cam=True, use_lidar=True,
                 ego_cond="none", ego_dim=256):
        super().__init__()
        # Perception is otherwise blind to the ego state: speed and command
        # reach the network only at the decoder, so attention cannot look
        # further ahead at speed or toward the turn at a junction.
        self.ego_cond = ego_cond
        if ego_cond == "film":
            self.film_c = FiLM(ego_dim, dim) if use_cam else None
            self.film_l = FiLM(ego_dim, dim) if use_lidar else None
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

    def forward(self, fc, fl, ego=None):
        """Without `cross` only the inter-modal attention is dropped; the rest of
        the block is untouched, which is what the `concat` and `late` ablations
        need in order to change exactly one thing."""
        if self.ego_cond == "film" and ego is not None:
            # Modulate before attention, so the ego state can steer what the two
            # branches read from each other rather than only what is decoded.
            if fc is not None and self.film_c is not None:
                fc = self.film_c(fc, ego)
            if fl is not None and self.film_l is not None:
                fl = self.film_l(fl, ego)
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


class AttentionPool(nn.Module):
    """Pool a token set with learned queries instead of averaging it.

    Mean pooling divides every token's contribution by the number of tokens, and
    the two branches do not have the same number: 440 camera tokens against 4096
    LiDAR pillars.  A single relevant object therefore arrives at the gate ten
    times weaker from LiDAR than from the camera purely because of how many
    tokens each branch happens to produce -- which is what the measured gate of
    0.995 shows.  Learned queries let the branch select what matters instead of
    diluting it, and several queries can specialise on different structure.

    Softmax attention is deliberate and does not touch the efficiency claim of
    Sec. 4-3-1: that claim is about the N_c x N_l cross-attention between the two
    branches, whereas this is M x N with M = 4, which costs nothing.
    """

    def __init__(self, dim, num_queries=4, num_heads=4):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.proj = nn.Linear(num_queries * dim, dim)

    def forward(self, tokens):
        q = self.q.expand(tokens.shape[0], -1, -1)
        out, _ = self.attn(q, tokens, tokens, need_weights=False)
        return self.proj(out.flatten(1))


class EGCAFusion(nn.Module):
    MODES = ("egca", "concat", "late", "camera_only", "lidar_only")

    def __init__(self, dim=256, num_blocks=4, num_heads=4, ffn_dim=512,
                 attention="linear", gate=True, gate_hidden=128,
                 sensor_dropout=0.15, mode="egca",
                 pooling="mean", gate_form="scalar", pool_queries=4,
                 ego_cond="none", ego_dim=256, goal_token=False,
                 token_output=False):
        super().__init__()
        # Defaults reproduce the original behaviour exactly, so every checkpoint
        # trained before these options existed still builds and loads.
        if pooling not in ("mean", "attention"):
            raise ValueError(f"unknown pooling {pooling!r}")
        if gate_form not in ("scalar", "vector"):
            raise ValueError(f"unknown gate_form {gate_form!r}")
        if ego_cond not in ("none", "film"):
            raise ValueError(f"unknown ego_cond {ego_cond!r}")
        self.pooling = pooling
        self.gate_form = gate_form
        self.ego_cond = ego_cond
        # Where the navigation goal enters.  Injecting it only at the decoder --
        # which is what the GRU does, at every one of its steps -- is the "late
        # goal conditioning" bottleneck LEAD identifies: the goal never meets the
        # perception features, so the policy leans on it as a steering shortcut
        # and collapses when it lands somewhere unusual.  As a token it is one
        # more thing the scene can be read against.
        # Set when a downstream readout consumes the tokens rather than z.
        self.token_output = token_output
        self.goal_token = goal_token
        if goal_token:
            self.goal_embed = nn.Sequential(
                nn.Linear(2, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))
        if mode not in self.MODES:
            raise ValueError(f"unknown fusion mode {mode!r}, expected {self.MODES}")
        self.mode = mode
        self.single = mode in ("camera_only", "lidar_only")
        self.cross = mode == "egca"
        self.blocks = nn.ModuleList(
            EGCABlock(dim, num_heads, ffn_dim, attention, cross=self.cross,
                      use_cam=mode != "lidar_only", use_lidar=mode != "camera_only",
                      ego_cond=ego_cond, ego_dim=ego_dim)
            for _ in range(num_blocks))
        if mode == "concat":
            self.merge = nn.Linear(2 * dim, dim)
        if pooling == "attention":
            # One pooler per branch: the two token sets differ in count and in
            # what a useful summary of them looks like.
            self.pool_c = (AttentionPool(dim, pool_queries, num_heads)
                           if mode != "lidar_only" else None)
            self.pool_l = (AttentionPool(dim, pool_queries, num_heads)
                           if mode != "camera_only" else None)
        self.use_gate = gate and mode == "egca"
        if gate:
            # A scalar gate can only interpolate between the two summaries, so
            # weighting one up necessarily weights the other down -- strictly
            # less expressive than the `concat` ablation's linear merge, which is
            # why the two score the same.  A per-dimension gate keeps the
            # reliability interpretation of Eq. 4.8 while letting each feature
            # take what it needs from either modality.
            out_dim = dim if gate_form == "vector" else 1
            self.gate_mlp = nn.Sequential(
                nn.Linear(2 * dim, gate_hidden), nn.ReLU(inplace=True),
                nn.Linear(gate_hidden, out_dim))
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

    def forward(self, fc, fl, force_drop=None, ego=None, goal=None):
        """Returns (context, gate, aux tokens, final tokens, per-modality
        summaries).  The last item is only used by the `late` ablation, which
        decodes each modality separately."""
        if not self.single:
            fc, fl = self.drop_modality(fc, fl, force=force_drop)
        n_extra = 0
        if self.goal_token and goal is not None:
            g_tok = self.goal_embed(goal).unsqueeze(1)          # B x 1 x d
            if fc is not None:
                fc = torch.cat([g_tok, fc], dim=1)
            if fl is not None:
                fl = torch.cat([g_tok, fl], dim=1)
            n_extra = 1
        aux = None
        for i, blk in enumerate(self.blocks):
            fc, fl = blk(fc, fl, ego=ego)
            if i == self.aux_block:
                # Strip the goal before the auxiliary heads: they reshape the
                # token sequence back into a spatial grid, and an extra token
                # makes that reshape silently wrong.
                aux = (None if fc is None else fc[:, n_extra:],
                       None if fl is None else fl[:, n_extra:])
        if self.pooling == "attention":                   # Eq. (4.7), revised
            zc = self.pool_c(fc) if fc is not None else None
            zl = self.pool_l(fl) if fl is not None else None
        else:
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
        # With a token-level readout the pooled context z is never consumed, so
        # the gate would compute a reliability weight that nothing applies -- the
        # central claim of Sec. 4-3-2 silently switched off.  Applying it to the
        # token sets instead keeps its meaning (an untrusted modality is scaled
        # down) without collapsing the spatial structure the readout exists to
        # preserve.  Broadcasting covers both gate forms: B x 1 scales a
        # modality uniformly, B x d scales it per channel.
        out_c, out_l = fc, fl
        if self.token_output and self.mode == "egca":
            gg = g.unsqueeze(1)
            if out_c is not None:
                out_c = gg * out_c
            if out_l is not None:
                out_l = (1.0 - gg) * out_l
        g_report = g.mean(dim=-1) if g.shape[-1] > 1 else g.squeeze(-1)
        return z, g_report, aux, (out_c, out_l), (zc, zl)
