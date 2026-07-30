"""Planning queries that read the fused tokens directly (Sec. 4-4, revised).

Why the pooled readout is being replaced
----------------------------------------
After fusion the network holds 440 camera tokens and 4096 LiDAR tokens, each
carrying where something is.  The original readout averages each set into one
256-d vector and hands that to a GRU, which destroys every spatial distinction
before the trajectory is decided.  The car that has to be avoided and the empty
road beside it arrive as the same average.

That bottleneck also plausibly explains the flat ablation table.  `egca`,
`concat` and `camera_only` all funnel through the same 256-d pipe, so whatever
the fusion mechanism produced upstream is compressed away before anything is
measured -- three different mechanisms are compared through an aperture too
narrow to show a difference between them.  Removing it is therefore not only a
performance change: it is what gives the architecture's own claim a chance to
be visible.

A small set of learned queries attends over the tokens instead.  Six queries
against ~4500 tokens costs nothing next to the fusion blocks themselves, and it
removes three known bottlenecks at once:

  * pooling dilution -- a query can look at the one cell that matters rather
    than averaging it against four thousand empty ones;
  * late goal conditioning -- with the goal present as a token, the queries read
    it alongside the scene rather than having it forced into every recurrent
    step, which is the failure LEAD attributes +6.7 DS to fixing;
  * lateral/longitudinal coupling -- a dedicated speed query predicts the target
    speed from the scene, so a collapsed trajectory no longer drags the speed to
    zero with it.  The old controller derived speed from the waypoint spacing,
    which made a standstill self-sustaining: stopped in, stopped out.

Target speed is classified over a small set of speeds rather than regressed.
The distribution is close to bimodal -- hold station, or proceed at the flow of
traffic -- and a regressor asked to straddle that predicts the mean of two modes
the expert never drove.
"""
import torch
import torch.nn as nn

from .attention import make_attention


class QueryReadout(nn.Module):
    """Fused scene tokens -> waypoints and a target-speed distribution."""

    def __init__(self, dim=256, horizon=4, num_layers=3, num_heads=4,
                 ffn_dim=512, attention="linear",
                 speed_bins=(0.0, 2.0, 4.0, 6.0, 8.0)):
        super().__init__()
        self.horizon = horizon
        n_q = horizon + 1                     # one query per waypoint, plus speed
        self.queries = nn.Parameter(torch.randn(1, n_q, dim) * 0.02)
        self.self_attn = nn.ModuleList(
            nn.MultiheadAttention(dim, num_heads, batch_first=True)
            for _ in range(num_layers))
        # The cross-attention honours the configured attention kind, so the
        # efficiency claim of Sec. 4-3-1 covers the readout too. Self-attention
        # among six queries is 6 x 6 and not worth a kernel trick.
        self.cross = nn.ModuleList(
            make_attention(attention, dim, num_heads) for _ in range(num_layers))
        self.ln_s = nn.ModuleList(nn.LayerNorm(dim) for _ in range(num_layers))
        self.ln_x = nn.ModuleList(nn.LayerNorm(dim) for _ in range(num_layers))
        self.ln_f = nn.ModuleList(nn.LayerNorm(dim) for _ in range(num_layers))
        self.ffn = nn.ModuleList(
            nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(),
                          nn.Linear(ffn_dim, dim))
            for _ in range(num_layers))
        self.wp_head = nn.Sequential(
            nn.Linear(dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 2))
        self.speed_head = nn.Linear(dim, len(speed_bins))
        self.register_buffer("speed_bins",
                             torch.tensor(speed_bins, dtype=torch.float32),
                             persistent=False)

    def forward(self, tokens, ego):
        """tokens: B x N x d fused scene, ego: B x d -> waypoints, speed logits."""
        b = tokens.shape[0]
        q = self.queries.expand(b, -1, -1) + ego.unsqueeze(1)
        for i in range(len(self.cross)):
            a, _ = self.self_attn[i](q, q, q, need_weights=False)
            q = self.ln_s[i](q + a)
            q = self.ln_x[i](q + self.cross[i](q, tokens))
            q = self.ln_f[i](q + self.ffn[i](q))
        waypoints = self.wp_head(q[:, :self.horizon])
        speed_logits = self.speed_head(q[:, self.horizon])
        return waypoints, speed_logits

    def expected_speed(self, logits):
        """Bin distribution -> one target speed, for the controller."""
        return (logits.softmax(dim=-1) * self.speed_bins).sum(dim=-1)


def speed_target(waypoints, wp_dt, bins):
    """Nearest speed bin implied by the ground-truth waypoint spacing.

    Derived from the labels rather than stored during collection, so no data has
    to be recollected and the target is exactly the quantity the old controller
    computed at inference -- what changes is that it is now read from the scene
    instead of from the network's own possibly-broken trajectory.
    """
    v = (waypoints[:, 1] - waypoints[:, 0]).norm(dim=-1) / wp_dt
    return (v.unsqueeze(-1) - bins.view(1, -1)).abs().argmin(dim=-1)
