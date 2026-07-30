"""GRU waypoint decoder (Sec. 4-4), Eqs. (4.10)-(4.12).

Waypoints are predicted differentially in the ego frame:
    h_k = GRU([w_{k-1}; x_goal], h_{k-1});  w_k = w_{k-1} + MLP(h_k)
"""
import torch
import torch.nn as nn


class WaypointDecoder(nn.Module):
    def __init__(self, ctx_dim=256, hidden_dim=256, horizon=4, use_goal=True):
        super().__init__()
        self.horizon = horizon
        # Whether the goal is still fed at every recurrent step.  With the goal
        # available to the fusion as a token, keeping it here as well leaves the
        # shortcut intact: the recurrent input is four numbers, two of which
        # point straight at the target, so the goal outweighs everything the
        # encoder learned.
        self.use_goal = use_goal
        self.init_mlp = nn.Sequential(                 # Eq. (4.10)
            nn.Linear(ctx_dim * 2, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim))
        self.gru = nn.GRUCell(input_size=4 if use_goal else 2,
                              hidden_size=hidden_dim)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 2))

    def forward(self, z, m, goal):
        """z: B x d fused context, m: B x d measurement embedding,
        goal: B x 2 next sparse goal in ego frame -> B x T x 2 waypoints."""
        b = z.shape[0]
        h = self.init_mlp(torch.cat([z, m], dim=-1))
        wp = z.new_zeros(b, 2)
        out = []
        for _ in range(self.horizon):
            step_in = torch.cat([wp, goal], dim=-1) if self.use_goal else wp
            h = self.gru(step_in, h)                         # Eq. (4.11)
            wp = wp + self.out(h)                            # Eq. (4.12)
            out.append(wp)
        return torch.stack(out, dim=1)


class MeasurementEncoder(nn.Module):
    """Ego-state + navigation command encoder, Eq. (4.3)."""

    NUM_COMMANDS = 4    # left, right, straight, follow-lane

    def __init__(self, embed_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1 + self.NUM_COMMANDS, 128), nn.ReLU(inplace=True),
            nn.Linear(128, embed_dim))

    def forward(self, speed, command):
        """speed: B x 1 (m/s), command: B (long) -> B x d."""
        onehot = torch.zeros(speed.shape[0], self.NUM_COMMANDS,
                             device=speed.device)
        onehot.scatter_(1, command.long().unsqueeze(1), 1.0)
        return self.mlp(torch.cat([speed / 12.0, onehot], dim=-1))
