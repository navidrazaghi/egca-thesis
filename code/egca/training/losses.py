"""Multi-task loss with homoscedastic uncertainty weighting (Eq. 4.16,
Kendall et al. 2018), in the numerically stable parametrization
s_k = log sigma_k^2:

    regression task (L1 waypoints, L1 depth):  0.5 * exp(-s_k) * L_k + 0.5 * s_k
    classification task (BEV cross-entropy) :        exp(-s_k) * L_k + 0.5 * s_k

The 1/2 factor in front of the likelihood term only appears for the Gaussian
(regression) heads; for the scaled-softmax classification head Kendall et al.
derive a weight of 1/sigma^2, so applying the regression form to a
cross-entropy term would be incorrect.  Optimizing s_k instead of sigma_k
keeps the weights positive without an explicit constraint.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyWeightedLoss(nn.Module):
    TASKS = ("wp", "seg", "depth", "speed")
    # likelihood-term coefficient: 1/2 for Gaussian heads, 1 for the softmax head
    COEF = {"wp": 0.5, "depth": 0.5, "seg": 1.0, "speed": 1.0}

    def __init__(self, use_seg=True, use_depth=True, wp_dt=0.5, speed_bins=None,
                 gate_supervision=0.0):
        super().__init__()
        self.use_seg, self.use_depth = use_seg, use_depth
        self.wp_dt = wp_dt
        # Fixed coefficient, deliberately outside the uncertainty weighting:
        # the gate target is not a task with unknown noise, it is a calibration
        # constraint whose scale is known (BCE, order 1), and letting a learned
        # log-variance shrink it away would quietly reproduce the unsupervised
        # gate that FINDINGS 4 measured as carrying no signal.
        self.gate_coef = float(gate_supervision)
        # The target-speed head only exists with the query readout; its weight is
        # allocated either way so a checkpoint stays loadable across the switch.
        self.register_buffer(
            "speed_bins",
            torch.tensor(list(speed_bins) if speed_bins else
                         [0.0, 2.0, 4.0, 6.0, 8.0], dtype=torch.float32),
            persistent=False)
        self.log_var = nn.ParameterDict({
            t: nn.Parameter(torch.zeros(())) for t in self.TASKS})

    def forward(self, out, batch):
        losses = {}
        # Eq. (3.21): L1 waypoint imitation loss
        losses["wp"] = F.l1_loss(out["waypoints"], batch["waypoints"])
        # Eq. (3.22): BEV semantic segmentation cross-entropy
        if self.use_seg and "bev_seg" in out:
            losses["seg"] = F.cross_entropy(out["bev_seg"], batch["bev_seg"])
        # Eq. (3.23): L1 depth loss on normalized inverse depth
        if self.use_depth and "depth" in out:
            losses["depth"] = F.l1_loss(out["depth"], batch["depth"])
        # Target speed, classified over bins.  The target is read off the
        # expert's own waypoints -- the same quantity the controller used to
        # derive from the network's predicted trajectory -- so nothing had to be
        # recollected, and the head learns it from the scene instead of
        # inheriting the trajectory's errors.
        if "speed_logits" in out:
            from ..models.query_readout import speed_target
            tgt = speed_target(batch["waypoints"], self.wp_dt, self.speed_bins)
            losses["speed"] = F.cross_entropy(out["speed_logits"], tgt)
        total = 0.0
        for k, l in losses.items():
            s = self.log_var[k]
            total = total + self.COEF[k] * torch.exp(-s) * l + 0.5 * s
        # Gate supervision (FINDINGS 4). Sensor dropout already manufactures
        # frames whose reliability is known by construction -- a dropped
        # modality carries zero information -- so on exactly those frames the
        # gate g (the *camera* weight in z = g z_c + (1-g) z_l) has a ground
        # truth: 0 when the camera was dropped, 1 when the LiDAR was. Frames
        # where nothing was dropped are left unsupervised; what the gate does
        # between clean sensors remains the network's to learn.
        if self.gate_coef > 0 and "drop_cam" in out:
            dropped = out["drop_cam"] | out["drop_lidar"]
            if bool(dropped.any()):
                g = out["gate"][dropped].clamp(1e-4, 1.0 - 1e-4)
                tgt = out["drop_lidar"][dropped].float()
                gate_l = F.binary_cross_entropy(g, tgt)
                losses["gate"] = gate_l
                total = total + self.gate_coef * gate_l
        losses["total"] = total
        return total, {k: float(v.detach()) for k, v in losses.items()}
