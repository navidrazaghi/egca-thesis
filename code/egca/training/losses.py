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
    TASKS = ("wp", "seg", "depth")
    # likelihood-term coefficient: 1/2 for Gaussian heads, 1 for the softmax head
    COEF = {"wp": 0.5, "depth": 0.5, "seg": 1.0}

    def __init__(self, use_seg=True, use_depth=True):
        super().__init__()
        self.use_seg, self.use_depth = use_seg, use_depth
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
        total = 0.0
        for k, l in losses.items():
            s = self.log_var[k]
            total = total + self.COEF[k] * torch.exp(-s) * l + 0.5 * s
        losses["total"] = total
        return total, {k: float(v.detach()) for k, v in losses.items()}
