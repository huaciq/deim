"""Exact one-dimensional W1 consistency on D-FINE's nonuniform FDR grid."""

import torch
from torch import nn


class WassersteinFDRLoss(nn.Module):
    """W1 residual to add to DFL, without changing the original DFL itself.

    Without explicit weights, target_corners is a continuous bin index in
    [0, reg_max]. D-FINE's bbox2distance instead returns the left bin index
    and two separate physical-grid interpolation weights; pass those weights
    to reuse exactly the same target distribution as its existing FGL loss.

    The default reduction matches the supplied standalone edge mean. With
    avg_factor, sum over edges / avg_factor matches DEIM's existing FGL
    normalization, including union matches and denoising groups.
    """

    def __init__(self, reg_max=32, loss_weight=0.5):
        super().__init__()
        if reg_max < 1 or loss_weight < 0:
            raise ValueError('W-FDR requires reg_max >= 1 and nonnegative loss_weight.')
        self.reg_max = reg_max
        self.loss_weight = float(loss_weight)

    def forward(self, pred_corners, target_corners, weight_grid, iou_weight=None,
                *, weight_left=None, weight_right=None, avg_factor=None):
        if (weight_left is None) != (weight_right is None):
            raise ValueError('Provide both DFL interpolation weights, or neither.')
        with torch.autocast(device_type=pred_corners.device.type, enabled=False):
            probability = pred_corners.float().softmax(dim=-1)
            target = target_corners.detach().float().reshape(-1).clamp(0, self.reg_max)
            # Keep the full last interval [reg_max-1, reg_max], including its endpoint.
            left = target.long().clamp(max=self.reg_max - 1)
            right = left + 1
            if weight_left is None:
                right_weight = target - left.float()
                left_weight = 1. - right_weight
            else:
                left_weight = weight_left.detach().float().reshape(-1)
                right_weight = weight_right.detach().float().reshape(-1)

            target_probability = torch.zeros_like(probability)
            target_probability.scatter_add_(1, left[:, None], left_weight[:, None])
            target_probability.scatter_add_(1, right[:, None], right_weight[:, None])
            difference = (probability.cumsum(-1) - target_probability.cumsum(-1))[:, :-1]
            grid = weight_grid.detach().float().reshape(-1)
            distance = (difference.abs() * (grid[1:] - grid[:-1])[None]).sum(-1)
            if iou_weight is not None:
                distance = distance * iou_weight.detach().float().reshape(-1)
            if avg_factor is not None:
                value = distance.sum() / avg_factor
            elif distance.numel():
                value = distance.mean()
            else:
                # Empty local matches must still give a finite, connected zero.
                value = distance.sum()
            return value * self.loss_weight
