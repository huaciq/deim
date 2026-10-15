"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/util/box_ops.py
"""

import torch
from torch import Tensor
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w.clamp(min=0.0)), (y_c - 0.5 * h.clamp(min=0.0)),
         (x_c + 0.5 * w.clamp(min=0.0)), (y_c + 0.5 * h.clamp(min=0.0))]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: Tensor) -> Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1: Tensor, boxes2: Tensor):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


def wasserstein_distance_normalized(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7):
    """
    Normalized Wasserstein Distance (NWD) for oriented/small object detection.

    NWD is scale-invariant and more suitable for small objects than IoU-based metrics.
    Based on "A Normalized Gaussian Wasserstein Distance for Tiny Object Detection"
    (https://arxiv.org/abs/2110.13389).

    Args:
        boxes1: [N, 4] in cxcywh format (normalized)
        boxes2: [M, 4] in cxcywh format (normalized)
        eps: small constant for numerical stability

    Returns:
        nwd: [N, M] matrix, values in [0, 1] where 1 means identical
    """
    N = boxes1.shape[0]
    M = boxes2.shape[0]

    # Extract cx, cy, w, h
    cx1, cy1, w1, h1 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    cx2, cy2, w2, h2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    # Compute pairwise center distances: [N, M]
    # Treat each box as 2D Gaussian: mean = (cx, cy), covariance diagonal (w^2, h^2)
    cx1 = cx1[:, None]  # [N, 1]
    cy1 = cy1[:, None]
    w1 = w1[:, None]
    h1 = h1[:, None]

    cx2 = cx2[None, :]  # [1, M]
    cy2 = cy2[None, :]
    w2 = w2[None, :]
    h2 = h2[None, :]

    # Wasserstein-2 distance between two 2D Gaussians with diagonal covariance:
    # W_2^2 = ||mu1 - mu2||^2 + ||sigma1 - sigma2||^2
    # where sigma is the std dev (w/2, h/2 treating w, h as ~2*std)

    # Center distance term
    center_dist_sq = (cx1 - cx2).pow(2) + (cy1 - cy2).pow(2)

    # Covariance distance term (using w, h as measures of spread)
    # Normalize by geometric mean of box scales to make it scale-invariant
    w_dist = (w1 / 2.0 - w2 / 2.0).pow(2)
    h_dist = (h1 / 2.0 - h2 / 2.0).pow(2)
    cov_dist_sq = w_dist + h_dist

    # Normalization constant: geometric mean of box diagonals
    # diagonal = sqrt(w^2 + h^2)
    diag1 = (w1.pow(2) + h1.pow(2)).sqrt()
    diag2 = (w2.pow(2) + h2.pow(2)).sqrt()
    normalize = (diag1 * diag2).clamp(min=eps)

    # Normalized Wasserstein distance
    wasserstein_dist = (center_dist_sq + cov_dist_sq).sqrt()
    nwd = torch.exp(- (wasserstein_dist / (normalize + eps)).pow(2) / 2.0)

    return nwd