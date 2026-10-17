"""
SAR Boundary-Aware Modules for Small Object Detection
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.

Modules for boundary-aware localization quality enhancement:
- BoundaryConsistencyMap (BCM): Extract boundary features from encoder
- ShapeAdaptiveScoring (SAQS): Shape-aware query quality scoring
- generate_gt_boundary_map: Generate GT boundary maps from boxes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_activation


class BoundaryConsistencyMap(nn.Module):
    """
    Boundary Consistency Map (BCM) Module

    Extracts boundary features from encoder output to evaluate query localization quality.
    Unlike IoU prediction (scalar), BCM provides spatial distribution of boundary information,
    which is more suitable for SAR's blurry boundaries.

    Key difference from SA-DETR:
    - SA-DETR uses external SAM (two-stage), we are end-to-end
    - SA-DETR uses full mask, we use only boundary (more lightweight)
    """

    def __init__(self, in_channels=256, hidden_dim=64, act='silu'):
        super().__init__()

        # Lightweight boundary extraction branch
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            get_activation(act),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            get_activation(act),
            nn.Conv2d(hidden_dim, 1, 1, bias=True)  # Single-channel boundary map
        )

        # Sobel-like initialization for edge detection bias
        self._init_boundary_conv()

    def _init_boundary_conv(self):
        """Initialize first conv with edge detection bias"""
        # Keep default initialization, let network learn SAR-specific boundaries
        pass

    def forward(self, encoder_feat):
        """
        Args:
            encoder_feat: [B, C, H, W] encoder output (stride 8 or 16)

        Returns:
            boundary_map: [B, 1, H, W] boundary response map
        """
        boundary_map = self.boundary_conv(encoder_feat)
        boundary_map = torch.sigmoid(boundary_map)  # [0, 1]
        return boundary_map

    def compute_query_boundary_score(self, boundary_map, query_boxes, feat_stride=8):
        """
        Compute boundary consistency score for each query.

        Args:
            boundary_map: [B, 1, H, W] boundary response
            query_boxes: [B, N, 4] in cxcywh format, normalized [0, 1]
            feat_stride: spatial stride of boundary_map relative to input

        Returns:
            boundary_scores: [B, N] boundary consistency score for each query
        """
        B, _, H, W = boundary_map.shape
        N = query_boxes.shape[1]

        # Convert normalized boxes to feature map coordinates
        boxes_feat = query_boxes.clone()
        boxes_feat[..., [0, 2]] = boxes_feat[..., [0, 2]] * W  # cx, w
        boxes_feat[..., [1, 3]] = boxes_feat[..., [1, 3]] * H  # cy, h

        # Extract boundary ROI for each query
        scores = []
        for b in range(B):
            batch_scores = []
            for n in range(N):
                cx, cy, w, h = boxes_feat[b, n]

                # Get box boundary coordinates
                x1 = (cx - w / 2).clamp(0, W - 1)
                y1 = (cy - h / 2).clamp(0, H - 1)
                x2 = (cx + w / 2).clamp(0, W - 1)
                y2 = (cy + h / 2).clamp(0, H - 1)

                # Extract ROI boundary
                x1_int, y1_int = int(x1), int(y1)
                x2_int, y2_int = int(x2.ceil()), int(y2.ceil())

                if x2_int <= x1_int or y2_int <= y1_int:
                    batch_scores.append(0.0)
                    continue

                roi = boundary_map[b, 0, y1_int:y2_int, x1_int:x2_int]

                if roi.numel() == 0:
                    batch_scores.append(0.0)
                    continue

                # Boundary score: mean response on box edges
                # Top/bottom edges
                edge_top = roi[0, :] if roi.shape[0] > 0 else torch.tensor([0.])
                edge_bottom = roi[-1, :] if roi.shape[0] > 0 else torch.tensor([0.])
                # Left/right edges
                edge_left = roi[:, 0] if roi.shape[1] > 0 else torch.tensor([0.])
                edge_right = roi[:, -1] if roi.shape[1] > 0 else torch.tensor([0.])

                edge_score = (edge_top.mean() + edge_bottom.mean() +
                             edge_left.mean() + edge_right.mean()) / 4.0

                batch_scores.append(edge_score.item())

            scores.append(batch_scores)

        return torch.tensor(scores, device=boundary_map.device, dtype=boundary_map.dtype)


class ShapeAdaptiveScoring(nn.Module):
    """
    Shape-Adaptive Query Scoring (SAQS) Module

    Dynamically adjusts query quality scoring based on target shape.
    Different shapes (point-like tanks vs linear bridges) have different localization challenges.

    Key difference from CGAQ-DETR:
    - CGAQ uses only corners (single feature)
    - We use shape-adaptive strategy (more flexible)
    """

    def __init__(self, hidden_dim=256, num_shape_types=3):
        super().__init__()

        # Shape classification head (lightweight)
        self.shape_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_shape_types)
        )

        # Learnable weights for different shape types
        # Shape types: 0=point-like (tanks), 1=linear (bridges), 2=irregular (harbors)
        self.register_buffer('shape_weights', torch.tensor([
            [0.3, 0.4, 0.3],  # point: cls, iou, boundary (focus on center)
            [0.3, 0.3, 0.4],  # linear: cls, iou, boundary (focus on boundary)
            [0.33, 0.33, 0.34]  # irregular: balanced
        ]))

    def forward(self, query_embed, cls_score, iou_score, boundary_score):
        """
        Args:
            query_embed: [B, N, C] query embeddings
            cls_score: [B, N] classification confidence
            iou_score: [B, N] IoU quality (from detection head)
            boundary_score: [B, N] boundary consistency (from BCM)

        Returns:
            adaptive_score: [B, N] shape-adaptive quality score
        """
        B, N, C = query_embed.shape

        # Predict shape type for each query
        shape_logits = self.shape_classifier(query_embed)  # [B, N, 3]
        shape_probs = F.softmax(shape_logits, dim=-1)  # [B, N, 3]

        # Compute weighted score for each shape type
        # scores_matrix: [B, N, 3] where dim-2 is [cls, iou, boundary]
        scores_matrix = torch.stack([cls_score, iou_score, boundary_score], dim=-1)

        # Shape-adaptive weighting: [B, N, 3] x [3, 3] -> [B, N, 3]
        weighted_scores = torch.matmul(shape_probs, self.shape_weights)  # [B, N, 3]

        # Final score: element-wise multiply and sum
        adaptive_score = (weighted_scores * scores_matrix).sum(dim=-1)  # [B, N]

        return adaptive_score


def extract_boundary_features(feature_map, boxes, output_size=7):
    """
    Extract boundary features from feature map given boxes.
    Used for Boundary Refinement Loss (BRL).

    Args:
        feature_map: [B, C, H, W]
        boxes: [B, N, 4] in xyxy format (feature map coordinates)
        output_size: ROI pooling output size

    Returns:
        boundary_feats: [B, N, C] extracted boundary features
    """
    # This is a placeholder - actual implementation would use ROIAlign
    # For simplicity, we'll use the box center features
    B, C, H, W = feature_map.shape
    N = boxes.shape[1]

    # Get box centers
    cx = ((boxes[..., 0] + boxes[..., 2]) / 2).clamp(0, W - 1).long()
    cy = ((boxes[..., 1] + boxes[..., 3]) / 2).clamp(0, H - 1).long()

    # Extract features at centers
    boundary_feats = []
    for b in range(B):
        feats = feature_map[b, :, cy[b], cx[b]].T  # [N, C]
        boundary_feats.append(feats)

    return torch.stack(boundary_feats, dim=0)  # [B, N, C]


def generate_gt_boundary_map(targets, feat_height, feat_width, feat_stride=8, line_width=2):
    """
    Generate ground-truth boundary map from bounding boxes.

    Args:
        targets: list of dicts, each containing:
            - 'boxes': [N, 4] in cxcywh format, normalized [0, 1]
            - 'labels': [N]
        feat_height: boundary map height (feature map resolution)
        feat_width: boundary map width
        feat_stride: spatial stride of feature map relative to input
        line_width: boundary line width in feature map pixels

    Returns:
        gt_boundary_maps: [B, 1, H, W] binary boundary maps
    """
    B = len(targets)
    device = targets[0]['boxes'].device if len(targets) > 0 and targets[0]['boxes'].numel() > 0 else 'cpu'

    gt_maps = []

    for target in targets:
        boxes = target['boxes']  # [N, 4] cxcywh normalized
        N = boxes.shape[0]

        # Initialize boundary map
        boundary_map = torch.zeros(feat_height, feat_width, device=device)

        if N == 0:
            gt_maps.append(boundary_map.unsqueeze(0))
            continue

        # Convert normalized boxes to feature map coordinates
        boxes_feat = boxes.clone()
        boxes_feat[:, [0, 2]] = boxes_feat[:, [0, 2]] * feat_width  # cx, w
        boxes_feat[:, [1, 3]] = boxes_feat[:, [1, 3]] * feat_height  # cy, h

        # Draw boundary for each box
        for i in range(N):
            cx, cy, w, h = boxes_feat[i]

            # Get box corners
            x1 = (cx - w / 2).clamp(0, feat_width - 1)
            y1 = (cy - h / 2).clamp(0, feat_height - 1)
            x2 = (cx + w / 2).clamp(0, feat_width - 1)
            y2 = (cy + h / 2).clamp(0, feat_height - 1)

            x1_int, y1_int = int(x1), int(y1)
            x2_int, y2_int = int(x2.ceil()), int(y2.ceil())

            # Draw boundary lines (edges of the box)
            # Top edge
            for lw in range(line_width):
                y_top = max(0, y1_int + lw)
                if y_top < feat_height:
                    boundary_map[y_top, x1_int:x2_int] = 1.0

            # Bottom edge
            for lw in range(line_width):
                y_bottom = min(feat_height - 1, y2_int - lw)
                if y_bottom >= 0:
                    boundary_map[y_bottom, x1_int:x2_int] = 1.0

            # Left edge
            for lw in range(line_width):
                x_left = max(0, x1_int + lw)
                if x_left < feat_width:
                    boundary_map[y1_int:y2_int, x_left] = 1.0

            # Right edge
            for lw in range(line_width):
                x_right = min(feat_width - 1, x2_int - lw)
                if x_right >= 0:
                    boundary_map[y1_int:y2_int, x_right] = 1.0

        gt_maps.append(boundary_map.unsqueeze(0))

    # Stack to [B, 1, H, W]
    return torch.stack(gt_maps, dim=0)
