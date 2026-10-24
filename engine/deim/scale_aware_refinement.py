"""
Scale-Aware Refinement (SAR) for SAR Object Detection
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.

Key innovation:
Adaptively select and refine box scales based on scattering characteristics.

Problem Analysis:
- 36.6% localization errors, many are scale estimation errors
- SAR targets have scale-dependent scattering patterns:
  * Small targets (ships): concentrated scattering, clear boundaries
  * Large targets (bridges): distributed scattering, fuzzy boundaries
- Single-scale features are suboptimal for multi-scale targets

Solution:
Instead of using fixed-scale features for all boxes, SAR:
1. Predicts optimal feature scale for each box
2. Adaptively pools features from the predicted scale
3. Refines box scale (width/height) based on pooled features
4. Maintains center position (only adjust size)

Advantages over standard multi-scale fusion:
- Query-specific scale selection (not all-to-all fusion)
- Physics-informed (scattering-scale relationship)
- Lightweight (single forward pass per scale)
"""

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign


class ScalePredictor(nn.Module):
    """
    Predict the optimal feature scale for each detection box.

    Given a query and its predicted box, determine which feature pyramid
    level (stride 8, 16, 32) contains the most relevant information.

    Architecture:
    - Input: query embedding + box size features
    - Output: scale distribution [B, Q, num_scales]
    """

    def __init__(self, hidden_dim=256, num_scales=3, fp32_ops=True):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        self.fp32_ops = fp32_ops

        # Box size encoder
        self.box_size_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),  # Only w, h (not cx, cy)
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(inplace=True)
        )

        # Query feature projection
        self.query_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True)
        )

        # Scale prediction
        self.scale_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 4 + hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_scales)
            # No softmax here, will be applied in forward
        )

        # Conservative initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, query_embed, pred_boxes):
        """
        Args:
            query_embed: [B, num_queries, C]
            pred_boxes: [B, num_queries, 4] (cxcywh normalized)

        Returns:
            scale_probs: [B, num_queries, num_scales] scale distribution
        """
        # Extract box size (w, h only)
        box_size = pred_boxes[..., 2:4]  # [B, Q, 2]
        box_size_feat = self.box_size_encoder(box_size)

        # Project query
        query_feat = self.query_proj(query_embed)

        # Fuse and predict
        fused = torch.cat([query_feat, box_size_feat], dim=-1)
        scale_logits = self.scale_predictor(fused)  # [B, Q, num_scales]

        # Softmax to get distribution
        if self.fp32_ops:
            scale_probs = F.softmax(scale_logits, dim=-1, dtype=torch.float32)
        else:
            scale_probs = F.softmax(scale_logits, dim=-1)

        return scale_probs


class AdaptiveScalePooling(nn.Module):
    """
    Pool features from multiple scales with adaptive weighting.

    Instead of pooling from all scales equally, use the predicted
    scale distribution to weight different scales.
    """

    def __init__(self, hidden_dim=256, output_size=7, fp32_ops=True):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.fp32_ops = fp32_ops

        # RoI pooling for each scale
        self.roi_align = RoIAlign(
            output_size=(output_size, output_size),
            spatial_scale=1.0,  # Will be set per-scale
            sampling_ratio=2
        )

    def forward(self, multi_scale_feats, pred_boxes, scale_probs, spatial_shapes):
        """
        Args:
            multi_scale_feats: list of [B, C, H, W] (3 scales)
            pred_boxes: [B, num_queries, 4] normalized (cxcywh)
            scale_probs: [B, num_queries, num_scales]
            spatial_shapes: list of [H, W] for each scale

        Returns:
            pooled_feat: [B, num_queries, C, out_size, out_size]
        """
        B, num_queries = pred_boxes.shape[:2]
        num_scales = len(multi_scale_feats)

        # Convert boxes from cxcywh to xyxy (still normalized)
        boxes_input = pred_boxes.float() if self.fp32_ops else pred_boxes
        boxes_xyxy = self._cxcywh_to_xyxy(boxes_input)  # [B, Q, 4]

        # Pool from each scale
        weighted_feat = None
        pooled_feats_per_scale = []

        for scale_idx, feat in enumerate(multi_scale_feats):
            H, W = spatial_shapes[scale_idx]

            # Denormalize boxes to feature map size
            boxes_denorm = boxes_xyxy.clone()
            boxes_denorm[..., [0, 2]] *= W  # x coords
            boxes_denorm[..., [1, 3]] *= H  # y coords

            # Prepare for RoI pooling: add batch index
            # RoI format: [batch_idx, x1, y1, x2, y2]
            batch_indices = torch.arange(B, device=feat.device)[:, None].expand(B, num_queries).reshape(-1)
            boxes_flat = boxes_denorm.reshape(-1, 4)
            # torch.cat promotes (fp32 index, fp16 boxes) to fp32, which then
            # mismatches the fp16 feature map in RoIAlign. Build the rois in the
            # dtype that the pooling path will actually use.
            rois_dtype = torch.float32 if self.fp32_ops else feat.dtype
            rois = torch.cat([batch_indices[:, None].to(rois_dtype), boxes_flat.to(rois_dtype)],
                             dim=1)  # [B*Q, 5]

            if self.fp32_ops:
                # Preserve the separately recorded BF16/FP32 experiment.
                with torch.autocast(device_type=feat.device.type, enabled=False):
                    pooled = self.roi_align(feat.float(), rois.float())
                    pooled = pooled.reshape(B, num_queries, self.hidden_dim, self.output_size, self.output_size)
                    scale_weight = scale_probs[:, :, scale_idx, None, None, None].float()
                    weighted = pooled * scale_weight
                    weighted_feat = weighted if weighted_feat is None else weighted_feat + weighted
            else:
                # Original SAR operations, dtypes and reduction order.
                pooled = self.roi_align(feat, rois)
                pooled = pooled.reshape(B, num_queries, self.hidden_dim, self.output_size, self.output_size)
                pooled_feats_per_scale.append(pooled)

        if not self.fp32_ops:
            pooled_feats = torch.stack(pooled_feats_per_scale, dim=2)
            scale_weights = scale_probs[:, :, :, None, None, None]
            weighted_feat = (pooled_feats * scale_weights).sum(dim=2)

        return weighted_feat

    def _cxcywh_to_xyxy(self, boxes):
        """Convert cxcywh to xyxy format."""
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return torch.stack([x1, y1, x2, y2], dim=-1)


class ScaleRefinementHead(nn.Module):
    """
    Refine box scale based on pooled features.

    Output: scale adjustment factors for width and height.
    """

    def __init__(self, hidden_dim=256, pooled_size=7, fp32_ops=True):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.pooled_size = pooled_size
        self.fp32_ops = fp32_ops

        # Feature processing
        self.feat_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, 1, 1),
            nn.GroupNorm(16, hidden_dim // 2),
            nn.ReLU(inplace=True)
        )

        # Global pooling + MLP
        self.refinement_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, 2)  # Scale factors for w, h
        )

        # Zero-init the final layer (start as identity)
        nn.init.zeros_(self.refinement_head[-1].weight)
        nn.init.zeros_(self.refinement_head[-1].bias)

    def forward(self, pooled_feat):
        """
        Args:
            pooled_feat: [B, num_queries, C, H, W]

        Returns:
            scale_delta: [B, num_queries, 2] adjustment for (w, h)
        """
        B, Q, C, H, W = pooled_feat.shape

        # Reshape for conv: [B*Q, C, H, W]
        feat = pooled_feat.reshape(B * Q, C, H, W)

        # Process features
        feat = self.feat_conv(feat)  # [B*Q, C/2, H, W]

        # Global average pooling
        feat = F.adaptive_avg_pool2d(feat, (1, 1))  # [B*Q, C/2, 1, 1]
        feat = feat.flatten(1)  # [B*Q, C/2]

        # Original AMP head unless the archived FP32 operation mode is selected.
        head_context = (torch.autocast(device_type=feat.device.type, enabled=False)
                        if self.fp32_ops else nullcontext())
        with head_context:
            head_input = feat.float() if self.fp32_ops else feat
            scale_delta = self.refinement_head(head_input).reshape(B, Q, 2)
            scale_delta = torch.tanh(scale_delta) * 0.3

        return scale_delta


class ScaleAwareRefinement(nn.Module):
    """
    Complete Scale-Aware Refinement (SAR) module.

    Pipeline:
    1. Predict optimal scale for each box
    2. Adaptively pool features from multi-scale pyramid
    3. Refine box width/height based on pooled features
    4. Keep center (cx, cy) unchanged (only adjust scale)

    Training supervision:
    - Scale prediction: supervised by GT box size vs. feature pyramid levels
    - Scale refinement: standard box regression loss
    """

    def __init__(self,
                 hidden_dim=256,
                 num_scales=3,
                 pooled_size=7,
                 fp32_ops=True):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        self.pooled_size = pooled_size
        self.fp32_ops = fp32_ops

        # === 1. Scale predictor ===
        self.scale_predictor = ScalePredictor(hidden_dim, num_scales, fp32_ops=fp32_ops)

        # === 2. Adaptive pooling ===
        self.adaptive_pooling = AdaptiveScalePooling(hidden_dim, pooled_size, fp32_ops=fp32_ops)

        # === 3. Refinement head ===
        self.refinement_head = ScaleRefinementHead(hidden_dim, pooled_size, fp32_ops=fp32_ops)

    def forward(self, query_embed, pred_boxes, multi_scale_feats, spatial_shapes):
        """
        Args:
            query_embed: [B, num_queries, C]
            pred_boxes: [B, num_queries, 4] normalized (cxcywh)
            multi_scale_feats: list of [B, C, H, W] (typically 3 scales)
            spatial_shapes: list of [H, W] for each scale

        Returns:
            refined_boxes: [B, num_queries, 4]
            scale_probs: [B, num_queries, num_scales] (for visualization/loss)
        """
        # Predict optimal scale
        scale_probs = self.scale_predictor(query_embed, pred_boxes)

        # Adaptive pooling
        pooled_feat = self.adaptive_pooling(
            multi_scale_feats, pred_boxes, scale_probs, spatial_shapes
        )

        # Predict scale adjustment
        scale_delta = self.refinement_head(pooled_feat)  # [B, Q, 2] for (w, h)

        # Apply refinement: only adjust w, h; keep cx, cy
        # Non-inplace operation to preserve gradient graph
        boxes_input = pred_boxes.float() if self.fp32_ops else pred_boxes
        cx_cy = boxes_input[..., :2]  # [B, Q, 2]
        wh = boxes_input[..., 2:4]  # [B, Q, 2]

        # Scale adjustment
        wh_refined = wh * (1 + scale_delta)

        # Clamp to valid range
        wh_refined = torch.clamp(wh_refined, min=0.01, max=1.0)

        # Concatenate back (non-inplace)
        refined_boxes = torch.cat([cx_cy, wh_refined], dim=-1)

        return refined_boxes, scale_probs

    def get_scale_supervision(self, pred_boxes, gt_boxes, spatial_shapes):
        """
        Generate scale supervision labels.

        Rule: assign GT box to the scale where its size best matches:
        - Small boxes (area < 32^2) → scale 0 (stride 8)
        - Medium boxes (32^2 ~ 96^2) → scale 1 (stride 16)
        - Large boxes (> 96^2) → scale 2 (stride 32)

        Args:
            pred_boxes: [B, num_queries, 4]
            gt_boxes: list of [num_gt, 4] per batch
            spatial_shapes: list of [H, W]

        Returns:
            scale_labels: [B, num_queries] in {0, 1, 2, -1}
                         -1 = ignore (unmatched query)
        """
        device = pred_boxes.device
        B, num_queries = pred_boxes.shape[:2]

        scale_labels = torch.full((B, num_queries), -1, dtype=torch.long, device=device)

        # Define scale thresholds (in normalized coordinates)
        # Assuming input size 640x640
        scale_thresholds = [32/640, 96/640]  # [0.05, 0.15]

        for batch_idx in range(B):
            if len(gt_boxes[batch_idx]) == 0:
                continue

            gt_wh = gt_boxes[batch_idx][..., 2:4]  # [num_gt, 2]
            gt_area = (gt_wh[..., 0] * gt_wh[..., 1]).sqrt()  # Geometric mean

            # Assign scale
            scale = torch.zeros(len(gt_area), dtype=torch.long, device=device)
            scale[gt_area >= scale_thresholds[1]] = 2  # Large
            scale[(gt_area >= scale_thresholds[0]) & (gt_area < scale_thresholds[1])] = 1  # Medium
            scale[gt_area < scale_thresholds[0]] = 0  # Small

            # This is a simplified supervision (need matcher indices for accurate assignment)
            # In practice, use matched queries from criterion
            # For now, mark as placeholder
            # scale_labels[batch_idx, :len(scale)] = scale

        return scale_labels


class SARLoss(nn.Module):
    """
    Loss for SAR training (optional, can use existing box loss).
    """

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, scale_probs, scale_labels):
        """
        Scale prediction loss (cross-entropy).

        Args:
            scale_probs: [B, num_queries, num_scales]
            scale_labels: [B, num_queries] in {0, 1, 2, -1}

        Returns:
            loss: scalar
        """
        valid_mask = scale_labels >= 0

        if valid_mask.sum() == 0:
            return scale_probs.sum() * 0.0

        scale_probs_valid = scale_probs[valid_mask]
        scale_labels_valid = scale_labels[valid_mask]

        loss = F.cross_entropy(scale_probs_valid, scale_labels_valid)

        return loss * self.loss_weight
