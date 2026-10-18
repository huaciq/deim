"""
Adaptive Feature Selection (AFS) for SAR Small Object Detection
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.

Key insight from error analysis:
- 12.4% tiny object miss rate (1038 missed out of 8369)
- Small objects (7x7 ~ 15x15 pixels) lose features during downsampling

Problem:
- Single-scale stride-8 feature cannot capture multi-scale small objects
- Different objects need different receptive fields

Solution:
- Multi-scale feature extraction with adaptive selection
- Learn which scale is most useful for each spatial position
- Lightweight (no wavelets, no extra backbone modifications)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_activation


class AdaptiveFeatureSelection(nn.Module):
    """
    Adaptive Feature Selection (AFS)

    Captures multi-scale features and adaptively selects the best scale
    for each spatial location using attention mechanism.

    Key components:
    1. Multi-scale depthwise convolutions (lightweight)
    2. Spatial attention to select best scale per position
    3. Residual fusion to preserve original features
    """

    def __init__(self, channels=256, scales=[1, 2, 3], act='silu'):
        """
        Args:
            channels: feature channels
            scales: list of dilation rates for multi-scale convolution
                    scale=1 means 3x3, scale=2 means 5x5, scale=3 means 7x7
            act: activation function
        """
        super().__init__()

        self.channels = channels
        self.scales = scales
        self.num_scales = len(scales)

        # ===== 1. Multi-scale feature extraction =====
        # Use depthwise convolution for efficiency
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                # Depthwise conv with different kernel sizes
                nn.Conv2d(
                    channels, channels,
                    kernel_size=scale * 2 + 1,  # scale 1->3x3, 2->5x5, 3->7x7
                    stride=1,
                    padding=scale,  # Same padding
                    groups=channels,  # Depthwise
                    bias=False
                ),
                nn.BatchNorm2d(channels),
                get_activation(act)
            ) for scale in scales
        ])

        # ===== 2. Scale attention =====
        # Learn which scale is most useful for each spatial position
        # Input: concatenated multi-scale features
        # Output: attention weights per scale [B, num_scales, H, W]
        self.scale_attention = nn.Sequential(
            nn.Conv2d(channels * self.num_scales, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            get_activation(act),
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.BatchNorm2d(channels // 4),
            get_activation(act),
            nn.Conv2d(channels // 4, self.num_scales, 1, bias=True),
        )

        # Temperature parameter for softmax (learnable)
        self.temperature = nn.Parameter(torch.ones(1))

        # ===== 3. Fusion =====
        # Fuse selected multi-scale features back to original
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

        # Initialize fusion as residual (zero-init)
        nn.init.zeros_(self.fusion[1].weight)

        # Learnable fusion weight
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        """
        Apply adaptive feature selection.

        Args:
            x: [B, C, H, W] input feature (stride 8 from encoder)

        Returns:
            out: [B, C, H, W] enhanced feature with multi-scale information
        """
        B, C, H, W = x.shape

        # ===== 1. Extract multi-scale features =====
        scale_features = []
        for scale_conv in self.scale_convs:
            scale_feat = scale_conv(x)  # [B, C, H, W]
            scale_features.append(scale_feat)

        # ===== 2. Compute attention weights =====
        # Concatenate all scales
        concat_feat = torch.cat(scale_features, dim=1)  # [B, C*num_scales, H, W]

        # Predict attention weights
        attn_logits = self.scale_attention(concat_feat)  # [B, num_scales, H, W]

        # Apply temperature and softmax
        attn_logits = attn_logits / self.temperature.clamp(min=0.1)
        attn_weights = F.softmax(attn_logits, dim=1)  # [B, num_scales, H, W]

        # ===== 3. Weighted fusion of multi-scale features =====
        # Multiply each scale by its attention weight and sum
        fused = torch.zeros_like(x)
        for i, scale_feat in enumerate(scale_features):
            # Broadcast attention: [B, 1, H, W] * [B, C, H, W]
            fused = fused + scale_feat * attn_weights[:, i:i+1, :, :]

        # ===== 4. Residual connection =====
        # Apply learnable alpha to balance original and multi-scale features
        fusion_out = self.fusion(fused)
        out = x + self.alpha.clamp(0, 1) * fusion_out

        return out

    def extra_repr(self):
        return f'channels={self.channels}, scales={self.scales}, num_scales={self.num_scales}'


class MultiScaleEnhancement(nn.Module):
    """
    Optional: Stack multiple AFS modules for stronger enhancement.

    For ablation: single AFS vs stacked AFS
    """

    def __init__(self, channels=256, num_layers=2, scales=[1, 2, 3], act='silu'):
        super().__init__()

        self.layers = nn.ModuleList([
            AdaptiveFeatureSelection(channels, scales, act)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
