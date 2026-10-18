"""
Detail Preservation Module (DPM) for SAR Small Object Detection
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.

Key insight from error analysis:
- Tiny objects have 12.4% miss rate (1038 missed out of 8369)
- Storage tank (point-like, 7x7 pixels) has worst recall: 83.9%

Problem:
- During encoder downsampling (stride 4 -> 8), small objects lose high-frequency details
- 7x7 pixel objects become indistinguishable from background noise at stride 8

Solution:
- Use wavelet decomposition to preserve high-frequency details
- Filter SAR-specific speckle noise while keeping object edges
- Fuse detail features with encoder output via learned gating
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_activation


class HaarWaveletDecomposition(nn.Module):
    """
    Haar Wavelet 2D Decomposition for detail extraction.

    Decomposes feature map into:
    - LL: Low-low (approximation, smooth regions)
    - LH: Low-high (horizontal edges)
    - HL: High-low (vertical edges)
    - HH: High-high (diagonal edges, corners)

    For small object detection, LH/HL/HH contain critical edge information.
    """

    def __init__(self):
        super().__init__()

        # Haar wavelet kernels (fixed, not learnable)
        # LL: average (low-pass)
        # LH: horizontal edge
        # HL: vertical edge
        # HH: diagonal edge
        self.register_buffer('haar_kernels', self._get_haar_kernels())

    def _get_haar_kernels(self):
        """
        Create 2D Haar wavelet kernels.

        Returns:
            kernels: [4, 1, 2, 2] - 4 kernels for LL, LH, HL, HH
        """
        # Normalized Haar basis
        ll = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) * 0.5
        lh = torch.tensor([[1, 1], [-1, -1]], dtype=torch.float32) * 0.5
        hl = torch.tensor([[1, -1], [1, -1]], dtype=torch.float32) * 0.5
        hh = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) * 0.5

        kernels = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # [4, 1, 2, 2]
        return kernels

    def forward(self, x):
        """
        Decompose feature map using Haar wavelets.

        Args:
            x: [B, C, H, W] input feature

        Returns:
            ll: [B, C, H//2, W//2] low-frequency approximation
            lh: [B, C, H//2, W//2] horizontal edges
            hl: [B, C, H//2, W//2] vertical edges
            hh: [B, C, H//2, W//2] diagonal edges
        """
        B, C, H, W = x.shape

        # Apply Haar kernels via depthwise convolution with stride 2
        # Process each channel separately
        x_unfold = x.reshape(B * C, 1, H, W)

        # Convolve with each Haar kernel
        wavelets = []
        for i in range(4):
            kernel = self.haar_kernels[i:i+1]  # [1, 1, 2, 2]
            w = F.conv2d(x_unfold, kernel, stride=2, padding=0)
            wavelets.append(w)

        # Reshape back to [B, C, H//2, W//2]
        ll = wavelets[0].reshape(B, C, H // 2, W // 2)
        lh = wavelets[1].reshape(B, C, H // 2, W // 2)
        hl = wavelets[2].reshape(B, C, H // 2, W // 2)
        hh = wavelets[3].reshape(B, C, H // 2, W // 2)

        return ll, lh, hl, hh


class SARNoiseFilter(nn.Module):
    """
    SAR-specific noise filtering for high-frequency components.

    SAR images suffer from multiplicative speckle noise, which appears
    as random high-frequency fluctuations. We need to suppress noise
    while preserving genuine object edges.

    Strategy:
    - Learn edge confidence maps from high-freq components
    - Use non-local similarity to distinguish structured edges from random noise
    - Apply soft gating to filter noise
    """

    def __init__(self, in_channels, act='silu'):
        super().__init__()

        # Edge confidence predictor
        # Predicts which high-freq responses are genuine edges vs noise
        self.edge_confidence = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 3, 1, 1, bias=False),  # 3 high-freq bands
            nn.BatchNorm2d(in_channels),
            get_activation(act),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            get_activation(act),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()  # Confidence in [0, 1]
        )

        # Noise suppression gate
        # Learns to adaptively filter noise based on local statistics
        self.noise_gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            get_activation(act),
            nn.Conv2d(in_channels // 2, in_channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, lh, hl, hh):
        """
        Filter speckle noise from high-frequency components.

        Args:
            lh, hl, hh: [B, C, H, W] horizontal, vertical, diagonal edges

        Returns:
            filtered: [B, C, H, W] noise-filtered high-freq features
        """
        # Concatenate all high-freq bands
        high_freq = torch.cat([lh, hl, hh], dim=1)  # [B, 3C, H, W]

        # Predict edge confidence
        edge_conf = self.edge_confidence(high_freq)  # [B, C, H, W]

        # Combine high-freq bands with confidence weighting
        combined = lh * edge_conf + hl * edge_conf + hh * edge_conf

        # Apply noise gate
        gate = self.noise_gate(combined)
        filtered = combined * gate

        return filtered


class DetailPreservationModule(nn.Module):
    """
    Detail Preservation Module (DPM)

    Preserves small object details during encoder downsampling.

    Pipeline:
    1. Wavelet decomposition of high-res feature (stride 4)
    2. SAR noise filtering on high-freq components
    3. Detail enhancement via learned projections
    4. Adaptive fusion with low-res encoder feature (stride 8)

    Key difference from HWD:
    - HWD only concatenates wavelets, doesn't filter noise
    - DPM specifically handles SAR speckle and small objects
    - Adaptive fusion weights details based on local content
    """

    def __init__(self, in_channels, out_channels, act='silu', detail_weight=0.5):
        """
        Args:
            in_channels: channels of high-res feature (stride 4)
            out_channels: channels of low-res feature (stride 8)
            act: activation function
            detail_weight: initial weight for detail branch (learnable)
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # ===== 1. Wavelet decomposition =====
        self.wavelet = HaarWaveletDecomposition()

        # ===== 2. Process low-frequency (LL) =====
        # LL contains coarse structure, project to output channels
        self.ll_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            get_activation(act)
        )

        # ===== 3. Process high-frequency (LH, HL, HH) =====
        # Project all high-freq bands together
        self.hf_proj = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            get_activation(act)
        )

        # Then filter noise (simplified - operate on combined projection)
        self.noise_filter = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            get_activation(act),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()  # Noise gate
        )

        # Detail enhancement
        self.detail_enhance = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            get_activation(act),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            get_activation(act)
        )

        # ===== 4. Adaptive fusion =====
        # Learn to weight detail branch based on content
        # Small objects need more detail, large objects need less
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            get_activation(act),
            nn.Conv2d(out_channels, out_channels, 1, bias=True),
            nn.Sigmoid()  # Gate in [0, 1]
        )

        # Final projection
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # Initialize fusion as residual (zero-init BN)
        nn.init.zeros_(self.fusion_proj[1].weight)

        # Learnable detail weight
        self.detail_alpha = nn.Parameter(torch.tensor(detail_weight))

    def forward(self, x_high_res, x_low_res):
        """
        Fuse detail-preserved high-res features with low-res encoder features.

        Args:
            x_high_res: [B, C_in, H*2, W*2] stride-4 feature from backbone
            x_low_res: [B, C_out, H, W] stride-8 feature from encoder

        Returns:
            fused: [B, C_out, H, W] detail-enhanced encoder feature
        """
        B, C_out, H, W = x_low_res.shape

        # ===== 1. Wavelet decomposition =====
        ll, lh, hl, hh = self.wavelet(x_high_res)  # Each: [B, C_in, H, W]

        # ===== 2. Process low-frequency =====
        ll_feat = self.ll_proj(ll)  # [B, C_out, H, W]

        # ===== 3. Process high-frequency =====
        # Concatenate all high-freq bands
        hf_concat = torch.cat([lh, hl, hh], dim=1)  # [B, 3*C_in, H, W]

        # Project to output channels
        hf_proj = self.hf_proj(hf_concat)  # [B, C_out, H, W]

        # Apply noise filtering gate
        noise_gate = self.noise_filter(hf_proj)  # [B, C_out, H, W]
        hf_filtered = hf_proj * noise_gate

        # Enhance details
        hf_enhanced = self.detail_enhance(hf_filtered)  # [B, C_out, H, W]

        # ===== 4. Adaptive fusion =====
        # Combine LL (structure) + HF (details)
        detail_branch = ll_feat + self.detail_alpha.clamp(0, 1) * hf_enhanced

        # Compute fusion gate
        gate_input = torch.cat([x_low_res, detail_branch], dim=1)
        fusion_gate = self.fusion_gate(gate_input)  # [B, C_out, H, W]

        # Apply gate to detail branch
        detail_gated = detail_branch * fusion_gate

        # Fuse with low-res feature
        fused_input = torch.cat([x_low_res, detail_gated], dim=1)
        fused_delta = self.fusion_proj(fused_input)

        # Residual connection
        fused = x_low_res + fused_delta

        return fused
