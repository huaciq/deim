"""
Scattering Pattern Isolation (SPI) for SAR Object Detection
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.

Key innovation:
Explicitly model SAR scattering mechanisms instead of treating SAR as grayscale images.

Physical foundation:
- SAR targets: Strong scattering (corner reflectors, metallic structures)
- Clutter: Diffuse scattering (sea surface, vegetation, buildings)
- Speckle noise: Random scattering (multiplicative noise)

Unlike polarimetric decomposition (requires multi-pol data), SPI learns to decompose
single-pol intensity into scattering patterns using deep features and statistical priors.

References:
- Cloude-Pottier decomposition (polarimetric SAR)
- CFAR detection (constant false alarm rate)
- Freeman-Durden scattering model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_activation


class LocalStatisticsExtractor(nn.Module):
    """
    Extract local statistical features to approximate scattering characteristics.

    SAR intensity statistics:
    - Mean: overall backscatter level
    - Variance: texture roughness
    - Coefficient of Variation (CV): speckle level
    - Higher-order moments: scattering mechanism indicators
    """

    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size
        self.channels = channels

        # Local pooling for statistics
        self.avg_pool = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size//2)

        # Feature extraction after statistics
        self.stat_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),  # mean + variance
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        """
        Extract local mean and variance.

        Args:
            x: [B, C, H, W] feature map

        Returns:
            stats: [B, C, H, W] statistical features
        """
        # Clamp input to avoid extreme values
        x = torch.clamp(x, min=-10, max=10)

        # Local mean
        mean = self.avg_pool(x)

        # Local variance: E[x^2] - E[x]^2
        x_squared = x * x
        mean_squared = self.avg_pool(x_squared)
        variance = mean_squared - mean * mean
        variance = F.relu(variance)  # Ensure non-negative

        # Add epsilon for numerical stability
        variance = variance + 1e-6

        # Concatenate statistics
        stats = torch.cat([mean, variance], dim=1)

        # Extract features from statistics
        out = self.stat_conv(stats)

        return out


class ScatteringDecomposer(nn.Module):
    """
    Decompose features into three scattering components:
    1. Strong scattering (targets): high intensity, low variance
    2. Diffuse scattering (clutter): medium intensity, high variance
    3. Random scattering (speckle): random, high coefficient of variation

    Inspired by polarimetric decomposition but adapted for single-pol + deep features.
    """

    def __init__(self, in_channels, act='silu'):
        super().__init__()

        # Analyze scattering characteristics
        self.analyzer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            get_activation(act),
            nn.Conv2d(in_channels, in_channels // 2, 1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            get_activation(act)
        )

        # Decompose into 3 scattering components
        self.decompose = nn.Sequential(
            nn.Conv2d(in_channels // 2, 3, 1, bias=True),  # 3 components
        )

    def forward(self, x):
        """
        Decompose into scattering components.

        Args:
            x: [B, C, H, W] input features

        Returns:
            scatter_probs: [B, 3, H, W] probability of each scattering type
                         [strong, diffuse, random]
        """
        # Clamp input for stability
        x = torch.clamp(x, min=-10, max=10)

        feat = self.analyzer(x)
        scatter_logits = self.decompose(feat)  # [B, 3, H, W]

        # Clamp logits to avoid overflow in softmax
        scatter_logits = torch.clamp(scatter_logits, min=-10, max=10)

        # Softmax to get probability distribution
        scatter_probs = F.softmax(scatter_logits, dim=1)

        return scatter_probs


class ScatteringPatternIsolation(nn.Module):
    """
    Scattering Pattern Isolation (SPI)

    Core innovation: Use SAR physics to guide feature enhancement

    Pipeline:
    1. Extract local statistical features (approximate scattering properties)
    2. Decompose into 3 scattering patterns: strong/diffuse/random
    3. Compare with learned target/clutter priors
    4. Enhance features matching target scattering, suppress clutter

    Key differences from attention mechanisms:
    - Attention: "what to focus on" (data-driven)
    - SPI: "which scattering type matches targets" (physics-informed)
    """

    def __init__(self, channels=256, act='silu'):
        super().__init__()

        self.channels = channels

        # === 1. Local statistics extraction ===
        self.stat_extractor = LocalStatisticsExtractor(channels, kernel_size=5)

        # === 2. Scattering decomposition ===
        self.decomposer = ScatteringDecomposer(channels, act=act)

        # === 3. Scattering priors (learnable) ===
        # Prior knowledge: targets have strong scattering, clutter is diffuse
        # Initialize with physical intuition, then let network fine-tune
        self.register_buffer('target_prior_init', torch.tensor([0.7, 0.2, 0.1]))  # [strong, diffuse, random]
        self.register_buffer('clutter_prior_init', torch.tensor([0.2, 0.6, 0.2]))

        # Learnable priors (deviation from initial)
        self.target_prior_delta = nn.Parameter(torch.zeros(3))
        self.clutter_prior_delta = nn.Parameter(torch.zeros(3))

        # === 4. Pattern-based feature gating ===
        # Use scattering similarity to modulate features
        self.pattern_gate = nn.Sequential(
            nn.Conv2d(channels + 3, channels, 1, bias=False),  # features + scatter_probs
            nn.BatchNorm2d(channels),
            get_activation(act),
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            get_activation(act),
            nn.Conv2d(channels, channels, 1, bias=True),
            nn.Sigmoid()  # Gate values in [0, 1]
        )

        # === 5. Fusion ===
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

        # Initialize fusion as residual (zero-init)
        nn.init.zeros_(self.fusion[1].weight)

        # Learnable fusion strength
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)

    def get_scattering_priors(self):
        """Get current learned scattering priors."""
        # Add learnable delta to initial priors
        target_prior = self.target_prior_init + self.target_prior_delta
        clutter_prior = self.clutter_prior_init + self.clutter_prior_delta

        # Softmax to ensure valid probability distribution
        target_prior = F.softmax(target_prior, dim=0)
        clutter_prior = F.softmax(clutter_prior, dim=0)

        return target_prior, clutter_prior

    def compute_scattering_similarity(self, scatter_probs):
        """
        Compute similarity to target scattering pattern.

        High similarity means the spatial location exhibits target-like scattering.
        Low similarity means clutter-like or noise-like scattering.

        Args:
            scatter_probs: [B, 3, H, W] scattering probabilities

        Returns:
            target_similarity: [B, 1, H, W] similarity to target pattern
            clutter_similarity: [B, 1, H, W] similarity to clutter pattern
        """
        target_prior, clutter_prior = self.get_scattering_priors()

        # Compute weighted similarity (dot product of probability distributions)
        # scatter_probs: [B, 3, H, W]
        # target_prior: [3]
        target_similarity = (scatter_probs * target_prior.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
        clutter_similarity = (scatter_probs * clutter_prior.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)

        return target_similarity, clutter_similarity

    def forward(self, x):
        """
        Apply scattering pattern isolation.

        Args:
            x: [B, C, H, W] encoder features (stride 8)

        Returns:
            out: [B, C, H, W] enhanced features
            scatter_probs: [B, 3, H, W] scattering decomposition (for visualization/supervision)
            target_similarity: [B, 1, H, W] target scattering similarity map
        """
        # Clamp input features for numerical stability
        x = torch.clamp(x, min=-10, max=10)

        # === 1. Extract local statistics ===
        stat_feat = self.stat_extractor(x)

        # === 2. Decompose scattering patterns ===
        scatter_probs = self.decomposer(stat_feat)  # [B, 3, H, W]

        # === 3. Compute scattering similarities ===
        target_sim, clutter_sim = self.compute_scattering_similarity(scatter_probs)

        # Clamp similarities to valid range
        target_sim = torch.clamp(target_sim, min=0, max=1)
        clutter_sim = torch.clamp(clutter_sim, min=0, max=1)

        # === 4. Generate pattern-aware gate ===
        # Concatenate features with scattering info
        gate_input = torch.cat([x, scatter_probs], dim=1)
        gate = self.pattern_gate(gate_input)  # [B, C, H, W]

        # Modulate gate by target similarity
        # High target similarity -> amplify features
        # High clutter similarity -> suppress features
        similarity_weight = target_sim - 0.5 * clutter_sim
        similarity_weight = torch.sigmoid(similarity_weight)  # Normalize to [0, 1]
        similarity_weight = torch.clamp(similarity_weight, min=0.1, max=1.5)  # Prevent extreme values

        modulated_gate = gate * similarity_weight

        # Clamp gate to prevent extreme amplification
        modulated_gate = torch.clamp(modulated_gate, min=0, max=2)

        # === 5. Apply gating to features ===
        gated_feat = x * (1.0 + modulated_gate)  # Multiplicative gating (preserves original)

        # Clamp gated features
        gated_feat = torch.clamp(gated_feat, min=-10, max=10)

        # === 6. Residual fusion ===
        fusion_delta = self.fusion(gated_feat)

        # Clamp fusion delta
        fusion_delta = torch.clamp(fusion_delta, min=-5, max=5)

        out = x + self.alpha.clamp(0, 1) * fusion_delta

        # Final clamp on output
        out = torch.clamp(out, min=-10, max=10)

        return out, scatter_probs, target_sim

    def extra_repr(self):
        target_prior, clutter_prior = self.get_scattering_priors()
        return (f'channels={self.channels}, '
                f'target_prior=[{target_prior[0]:.2f}, {target_prior[1]:.2f}, {target_prior[2]:.2f}], '
                f'clutter_prior=[{clutter_prior[0]:.2f}, {clutter_prior[1]:.2f}, {clutter_prior[2]:.2f}]')


class SPILoss(nn.Module):
    """
    Optional supervision for scattering decomposition.

    Idea: Use GT boxes to provide weak supervision
    - Inside GT boxes: should have high target_similarity
    - Outside GT boxes (background): should have high clutter_similarity

    This helps the network learn meaningful scattering patterns faster.
    """

    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight

    def forward(self, target_similarity, gt_boxes, img_size):
        """
        Encourage high target similarity inside GT boxes.

        Args:
            target_similarity: [B, 1, H, W]
            gt_boxes: list of [num_gt, 4] boxes per image (cxcywh normalized)
            img_size: (H_feat, W_feat)

        Returns:
            loss: scalar
        """
        B = target_similarity.shape[0]
        H, W = img_size

        total_loss = 0.0
        num_valid = 0

        for b in range(B):
            if len(gt_boxes[b]) == 0:
                continue

            # Create target mask from GT boxes
            target_mask = torch.zeros(1, 1, H, W, device=target_similarity.device)

            boxes = gt_boxes[b]  # [num_gt, 4] cxcywh normalized
            for box in boxes:
                cx, cy, w, h = box

                # Convert to feature map coordinates
                x1 = max(0, int((cx - w/2) * W))
                x2 = min(W, int((cx + w/2) * W))
                y1 = max(0, int((cy - h/2) * H))
                y2 = min(H, int((cy + h/2) * H))

                target_mask[0, 0, y1:y2, x1:x2] = 1.0

            # Loss: high similarity inside boxes, low outside
            similarity = target_similarity[b]

            # Inside boxes: encourage high similarity
            inside_loss = F.binary_cross_entropy(
                similarity * target_mask,
                target_mask,
                reduction='sum'
            )

            # Outside boxes: encourage low similarity (optional, usually noisy)
            # outside_loss = F.binary_cross_entropy(
            #     similarity * (1 - target_mask),
            #     torch.zeros_like(target_mask),
            #     reduction='sum'
            # )

            total_loss += inside_loss
            num_valid += target_mask.sum().item()

        if num_valid > 0:
            loss = total_loss / num_valid * self.weight
        else:
            loss = target_similarity.sum() * 0.0  # Zero loss but keep gradient flow

        return loss
