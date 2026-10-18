"""
Scatter-Aware Clutter Suppression (SCAS) for SAR Object Detection
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.

Key insight from error analysis:
- 63.4% of false positives come from background clutter (1057 FPs)
- Mean FP score is 0.447 (high), indicating the network cannot distinguish clutter

Solution:
- Analyze SAR scatter intensity to identify clutter patterns
- Learn clutter prototypes (sea surface, urban structures, speckle noise)
- Suppress queries likely to be background clutter based on scatter characteristics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_activation


class ScatterAwareClutterSuppression(nn.Module):
    """
    Scatter-Aware Clutter Suppression (SCAS)

    Analyzes SAR scatter patterns to suppress background clutter false positives.
    Unlike pure classification confidence, this leverages SAR imaging physics.

    Key components:
    1. Scatter intensity encoder: extract intensity patterns from encoder features
    2. Clutter prototype learning: learn typical patterns of sea/urban/speckle clutter
    3. Spatial context: use query position to assess clutter likelihood
    4. Suppression gating: dynamically weight queries based on clutter likelihood
    """

    def __init__(self, hidden_dim=256, num_clutter_types=3, act='silu', dropout=0.1):
        """
        Args:
            hidden_dim: feature dimension
            num_clutter_types: number of clutter prototypes (sea, urban, speckle)
            act: activation function
            dropout: dropout rate for suppression head
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_clutter_types = num_clutter_types

        # ===== 1. Scatter intensity encoder =====
        # Extract single-channel intensity map from encoder features
        # High intensity -> strong scatterer (could be target or clutter)
        self.scatter_encoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 4),
            get_activation(act),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 8),
            get_activation(act),
            nn.Conv2d(hidden_dim // 8, 1, kernel_size=1, bias=True),
            nn.Sigmoid()  # Normalize to [0, 1]
        )

        # ===== 2. Clutter prototype bank =====
        # Learnable prototypes representing typical clutter patterns
        # Type 0: Sea surface clutter (uniform high backscatter)
        # Type 1: Urban clutter (structured strong scatterers)
        # Type 2: Speckle noise (random scattered points)
        self.clutter_prototypes = nn.Parameter(torch.randn(num_clutter_types, hidden_dim))
        nn.init.orthogonal_(self.clutter_prototypes)  # Initialize as orthogonal vectors

        # Prototype attention: learn to weight different clutter types
        self.prototype_attention = nn.Sequential(
            nn.Linear(hidden_dim, num_clutter_types),
            nn.Softmax(dim=-1)
        )

        # ===== 3. Spatial context encoder =====
        # Query position affects clutter likelihood (e.g., image edges often have more clutter)
        self.spatial_encoder = nn.Sequential(
            nn.Linear(2, 32),  # 2D position
            nn.ReLU(inplace=True),
            nn.Linear(32, 16)
        )

        # ===== 4. Intensity context encoder =====
        # Local intensity statistics around query position
        self.intensity_context = nn.Sequential(
            nn.Linear(4, 16),  # mean, std, min, max of local intensity
            nn.ReLU(inplace=True),
            nn.Linear(16, 16)
        )

        # ===== 5. Suppression decision head =====
        # Combine all features to predict clutter likelihood
        fusion_dim = hidden_dim + num_clutter_types + 16 + 16  # query + clutter_sim + spatial + intensity

        self.suppression_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            get_activation(act),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            get_activation(act),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output: 0=suppress (clutter), 1=keep (target)
        )

        # Temperature for soft gating (learnable)
        self.temperature = nn.Parameter(torch.ones(1))

    def extract_local_intensity_stats(self, scatter_map, reference_points, kernel_size=5):
        """
        Extract local intensity statistics around each query position.

        Args:
            scatter_map: [B, 1, H, W] intensity map
            reference_points: [B, num_queries, 2] normalized xy in [0, 1]
            kernel_size: local region size

        Returns:
            stats: [B, num_queries, 4] (mean, std, min, max)
        """
        B, _, H, W = scatter_map.shape
        num_queries = reference_points.shape[1]

        # Normalize to grid coordinates [-1, 1]
        grid = reference_points * 2 - 1  # [B, num_queries, 2]

        # Extract local patches
        # Use unfold-like operation via grid_sample with offsets
        half_k = kernel_size // 2
        offsets = torch.stack([
            torch.arange(-half_k, half_k + 1, device=scatter_map.device).float() / W,
            torch.arange(-half_k, half_k + 1, device=scatter_map.device).float() / H
        ])  # [2, kernel_size]

        # Create sampling grid for each query
        grid_expanded = grid.unsqueeze(2).unsqueeze(2)  # [B, num_queries, 1, 1, 2]

        # Sample at center (simple version - full patch sampling would be more accurate but slower)
        sampled = F.grid_sample(scatter_map, grid.unsqueeze(2), align_corners=False)
        sampled = sampled.squeeze(-1).squeeze(1)  # [B, num_queries]

        # Approximate local statistics (for efficiency, we use a simplified version)
        # In practice, you might want to sample a full patch and compute exact stats
        # For now, we estimate based on center value and neighbors
        stats_mean = sampled
        stats_std = torch.ones_like(sampled) * 0.1  # Placeholder - could sample neighbors
        stats_min = torch.clamp(sampled - 0.1, 0, 1)
        stats_max = torch.clamp(sampled + 0.1, 0, 1)

        stats = torch.stack([stats_mean, stats_std, stats_min, stats_max], dim=-1)
        return stats

    def compute_clutter_similarity(self, query_embeddings):
        """
        Compute similarity between query embeddings and clutter prototypes.

        Args:
            query_embeddings: [B, num_queries, C]

        Returns:
            clutter_scores: [B, num_queries, num_clutter_types]
        """
        B, num_queries, C = query_embeddings.shape

        # Normalize for cosine similarity
        query_norm = F.normalize(query_embeddings, dim=-1)  # [B, num_queries, C]
        proto_norm = F.normalize(self.clutter_prototypes, dim=-1)  # [num_clutter_types, C]

        # Compute similarity: higher means more similar to clutter
        similarity = torch.matmul(query_norm, proto_norm.T)  # [B, num_queries, num_clutter_types]

        # Apply temperature scaling
        similarity = similarity / self.temperature.clamp(min=0.1)
        similarity = torch.sigmoid(similarity)  # [0, 1]

        return similarity

    def forward(self, encoder_features, query_embeddings, reference_points):
        """
        Compute suppression weights for each query based on clutter likelihood.

        Args:
            encoder_features: [B, C, H, W] encoder output (stride 8)
            query_embeddings: [B, num_queries, C] query content embeddings
            reference_points: [B, num_queries, 2] normalized xy positions in [0, 1]

        Returns:
            suppression_weights: [B, num_queries] in [0, 1]
                - 1.0 = keep (high confidence it's a target)
                - 0.0 = suppress (high confidence it's clutter)
        """
        B, C, H, W = encoder_features.shape
        num_queries = query_embeddings.shape[1]

        # ===== 1. Extract scatter intensity map =====
        scatter_map = self.scatter_encoder(encoder_features)  # [B, 1, H, W]

        # ===== 2. Sample intensity at query positions =====
        grid = reference_points * 2 - 1  # Normalize to [-1, 1] for grid_sample
        grid_unsqueezed = grid.unsqueeze(2)  # [B, num_queries, 1, 2]

        sampled_intensity = F.grid_sample(
            scatter_map, grid_unsqueezed,
            mode='bilinear', align_corners=False
        ).squeeze(-1).squeeze(1)  # [B, num_queries]

        # ===== 3. Extract local intensity statistics =====
        intensity_stats = self.extract_local_intensity_stats(
            scatter_map, reference_points, kernel_size=5
        )  # [B, num_queries, 4]

        intensity_context = self.intensity_context(intensity_stats)  # [B, num_queries, 16]

        # ===== 4. Compute clutter similarity =====
        clutter_sim = self.compute_clutter_similarity(query_embeddings)  # [B, num_queries, num_clutter_types]

        # ===== 5. Encode spatial context =====
        spatial_context = self.spatial_encoder(reference_points)  # [B, num_queries, 16]

        # ===== 6. Fuse all features =====
        # High scatter + high clutter similarity + edge position -> likely clutter
        combined = torch.cat([
            query_embeddings,      # [B, num_queries, C]
            clutter_sim,           # [B, num_queries, num_clutter_types]
            spatial_context,       # [B, num_queries, 16]
            intensity_context      # [B, num_queries, 16]
        ], dim=-1)  # [B, num_queries, C + num_clutter_types + 32]

        # ===== 7. Predict suppression weights =====
        suppression_weights = self.suppression_head(combined).squeeze(-1)  # [B, num_queries]

        return suppression_weights, scatter_map

    def apply_suppression(self, pred_logits, suppression_weights):
        """
        Apply suppression weights to prediction logits.

        Args:
            pred_logits: [B, num_queries, num_classes]
            suppression_weights: [B, num_queries]

        Returns:
            suppressed_logits: [B, num_queries, num_classes]
        """
        # Reshape weights for broadcasting
        weights = suppression_weights.unsqueeze(-1)  # [B, num_queries, 1]

        # Apply multiplicative suppression
        # Queries with low weight will have reduced classification scores
        suppressed_logits = pred_logits * weights

        return suppressed_logits
