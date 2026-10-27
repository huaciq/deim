"""SLQC: local scattering-layout interaction for classification queries.

The module returns a query residual, not logits and not a refined box. The
caller adds ``linear(delta_query, classification_weight, bias=None)`` to the
existing final logits, preserving the detector's localization-quality term.

The design draws on structure-guided representation (PGD, arXiv:2411.12301)
and relative structural relationships (SAR2Struct, arXiv:2506.06757). Its local
tokens are learned feature samples, not supervised nose/wing/engine points.
No external implementation is copied and no physical scattering decomposition
is asserted. Current predicted boxes, never GT boxes, determine ROI sampling.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScatteringLayoutQueryClassifier(nn.Module):
    """Build an inference-time classification-query residual from local ROIs.

    Args:
        hidden_dim: Query and input feature channel count.
        token_dim: Low-dimensional ROI projection and attention width.
        roi_size: Samples per ROI side, at bin centers.
        num_levels: First feature levels used, normally P3 and P4.
        mode: ``layout`` uses normalized coordinates, second-order terms,
            relative radial distance/direction and level identity. ``roi``
            uses only content, providing a permutation-invariant ROI control.
        alpha_init: Residual scale. Zero preserves the initial classification
            path while keeping nonzero initialization in the other projections.

    Sampling uses align_corners=False and border extension: sampling points
    outside the image use the nearest border pixel, rather than a hard clamp
    of the reference-box geometry. The fraction of bin centers inside [0,1]^2
    multiplies the learned reliability, so fully outside boxes contribute no
    residual. Box coordinates are detached before this entire calculation.

    Attention, grid construction and convolution/projection calculations use
    FP32 even under AMP. The returned residual has the query dtype. Feature
    tensors are projected once per level; grid_sample handles all queries in
    one grid per image, with no B*Q replication of full feature maps.
    """

    def __init__(self, hidden_dim=256, token_dim=64, roi_size=4, num_levels=2,
                 mode='layout', alpha_init=0.0):
        super().__init__()
        if hidden_dim < 1 or token_dim < 1 or roi_size < 1 or num_levels < 1:
            raise ValueError('SLQC dimensions, ROI size and level count must be positive.')
        if mode not in ('layout', 'roi'):
            raise ValueError(f'Unknown SLQC mode {mode!r}.')
        self.hidden_dim = int(hidden_dim)
        self.token_dim = int(token_dim)
        self.roi_size = int(roi_size)
        self.num_levels = int(num_levels)
        self.mode = mode
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        self.feature_projs = nn.ModuleList([
            nn.Conv2d(hidden_dim, token_dim, 1) for _ in range(num_levels)
        ])
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, token_dim)
        self.token_norm = nn.LayerNorm(token_dim)
        self.key_proj = nn.Linear(token_dim, token_dim)
        self.value_proj = nn.Linear(token_dim, token_dim)
        self.context_norm = nn.LayerNorm(token_dim)
        self.up_proj = nn.Linear(token_dim, hidden_dim)
        self.reliability = nn.Sequential(
            nn.Linear(2 * token_dim + 2, 32), nn.SiLU(), nn.Linear(32, 1)
        )

        centers = (torch.arange(roi_size, dtype=torch.float32) + .5) / roi_size
        yy, xx = torch.meshgrid(centers, centers, indexing='ij')
        unit_grid = torch.stack((xx, yy), dim=-1)
        self.register_buffer('unit_grid', unit_grid, persistent=False)
        if mode == 'layout':
            xy = (unit_grid.reshape(-1, 2) - .5) * 2
            x, y = xy.unbind(-1)
            r2 = x.square() + y.square()
            # For an odd ROI the center has zero direction terms by convention.
            angular_denominator = r2.clamp_min(1e-6)
            basic = torch.stack((x, y, x.square(), y.square(), x * y,
                                 torch.sqrt(r2),
                                 (x.square() - y.square()) / angular_denominator,
                                 2 * x * y / angular_denominator), dim=-1)
            layout = []
            for level in range(num_levels):
                level_value = 0.0 if num_levels == 1 else 2 * level / (num_levels - 1) - 1
                layout.append(torch.cat((basic, basic.new_full((len(basic), 1), level_value)), dim=-1))
            self.register_buffer('layout_features', torch.cat(layout), persistent=False)
            self.layout_encoder = nn.Sequential(
                nn.Linear(9, token_dim), nn.SiLU(), nn.Linear(token_dim, token_dim)
            )
        else:
            # Do not register a trainable but bypassed position encoder: all
            # registered parameters remain on the loss graph in either mode.
            self.layout_encoder = None

    def _sampling_grid(self, boxes):
        """Return detached grid B x (Q*R) x R x 2 and in-frame fractions."""
        reference = boxes.detach().float()
        centers = reference[..., :2]
        extent = reference[..., 2:].clamp_min(1e-6)
        sample_xy = centers[:, :, None, None] + (
            self.unit_grid.float()[None, None] - .5
        ) * extent[:, :, None, None]
        inside = ((sample_xy >= 0) & (sample_xy <= 1)).all(dim=-1)
        valid_fraction = inside.float().mean(dim=(-1, -2), keepdim=False).unsqueeze(-1)
        b, q = boxes.shape[:2]
        grid = (sample_xy * 2 - 1).reshape(b, q * self.roi_size, self.roi_size, 2)
        return grid, valid_fraction

    def sample_roi_tokens(self, boxes, features):
        """Expose sampled projected tokens for geometry regression tests.

        Returns ``tokens[B,Q,L*R*R,D], in_frame_fraction[B,Q,1]``. The
        sampler uses only the first num_levels inputs and never reads labels.
        """
        if len(features) < self.num_levels:
            raise ValueError(f'SLQC requires {self.num_levels} feature levels.')
        b, q = boxes.shape[:2]
        grid, valid_fraction = self._sampling_grid(boxes)
        tokens = []
        for projector, feature in zip(self.feature_projs, features[:self.num_levels]):
            projected = projector(feature.float())
            sampled = F.grid_sample(projected, grid, mode='bilinear',
                                    padding_mode='border', align_corners=False)
            sampled = sampled.reshape(b, self.token_dim, q, self.roi_size, self.roi_size)
            tokens.append(sampled.permute(0, 2, 3, 4, 1).reshape(
                b, q, self.roi_size * self.roi_size, self.token_dim))
        return torch.cat(tokens, dim=2), valid_fraction

    def classify_tokens(self, queries, tokens, valid_fraction=None):
        """Shared content/layout interaction; roi mode is order invariant.

        This helper accepts already sampled tokens, so the role of layout can
        be tested independently from feature-map geometry and interpolation.
        """
        original_dtype = queries.dtype
        with torch.autocast(device_type=queries.device.type, enabled=False):
            query = self.query_proj(self.query_norm(queries.float()))
            normalized_tokens = self.token_norm(tokens.float())
            keys = self.key_proj(normalized_tokens)
            values = self.value_proj(normalized_tokens)
            if self.layout_encoder is not None:
                position = self.layout_encoder(self.layout_features.float())
                keys = keys + position[None, None]
            logits = (query.unsqueeze(-2) * keys).sum(-1) / math.sqrt(self.token_dim)
            attention = logits.softmax(dim=-1)
            context = (attention.unsqueeze(-1) * values).sum(-2)
            if valid_fraction is None:
                valid_fraction = query.new_ones((*query.shape[:2], 1))
            else:
                valid_fraction = valid_fraction.float()
            entropy = -(attention * attention.clamp_min(1e-8).log()).sum(-1, keepdim=True)
            entropy = entropy / max(math.log(tokens.shape[-2]), 1.0)
            confidence_input = torch.cat((query, context, 1 - entropy, valid_fraction), dim=-1)
            reliability = self.reliability(confidence_input).sigmoid() * valid_fraction
            delta = self.alpha.float() * reliability * self.up_proj(self.context_norm(context))
        return delta.to(original_dtype)

    def forward(self, queries, boxes, features):
        if queries.shape[:2] != boxes.shape[:2] or boxes.shape[-1] != 4:
            raise ValueError('SLQC queries and normalized cxcywh boxes must align.')
        if len(features) < self.num_levels:
            raise ValueError(f'SLQC requires {self.num_levels} feature levels.')
        if queries.numel() == 0:
            # DDP ranks retain a connected, finite zero-gradient graph even
            # when a caller presents an empty query dimension.
            zero = sum(parameter.float().sum() * 0 for parameter in self.parameters())
            zero = zero + sum(feature.float().sum() * 0 for feature in features[:self.num_levels])
            return (queries.float() * 0 + zero).to(queries.dtype)
        with torch.autocast(device_type=queries.device.type, enabled=False):
            tokens, valid_fraction = self.sample_roi_tokens(boxes, features)
            return self.classify_tokens(queries, tokens, valid_fraction)
