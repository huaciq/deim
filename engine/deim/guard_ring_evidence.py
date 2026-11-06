"""Guard-ring evidence attention (GREA) for SAR detection queries.

Constant false alarm detection compares a test cell with a reference ring, with a
guard band between them so target energy does not leak into the clutter estimate.
GREA gives each decoder query that structure:

* target evidence T comes from points constrained inside the box;
* clutter evidence R comes from points in a ring outside a guard band;
* contrast C is learned from T - R, T and R;
* a query- and class-conditioned gate mixes the three, so a class that needs its own
  appearance and one that needs context can weigh them differently.

Evidence is sampled from low-level backbone features, before AIFI and the FPN add
wide context. The residual scale starts at zero, so the module starts at the baseline.
See docs/sardet100k_grea_design.md.
"""

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['LocalEvidencePyramid', 'GuardRingEvidence']


class LocalEvidencePyramid(nn.Module):
    """Project low-level backbone stages to a common evidence dimension."""

    def __init__(self, channels, dim):
        super().__init__()
        self.projections = nn.ModuleList(
            nn.Sequential(OrderedDict([('conv', nn.Conv2d(c, dim, 1, bias=False)), ('norm', nn.BatchNorm2d(dim))]))
            for c in channels)

    def forward(self, features):
        return [projection(feature) for projection, feature in zip(self.projections, features)]


class GuardRingEvidence(nn.Module):
    def __init__(self, hidden_dim, num_classes, levels=2, dim=128, points=8, ring=(1.25, 2.), act=nn.SiLU):
        super().__init__()
        if not 1. < ring[0] < ring[1]:
            raise ValueError('The reference ring must lie outside the box: 1 < inner < outer.')
        self.ring = (float(ring[0]), float(ring[1]))
        self.points = points
        self.core_offsets = nn.Linear(hidden_dim, 2 * points)
        self.core_weights = nn.Linear(hidden_dim, points)
        self.ring_radii = nn.Linear(hidden_dim, points)
        self.ring_weights = nn.Linear(hidden_dim, points)
        for layer in (self.core_offsets, self.core_weights, self.ring_radii, self.ring_weights):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        angles = torch.arange(points, dtype=torch.float32) * (2 * math.pi / points)
        directions = torch.stack((angles.cos(), angles.sin()), -1)
        # Start the core points on a circle at half the distance to the box edge.
        self.core_offsets.bias.data.copy_(torch.atanh(directions * .5).flatten())
        # L-infinity unit directions: radius 1 lies exactly on the box edge.
        self.register_buffer('directions', directions / directions.abs().amax(-1, keepdim=True), persistent=False)
        self.target_projection = nn.Linear(levels * dim, dim)
        self.clutter_projection = nn.Linear(levels * dim, dim)
        self.contrast = nn.Sequential(nn.Linear(3 * dim, dim), act(), nn.Linear(dim, dim))
        self.gate = nn.Sequential(nn.Linear(hidden_dim + num_classes, 64), act(), nn.Linear(64, 3))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.output = nn.Linear(dim, hidden_dim)
        self.alpha = nn.Parameter(torch.zeros(()))

    def sample_points(self, query, boxes, guard):
        """Return core and ring points in normalized image coordinates, [B, Q, K, 2]."""
        batch, count = query.shape[:2]
        center, half = boxes[..., None, :2], boxes[..., None, 2:] / 2
        offsets = torch.tanh(self.core_offsets(query)).reshape(batch, count, self.points, 2)
        core = center + offsets * half
        inner, outer = self.ring
        radii = inner + (outer - inner) * torch.sigmoid(self.ring_radii(query))
        ring = center + self.directions * (half * radii[..., None] + guard)
        return core, ring

    @staticmethod
    def _pool(levels, points, logits):
        """Attention-pool every level at the given points, [B, Q, L * dim]."""
        outside = (points < 0).any(-1) | (points > 1).any(-1)
        # A query whose points are all outside keeps the border samples.
        usable = outside & ~outside.all(-1, keepdim=True)
        weights = logits.masked_fill(usable, -1e4).softmax(-1)
        grid = 2 * points - 1
        pooled = [(F.grid_sample(level, grid, mode='bilinear', padding_mode='border', align_corners=False) * weights.unsqueeze(1)).sum(-1).transpose(1, 2)
                  for level in levels]
        return torch.cat(pooled, -1)

    def forward(self, query, boxes, levels, class_scores):
        with torch.autocast(device_type=query.device.type, enabled=False):
            source = query.float()
            maps = [level.float() for level in levels]
            reference = boxes.detach().float()
            # At least one cell of the finest map separates the ring from the box.
            guard = source.new_tensor([1. / maps[0].shape[-1], 1. / maps[0].shape[-2]])
            core, ring = self.sample_points(source, reference, guard)
            target = self.target_projection(self._pool(maps, core, self.core_weights(source)))
            clutter = self.clutter_projection(self._pool(maps, ring, self.ring_weights(source)))
            contrast = self.contrast(torch.cat((target - clutter, target, clutter), -1))
            gate = self.gate(torch.cat((source, class_scores.detach().float().sigmoid()), -1)).softmax(-1)
            fused = gate[..., 0:1] * target + gate[..., 1:2] * clutter + gate[..., 2:3] * contrast
            delta = self.alpha.float() * self.output(fused)
        return delta.to(dtype=query.dtype), target, gate
