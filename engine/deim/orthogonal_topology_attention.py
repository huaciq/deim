"""Orthogonal scatter-topology deformable cross attention (OST-DA / TACA)."""

import math

import torch
from torch import nn
import torch.nn.functional as F

from .utils import deformable_attention_core_func


class OrthogonalTopologyDeformAttn(nn.Module):
    """Half free 2-D heads, half orthogonal axial heads, then channel gating.

    Accept the attachment's flat [B,S,D] value or D-FINE's per-level
    [B,H,D/H,H_l*W_l] values. The latter are reassembled before value_proj.
    Reference boxes may be shared across levels ([B,Q,1,4]) or per-level.
    Projection layers follow AMP; trigonometry, grids and bilinear sampling
    run in FP32. No angle labels or additional losses are used.
    """

    def __init__(self, d_model=256, n_levels=3, n_heads=8, n_points=4):
        super().__init__()
        if isinstance(n_points, (list, tuple)):
            if len(n_points) != n_levels or len(set(n_points)) != 1:
                raise ValueError('OST-DA requires the same even point count at every level (e.g. num_points: 4).')
            n_points = n_points[0]
        if n_heads <= 0 or n_heads % 2 or n_points <= 0 or n_points % 2:
            raise ValueError('OST-DA requires positive even n_heads and n_points.')
        if d_model % n_heads:
            raise ValueError('OST-DA requires d_model divisible by n_heads.')
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.n_free_heads = n_heads // 2
        self.n_topo_heads = n_heads // 2

        self.angle_proj = nn.Linear(d_model, self.n_topo_heads)
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.topo_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.Sigmoid())
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.zeros_(self.sampling_offsets.weight)
        angles = torch.arange(self.n_heads, dtype=torch.float32) * (2 * math.pi / self.n_heads)
        grid = torch.stack((angles.cos(), angles.sin()), dim=-1)
        grid = grid / grid.abs().amax(-1, keepdim=True)
        grid = grid.view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        grid *= torch.arange(1, self.n_points + 1).view(1, 1, -1, 1)
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid.flatten())
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.zeros_(self.angle_proj.weight)
        nn.init.zeros_(self.angle_proj.bias)

    def sampling_geometry(self, query, reference_points):
        """Build differentiable head/level/point coordinates and attention weights."""
        if reference_points.shape[-1] != 4:
            raise ValueError('OST-DA requires cxcywh reference boxes.')
        batch, queries = query.shape[:2]
        raw = self.sampling_offsets(query).reshape(batch, queries, self.n_heads, self.n_levels, self.n_points, 2)
        angle_logits = self.angle_proj(query)
        weight_logits = self.attention_weights(query)
        with torch.autocast(device_type=query.device.type, enabled=False):
            raw = raw.float()
            theta = math.pi * angle_logits.float().sigmoid()
            cosine, sine = theta.cos()[..., None, None], theta.sin()[..., None, None]
            fuselage = torch.stack((cosine, sine), dim=-1)
            wing = torch.stack((-sine, cosine), dim=-1)
            topology = raw[:, :, self.n_free_heads:]
            radius = topology[..., :1]
            perturbation = .25 * topology[..., 1:].tanh()
            half = self.n_points // 2
            body_offsets = radius[..., :half, :] * fuselage + perturbation[..., :half, :] * wing
            wing_offsets = radius[..., half:, :] * wing + perturbation[..., half:, :] * fuselage
            offsets = torch.cat((raw[:, :, :self.n_free_heads],
                                 torch.cat((body_offsets, wing_offsets), dim=-2)), dim=2)
            reference = reference_points.float()
            locations = (reference[:, :, None, :, None, :2]
                         + offsets / self.n_points * reference[:, :, None, :, None, 2:] * .5)
            weights = weight_logits.float().reshape(batch, queries, self.n_heads, self.n_levels * self.n_points)
            weights = F.softmax(weights, dim=-1).reshape(batch, queries, self.n_heads, self.n_levels, self.n_points)
        return locations, weights

    def forward(self, query, reference_points, value, value_spatial_shapes):
        if isinstance(value, (list, tuple)):
            value = torch.cat(value, dim=-1).permute(0, 3, 1, 2).flatten(2)
        batch, values = value.shape[:2]
        projected = self.value_proj(value).reshape(batch, values, self.n_heads, self.d_model // self.n_heads)
        locations, weights = self.sampling_geometry(query, reference_points)
        output = self._ms_deform_attn_core(projected, value_spatial_shapes, locations, weights)
        gate = self.topo_gate(output)
        return self.output_proj(output * gate).to(dtype=query.dtype)

    @staticmethod
    def _ms_deform_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights):
        with torch.autocast(device_type=value.device.type, enabled=False):
            return deformable_attention_core_func(
                value.float(), value_spatial_shapes, sampling_locations.float(), attention_weights.float()).contiguous()
