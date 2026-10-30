"""OST-DA v2: level-specific density, mirrored axial samples and a residual gate."""

import math

import torch
from torch import nn
import torch.nn.functional as F

from .utils import deformable_attention_core_func_v2


class SymmetricOrthogonalTopologyDeformAttn(nn.Module):
    """Keep half the heads free; generate the other half from signed axial pairs.

    A topology head predicts K/2 scalar steps per level, rather than K independent
    2-D offsets. Every step produces an exact +/- pair. For K=4m+2, alternate
    which axis receives the extra pair between adjacent topology heads.
    The packed point dimension supports different even K at different levels.
    """

    def __init__(self, d_model=256, n_levels=3, n_heads=8, n_points=(4, 6, 4)):
        super().__init__()
        if n_levels <= 0:
            raise ValueError('OST-DA v2 requires positive n_levels.')
        points = list(n_points) if isinstance(n_points, (list, tuple)) else [n_points] * n_levels
        if len(points) != n_levels or any(not isinstance(k, int) or k < 4 or k % 2 for k in points):
            raise ValueError('OST-DA v2 requires one even point count >= 4 per level (e.g. [4, 6, 4]).')
        if n_heads <= 0 or n_heads % 2 or d_model <= 0 or d_model % n_heads:
            raise ValueError('OST-DA v2 requires positive even n_heads and d_model divisible by n_heads.')
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points_list = points
        self.total_points = sum(points)
        self.n_free_heads = n_heads // 2
        self.n_topo_heads = n_heads // 2
        self.n_pairs_list = [k // 2 for k in points]

        # Separate names prevent COCO's asymmetric 2-D offsets from replacing
        # the symmetric initialization during same-name/same-shape tuning.
        self.free_sampling_offsets = nn.Linear(d_model, self.n_free_heads * self.total_points * 2)
        self.topo_steps = nn.Linear(d_model, self.n_topo_heads * sum(self.n_pairs_list))
        self.angle_proj = nn.Linear(d_model, self.n_topo_heads)
        self.attention_weights = nn.Linear(d_model, n_heads * self.total_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.topo_gate_linear = nn.Linear(d_model, d_model)
        self.gate_alpha = nn.Parameter(torch.zeros(1, 1, d_model))
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    @staticmethod
    def axis_pair_counts(n_pairs, head):
        body = (n_pairs + 1) // 2 if head % 2 == 0 else n_pairs // 2
        return body, n_pairs - body

    def _reset_parameters(self):
        nn.init.zeros_(self.free_sampling_offsets.weight)
        angles = torch.arange(self.n_free_heads, dtype=torch.float32) * (2 * math.pi / self.n_heads)
        direction = torch.stack((angles.cos(), angles.sin()), dim=-1)
        direction = direction / direction.abs().amax(-1, keepdim=True)
        radii = torch.cat([torch.arange(1, k + 1, dtype=torch.float32) for k in self.n_points_list])
        free_grid = direction[:, None, :] * radii[None, :, None]
        nn.init.zeros_(self.topo_steps.weight)
        steps = []
        for head in range(self.n_topo_heads):
            levels = []
            for pairs in self.n_pairs_list:
                body, wing = self.axis_pair_counts(pairs, head)
                levels.append(torch.cat((torch.arange(1, body + 1), torch.arange(1, wing + 1))))
            steps.append(torch.cat(levels))
        with torch.no_grad():
            self.free_sampling_offsets.bias.copy_(free_grid.flatten())
            self.topo_steps.bias.copy_(torch.stack(steps).flatten())
        nn.init.zeros_(self.angle_proj.weight)
        nn.init.zeros_(self.angle_proj.bias)
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        # topo_gate_linear keeps its normal initialization: alpha receives a
        # gradient on the first step, then the gate weights can learn as well.
        nn.init.zeros_(self.gate_alpha)

    def sampling_geometry(self, query, reference_points):
        """Return packed [B,Q,H,sum(K),2] locations and [B,Q,H,sum(K)] weights."""
        if reference_points.shape[-1] != 4 or reference_points.shape[-2] not in (1, self.n_levels):
            raise ValueError('OST-DA v2 requires shared or per-level cxcywh reference boxes.')
        batch, queries = query.shape[:2]
        free = self.free_sampling_offsets(query).reshape(batch, queries, self.n_free_heads, self.total_points, 2)
        steps = self.topo_steps(query).reshape(batch, queries, self.n_topo_heads, sum(self.n_pairs_list), 1)
        angle_logits = self.angle_proj(query)
        weight_logits = self.attention_weights(query)
        with torch.autocast(device_type=query.device.type, enabled=False):
            theta = math.pi * angle_logits.float().sigmoid()
            body_axis = torch.stack((theta.cos(), theta.sin()), dim=-1)
            wing_axis = torch.stack((-theta.sin(), theta.cos()), dim=-1)
            free_levels = free.float().split(self.n_points_list, dim=-2)
            step_levels = steps.float().split(self.n_pairs_list, dim=-2)
            references = reference_points.float().expand(-1, -1, self.n_levels, -1)
            locations = []
            for level, count in enumerate(self.n_points_list):
                topology = []
                for head in range(self.n_topo_heads):
                    body_pairs, _ = self.axis_pair_counts(self.n_pairs_list[level], head)
                    radii = step_levels[level][:, :, head]
                    body = radii[..., :body_pairs, :] * body_axis[:, :, head, None, :]
                    wing = radii[..., body_pairs:, :] * wing_axis[:, :, head, None, :]
                    # Interleave +/- partners: body+, body-, ..., wing+, wing-.
                    body = torch.stack((body, -body), dim=-2).flatten(-3, -2)
                    wing = torch.stack((wing, -wing), dim=-2).flatten(-3, -2)
                    topology.append(torch.cat((body, wing), dim=-2))
                offsets = torch.cat((free_levels[level], torch.stack(topology, dim=2)), dim=2)
                reference = references[:, :, None, level, None, :]
                locations.append(reference[..., :2] + offsets / count * reference[..., 2:] * .5)
            locations = torch.cat(locations, dim=-2)
            weights = F.softmax(weight_logits.float().reshape(batch, queries, self.n_heads, self.total_points), dim=-1)
        return locations, weights

    def forward(self, query, reference_points, value, value_spatial_shapes):
        if isinstance(value, (list, tuple)):
            value = torch.cat(value, dim=-1).permute(0, 3, 1, 2).flatten(2)
        batch, values = value.shape[:2]
        projected = self.value_proj(value).reshape(batch, values, self.n_heads, self.d_model // self.n_heads)
        locations, weights = self.sampling_geometry(query, reference_points)
        output = self._ms_deform_attn_core(projected, value_spatial_shapes, locations, weights, self.n_points_list)
        gate = torch.tanh(self.topo_gate_linear(output))
        output = self.output_proj(output * (1.0 + self.gate_alpha * gate))
        return output.to(dtype=query.dtype)

    @staticmethod
    def _ms_deform_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights, num_points_list):
        with torch.autocast(device_type=value.device.type, enabled=False):
            return deformable_attention_core_func_v2(
                value.float(), value_spatial_shapes, sampling_locations.float(), attention_weights.float(),
                num_points_list, method='default', value_shape='reshape').contiguous()
