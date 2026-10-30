"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
"""

import math
import copy
import functools
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import List

from .dfine_utils import weighting_function, distance2bbox
from .denoising import get_contrastive_denoising_training_group
from .utils import deformable_attention_core_func_v2, get_activation, inverse_sigmoid
from .utils import bias_init_with_prob
from .scatter_clutter_suppression import ScatterAwareClutterSuppression
from .background_rejection import SimpleBackgroundRejection
from .boundary_refinement import BoundaryRefinementLoss
from .scale_aware_refinement import ScaleAwareRefinement
from .scattering_signature import ScatteringSignatureHead
from .scattering_layout import ScatteringLayoutQueryClassifier
from .query_prototypes import QueryPrototypeHead
from .orthogonal_topology_attention import OrthogonalTopologyDeformAttn
from ..core import register

__all__ = ['DFINETransformer']


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MSDeformableAttention(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        method='default',
        offset_scale=0.5,
    ):
        """Multi-Scale Deformable Attention
        """
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale

        if isinstance(num_points, list):
            assert len(num_points) == num_levels, ''
            num_points_list = num_points
        else:
            num_points_list = [num_points for _ in range(num_levels)]

        self.num_points_list = num_points_list

        num_points_scale = [1/n for n in num_points_list for _ in range(n)]
        self.register_buffer('num_points_scale', torch.tensor(num_points_scale, dtype=torch.float32))

        self.total_points = num_heads * sum(num_points_list)
        self.method = method

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)

        self.ms_deformable_attn_core = functools.partial(deformable_attention_core_func_v2, method=self.method)

        self._reset_parameters()

        if method == 'discrete':
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])
        scaling = torch.concat([torch.arange(1, n + 1) for n in self.num_points_list]).reshape(1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)


    def forward(self,
                query: torch.Tensor,
                reference_points: torch.Tensor,
                value: torch.Tensor,
                value_spatial_shapes: List[int]):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]

        sampling_offsets: torch.Tensor = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.reshape(bs, Len_q, self.num_heads, sum(self.num_points_list), 2)

        attention_weights = self.attention_weights(query).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list))
        attention_weights = F.softmax(attention_weights, dim=-1)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            # reference_points [8, 480, None, 1,  4]
            # sampling_offsets [8, 480, 8,    12, 2]
            num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
            offset = sampling_offsets * num_points_scale * reference_points[:, :, None, :, 2:] * self.offset_scale
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights, self.num_points_list)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation='relu',
                 n_levels=4,
                 n_points=4,
                 cross_attn_method='default',
                 layer_scale=None,
                 use_ost_da=False):
        super(TransformerDecoderLayer, self).__init__()
        if use_ost_da and cross_attn_method != 'default':
            raise ValueError('OST-DA requires differentiable bilinear cross attention (cross_attn_method: default).')
        if layer_scale is not None:
            dim_feedforward = round(layer_scale * dim_feedforward)
            d_model = round(layer_scale * d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points, \
                                                method=cross_attn_method)
        self.dropout2 = nn.Dropout(dropout)

        # gate
        self.gateway = Gate(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self._reset_parameters()

        if use_ost_da:
            # Preserve original layer initialization and subsequent RNG state.
            with torch.random.fork_rng(devices=[]):
                self.cross_attn = OrthogonalTopologyDeformAttn(d_model, n_levels, n_head, n_points)

    def _reset_parameters(self):
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                target,
                reference_points,
                value,
                spatial_shapes,
                attn_mask=None,
                query_pos_embed=None):

        # self attention
        q = k = self.with_pos_embed(target, query_pos_embed)

        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = target + self.dropout1(target2)
        target = self.norm1(target)

        # cross attention
        target2 = self.cross_attn(\
            self.with_pos_embed(target, query_pos_embed),
            reference_points,
            value,
            spatial_shapes)

        target = self.gateway(target, self.dropout2(target2))

        # ffn
        target2 = self.forward_ffn(target)
        target = target + self.dropout4(target2)
        target = self.norm3(target.clamp(min=-65504, max=65504))

        return target


class Gate(nn.Module):
    def __init__(self, d_model):
        super(Gate, self).__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        bias = bias_init_with_prob(0.5)
        init.constant_(self.gate.bias, bias)
        init.constant_(self.gate.weight, 0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x1, x2):
        gate_input = torch.cat([x1, x2], dim=-1)
        gates = torch.sigmoid(self.gate(gate_input))
        gate1, gate2 = gates.chunk(2, dim=-1)
        return self.norm(gate1 * x1 + gate2 * x2)


class Integral(nn.Module):
    """
    A static layer that calculates integral results from a distribution.

    This layer computes the target location using the formula: `sum{Pr(n) * W(n)}`,
    where Pr(n) is the softmax probability vector representing the discrete
    distribution, and W(n) is the non-uniform Weighting Function.

    Args:
        reg_max (int): Max number of the discrete bins. Default is 32.
                       It can be adjusted based on the dataset or task requirements.
    """

    def __init__(self, reg_max=32):
        super(Integral, self).__init__()
        self.reg_max = reg_max

    def forward(self, x, project):
        shape = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, project.to(x.device)).reshape(-1, 4)
        return x.reshape(list(shape[:-1]) + [-1])


class BoxLocalEvidenceSampler(nn.Module):
    """Shared CLCR/SDCP box-relative core and perimeter sampling geometry."""

    def __init__(self):
        super().__init__()
        core = torch.linspace(-.375, .375, 3)
        yy, xx = torch.meshgrid(core, core, indexing='ij')
        inside = torch.stack((xx, yy), dim=-1).reshape(9, 2)
        along = torch.tensor([-1. / 3., 0., 1. / 3.]) * 1.05
        outer = torch.full_like(along, .525)
        ring = torch.cat((
            torch.stack((along, -outer), dim=-1),
            torch.stack((along, outer), dim=-1),
            torch.stack((-outer, along), dim=-1),
            torch.stack((outer, along), dim=-1),
        ))
        self.register_buffer('sample_offsets', torch.cat((inside, ring)), persistent=False)

    def pool_local_evidence(self, boxes, feature):
        """Return pooled core/perimeter features [B,Q,C], preserving box gradients."""
        with torch.autocast(device_type=feature.device.type, enabled=False):
            reference = boxes.float()
            points = reference[..., None, :2] + reference[..., None, 2:] * self.sample_offsets.float()
            sampled = F.grid_sample(feature.float(), 2 * points - 1, mode='bilinear',
                                    padding_mode='border', align_corners=False)
            inside = sampled[..., :9].mean(-1).transpose(1, 2)
            ring = sampled[..., 9:].mean(-1).transpose(1, 2)
        return inside, ring


class SDCPCosineClassifier(nn.Module):
    """Query + pooled box core + log aspect embedding, followed by cosine logits.

    Prototype rows retain the original classifier's weight key and shape for
    COCO initialization. There is no class bias or extra residual gate. Tau is
    a fixed configurable positive scale, not a learned amplitude shortcut.
    """

    def __init__(self, classifier_weight, feature_dim, aspect_dim=16, temperature=10., act='relu'):
        super().__init__()
        if temperature <= 0:
            raise ValueError('SDCP temperature must be positive.')
        self.weight = nn.Parameter(classifier_weight.detach().clone())
        self.aspect_embedding = MLP(1, aspect_dim, aspect_dim, 2, act=act)
        self.local_projection = nn.Linear(feature_dim + aspect_dim, classifier_weight.shape[1])
        self.temperature = float(temperature)

    @staticmethod
    def log_aspect_ratio(boxes, image_size):
        # cxcywh is normalized separately by W/H; restore their pixel ratio.
        dimensions = boxes.float()[..., 2:] * image_size.float().flip(0)
        return (dimensions[..., :1].clamp_min(1e-6).log()
                - dimensions[..., 1:].clamp_min(1e-6).log())

    def forward(self, queries, inside, boxes, image_size):
        with torch.autocast(device_type=queries.device.type, enabled=False):
            aspect = self.aspect_embedding(self.log_aspect_ratio(boxes, image_size))
            fused = queries.float() + self.local_projection(torch.cat((inside.float(), aspect), dim=-1))
            scores = self.temperature * F.linear(F.normalize(fused, dim=-1, eps=1e-6),
                                                 F.normalize(self.weight.float(), dim=-1, eps=1e-6))
        return scores.to(dtype=queries.dtype)


class CLCR(BoxLocalEvidenceSampler):
    """Class-specific reliability residual from competition, local and edge evidence.

    Sampling uses the current normalized cxcywh box, nine points in its 0.75
    core and twelve distinct points on its 1.05 perimeter (three per side).
    All queries share one grid_sample call per layer; the feature map is never
    replicated per query. Logits, boxes, feature maps and edge statistics stay
    differentiable. Only beta is zero-initialized, so it can learn immediately.
    """

    def __init__(self, feature_dim, edge_dim, hidden_dim=64, act='relu'):
        super().__init__()
        self.local_mlp = MLP(2 * feature_dim, hidden_dim, 16, 2, act=act)
        # Shared across classes, with a distinct four-vector for every class.
        self.calibration = MLP(4 + 16 + edge_dim, hidden_dim, 1, 2, act=act)
        self.beta = nn.Parameter(torch.zeros(()))

    @staticmethod
    def competition_statistics(scores, eps=1e-6):
        """Return [p_c, p_c-max(other), p_c/(sum(p)+eps), normalized entropy]."""
        with torch.autocast(device_type=scores.device.type, enabled=False):
            logits = scores.float()
            classes = logits.shape[-1]
            probabilities = logits.sigmoid()
            values, indices = probabilities.topk(2, dim=-1)
            class_ids = torch.arange(classes, device=scores.device)
            competitors = torch.where(class_ids == indices[..., :1],
                                      values[..., 1:2], values[..., :1])
            margin = probabilities - competitors
            ratio = probabilities / (probabilities.sum(-1, keepdim=True) + eps)
            # log(sigmoid) avoids underflow when all logits are very negative.
            log_normalized = F.log_softmax(F.logsigmoid(logits), dim=-1)
            entropy = -(log_normalized.exp() * log_normalized).sum(-1, keepdim=True)
            entropy = entropy / math.log(classes)
            return torch.stack((probabilities, margin, ratio, entropy.expand_as(probabilities)), dim=-1)

    def forward(self, scores, edge_statistics, boxes, feature, local_evidence=None):
        competition = self.competition_statistics(scores)
        inside, ring = self.pool_local_evidence(boxes, feature) if local_evidence is None else local_evidence
        local = self.local_mlp(torch.cat((inside, inside - ring), dim=-1))
        classes = scores.shape[-1]
        evidence = torch.cat((
            competition,
            local.unsqueeze(2).expand(-1, -1, classes, -1),
            edge_statistics.unsqueeze(2).expand(-1, -1, classes, -1),
        ), dim=-1)
        correction = self.calibration(evidence).squeeze(-1)
        return self.beta.to(dtype=scores.dtype) * correction.to(dtype=scores.dtype)


class CLCRLite(nn.Module):
    """Eight distribution/geometry statistics; no image feature or spatial sampling.

    Inputs per class: p, competition margin, normalized class entropy, mean/std
    of the four LQE top-k means, log(w), log(h), log(w/h). The original LQE
    keeps its full statistics. One 162-parameter head (including beta) is shared
    across decoder layers and classes. Zero only the output layer: zeroing both
    layers or combining a zero output with beta=0 would prevent useful learning.
    """

    def __init__(self):
        super().__init__()
        self.calibration = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 1))
        init.zeros_(self.calibration[-1].weight)
        init.zeros_(self.calibration[-1].bias)
        self.beta = nn.Parameter(torch.ones(()))

    @staticmethod
    def input_statistics(scores, edge_statistics, boxes):
        with torch.autocast(device_type=scores.device.type, enabled=False):
            competition = CLCR.competition_statistics(scores)[..., [0, 1, 3]]
            # LQE packs [top-k probabilities, their mean] separately for each edge.
            concentration = edge_statistics.float().reshape(*scores.shape[:2], 4, -1)[..., -1]
            edge = torch.stack((concentration.mean(-1), concentration.std(-1, unbiased=False)), dim=-1)
            log_wh = boxes.float()[..., 2:].clamp_min(1e-6).log()
            scale = torch.cat((log_wh, log_wh[..., :1] - log_wh[..., 1:]), dim=-1)
            geometry = torch.cat((edge, scale), dim=-1)
            return torch.cat((competition, geometry.unsqueeze(2).expand(-1, -1, scores.shape[-1], -1)), dim=-1)

    def forward(self, scores, edge_statistics, boxes):
        evidence = self.input_statistics(scores, edge_statistics, boxes)
        correction = self.calibration(evidence).squeeze(-1)
        return self.beta.to(dtype=scores.dtype) * correction.to(dtype=scores.dtype)


class LQE(nn.Module):
    def __init__(self, k, hidden_dim, num_layers, reg_max, act='relu'):
        super(LQE, self).__init__()
        self.k = k
        self.reg_max = reg_max
        self.reg_conf = MLP(4 * (k + 1), hidden_dim, 1, num_layers, act=act)
        init.constant_(self.reg_conf.layers[-1].bias, 0)
        init.constant_(self.reg_conf.layers[-1].weight, 0)
        self.clcr = None

    def forward(self, scores, pred_corners, boxes=None, local_feature=None, local_evidence=None, clcr_lite=None):
        B, L, _ = pred_corners.size()
        prob = F.softmax(pred_corners.reshape(B, L, 4, self.reg_max+1), dim=-1)
        prob_topk, _ = prob.topk(self.k, dim=-1)
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        edge_statistics = stat.reshape(B, L, -1)
        quality_score = self.reg_conf(edge_statistics)
        calibrated = scores + quality_score
        if self.clcr is not None:
            # Competition is computed from raw class logits, before the scalar LQE.
            if local_evidence is None:
                calibrated = calibrated + self.clcr(scores, edge_statistics, boxes, local_feature)
            else:
                calibrated = calibrated + self.clcr(scores, edge_statistics, boxes, local_feature,
                                                    local_evidence=local_evidence)
        if clcr_lite is not None:
            calibrated = calibrated + clcr_lite(scores, edge_statistics, boxes)
        return calibrated


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder implementing Fine-grained Distribution Refinement (FDR).

    This decoder refines object detection predictions through iterative updates across multiple layers,
    utilizing attention mechanisms, location quality estimators, and distribution refinement techniques
    to improve bounding box accuracy and robustness.
    """

    def __init__(self, hidden_dim, decoder_layer, decoder_layer_wide, num_layers, num_head, reg_max, reg_scale, up,
                 eval_idx=-1, layer_scale=2, act='relu'):
        super(TransformerDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_scale = layer_scale
        self.num_head = num_head
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.up, self.reg_scale, self.reg_max = up, reg_scale, reg_max
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(self.eval_idx + 1)] \
                    + [copy.deepcopy(decoder_layer_wide) for _ in range(num_layers - self.eval_idx - 1)])
        self.lqe_layers = nn.ModuleList([copy.deepcopy(LQE(4, 64, 2, reg_max, act=act)) for _ in range(num_layers)])
        self.use_clcr = False
        self.use_clcr_lite = False
        self.clcr_lite = None
        self.use_sdcp = False
        self.sdcp_sampler = None

    def value_op(self, memory, value_proj, value_scale, memory_mask, memory_spatial_shapes):
        """
        Preprocess values for MSDeformableAttention.
        """
        value = value_proj(memory) if value_proj is not None else memory
        value = F.interpolate(memory, size=value_scale) if value_scale is not None else value
        if memory_mask is not None:
            value = value * memory_mask.to(value.dtype).unsqueeze(-1)
        value = value.reshape(value.shape[0], value.shape[1], self.num_head, -1)
        split_shape = [h * w for h, w in memory_spatial_shapes]
        return value.permute(0, 2, 3, 1).split(split_shape, dim=-1)

    def convert_to_deploy(self):
        self.project = weighting_function(self.reg_max, self.up, self.reg_scale, deploy=True)
        self.layers = self.layers[:self.eval_idx + 1]
        self.lqe_layers = nn.ModuleList([nn.Identity()] * (self.eval_idx) + [self.lqe_layers[self.eval_idx]])

    def forward(self,
                target,
                ref_points_unact,
                memory,
                spatial_shapes,
                bbox_head,
                score_head,
                query_pos_head,
                pre_bbox_head,
                integral,
                up,
                reg_scale,
                attn_mask=None,
                memory_mask=None,
                dn_meta=None,
                return_query_features=False,
                image_size=None):
        output = target
        output_detach = pred_corners_undetach = 0
        value = self.value_op(memory, None, None, memory_mask, spatial_shapes)
        local_feature = None
        if self.use_clcr or self.use_sdcp:
            height, width = spatial_shapes[0]
            local_feature = memory[:, :height * width].transpose(1, 2).reshape(
                memory.shape[0], self.hidden_dim, height, width)

        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_pred_corners = []
        dec_out_refs = []
        if not hasattr(self, 'project'):
            project = weighting_function(self.reg_max, up, reg_scale)
        else:
            project = self.project

        ref_points_detach = F.sigmoid(ref_points_unact)

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)

            # TODO Adjust scale if needed for detachable wider layers
            if i >= self.eval_idx + 1 and self.layer_scale > 1:
                query_pos_embed = F.interpolate(query_pos_embed, scale_factor=self.layer_scale)
                value = self.value_op(memory, None, query_pos_embed.shape[-1], memory_mask, spatial_shapes)
                output = F.interpolate(output, size=query_pos_embed.shape[-1])
                output_detach = output.detach()

            output = layer(output, ref_points_input, value, spatial_shapes, attn_mask, query_pos_embed)

            if i == 0 :
                # Initial bounding box predictions with inverse sigmoid refinement
                pre_bboxes = F.sigmoid(pre_bbox_head(output) + inverse_sigmoid(ref_points_detach))
                if self.use_sdcp:
                    pre_scores = None
                    if self.training:
                        pre_inside, _ = self.sdcp_sampler.pool_local_evidence(pre_bboxes, local_feature)
                        pre_scores = score_head[0](output, pre_inside, pre_bboxes, image_size)
                else:
                    pre_scores = score_head[0](output)
                ref_points_initial = pre_bboxes.detach()

            # Refine bounding box corners using FDR, integrating previous layer's corrections
            pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach
            inter_ref_bbox = distance2bbox(ref_points_initial, integral(pred_corners, project), reg_scale)

            if self.training or i == self.eval_idx:
                if self.use_sdcp:
                    local_evidence = self.sdcp_sampler.pool_local_evidence(inter_ref_bbox, local_feature)
                    raw_scores = score_head[i](output, local_evidence[0], inter_ref_bbox, image_size)
                    scores = self.lqe_layers[i](raw_scores, pred_corners, inter_ref_bbox, local_feature,
                                                local_evidence=local_evidence, clcr_lite=self.clcr_lite)
                elif self.use_clcr or self.use_clcr_lite:
                    scores = score_head[i](output)
                    scores = self.lqe_layers[i](scores, pred_corners, inter_ref_bbox, local_feature,
                                                clcr_lite=self.clcr_lite)
                else:
                    scores = score_head[i](output)
                    scores = self.lqe_layers[i](scores, pred_corners)
                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)

                if not self.training:
                    break

            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()

        result = (torch.stack(dec_out_bboxes), torch.stack(dec_out_logits),
                  torch.stack(dec_out_pred_corners), torch.stack(dec_out_refs), pre_bboxes, pre_scores)
        if return_query_features:
            result += (output,)
        if self.use_sdcp and self.training:
            result += (raw_scores,)
        return result


@register()
class DFINETransformer(nn.Module):
    __share__ = ['num_classes', 'eval_spatial_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_points=4,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learn_query_content=False,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2,
                 aux_loss=True,
                 cross_attn_method='default',
                 query_select_method='default',
                 reg_max=32,
                 reg_scale=4.,
                 layer_scale=1,
                 mlp_act='relu',
                 use_scas=False,
                 scas_num_clutter_types=3,
                 use_sbr=False,
                 sbr_neg_iou_threshold=0.3,
                 sbr_ignore_iou_threshold=0.5,
                 sbr_suppression_mode='multiplicative',
                 sbr_suppression_strength=0.8,
                 sbr_apply_in_train=True,
                 sbr_bg_label_metric='iou',
                 use_brl=False,
                 brl_use_boundary_attention=False,
                 use_sar=False,
                 sar_num_scales=3,
                 sar_pooled_size=7,
                 sar_fp32_ops=True,
                 use_scs=False,
                 scs_hidden_dim=128,
                 use_slqc=False,
                 slqc_token_dim=64,
                 slqc_roi_size=4,
                 slqc_num_levels=2,
                 slqc_mode='layout',
                 use_mpq=False,
                 mpq_projection_dim=128,
                 mpq_prototype_counts=None,
                 mpq_prototype_temperature=0.1,
                 mpq_class_temperature=0.1,
                 use_clcr=False,
                 clcr_hidden_dim=64,
                 use_sdcp=False,
                 sdcp_aspect_dim=16,
                 sdcp_temperature=10.,
                 use_clcr_lite=False,
                 use_ost_da=False,
                 ):
        super().__init__()
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)
        if use_clcr and use_clcr_lite:
            raise ValueError('CLCR and CLCR-Lite are alternative calibrators; enable only one.')
        if use_clcr_lite and num_classes < 2:
            raise ValueError('CLCR-Lite requires at least two competing classes.')

        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        scaled_dim = round(layer_scale*hidden_dim)
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.reg_max = reg_max

        assert query_select_method in ('default', 'one2many', 'agnostic'), ''
        assert cross_attn_method in ('default', 'discrete'), ''
        self.cross_attn_method = cross_attn_method
        self.use_ost_da = use_ost_da
        self.query_select_method = query_select_method

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # Transformer module
        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)
        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, \
            activation, num_levels, num_points, cross_attn_method=cross_attn_method, use_ost_da=use_ost_da)
        decoder_layer_wide = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, \
            activation, num_levels, num_points, cross_attn_method=cross_attn_method,
            layer_scale=layer_scale, use_ost_da=use_ost_da)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, decoder_layer_wide, num_layers, nhead,
                                          reg_max, self.reg_scale, self.up, eval_idx, layer_scale, act=activation)
      # denoising
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(num_classes+1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[:-1])

        # decoder embedding
        self.learn_query_content = learn_query_content
        if learn_query_content:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2, act=mlp_act)

        # if num_select_queries != self.num_queries:
        #     layer = TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward, activation='gelu')
        #     self.encoder = TransformerEncoder(layer, 1)

        self.enc_output = nn.Sequential(OrderedDict([
            ('proj', nn.Linear(hidden_dim, hidden_dim)),
            ('norm', nn.LayerNorm(hidden_dim,)),
        ]))

        if query_select_method == 'agnostic':
            self.enc_score_head = nn.Linear(hidden_dim, 1)
        else:
            self.enc_score_head = nn.Linear(hidden_dim, num_classes)

        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)

        # decoder head
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.dec_score_head = nn.ModuleList(
            [nn.Linear(hidden_dim, num_classes) for _ in range(self.eval_idx + 1)]
          + [nn.Linear(scaled_dim, num_classes) for _ in range(num_layers - self.eval_idx - 1)])
        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)
        self.dec_bbox_head = nn.ModuleList(
            [MLP(hidden_dim, hidden_dim, 4 * (self.reg_max+1), 3, act=mlp_act) for _ in range(self.eval_idx + 1)]
          + [MLP(scaled_dim, scaled_dim, 4 * (self.reg_max+1), 3, act=mlp_act) for _ in range(num_layers - self.eval_idx - 1)])
        self.integral = Integral(self.reg_max)

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer('anchors', anchors)
            self.register_buffer('valid_mask', valid_mask)
        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            self.anchors, self.valid_mask = self._generate_anchors()

        # Optional Scatter-Aware Clutter Suppression (SCAS) for SAR
        self.use_scas = use_scas
        if use_scas:
            self.scas = ScatterAwareClutterSuppression(
                hidden_dim=hidden_dim,
                num_clutter_types=scas_num_clutter_types,
                act=activation,
                dropout=dropout
            )
        else:
            self.scas = None

        # Optional Simple Background Rejection (SBR) for SAR
        self.use_sbr = use_sbr
        if use_sbr:
            self.sbr = SimpleBackgroundRejection(
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                neg_iou_threshold=sbr_neg_iou_threshold,
                ignore_iou_threshold=sbr_ignore_iou_threshold,
                suppression_mode=sbr_suppression_mode,
                suppression_strength=sbr_suppression_strength,
                apply_in_train=sbr_apply_in_train,
                bg_label_metric=sbr_bg_label_metric
            )
        else:
            self.sbr = None

        # Optional Boundary Refinement Loss (BRL)
        self.use_brl = use_brl
        if use_brl:
            self.brl = BoundaryRefinementLoss(
                hidden_dim=hidden_dim,
                use_boundary_attention=brl_use_boundary_attention,
                iou_loss_weight=2.0
            )
        else:
            self.brl = None

        # Optional Scale-Aware Refinement (SAR)
        self.use_sar = use_sar
        if use_sar:
            self.sar = ScaleAwareRefinement(
                hidden_dim=hidden_dim,
                num_scales=sar_num_scales,
                pooled_size=sar_pooled_size,
                fp32_ops=sar_fp32_ops
            )
        else:
            self.sar = None

        self._reset_parameters(feat_channels)

        self.use_scs = use_scs
        self.scs_head = None
        if use_scs:
            final_dim = hidden_dim if num_layers <= self.eval_idx + 1 else scaled_dim
            # A training-only head must not perturb common model initialization
            # or the seed's subsequent data/denoising random sequence.
            with torch.random.fork_rng(devices=[]):
                self.scs_head = ScatteringSignatureHead(final_dim, scs_hidden_dim)

        self.slqc = None
        if use_slqc:
            if layer_scale != 1:
                raise ValueError('SLQC currently requires layer_scale=1, as in the frozen DEIM-M protocol.')
            if slqc_num_levels > num_levels:
                raise ValueError('SLQC cannot sample more feature levels than the decoder receives.')
            with torch.random.fork_rng(devices=[]):
                self.slqc = ScatteringLayoutQueryClassifier(
                    hidden_dim, slqc_token_dim, slqc_roi_size, slqc_num_levels, slqc_mode)

        self.mpq_head = None
        if use_mpq:
            final_dim = hidden_dim if num_layers <= self.eval_idx + 1 else scaled_dim
            with torch.random.fork_rng(devices=[]):
                self.mpq_head = QueryPrototypeHead(
                    final_dim, num_classes, mpq_projection_dim, mpq_prototype_counts,
                    mpq_prototype_temperature, mpq_class_temperature)

        self.use_clcr = use_clcr
        if use_clcr:
            if num_classes < 2 or feat_strides[0] != 8:
                raise ValueError('CLCR requires at least two classes and a stride-8 first feature.')
            # Add calibration after baseline initialization, keeping all pretrained
            # LQE names/weights and the subsequent data/denoising RNG unchanged.
            with torch.random.fork_rng(devices=[]):
                for lqe in self.decoder.lqe_layers:
                    lqe.clcr = CLCR(hidden_dim, 4 * (lqe.k + 1), clcr_hidden_dim, act=activation)
            self.decoder.use_clcr = True

        self.use_clcr_lite = use_clcr_lite
        if use_clcr_lite:
            # Shared across all LQE layers: 161 MLP parameters + one beta total.
            # Register exactly once so state_dict-based EMA updates it once.
            with torch.random.fork_rng(devices=[]):
                self.decoder.clcr_lite = CLCRLite()
            self.decoder.use_clcr_lite = True

        self.use_sdcp = use_sdcp
        if use_sdcp:
            if num_classes < 2 or feat_strides[0] != 8:
                raise ValueError('SDCP requires at least two classes and a stride-8 first feature.')
            with torch.random.fork_rng(devices=[]):
                self.dec_score_head = nn.ModuleList([
                    SDCPCosineClassifier(head.weight, hidden_dim, sdcp_aspect_dim, sdcp_temperature, act=activation)
                    for head in self.dec_score_head
                ])
            self.decoder.use_sdcp = True
            self.decoder.sdcp_sampler = BoxLocalEvidenceSampler()

    def convert_to_deploy(self):
        self.scs_head = None
        self.use_scs = False
        self.mpq_head = None
        self.dec_score_head = nn.ModuleList([nn.Identity()] * (self.eval_idx) + [self.dec_score_head[self.eval_idx]])
        self.dec_bbox_head = nn.ModuleList(
            [self.dec_bbox_head[i] if i <= self.eval_idx else nn.Identity() for i in range(len(self.dec_bbox_head))]
        )

    def _reset_parameters(self, feat_channels):
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        init.constant_(self.pre_bbox_head.layers[-1].weight, 0)
        init.constant_(self.pre_bbox_head.layers[-1].bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            if hasattr(reg_, 'layers'):
                init.constant_(reg_.layers[-1].weight, 0)
                init.constant_(reg_.layers[-1].bias, 0)

        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learn_query_content:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        for m, in_channels in zip(self.input_proj, feat_channels):
            if in_channels != self.hidden_dim:
                init.xavier_uniform_(m[0].weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(OrderedDict([
                        ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),
                        ('norm', nn.BatchNorm2d(self.hidden_dim,))])
                    )
                )

        in_channels = feat_channels[-1]

        for _ in range(self.num_levels - len(feat_channels)):
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(OrderedDict([
                        ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                        ('norm', nn.BatchNorm2d(self.hidden_dim))])
                    )
                )
                in_channels = self.hidden_dim

    def _get_encoder_input(self, feats: List[torch.Tensor]):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])

        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        return feat_flatten, spatial_shapes

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype=torch.float32,
                          device='cpu'):
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])

        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            lvl_anchors = torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4)
            anchors.append(lvl_anchors)

        anchors = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask


    def _get_decoder_input(self,
                           memory: torch.Tensor,
                           spatial_shapes,
                           denoising_logits=None,
                           denoising_bbox_unact=None):

        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors = self.anchors
            valid_mask = self.valid_mask
        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)

        # memory = torch.where(valid_mask, memory, 0)
        # TODO fix type error for onnx export
        memory = valid_mask.to(memory.dtype) * memory

        output_memory :torch.Tensor = self.enc_output(memory)
        enc_outputs_logits :torch.Tensor = self.enc_score_head(output_memory)

        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        enc_topk_memory, enc_topk_logits, enc_topk_anchors = \
            self._select_topk(output_memory, enc_outputs_logits, anchors, self.num_queries)

        enc_topk_bbox_unact :torch.Tensor = self.enc_bbox_head(enc_topk_memory) + enc_topk_anchors

        if self.training:
            enc_topk_bboxes = F.sigmoid(enc_topk_bbox_unact)
            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_logits_list.append(enc_topk_logits)

        # if self.num_select_queries != self.num_queries:
        #     raise NotImplementedError('')

        if self.learn_query_content:
            content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
        else:
            content = enc_topk_memory.detach()

        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()

        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            content = torch.concat([denoising_logits, content], dim=1)

        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list

    def _select_topk(self, memory: torch.Tensor, outputs_logits: torch.Tensor, outputs_anchors_unact: torch.Tensor, topk: int):
        if self.query_select_method == 'default':
            _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)

        elif self.query_select_method == 'one2many':
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes

        elif self.query_select_method == 'agnostic':
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)

        topk_ind: torch.Tensor

        topk_anchors = outputs_anchors_unact.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_anchors_unact.shape[-1]))

        topk_logits = outputs_logits.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1])) if self.training else None

        topk_memory = memory.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))

        return topk_memory, topk_logits, topk_anchors

    def forward(self, feats, targets=None, image_size=None):
        # input projection and embedding
        memory, spatial_shapes = self._get_encoder_input(feats)
        if self.use_sdcp:
            if image_size is None:
                image_size = [side * self.feat_strides[0] for side in spatial_shapes[0]]
            image_size = torch.as_tensor(image_size, dtype=torch.float32, device=memory.device)

        # prepare denoising training
        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(targets, \
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=1.0,
                    )
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list = \
            self._get_decoder_input(memory, spatial_shapes, denoising_logits, denoising_bbox_unact)

        # decoder
        need_queries = self.slqc is not None or (self.training and (self.use_scs or self.mpq_head is not None))
        decoder_results = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            self.pre_bbox_head,
            self.integral,
            self.up,
            self.reg_scale,
            attn_mask=attn_mask,
            dn_meta=dn_meta,
            return_query_features=need_queries,
            image_size=image_size)

        out_bboxes, out_logits, out_corners, out_refs, pre_bboxes, pre_logits = decoder_results[:6]
        final_queries = decoder_results[6] if need_queries else None
        sdcp_logits = decoder_results[-1] if self.use_sdcp and self.training else None

        if self.training and dn_meta is not None:
            # the output from the first decoder layer, only one
            dn_pre_logits, pre_logits = torch.split(pre_logits, dn_meta['dn_num_split'], dim=1)
            dn_pre_bboxes, pre_bboxes = torch.split(pre_bboxes, dn_meta['dn_num_split'], dim=1)

            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)

            dn_out_corners, out_corners = torch.split(out_corners, dn_meta['dn_num_split'], dim=2)
            dn_out_refs, out_refs = torch.split(out_refs, dn_meta['dn_num_split'], dim=2)
            if final_queries is not None:
                _, final_queries = torch.split(final_queries, dn_meta['dn_num_split'], dim=1)
            if sdcp_logits is not None:
                _, sdcp_logits = torch.split(sdcp_logits, dn_meta['dn_num_split'], dim=1)

        localization_teacher_logits = out_logits[-1]
        if self.slqc is not None:
            # Reuse the projected encoder memory. These are views, not B*Q
            # copies of full maps. Sampling boxes are detached inside SLQC.
            batch, _, channels = memory.shape
            local_features, offset = [], 0
            for height, width in spatial_shapes[:self.slqc.num_levels]:
                height, width = int(height), int(width)
                count = height * width
                feature = memory[:, offset:offset + count].transpose(1, 2)
                local_features.append(feature.reshape(batch, channels, height, width))
                offset += count
            delta_query = self.slqc(final_queries, out_bboxes[-1], local_features)
            score_index = self.num_layers - 1 if self.training else self.eval_idx
            # Reuse the original classifier and preserve its bias and the
            # already-added LQE quality score. Only final ordinary logits
            # change; bbox/FDR, DN, encoder and shallow aux paths stay intact.
            residual_logits = F.linear(delta_query, self.dec_score_head[score_index].weight, bias=None)
            final_logits = out_logits[-1] + residual_logits
            out_logits = torch.cat((out_logits[:-1], final_logits.unsqueeze(0)), dim=0)


        if self.training:
            out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1], 'pred_corners': out_corners[-1],
                   'ref_points': out_refs[-1], 'up': self.up, 'reg_scale': self.reg_scale}
            if self.use_sdcp:
                out.update(sdcp_logits=sdcp_logits, sdcp_image_size=image_size)
        else:
            # Diagnosis only: expose the distribution and its decoding reference.
            out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1],
                   'pred_corners': out_corners[-1], 'ref_points': out_refs[-1],
                   'up': self.up, 'reg_scale': self.reg_scale}

        if self.use_scs and self.training:
            out['pred_scs'] = self.scs_head(final_queries)

        if self.mpq_head is not None and self.training:
            out.update(self.mpq_head(final_queries))

        # Apply SCAS (Scatter-Aware Clutter Suppression) if enabled
        if self.scas is not None:
            # Reconstruct stride-8 encoder feature from memory
            # memory: list of [B, H*W, C] for each level
            # spatial_shapes: [num_levels, 2] (H, W for each level)

            # Check if memory has correct shape (profiler may pass 2D tensor)
            if memory[0].dim() == 3:  # Normal case: [B, H*W, C]
                B = memory[0].shape[0]
                H_s8, W_s8 = spatial_shapes[0].tolist()  # Stride-8 is first level
                C = memory[0].shape[-1]
                encoder_feat_s8 = memory[0].permute(0, 2, 1).reshape(B, C, H_s8, W_s8)  # [B, C, H, W]

                query_embeds = init_ref_contents  # [B, num_queries, C]
                ref_points = torch.sigmoid(init_ref_points_unact)  # [B, num_queries, 2]

                suppression_weights, scatter_map = self.scas(
                    encoder_feat_s8, query_embeds, ref_points
                )

                # Apply suppression to final logits
                out['pred_logits'] = self.scas.apply_suppression(
                    out['pred_logits'], suppression_weights
                )

                # Store for visualization/analysis (only during training)
                if self.training:
                    out['scas_weights'] = suppression_weights
                    out['scatter_map'] = scatter_map

        # Apply SBR (Simple Background Rejection) if enabled
        if self.sbr is not None:
            # Handle denoising: init_ref_* contains both dn and regular queries
            # but out['pred_boxes'] only contains regular queries
            if self.training and dn_meta is not None:
                # Split to get only regular queries (matching out['pred_boxes'])
                _, query_embeds = torch.split(init_ref_contents, dn_meta['dn_num_split'], dim=1)
                _, ref_points_unact = torch.split(init_ref_points_unact, dn_meta['dn_num_split'], dim=1)
                ref_points = torch.sigmoid(ref_points_unact)
            else:
                # No denoising, use all queries
                query_embeds = init_ref_contents
                ref_points = torch.sigmoid(init_ref_points_unact)

            pred_boxes = out['pred_boxes']  # [B, num_queries, 4 or more]
            pred_logits = out['pred_logits']  # [B, num_queries, num_classes]

            # Ensure pred_boxes is exactly 4-dim (cxcywh)
            if pred_boxes.shape[-1] > 4:
                pred_boxes = pred_boxes[..., :4]

            # Compute background scores
            bg_scores = self.sbr(query_embeds, ref_points, pred_boxes, pred_logits)

            # Apply suppression to logits only when allowed.
            # During training this rewrites the logits seen by the Hungarian
            # matcher, which creates a feedback loop (suppress -> matching
            # changes -> supervision labels change -> suppress again) and
            # leaves aux layers unmatched. With apply_in_train=False the head
            # is trained purely as an auxiliary objectness target and only
            # used for re-ranking at inference.
            if (not self.training) or self.sbr.apply_in_train:
                out['pred_logits'] = self.sbr.apply_suppression(pred_logits, bg_scores)

            # Store background scores for loss computation
            out['bg_scores'] = bg_scores

        # Apply BRL (Boundary Refinement Loss) if enabled
        if self.brl is not None:
            # Handle denoising
            if self.training and dn_meta is not None:
                _, query_embeds_brl = torch.split(init_ref_contents, dn_meta['dn_num_split'], dim=1)
            else:
                query_embeds_brl = init_ref_contents

            pred_boxes_brl = out['pred_boxes']
            if pred_boxes_brl.shape[-1] > 4:
                pred_boxes_brl = pred_boxes_brl[..., :4]

            # Predict IoU quality
            pred_iou, box_delta = self.brl(query_embeds_brl, pred_boxes_brl, boundary_feat=None)

            # Store for loss computation
            out['pred_iou'] = pred_iou

            # Optional: apply box refinement (if boundary attention enabled)
            if box_delta is not None:
                out['pred_boxes'] = pred_boxes_brl + box_delta

        # Apply SAR (Scale-Aware Refinement) if enabled
        if self.sar is not None:
            # Fail loudly instead of silently skipping the module. A silent skip
            # trains an unrefined detector that looks entirely normal, which has
            # already produced one uninterpretable result in this repository.
            if memory.dim() != 3:
                raise RuntimeError(
                    f'SAR is enabled but memory has shape {tuple(memory.shape)}; '
                    'expected the flattened [B, sum(H*W), C] layout.')
            # Handle denoising
            if self.training and dn_meta is not None:
                _, query_embeds_sar = torch.split(init_ref_contents, dn_meta['dn_num_split'], dim=1)
            else:
                query_embeds_sar = init_ref_contents

            pred_boxes_sar = out['pred_boxes']
            if pred_boxes_sar.shape[-1] > 4:
                pred_boxes_sar = pred_boxes_sar[..., :4]

            # memory is flattened: [B, sum(H*W), C] across all pyramid levels.
            # Split it back into per-level feature maps using spatial_shapes.
            B_sar, _, C_sar = memory.shape
            multi_scale_feats = []
            offset = 0
            for H_lvl, W_lvl in spatial_shapes:
                H_lvl, W_lvl = int(H_lvl), int(W_lvl)
                numel = H_lvl * W_lvl
                feat = memory[:, offset:offset + numel, :]  # [B, H*W, C]
                feat = feat.permute(0, 2, 1).reshape(B_sar, C_sar, H_lvl, W_lvl).contiguous()
                multi_scale_feats.append(feat)
                offset += numel

            if len(multi_scale_feats) > 0:
                # Scale-aware refinement; refined boxes flow into the box
                # regression / GIoU losses, so all SAR params receive gradient.
                refined_boxes, _ = self.sar(
                    query_embeds_sar,
                    pred_boxes_sar,
                    multi_scale_feats,
                    spatial_shapes
                )
                out['pred_boxes'] = refined_boxes

        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss2(out_logits[:-1], out_bboxes[:-1], out_corners[:-1], out_refs[:-1],
                                                     out_corners[-1], localization_teacher_logits)
            out['enc_aux_outputs'] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list)
            out['pre_outputs'] = {'pred_logits': pre_logits, 'pred_boxes': pre_bboxes}
            out['enc_meta'] = {'class_agnostic': self.query_select_method == 'agnostic'}

            if dn_meta is not None:
                out['dn_outputs'] = self._set_aux_loss2(dn_out_logits, dn_out_bboxes, dn_out_corners, dn_out_refs,
                                                        dn_out_corners[-1], dn_out_logits[-1])
                out['dn_pre_outputs'] = {'pred_logits': dn_pre_logits, 'pred_boxes': dn_pre_bboxes}
                out['dn_meta'] = dn_meta

        return out


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class, outputs_coord)]


    @torch.jit.unused
    def _set_aux_loss2(self, outputs_class, outputs_coord, outputs_corners, outputs_ref,
                       teacher_corners=None, teacher_logits=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_corners': c, 'ref_points': d,
                     'teacher_corners': teacher_corners, 'teacher_logits': teacher_logits}
                for a, b, c, d in zip(outputs_class, outputs_coord, outputs_corners, outputs_ref)]
