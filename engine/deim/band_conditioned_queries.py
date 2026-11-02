"""Complete band-conditioned instance/class evidence decoder (BCCQ).

Class-query learning is inspired by INTR (ICLR 2024) and Classification
Matters (ECCV 2024). BCCQ adds instance/class-conditioned source routing
over spatial features and signed Haar bands. This is an original adaptation,
not an implementation of either reference detector.
"""

import math
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .radar_enhancement import HaarDWT2D


SOURCE_NAMES = ('p3', 'p4', 'll', 'lh', 'hl', 'hh')


def joint_detection_logits(type_logits, quality_logits):
    """Stable logit(sigmoid(quality) * softmax(type)), evaluated in FP32."""
    with torch.autocast(device_type=type_logits.device.type, enabled=False):
        log_score = F.logsigmoid(quality_logits.float()) + F.log_softmax(type_logits.float(), -1)
        # Saturated logits can round log_score to exactly zero in FP32.
        log_score = log_score.clamp_max(-torch.finfo(torch.float32).eps)
        return log_score - torch.log(-torch.expm1(log_score))


class BandClassQueryLayer(nn.Module):
    def __init__(self, dim, heads, feedforward):
        super().__init__()
        self.dim, self.heads = dim, heads
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=0., batch_first=True)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)
        self.gate_query = nn.Linear(dim, dim)
        self.gate_key = nn.Linear(dim, dim)
        self.linear1 = nn.Linear(dim, feedforward)
        self.linear2 = nn.Linear(feedforward, dim)
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(dim) for _ in range(3))

    def project_sources(self, sources):
        """K/V projections before pooling: share maps across all instances/classes."""
        shapes = [source.shape[-2:] for source in sources]
        counts = [h * w for h, w in shapes]
        flat = torch.cat([source.flatten(2).transpose(1, 2) for source in sources], 1)
        keys, values = self.key(flat), self.value(flat)
        key_parts, value_parts = keys.split(counts, 1), values.split(counts, 1)
        return tuple(torch.cat((source, key.transpose(1, 2).reshape_as(source),
                                value.transpose(1, 2).reshape_as(source)), 1)
                     for source, key, value in zip(sources, key_parts, value_parts))


class BandConditionedClassQueries(nn.Module):
    """Four-layer instance-specific class queries with six complete evidence sources."""
    def __init__(self, input_dim, num_classes, num_layers, eval_idx, dim=128,
                 heads=4, feedforward=512, roi_size=5, query_chunk_size=64,
                 activation_checkpoint=True):
        super().__init__()
        if dim % heads or dim % 4 or num_classes < 2 or roi_size < 2 or query_chunk_size < 1:
            raise ValueError('BCCQ requires dim divisible by heads and 4, >=2 classes/ROI size, positive chunk size.')
        self.dim, self.heads, self.num_classes = dim, heads, num_classes
        self.num_layers, self.eval_idx = num_layers, eval_idx
        self.roi_size, self.query_chunk_size = roi_size, query_chunk_size
        self.activation_checkpoint = activation_checkpoint
        self.source_names = SOURCE_NAMES
        self.input_proj = nn.ModuleList([nn.Conv2d(input_dim, dim, 1, bias=False) for _ in range(2)])
        self.dwt = HaarDWT2D()
        self.instance_proj = nn.Linear(input_dim, dim)
        self.instance_norm = nn.LayerNorm(dim)
        self.class_embed = nn.Embedding(num_classes, dim)
        self.source_embed = nn.Embedding(len(SOURCE_NAMES), dim)
        self.box_position = nn.Sequential(nn.Linear(4, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.summary = nn.Sequential(nn.Linear(2 * dim, dim), nn.SiLU())
        self.layers = nn.ModuleList([BandClassQueryLayer(dim, heads, feedforward) for _ in range(num_layers)])
        self.presence = nn.Linear(dim, 1, bias=False)
        self.quality_heads = nn.ModuleList([nn.Linear(input_dim, 1) for _ in range(num_layers)])
        for head in self.quality_heads:
            nn.init.constant_(head.bias, math.log(.01 / .99))
        nn.init.normal_(self.class_embed.weight, std=.02)
        nn.init.normal_(self.source_embed.weight, std=.02)
        axis = torch.linspace(-.4, .4, roi_size)
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        offsets = torch.stack((xx, yy), -1).flatten(0, 1)
        self.register_buffer('roi_offsets', offsets)
        frequencies = 2 * math.pi * torch.exp(-math.log(10000.) * torch.arange(dim // 4) / (dim // 4))
        angles = offsets[..., None] * frequencies
        self.register_buffer('relative_position', torch.cat((angles[:, 0].sin(), angles[:, 0].cos(),
                                                            angles[:, 1].sin(), angles[:, 1].cos()), -1))

    def prepare_sources(self, memory, spatial_shapes):
        """Projected P3/P4 and signed P3 LL/LH/HL/HH, computed once per forward."""
        sources, offset = [], 0
        for index, (height, width) in enumerate(spatial_shapes[:2]):
            feature = memory[:, offset:offset + height * width].transpose(1, 2)
            feature = feature.reshape(memory.shape[0], memory.shape[-1], height, width)
            sources.append(self.input_proj[index](feature))
            offset += height * width
        p3, p4 = sources
        with torch.autocast(device_type=p3.device.type, enabled=False):
            bands = tuple(band.to(p3.dtype) for band in self.dwt(p3.float()))
        height, width = p3.shape[-2:]
        # Odd P3 sizes are replicated before DWT. Image-normalized coordinates
        # must first be mapped to that padded extent, including its half-cell centers.
        band_scale = (width / (width + width % 2), height / (height + height % 2))
        scales = ((1., 1.), (1., 1.)) + (band_scale,) * 4
        return (p3, p4, *bands), scales

    def initialize_queries(self, instances):
        base = self.instance_norm(self.instance_proj(instances.detach()))
        return base.unsqueeze(2) + self.class_embed.weight.to(base.dtype)[None, None]

    def sample_source(self, packed, boxes, scale):
        with torch.autocast(device_type=packed.device.type, enabled=False):
            grid = boxes.detach().float()[..., None, :2] + (
                boxes.detach().float()[..., None, 2:] * self.roi_offsets.float())
            grid = grid * grid.new_tensor(scale) * 2. - 1.
            pooled = F.grid_sample(packed.float(), grid, mode='bilinear',
                                   padding_mode='border', align_corners=False)
        return pooled.permute(0, 2, 3, 1).to(packed.dtype)

    def _chunk(self, layer_index, scales, states, boxes, *packed_sources):
        layer = self.layers[layer_index]
        batch, count, classes, dim = states.shape
        position = self.box_position(boxes.detach()).to(states.dtype).unsqueeze(2)
        inputs = states.reshape(batch * count, classes, dim)
        query = (states + position).reshape_as(inputs)
        attended = layer.self_attn(query, query, inputs, need_weights=False)[0]
        states = layer.norm1(inputs + attended).reshape(batch, count, classes, dim)

        pooled = torch.stack([self.sample_source(source, boxes, scale)
                              for source, scale in zip(packed_sources, scales)], 2)
        visual, keys, values = pooled.split(dim, -1)
        with torch.autocast(device_type=states.device.type, enabled=False):
            mean = visual.float().mean(-2)
            rms = (visual.float().square().mean(-2) + 1e-6).sqrt()
        summaries = self.summary(torch.cat((mean, rms), -1))
        source_position = self.source_embed.weight.to(keys.dtype)
        keys = keys + self.relative_position.to(keys.dtype)[None, None, None] + source_position[None, None, :, None]

        head_dim = dim // self.heads
        q = layer.query(states + position).reshape(batch, count, classes, self.heads, head_dim)
        k = keys.reshape(batch, count, len(SOURCE_NAMES), -1, self.heads, head_dim)
        v = values.reshape_as(k)
        gq = layer.gate_query(states + position).reshape_as(q)
        gk = (layer.gate_key(summaries) + source_position).reshape(
            batch, count, len(SOURCE_NAMES), self.heads, head_dim)
        with torch.autocast(device_type=states.device.type, enabled=False):
            spatial = torch.einsum('bqchd,bqsnhd->bqhcsn', q.float(), k.float()) / math.sqrt(head_dim)
            spatial = spatial.softmax(-1)
            gates = torch.einsum('bqchd,bqshd->bqhcs', gq.float(), gk.float()) / math.sqrt(head_dim)
            gates = gates.softmax(-1)
            evidence = torch.einsum('bqhcsn,bqsnhd->bqchsd', spatial, v.float())
            evidence = (evidence * gates.permute(0, 1, 3, 2, 4).unsqueeze(-1)).sum(-2)
            evidence = evidence.flatten(-2).to(states.dtype)
            mean_gates = gates.mean(2)
            entropy = -(gates * gates.clamp_min(1e-12).log()).sum(-1).mean(2)
        states = layer.norm2(states + layer.output(evidence))
        states = layer.norm3(states + layer.linear2(F.silu(layer.linear1(states))))
        return states, mean_gates, entropy

    def step(self, layer_index, states, boxes, source_cache):
        sources, scales = source_cache
        packed = self.layers[layer_index].project_sources(sources)
        state_parts, gate_parts, entropy_parts = [], [], []
        function = partial(self._chunk, layer_index, scales)
        for start in range(0, states.shape[1], self.query_chunk_size):
            state = states[:, start:start + self.query_chunk_size]
            reference = boxes[:, start:start + self.query_chunk_size].detach()
            if self.activation_checkpoint and self.training and torch.is_grad_enabled():
                updated, gates, entropy = checkpoint(function, state, reference, *packed, use_reentrant=False)
            else:
                updated, gates, entropy = function(state, reference, *packed)
            state_parts.append(updated)
            gate_parts.append(gates.detach())
            entropy_parts.append(entropy.detach())
        return torch.cat(state_parts, 1), torch.cat(gate_parts, 1), torch.cat(entropy_parts, 1)

    def classify(self, states):
        return self.presence(states).squeeze(-1)

    def convert_to_deploy(self):
        self.layers = self.layers[:self.eval_idx + 1]
        self.quality_heads = nn.ModuleList([nn.Identity()] * self.eval_idx + [self.quality_heads[self.eval_idx]])

