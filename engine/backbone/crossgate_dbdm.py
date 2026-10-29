"""CrossGate-DBDM downsampling, implemented from the supplied modified design.

This is an independent SAR adaptation, not an official DBCFS-Net reproduction.
Four sampling phases in each 2x2 block have learned, edge-conditioned offsets.
The original rigid path keeps its parameter names for COCO checkpoint loading.
"""

from collections import OrderedDict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossGateDBDM(nn.Module):
    """Stride-2 downsampling with Sobel-guided sampling and channel gating.

    output = residual + res_scale * gate * (dynamic - residual).
    Zero initial res_scale preserves the original pretrained downsampler;
    the offset head starts at zero and learns shifts in input-pixel units.
    Odd sizes use the same ceil(H/2), ceil(W/2) extent as 3x3/stride-2/pad-1.
    """

    def __init__(self, in_channels, out_channels, residual=None,
                 max_offset=1.0, res_scale_init=0.0):
        super().__init__()
        self.max_offset = float(max_offset)
        if residual is None:
            groups = in_channels if in_channels == out_channels else 1
            self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2,
                                  padding=1, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)
            self.act = nn.Identity()
            self.lab = nn.Identity()
        else:
            # Reuse the original HGNet ConvBNAct children at their old keys.
            self.conv = residual.conv
            self.bn = residual.bn
            self.act = residual.act
            self.lab = residual.lab

        sobel = torch.tensor([
            [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
            [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
        ]).unsqueeze(1) / 8.
        self.register_buffer('sobel', sobel)
        self.register_buffer('sample_phases', torch.tensor([[0., 0.], [1., 0.],
                                                           [0., 1.], [1., 1.]]))
        hidden = max(8, min(32, in_channels // 2))
        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_channels + 1, hidden, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, 8, 1),
        )
        nn.init.zeros_(self.offset_conv[-1].weight)
        nn.init.zeros_(self.offset_conv[-1].bias)
        self.fuse = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels * 4, out_channels, 1, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
            ('act', nn.SiLU()),
        ]))
        gate_hidden = max(8, out_channels // 4)
        self.cross_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * 2 + 1, gate_hidden, 1),
            nn.SiLU(),
            nn.Conv2d(gate_hidden, out_channels, 1),
            nn.Sigmoid(),
        )
        self.res_scale = nn.Parameter(torch.tensor(float(res_scale_init)))

    def _edge_energy(self, features):
        # Keep the squared Sobel energy in FP32 under AMP. Replicate padding
        # avoids inventing an image-border edge for a constant feature map.
        with torch.autocast(device_type=features.device.type, enabled=False):
            gray = features.float().mean(1, keepdim=True)
            gradients = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode='replicate'),
                                 self.sobel.float())
            epsilon = 1e-6
            energy = (gradients.square().sum(1, keepdim=True) + epsilon).sqrt() - math.sqrt(epsilon)
            return energy / energy.mean((2, 3), keepdim=True).clamp_min(epsilon)

    def _sample_blocks(self, features, offsets):
        """Sample four local phases; offsets are (B,8,h,w), ordered (x,y)."""
        batch, _, height, width = features.shape
        out_height, out_width = offsets.shape[-2:]
        with torch.autocast(device_type=features.device.type, enabled=False):
            y, x = torch.meshgrid(
                torch.arange(out_height, device=features.device, dtype=torch.float32) * 2,
                torch.arange(out_width, device=features.device, dtype=torch.float32) * 2,
                indexing='ij')
            base = torch.stack((x, y), dim=-1)[None, None]
            shifts = offsets.float().reshape(batch, 4, 2, out_height, out_width).permute(0, 1, 3, 4, 2)
            coordinates = base + self.sample_phases.float()[None, :, None, None] + shifts
            # align_corners=False: normalize input-pixel centers, including
            # singleton axes. Border sampling handles odd final blocks.
            scale = coordinates.new_tensor([width, height])
            grids = (coordinates * 2 + 1) / scale - 1
            source = features.float()
            samples = [F.grid_sample(source, grids[:, phase], mode='bilinear',
                                     padding_mode='border', align_corners=False)
                       for phase in range(4)]
            return torch.cat(samples, dim=1)

    def forward(self, features):
        residual = self.lab(self.act(self.bn(self.conv(features))))
        edge = self._edge_energy(features)
        offsets = self.offset_conv(torch.cat((features, edge.to(dtype=features.dtype)), dim=1))
        offsets = offsets.tanh() * self.max_offset
        dynamic = self.fuse(self._sample_blocks(features, offsets))
        edge_down = F.adaptive_avg_pool2d(edge, residual.shape[-2:]).to(dtype=dynamic.dtype)
        gate = self.cross_gate(torch.cat((dynamic, residual.to(dtype=dynamic.dtype), edge_down), dim=1))
        correction = gate * (dynamic - residual.to(dtype=dynamic.dtype))
        return residual + self.res_scale.to(dtype=residual.dtype) * correction.to(dtype=residual.dtype)
