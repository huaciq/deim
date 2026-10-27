"""Stride-4 detail injection with isolated plain, strip and local controls.

StripDetailTransform adapts the StripBlock/Attention operator of Strip R-CNN
(Yuan et al., AAAI 2026; arXiv:2501.03775). It does not reproduce StripNet or
its detection head. Attribution and upstream license: docs/strip_rcnn_notice.md.
The adapted strip operator is licensed CC-BY-NC-4.0.
"""
from collections import OrderedDict

import torch
from torch import nn


class StripDetailTransform(nn.Module):
    """Residual spatial gating; ``local`` replaces only the two strip kernels.

    Serial orthogonal strips have a two-dimensional effective receptive field;
    they are neither a cross-shaped mask nor a rotation-equivariant operator.
    The local control retains projections, gating and depth with 5x5/3x3 DW
    kernels. At C=96 it has 384 fewer parameters than the K=19 strip variant.
    """

    def __init__(self, channels, mode='strip', strip_kernel=19):
        super().__init__()
        if mode not in ('strip', 'local'):
            raise ValueError('Detail transform mode must be strip or local.')
        if strip_kernel < 1 or strip_kernel % 2 == 0:
            raise ValueError('Strip kernel must be a positive odd integer.')
        kernels = ((1, strip_kernel), (strip_kernel, 1)) if mode == 'strip' else ((5, 5), (3, 3))
        self.proj_in = nn.Conv2d(channels, channels, 1)
        self.activation = nn.GELU()
        self.context = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2, groups=channels),
            *[nn.Conv2d(channels, channels, kernel, padding=tuple(k // 2 for k in kernel),
                        groups=channels) for kernel in kernels],
            nn.Conv2d(channels, channels, 1),
        )
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, features):
        local = self.activation(self.proj_in(features))
        return features + self.proj_out(local * self.context(local))


class ShallowDetailInjection(nn.Module):
    """Add a single S2 branch to projected S3; no new target or auxiliary loss.

    D is DW3x3/s2 -> PW1x1 -> BN, with a zero-initialized final BN affine.
    Only that affine is zeroed, so its first gradient opens the residual path.
    Instantiate D before the optional transform to share D's initialization
    across experiments. The encoder isolates construction from the base RNG.
    """
    MODES = ('plain', 'strip', 'local')

    def __init__(self, in_channels, out_channels, mode='plain', strip_kernel=19):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f'S2 injection mode must be one of {self.MODES}.')
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError('S2 injection channels must be positive.')
        self.mode = mode
        self.downsample = nn.Sequential(OrderedDict([
            ('depthwise', nn.Conv2d(in_channels, in_channels, 3, stride=2,
                                    padding=1, groups=in_channels, bias=False)),
            ('projection', nn.Conv2d(in_channels, out_channels, 1, bias=False)),
            ('norm', nn.BatchNorm2d(out_channels)),
        ]))
        nn.init.zeros_(self.downsample.norm.weight)
        nn.init.zeros_(self.downsample.norm.bias)
        self.structure = (nn.Identity() if mode == 'plain' else
                          StripDetailTransform(in_channels, mode, strip_kernel))
        self.last_metrics = {}

    def forward(self, detail, main):
        residual = self.downsample(self.structure(detail))
        if residual.shape != main.shape:
            raise ValueError(f'S2 residual {tuple(residual.shape)} must match projected S3 '
                             f'{tuple(main.shape)}; use stride-4 S2 and stride-8 S3.')
        if self.training:
            with torch.no_grad():
                # Same element count: this is also RMS(residual)/RMS(main).
                ratio = (torch.linalg.vector_norm(residual.detach(), dtype=torch.float32) /
                         torch.linalg.vector_norm(main.detach(), dtype=torch.float32).clamp_min(1e-12))
                self.last_metrics = {'s2_residual_ratio': ratio}
        else:
            self.last_metrics = {}
        return main + residual
