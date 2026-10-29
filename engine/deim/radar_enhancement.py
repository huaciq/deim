"""Same-resolution REM adapter inspired by MARSS (CVPR 2026), Sec. 3.1/Fig. 3.

Reference: https://openaccess.thecvf.com/content/CVPR2026/html/
Chen_MARSS_Radar_Semantic_Segmentation_via_Modular_Attention_and_State_Space_CVPR_2026_paper.html

The four convolution branches follow the paper. Its figure concatenates the
subbands; this detector adapter instead uses signed Haar reconstruction and a
zero-initialized residual scale to retain the existing encoder's feature sizes.
This is a paper-based adaptation, not a copy of an official implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_activation


def _haar_kernels():
    # LL, LH (row variation), HL (column variation), HH; orthonormal scaling.
    return torch.tensor([
        [[1., 1.], [1., 1.]],
        [[1., 1.], [-1., -1.]],
        [[1., -1.], [1., -1.]],
        [[1., -1.], [-1., 1.]],
    ]).unsqueeze(1) * .5


class HaarDWT2D(nn.Module):
    """Fixed, per-channel stride-2 Haar analysis returning LL/LH/HL/HH."""

    def __init__(self):
        super().__init__()
        self.register_buffer('kernels', _haar_kernels())

    def forward(self, features):
        batch, channels, height, width = features.shape
        if height % 2 or width % 2:
            features = F.pad(features, (0, width % 2, 0, height % 2), mode='replicate')
        weight = self.kernels.to(dtype=features.dtype).repeat(channels, 1, 1, 1)
        bands = F.conv2d(features, weight, stride=2, groups=channels)
        return bands.reshape(batch, channels, 4, *bands.shape[-2:]).unbind(2)


class HaarIDWT2D(nn.Module):
    """Fixed Haar synthesis; crop replicated padding to the original size."""

    def __init__(self):
        super().__init__()
        self.register_buffer('kernels', _haar_kernels())

    def forward(self, bands, output_size=None):
        batch, channels, height, width = bands[0].shape
        coefficients = torch.stack(bands, dim=2).reshape(batch, channels * 4, height, width)
        weight = self.kernels.to(dtype=coefficients.dtype).repeat(channels, 1, 1, 1)
        restored = F.conv_transpose2d(coefficients, weight, stride=2, groups=channels)
        if output_size is not None:
            restored = restored[..., :output_size[0], :output_size[1]]
        return restored


def _conv_norm_act(channels, kernel_size, act):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    return nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size,
                  padding=tuple(size // 2 for size in kernel_size), bias=False),
        nn.BatchNorm2d(channels),
        get_activation(act),
    )


class RadarEnhancementModule(nn.Module):
    """REM with four distinct subband branches and a same-shape residual.

    LL uses 1x1; LH/HL use opposite axial orders followed by 3x3; HH uses 3x3.
    Each branch starts with the 1x1 projection shown in the paper's Fig. 3.
    Only alpha starts at zero, so the adapter is initially an identity while
    the residual scale can learn immediately. Feature dtype is preserved.
    """

    def __init__(self, channels, act='silu'):
        super().__init__()
        self.dwt = HaarDWT2D()
        self.idwt = HaarIDWT2D()
        self.ll = _conv_norm_act(channels, 1, act)
        self.lh = nn.Sequential(
            _conv_norm_act(channels, 1, act),
            _conv_norm_act(channels, (1, 3), act),
            _conv_norm_act(channels, (3, 1), act),
            _conv_norm_act(channels, 3, act),
        )
        self.hl = nn.Sequential(
            _conv_norm_act(channels, 1, act),
            _conv_norm_act(channels, (3, 1), act),
            _conv_norm_act(channels, (1, 3), act),
            _conv_norm_act(channels, 3, act),
        )
        self.hh = nn.Sequential(
            _conv_norm_act(channels, 1, act),
            _conv_norm_act(channels, 3, act),
        )
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, features):
        ll, lh, hl, hh = self.dwt(features)
        restored = self.idwt((self.ll(ll), self.lh(lh), self.hl(hl), self.hh(hh)),
                             features.shape[-2:])
        return features + self.alpha.to(dtype=features.dtype) * restored.to(dtype=features.dtype)
