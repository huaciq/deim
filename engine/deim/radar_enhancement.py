"""Same-resolution REM adapter inspired by MARSS (CVPR 2026), Sec. 3.1/Fig. 3.

Reference: https://openaccess.thecvf.com/content/CVPR2026/html/
Chen_MARSS_Radar_Semantic_Segmentation_via_Modular_Attention_and_State_Space_CVPR_2026_paper.html

The four convolution branches follow the paper. Its figure concatenates the
subbands; this detector adapter instead uses signed Haar reconstruction and a
zero-initialized residual scale to retain the existing encoder's feature sizes.
This is a paper-based adaptation, not a copy of an official implementation.

PGD-REM below is a separate detector variant: it keeps REM's directional
branches and adds an LL-derived gate within the same Haar analysis/synthesis.
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


class SpaceToDepth2D(nn.Module):
    """Split each 2x2 block into four raw spatial phases, without mixing them."""

    def forward(self, features):
        batch, channels, height, width = features.shape
        if height % 2 or width % 2:
            features = F.pad(features, (0, width % 2, 0, height % 2), mode='replicate')
        phases = F.pixel_unshuffle(features, 2)
        return phases.reshape(batch, channels, 4, *phases.shape[-2:]).unbind(2)


class DepthToSpace2D(nn.Module):
    """Invert SpaceToDepth2D with the same channel/phase order and odd crop."""

    def forward(self, phases, output_size=None):
        batch, channels, height, width = phases[0].shape
        packed = torch.stack(phases, dim=2).reshape(batch, channels * 4, height, width)
        restored = F.pixel_shuffle(packed, 2)
        if output_size is not None:
            restored = restored[..., :output_size[0], :output_size[1]]
        return restored


def _conv_norm_act(channels, kernel_size, act, out_channels=None):
    out_channels = channels if out_channels is None else out_channels
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    return nn.Sequential(
        nn.Conv2d(channels, out_channels, kernel_size,
                  padding=tuple(size // 2 for size in kernel_size), bias=False),
        nn.BatchNorm2d(out_channels),
        get_activation(act),
    )


class RadarEnhancementModule(nn.Module):
    """REM with four distinct subband branches and a same-shape residual.

    LL uses 1x1; LH/HL use opposite axial orders followed by 3x3; HH uses 3x3.
    Each branch starts with the 1x1 projection shown in the paper's Fig. 3.
    Only alpha starts at zero, so the adapter is initially an identity while
    the residual scale can learn immediately. Feature dtype is preserved.
    Optional controls change only the analysis/synthesis basis (space_to_depth)
    or the explicit axial strip pair (haar_square); the reported default stays
    haar_directional.
    """

    def __init__(self, channels, act='silu', variant='haar_directional'):
        super().__init__()
        if variant not in ('haar_directional', 'space_to_depth', 'haar_square'):
            raise ValueError('REM variant must be haar_directional, space_to_depth or haar_square.')
        self.variant = variant
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

        # Build the reported REM first so common tensors keep their exact
        # initialization. The default path and state_dict remain unchanged.
        if variant == 'space_to_depth':
            self.dwt = SpaceToDepth2D()
            self.idwt = DepthToSpace2D()
        elif variant == 'haar_square':
            # Each strip pair costs 6*C*C weights. Square 3x3 -> 1x1 costs
            # 10*C*M, hence M ~= 3*C/5. Preserve depth and 5x5 branch RF.
            middle = max(1, round(3 * channels / 5))
            with torch.random.fork_rng(devices=[]):
                for branch in (self.lh, self.hl):
                    branch[1] = _conv_norm_act(channels, 3, act, out_channels=middle)
                    branch[2] = _conv_norm_act(middle, 1, act, out_channels=channels)

    def forward(self, features):
        ll, lh, hl, hh = self.dwt(features)
        restored = self.idwt((self.ll(ll), self.lh(lh), self.hl(hl), self.hh(hh)),
                             features.shape[-2:])
        return features + self.alpha.to(dtype=features.dtype) * restored.to(dtype=features.dtype)


class LowFrequencyStructureGate(nn.Module):
    """Compare compressed LL with a smooth prior; emit three full-width gates.

    Only this side branch uses a bottleneck. The depthwise 5x5 kernel starts
    as an averaging filter and remains learnable. Its receptive field is local
    on LL; it is not an FFT or a whole-image global operator. The learned
    gates are bounded in [0.5, 1.5], without a hand-coded foreground mask.
    """

    def __init__(self, channels, prior_channels=64):
        super().__init__()
        self.reduce = nn.Conv2d(channels, prior_channels, 1)
        self.smooth = nn.Conv2d(
            prior_channels, prior_channels, 5, padding=2,
            padding_mode='replicate', groups=prior_channels, bias=False)
        nn.init.constant_(self.smooth.weight, 1. / 25.)
        self.gate = nn.Conv2d(2 * prior_channels, 3 * channels, 1)

    def forward(self, ll):
        compressed = self.reduce(ll)
        prior = self.smooth(compressed)
        comparison = torch.cat((compressed, prior - compressed), dim=1)
        return (0.5 + torch.sigmoid(self.gate(comparison))).chunk(3, dim=1)


class PriorGatedDirectionalREM(RadarEnhancementModule):
    """PGD-REM: one full-channel DWT/IDWT pair and one zero-init alpha.

    Reuse REM's LH (1x1 -> 1x3 -> 3x1 -> 3x3), HL (opposite axial
    order), and HH (1x1 -> 3x3) Conv/BN/activation chains. LL_base uses
    a plain 1x1 convolution. All four reconstruction bands keep `channels`;
    the compressed LL branch only predicts spatial, per-channel gates.
    """

    def __init__(self, channels, prior_channels=64, act='silu'):
        super().__init__(channels, act=act)
        self.ll = nn.Conv2d(channels, channels, 1)
        self.prior_gate = LowFrequencyStructureGate(channels, prior_channels)

    def forward(self, features):
        ll, lh, hl, hh = self.dwt(features)
        gamma_lh, gamma_hl, gamma_hh = self.prior_gate(ll)
        restored = self.idwt((
            self.ll(ll),
            gamma_lh * self.lh(lh),
            gamma_hl * self.hl(hl),
            gamma_hh * self.hh(hh),
        ), features.shape[-2:])
        return features + self.alpha.to(dtype=features.dtype) * restored.to(dtype=features.dtype)
