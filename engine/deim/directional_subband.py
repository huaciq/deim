"""DSG: directional subband residual feature modulation.

The fixed Haar basis, per-band processing and residual fusion follow common
wavelet-network principles (WTConv, ECCV 2024; WSF-YOLO, Remote Sensing 2026).
The task motivation also relates to PGD's soft scattering-structure guidance.
This implementation does not estimate physical scattering mechanisms or
recover calibrated reflectivity. No external implementation is copied here.

Haar helpers retain coefficient signs. LH measures row variation (horizontal
edges), HL measures column variation (vertical edges), and HH is diagonal.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def haar_dwt2(features):
    """One-level orthonormal Haar transform, returning (B,4,C,h,w), (H,W).

    A replicate pad handles an odd final row/column; the original extent is
    retained for exact reconstruction/cropping. The 4 bands are LL,LH,HL,HH.
    """
    height, width = features.shape[-2:]
    if height % 2 or width % 2:
        features = F.pad(features, (0, width % 2, 0, height % 2), mode='replicate')
    a = features[..., 0::2, 0::2]
    b = features[..., 0::2, 1::2]
    c = features[..., 1::2, 0::2]
    d = features[..., 1::2, 1::2]
    bands = torch.stack(((a + b + c + d) * .5,
                         (a + b - c - d) * .5,
                         (a - b + c - d) * .5,
                         (a - b - c + d) * .5), dim=1)
    return bands, (height, width)


def haar_iwt2(bands, original_size=None):
    """Invert signed (B,4,C,h,w) coefficients; optionally crop to (H,W)."""
    ll, lh, hl, hh = bands.unbind(1)
    a = (ll + lh + hl + hh) * .5
    b = (ll + lh - hl - hh) * .5
    c = (ll - lh + hl - hh) * .5
    d = (ll - lh - hl + hh) * .5
    batch, channels, height, width = ll.shape
    output = torch.stack((a, b, c, d), dim=-1)
    output = output.reshape(batch, channels, height, width, 2, 2)
    output = output.permute(0, 1, 2, 4, 3, 5).reshape(batch, channels, height * 2, width * 2)
    if original_size is not None:
        output = output[..., :original_size[0], :original_size[1]]
    return output


class DirectionalSubbandGate(nn.Module):
    """Same-shape stride-8 adapter with explicitly isolated ablation modes.

    ``directional`` uses separate LH/HL/HH channel-spatial gates; ``shared``
    averages those gates before using the same gate on all three directions.
    ``ll_only`` processes only LL (including its gate conditions), with zero
    HF coefficients at reconstruction. Its four filters all process LL so
    it has the same parameter budget and no unused trainable branch.
    ``spatial`` uses the same filters/gate on the full spatial grid without
    DWT/IWT; this matches parameter count, not operation count or latency.

    All projections are nonzero-initialized; alpha is the sole zero-initialized
    residual multiplier. At alpha=0 this is an exact identity, and alpha can
    learn on the first update. Subband/energy arithmetic and this adapter's
    projections run in FP32 under AMP; output retains the input dtype.
    """
    MODES = ('directional', 'shared', 'll_only', 'spatial')

    def __init__(self, channels=256, bottleneck=64, mode='directional', alpha_init=0.0):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f'Unsupported DSG mode {mode!r}; choose one of {self.MODES}.')
        if channels <= 0 or bottleneck <= 0:
            raise ValueError('DSG channel dimensions must be positive.')
        self.channels = channels
        self.bottleneck = bottleneck
        self.mode = mode
        self.reduce = nn.Conv2d(channels, bottleneck, 1, bias=False)
        self.norm = nn.GroupNorm(math.gcd(8, bottleneck), bottleneck)
        self.band_filters = nn.ModuleList([
            # No HF bias: a zero detail band must not acquire a synthetic
            # periodic component merely from a learned constant offset.
            nn.Conv2d(bottleneck, bottleneck, 3, padding=1, groups=bottleneck, bias=False)
            for _ in range(4)
        ])
        gate_hidden = max(1, bottleneck * 3 // 2)
        self.gate = nn.Sequential(
            nn.Conv2d(bottleneck + 3, gate_hidden, 1),
            nn.SiLU(),
            nn.Conv2d(gate_hidden, bottleneck * 3, 1),
        )
        self.expand = nn.Conv2d(bottleneck, channels, 1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def _gates(self, low, details):
        # details: B,3,C,H,W. Gate conditions preserve orientation while using
        # relative energies to avoid equating high amplitude with foreground.
        batch, directions, channels, height, width = details.shape
        energy = details.square().mean(2)
        energy = F.avg_pool2d(energy, 3, stride=1, padding=1)
        energy = energy / energy.mean(1, keepdim=True).clamp_min(1e-6)
        gates = self.gate(torch.cat((low, energy), dim=1))
        gates = gates.reshape(batch, directions, channels, height, width).sigmoid()
        if self.mode == 'shared':
            gates = gates.mean(1, keepdim=True).expand_as(gates)
        return gates

    def forward(self, features):
        input_dtype = features.dtype
        # Do not clamp, mask or replace nonfinite inputs: the training engine
        # must still observe numerical failures and apply its existing policy.
        with torch.autocast(device_type=features.device.type, enabled=False):
            reduced = F.silu(self.norm(self.reduce(features.float())))
            if self.mode == 'spatial':
                branches = [operation(reduced) for operation in self.band_filters]
                details = torch.stack(branches[1:], dim=1)
                gates = self._gates(branches[0], details)
                restored = branches[0] + (gates * details).sum(1) / math.sqrt(3.)
            else:
                bands, original_size = haar_dwt2(reduced)
                if self.mode == 'll_only':
                    # No HF values enter this mode, including gate statistics.
                    branches = [operation(bands[:, 0]) for operation in self.band_filters]
                    details = torch.stack(branches[1:], dim=1)
                    gates = self._gates(branches[0], details)
                    low = branches[0] + (gates * details).sum(1) / math.sqrt(3.)
                    filtered = torch.stack((low, torch.zeros_like(low),
                                            torch.zeros_like(low), torch.zeros_like(low)), dim=1)
                else:
                    low = self.band_filters[0](bands[:, 0])
                    details = torch.stack([operation(bands[:, i + 1])
                                           for i, operation in enumerate(self.band_filters[1:])], dim=1)
                    gates = self._gates(bands[:, 0], bands[:, 1:])
                    filtered = torch.cat((low[:, None], gates * details), dim=1)
                restored = haar_iwt2(filtered, original_size)
            delta = self.expand(restored)
            output = features.float() + self.alpha * delta
        return output.to(dtype=input_dtype)
