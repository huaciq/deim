"""DFGF-inspired PACFG adapter for same-resolution detector features.

Reference: SPJFNet (AAAI 2026), https://arxiv.org/html/2508.04041,
Dual-Frequency Guidance Framework. The supplied PACFG modification has no
available source; this is an independent implementation of its frequency
guidance idea, not a reproduction of that modification or an InfSA module.

Feature priors replace image-gamma guidance and restoration supervision.
Fourier phase here describes spatial features, not complex SAR echo phase.
"""

import math

import torch
import torch.nn as nn

from .directional_subband import haar_dwt2, haar_iwt2


class PhaseAlignedCrossFrequencyGuidance(nn.Module):
    """Self-prior Fourier correction, cross-band gating and local refinement.

    A bottleneck Haar decomposition isolates LL for global Fourier processing.
    The phase step follows a circular phase difference, without differentiating
    an undefined angle at zero magnitude. Amplitude gain is bounded, and the
    pointwise frequency gate preserves conjugate symmetry for real features.
    The complete adapter runs in FP32 under AMP, then restores the input dtype.
    """

    def __init__(self, channels, bottleneck=64):
        super().__init__()
        self.reduce = nn.Conv2d(channels, bottleneck, 1, bias=False)
        self.norm = nn.GroupNorm(math.gcd(8, bottleneck), bottleneck)
        self.prior = nn.Sequential(
            nn.Conv2d(bottleneck, bottleneck, 3, padding=1, groups=bottleneck, bias=False),
            nn.SiLU(),
            nn.Conv2d(bottleneck, bottleneck, 1, bias=False),
        )
        self.frequency_gate = nn.Sequential(
            nn.Conv2d(bottleneck * 3, bottleneck, 1),
            nn.SiLU(),
            nn.Conv2d(bottleneck, bottleneck, 1),
            nn.Sigmoid(),
        )
        self.subband_gate = nn.Sequential(
            nn.Conv2d(bottleneck * 5, bottleneck, 1),
            nn.SiLU(),
            nn.Conv2d(bottleneck, bottleneck * 3, 1),
            nn.Sigmoid(),
        )
        self.local_refine = nn.Sequential(
            nn.Conv2d(bottleneck, bottleneck, 3, padding=1, groups=bottleneck, bias=False),
            nn.SiLU(),
            nn.Conv2d(bottleneck, bottleneck, 1, bias=False),
        )
        self.expand = nn.Conv2d(bottleneck, channels, 1, bias=False)
        self.alpha = nn.Parameter(torch.zeros(()))

    def _enhance_low(self, low, prior):
        spectrum = torch.fft.fft2(low, norm='ortho')
        prior_spectrum = torch.fft.fft2(prior, norm='ortho')
        amplitude, prior_amplitude = spectrum.abs(), prior_spectrum.abs()
        log_amplitude = amplitude.log1p()
        prior_log_amplitude = prior_amplitude.log1p()
        # prior * conj(current) gives the direction from current to prior.
        cross = prior_spectrum * spectrum.conj()
        denominator = (amplitude.square() + 1e-6).sqrt() * (prior_amplitude.square() + 1e-6).sqrt()
        cosine = cross.real / denominator
        sine = cross.imag / denominator
        gate = self.frequency_gate(torch.cat((log_amplitude, prior_log_amplitude, cosine), dim=1))
        gain = (.5 * gate * (prior_log_amplitude - log_amplitude).tanh()).exp()
        phase_step = (math.pi / 4) * gate * sine
        rotation = torch.complex(phase_step.cos(), phase_step.sin())
        corrected = spectrum * gain * rotation
        return torch.fft.ifft2(corrected, norm='ortho').real

    def forward(self, features):
        input_dtype = features.dtype
        with torch.autocast(device_type=features.device.type, enabled=False):
            reduced = torch.nn.functional.silu(self.norm(self.reduce(features.float())))
            bands, original_size = haar_dwt2(reduced)
            low, details = bands[:, 0], bands[:, 1:]
            enhanced_low = self._enhance_low(low, self.prior(low))
            conditions = torch.cat((low, enhanced_low, *details.abs().unbind(1)), dim=1)
            gate = self.subband_gate(conditions).reshape_as(details)
            enhanced_details = details * (.5 + gate)
            restored = haar_iwt2(torch.cat((enhanced_low[:, None], enhanced_details), dim=1), original_size)
            delta = self.expand(restored + self.local_refine(reduced))
            output = features.float() + self.alpha * delta
        return output.to(dtype=input_dtype)
