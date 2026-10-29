"""PAFP adapter, independently implemented from the supplied modification.

The low-frequency-guided purification backbone follows LFP in NS-FPN,
Seeing Through the Noise (CVPR 2026), Sec. 3.1:
https://arxiv.org/html/2508.06878

Spatial/channel priors, soft adaptive thresholds, a Gaussian expert mixture
and residual calibration implement the supplied PAFP idea. This is not a
claim to reproduce unavailable third-party PAFP source or the full NS-FPN.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .directional_subband import haar_dwt2, haar_iwt2


class GaussianExpert(nn.Module):
    """Normalized depthwise Gaussian filter with a positive learned sigma."""

    def __init__(self, kernel_size, sigma):
        super().__init__()
        coordinate = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        self.register_buffer('squared_radius', coordinate[:, None].square() + coordinate[None, :].square())
        self.raw_sigma = nn.Parameter(torch.tensor(math.log(math.expm1(sigma - .25))))
        self.padding = kernel_size // 2

    def kernel(self):
        sigma = F.softplus(self.raw_sigma) + .25
        weight = (-self.squared_radius / (2 * sigma.square())).exp()
        return weight / weight.sum()

    def forward(self, features):
        channels = features.shape[1]
        weight = self.kernel()[None, None].expand(channels, 1, -1, -1).contiguous()
        padded = F.pad(features, (self.padding,) * 4, mode='replicate')
        return F.conv2d(padded, weight, groups=channels)


class PriorAdaptiveFrequencyPurification(nn.Module):
    """Same-shape purification with low-frequency-conditioned controls.

    Adaptive thresholds act on per-sample/per-band RMS-normalized magnitudes.
    A differentiable sigmoid retains confident detail coefficients while a
    low-frequency-selected Gaussian mixture smooths less confident ones.
    Signed coefficients are retained throughout. Alpha alone starts at zero;
    wavelet/filter arithmetic runs in FP32 under AMP and returns input dtype.
    """

    def __init__(self, channels, bottleneck=64):
        super().__init__()
        self.bottleneck = bottleneck
        self.reduce = nn.Conv2d(channels, bottleneck, 1, bias=False)
        self.norm = nn.GroupNorm(math.gcd(8, bottleneck), bottleneck)
        self.spatial_prior = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3), nn.Sigmoid())
        hidden = max(8, bottleneck // 4)
        self.channel_prior = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(bottleneck, hidden, 1), nn.SiLU(),
            nn.Conv2d(hidden, bottleneck * 3, 1), nn.Sigmoid(),
        )
        self.threshold_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(bottleneck, hidden, 1), nn.SiLU(),
            nn.Conv2d(hidden, 3, 1), nn.Sigmoid(),
        )
        self.expert_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(bottleneck, hidden, 1), nn.SiLU(),
            nn.Conv2d(hidden, 3, 1),
        )
        self.gaussians = nn.ModuleList([
            GaussianExpert(3, .6), GaussianExpert(5, 1.), GaussianExpert(7, 1.4),
        ])
        self.refine = nn.Sequential(
            nn.Conv2d(bottleneck, bottleneck, 3, padding=1, groups=bottleneck, bias=False),
            nn.SiLU(), nn.Conv2d(bottleneck, bottleneck, 1, bias=False),
        )
        self.expand = nn.Conv2d(bottleneck, channels, 1, bias=False)
        self.alpha = nn.Parameter(torch.zeros(()))

    def _purify_high(self, low, details):
        batch, _, channels, height, width = details.shape
        spatial = self.spatial_prior(torch.cat((low.mean(1, keepdim=True), low.amax(1, keepdim=True)), dim=1))
        channel = self.channel_prior(low).reshape(batch, 3, channels, 1, 1)
        modulated = details * spatial[:, None] * channel
        threshold = (.1 + .9 * self.threshold_head(low)).reshape(batch, 3, 1, 1, 1)
        rms = (modulated.square().mean((2, 3, 4), keepdim=True) + 1e-6).sqrt()
        keep = ((modulated.abs() / rms - threshold) * 10.).sigmoid()
        mixture = self.expert_head(low).softmax(1)
        flattened = modulated.reshape(batch * 3, channels, height, width)
        smoothed = torch.zeros_like(modulated)
        for index, expert in enumerate(self.gaussians):
            filtered = expert(flattened).reshape_as(modulated)
            weight = mixture[:, index:index + 1, :, :].unsqueeze(1)
            smoothed = smoothed + weight * filtered
        return keep * modulated + (1 - keep) * smoothed

    def forward(self, features):
        input_dtype = features.dtype
        with torch.autocast(device_type=features.device.type, enabled=False):
            reduced = F.silu(self.norm(self.reduce(features.float())))
            bands, original_size = haar_dwt2(reduced)
            low = bands[:, 0]
            purified = self._purify_high(low, bands[:, 1:])
            restored = haar_iwt2(torch.cat((low[:, None], purified), dim=1), original_size)
            delta = self.expand(restored - reduced + self.refine(restored))
            output = features.float() + self.alpha * delta
        return output.to(dtype=input_dtype)
