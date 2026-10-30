"""White top-hat/CoV guided residual reconstruction on projected SAR features."""

import torch
from torch import nn
import torch.nn.functional as F


class SPIModule(nn.Module):
    """Scattering Physics-Informed module, independent of the older SPI isolator.

    Each projected pyramid level has its own module. Statistics and GroupNorm
    run in FP32 to keep squared features and near-zero means stable under AMP.
    CNN branches follow the surrounding autocast context. Channelwise alpha
    and beta start at zero, preserving the incoming feature and its dtype.
    """

    def __init__(self, channels=256, kernel_size=3, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.k = kernel_size
        self.pad = kernel_size // 2
        self.eps = eps
        self.norm_peak = nn.GroupNorm(32, channels)
        self.norm_cov = nn.GroupNorm(32, channels)
        self.phys_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.bridge_h = nn.Conv2d(channels, channels, (1, 5), padding=(0, 2),
                                  groups=channels, bias=False)
        self.bridge_v = nn.Conv2d(channels, channels, (5, 1), padding=(2, 0),
                                  groups=channels, bias=False)
        self.bridge_pw = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        self.alpha = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def _physical_statistics(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            values = x.float()
            eroded = -F.max_pool2d(-values, self.k, stride=1, padding=self.pad)
            opened = F.max_pool2d(eroded, self.k, stride=1, padding=self.pad)
            peak = F.relu(values - opened)
            mean = F.avg_pool2d(values.abs(), self.k, stride=1, padding=self.pad)
            mean_square = F.avg_pool2d(values.square(), self.k, stride=1, padding=self.pad)
            deviation = (F.relu(mean_square - mean.square()) + self.eps).sqrt()
            cov = deviation / (mean + self.eps)
        return peak, cov

    def forward(self, x):
        peak, cov = self._physical_statistics(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            prior = torch.cat((self.norm_peak(peak), self.norm_cov(cov)), dim=1)
        mask = self.phys_gate(prior)
        salient = x * mask.to(dtype=x.dtype)
        bridge = self.bridge_pw(self.bridge_h(salient) + self.bridge_v(salient))
        return (x + self.alpha.to(dtype=x.dtype) * salient
                + self.beta.to(dtype=x.dtype) * bridge.to(dtype=x.dtype))
