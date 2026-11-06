"""Stride-4 pyramid level built after the original neck, so the three original levels keep their COCO weights."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FineLevelNeck(nn.Module):
    """P2 = fuse(project(S2), upsample(project(P3))). Only depthwise and 1x1 convolutions at stride 4."""

    def __init__(self, s2_channels, hidden_dim, dim=128):
        super().__init__()
        self.s2_conv = nn.Conv2d(s2_channels, dim, 1, bias=False)
        self.s2_norm = nn.BatchNorm2d(dim)
        self.p3_conv = nn.Conv2d(hidden_dim, dim, 1, bias=False)
        self.p3_norm = nn.BatchNorm2d(dim)
        self.fuse_conv = nn.Conv2d(2 * dim, hidden_dim, 1, bias=False)
        self.fuse_norm = nn.BatchNorm2d(hidden_dim)
        self.local1 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False)
        self.local1_norm = nn.BatchNorm2d(hidden_dim)
        self.mix1 = nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False)
        self.mix1_norm = nn.BatchNorm2d(hidden_dim)
        self.local2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False)
        self.local2_norm = nn.BatchNorm2d(hidden_dim)
        self.out = nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False)
        self.out_norm = nn.BatchNorm2d(hidden_dim)
        self.act = nn.SiLU(inplace=True)

    def forward(self, s2, p3):
        coarse = F.interpolate(self.p3_norm(self.p3_conv(p3)), size=s2.shape[-2:], mode='bilinear', align_corners=False)
        x = self.act(self.fuse_norm(self.fuse_conv(torch.cat([self.s2_norm(self.s2_conv(s2)), coarse], 1))))
        x = x + self.act(self.mix1_norm(self.mix1(self.act(self.local1_norm(self.local1(x))))))
        x = x + self.act(self.out_norm(self.out(self.act(self.local2_norm(self.local2(x))))))
        return x
