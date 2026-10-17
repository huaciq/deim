"""
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import torch.nn as nn
from ..core import register


__all__ = ['DEIM', ]


@register()
class DEIM(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, x, targets=None):
        x = self.backbone(x)

        # Handle encoder output (may include boundary_output for BCM)
        encoder_out = self.encoder(x)
        if isinstance(encoder_out, tuple):
            # BCM enabled: (features, boundary_map)
            x, boundary_map = encoder_out
        else:
            # BCM disabled: just features
            x = encoder_out
            boundary_map = None

        # Pass to decoder
        decoder_out = self.decoder(x, targets)

        # Attach boundary_map to decoder output for loss computation
        if boundary_map is not None and isinstance(decoder_out, dict):
            decoder_out['boundary_map'] = boundary_map

        return decoder_out

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
