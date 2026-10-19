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

        # Handle encoder output (may include boundary_output and/or spi_outputs)
        encoder_out = self.encoder(x)

        boundary_map = None
        spi_outputs = None

        if isinstance(encoder_out, tuple):
            if len(encoder_out) == 2:
                # Could be (features, boundary_map) or (features, spi_outputs)
                x, extra = encoder_out
                if isinstance(extra, dict):
                    spi_outputs = extra
                else:
                    boundary_map = extra
            elif len(encoder_out) == 3:
                # (features, boundary_map, spi_outputs)
                x, boundary_map, spi_outputs = encoder_out
        else:
            x = encoder_out

        # Pass to decoder
        decoder_out = self.decoder(x, targets)

        # Attach boundary_map and spi_outputs to decoder output for loss computation
        if isinstance(decoder_out, dict):
            if boundary_map is not None:
                decoder_out['boundary_map'] = boundary_map
            if spi_outputs is not None:
                decoder_out.update(spi_outputs)

        return decoder_out

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
