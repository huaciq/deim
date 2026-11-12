"""E013-v2: align and fuse projected orientation evidence before one encoder body.

The two views share HGNetv2 and the existing 1x1 projections. AIFI and FPN/PAN
operate once in the original image frame. This retains orientation-dependent
responses without imposing flip invariance or duplicating cross-scale mixing.

HGNetv2-B2's sampling compensation is fixed by geometry/calibration, not val AP:
three input pixels and one feature cell. It aligns the dominant stem path;
the stem's other path has a different centre, so this is approximate, not an
exact equivariance claim. Zero-filled aligned cells cannot compete in the gate.
"""
from contextlib import contextmanager, nullcontext

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .orientation_fusion import OrientationSharedFusion


@contextmanager
def preserve_batchnorm_buffers(modules):
    """Use disposable BN/SyncBN buffers during activation recomputation.

    Batch statistics, training mode, affine gradients and SyncBN collectives
    remain unchanged. Swap buffer references instead of restoring in-place:
    tensors saved by autograd must not have their version counters modified.
    The original running state has already been updated by the real forwards.
    """
    saved, visited = [], set()
    try:
        for root in modules:
            for module in root.modules():
                if id(module) in visited or not isinstance(module, nn.modules.batchnorm._BatchNorm):
                    continue
                visited.add(id(module))
                for name in ('running_mean', 'running_var', 'num_batches_tracked'):
                    buffer = getattr(module, name)
                    if buffer is not None:
                        saved.append((module, name, buffer))
                        setattr(module, name, buffer.detach().clone())
        yield
    finally:
        for module, name, buffer in saved:
            setattr(module, name, buffer)


class ProjectedOrientationFusion(OrientationSharedFusion):
    def __init__(self, views=('vflip',), hidden_dim=256, num_levels=3, groups=8,
                 reduction=4, init_bias=2., checkpoint_views=True):
        if tuple(views) != ('vflip',):
            raise ValueError('E013-v2 uses the original image and one vertical-flip view')
        super().__init__(views=views, hidden_dim=hidden_dim, num_levels=num_levels,
                         groups=groups, reduction=reduction, init_bias=init_bias,
                         image_shift=3, cell_shift=1, checkpoint_views=checkpoint_views)
        self.last_metrics = {}

    def _contexts(self, modules):
        return nullcontext(), preserve_batchnorm_buffers(modules)

    @torch.no_grad()
    def _record_weights(self, weight, level, targets):
        # These are descriptive diagnostics, not an acceptance threshold.
        name = f'orientation_vflip_p{level + 3}'
        self.last_metrics[name] = weight[:, 1].float().mean()
        if targets is not None:
            h, w = weight.shape[-2:]
            per_location = weight[:, 1, :, 0].float().mean(1)
            sampled = []
            for batch_index, target in enumerate(targets):
                boxes = target['boxes']  # training cxcywh, normalised to the image
                if len(boxes):
                    ix = (boxes[:, 0] * w).long().clamp(0, w - 1)
                    iy = (boxes[:, 1] * h).long().clamp(0, h - 1)
                    sampled.append(per_location[batch_index, iy, ix])
            self.last_metrics[name + '_gt_center'] = (
                torch.cat(sampled).mean() if sampled else per_location.new_zeros(()))

    def forward(self, images, encode, checkpoint_modules=(), targets=None):
        """Encode views sequentially in BOTH modes; only storage/recompute varies.

        Each real forward contributes once to BN running statistics (two updates
        per training batch). In DDP, SyncBN synchronises each view across ranks.
        The global number of supervised images and all detection losses stay fixed.
        """
        def encode_tuple(view):
            return tuple(encode(view))

        per_view = []
        for view in (images, self.view(images, 'vflip')):
            if self.checkpoint_views and self.training and torch.is_grad_enabled():
                if not checkpoint_modules:
                    raise ValueError('Pass the encoded modules to preserve their BN state on recomputation')
                features = checkpoint(
                    encode_tuple, view, use_reentrant=False,
                    context_fn=lambda: self._contexts(checkpoint_modules))
            else:
                features = encode_tuple(view)
            per_view.append(features)

        fused, weights = [], []
        self.last_metrics = {}
        for level, gate in enumerate(self.gates):
            original, turned = per_view[0][level], per_view[1][level]
            aligned = self.align(turned, 'vflip')
            b, c, h, w = original.shape
            # Mask the artificial row created by the feature-grid correction.
            # This is support validity, not a claim that border receptive fields
            # are unaffected by the ordinary padding of the backbone.
            valid_flip = self.align(turned.new_ones((1, 1, h, w)), 'vflip').bool()
            valid = torch.stack((torch.ones_like(valid_flip), valid_flip), dim=1)
            valid = valid.view(1, 2, 1, 1, h, w)
            logits = gate(torch.cat((original, aligned), dim=1))
            logits = logits.view(b, 2, self.groups, 1, h, w).float()
            weight = logits.masked_fill(~valid, -torch.inf).softmax(1)
            values = torch.stack((original, aligned), dim=1).view(b, 2, self.groups, c // self.groups, h, w)
            fused.append((weight.to(values.dtype) * values).sum(1).reshape(b, c, h, w))
            weights.append(weight.detach().mean((0, 2, 3, 4, 5)))
            self._record_weights(weight.detach(), level, targets)
        self.last_weights = torch.stack(weights)
        return fused
