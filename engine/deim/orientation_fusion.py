"""Orientation-shared encoding with orientation evidence fusion (SARDet-100K, E013).

Diagnosis (docs/sardet100k_d16_d17_source_orientation.md, D17): the detector is locked to the image orientation it was
trained on. With frozen weights a vertical flip of the val images lowers AP from 65.3 to 38.2 and a transposition to 25.1,
for targets with and without a training near-duplicate alike, and a target that the training set only shows in another
orientation is detected no better than one it never shows.

Structure:
  1. the image is expanded into K views (itself plus flips / transpositions);
  2. ONE backbone and ONE encoder (shared weights) encode all views as one batch;
  3. each view's feature pyramid is mapped back to the original orientation and grid;
  4. per pyramid level and location, a gate computed from all K aligned features gives every channel group a softmax
     weight over the views; the fused pyramid goes to the unchanged decoder.
The orientation-specific features stay available up to the gate, so an orientation cue that is real (a stable imaging
direction) can still be used; what changes is that every view supervises the same filters.

Grid alignment. Stride-2 convolutions on even sizes and the HGNetv2 stem sample asymmetrically, so the pyramid of a flipped
image, flipped back, sits almost one cell away from the pyramid of the image itself. Shifting the flipped image by
`image_shift` pixels before encoding and the flipped-back features by `cell_shift` cells removes this. The defaults
(3 pixels, 1 cell) were calibrated on HGNetv2-B2 with flip-symmetric kernels: the backbone pyramids then agree to within
1% (36-85% without the shifts); tests/test_orientation_fusion.py checks this. The encoder's nearest-neighbour upsampling
adds a half-cell bias to the top-down path that is mirrored in a flipped view; no shift removes it, and it is left as is.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['OrientationSharedFusion']

# (transpose, flip along width, flip along height), applied to the image in this order
VIEWS = {'hflip': (0, 1, 0), 'vflip': (0, 0, 1), 'rot180': (0, 1, 1),
         'transpose': (1, 0, 0), 'rot90a': (1, 1, 0), 'rot90b': (1, 0, 1), 'antitranspose': (1, 1, 1)}


def shift(x, amount, dim):
    """Move the content of `x` by `amount` positions along `dim` (towards larger indices if positive), zero filled."""
    if amount == 0:
        return x
    pad = [0, 0, 0, 0]
    index = {-1: 0, -2: 2}[dim] + (0 if amount > 0 else 1)
    pad[index] = abs(amount)
    x = F.pad(x, pad)
    size = x.shape[dim] - abs(amount)
    return x.narrow(dim, 0 if amount > 0 else abs(amount), size)


class OrientationSharedFusion(nn.Module):
    def __init__(self, views=('vflip',), hidden_dim=256, num_levels=3, groups=8, reduction=4, init_bias=2.,
                 image_shift=3, cell_shift=1, checkpoint_views=False):
        super().__init__()
        unknown = [v for v in views if v not in VIEWS]
        if unknown or not views:
            raise ValueError(f'views must be a non-empty subset of {sorted(VIEWS)}, got {list(views)}')
        if hidden_dim % groups:
            raise ValueError('hidden_dim must be divisible by groups')
        self.views = list(views)
        self.groups = groups
        self.image_shift = image_shift
        self.cell_shift = cell_shift
        # Training memory: with gradient checkpointing every view is encoded without storing activations and re-encoded,
        # one view at a time, during the backward pass. Same gradients; one more encoder forward pass per view.
        self.checkpoint_views = checkpoint_views
        k = len(self.views) + 1
        self.gates = nn.ModuleList()
        for _ in range(num_levels):
            gate = nn.Sequential(nn.Conv2d(k * hidden_dim, hidden_dim // reduction, 1, bias=False), nn.BatchNorm2d(hidden_dim // reduction),
                                 nn.ReLU(inplace=True), nn.Conv2d(hidden_dim // reduction, k * groups, 1))
            # Start content independent, with the un-transformed view dominant: the decoder first sees nearly the
            # pyramid it was initialised for.
            nn.init.zeros_(gate[-1].weight)
            with torch.no_grad():
                gate[-1].bias.zero_()
                gate[-1].bias[:groups] = init_bias
            self.gates.append(gate)
        self.last_weights = None

    def view(self, images, name):
        t = VIEWS[name]
        x = images.transpose(-1, -2) if t[0] else images
        if t[1]:
            x = shift(x.flip(-1), self.image_shift, -1)
        if t[2]:
            x = shift(x.flip(-2), self.image_shift, -2)
        return x

    def align(self, feature, name):
        """Feature map of a view -> the orientation and grid of the un-transformed image."""
        t = VIEWS[name]
        x = feature
        if t[2]:
            x = shift(x, -self.cell_shift, -2).flip(-2)
        if t[1]:
            x = shift(x, -self.cell_shift, -1).flip(-1)
        if t[0]:
            x = x.transpose(-1, -2)
        return x

    def forward(self, images, encode):
        """`encode` maps an image batch to the list of encoder feature maps."""
        if any(VIEWS[v][0] for v in self.views) and images.shape[-1] != images.shape[-2]:
            raise ValueError('transposed views need square inputs')
        batch = images.shape[0]
        k = len(self.views) + 1
        if self.checkpoint_views and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            per_view = [checkpoint(lambda x: tuple(encode(x)), x, use_reentrant=False) for x in [images] + [self.view(images, v) for v in self.views]]
            levels = [[p[n] for p in per_view] for n in range(len(self.gates))]
        else:
            levels = [list(level.split(batch)) for level in encode(torch.cat([images] + [self.view(images, v) for v in self.views]))]
        fused, weights = [], []
        for gate, parts in zip(self.gates, levels):
            parts = [parts[0]] + [self.align(p, v) for p, v in zip(parts[1:], self.views)]
            b, c, h, w = parts[0].shape
            weight = gate(torch.cat(parts, 1)).view(b, k, self.groups, 1, h, w).softmax(1)
            stacked = torch.stack(parts, 1).view(b, k, self.groups, c // self.groups, h, w)
            fused.append((weight * stacked).sum(1).view(b, c, h, w))
            weights.append(weight.detach().mean((0, 2, 3, 4, 5)))
        self.last_weights = torch.stack(weights)   # (levels, views) mean gate weight, for monitoring
        return fused
