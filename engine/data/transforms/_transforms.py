"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as NF

import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

import PIL
import PIL.Image

from typing import Any, Dict, List, Optional

from .._misc import convert_to_tv_tensor, _boxes_keys
from .._misc import Image, Video, Mask, BoundingBoxes
from .._misc import SanitizeBoundingBoxes

from ...core import register
torchvision.disable_beta_transforms_warning()


RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomZoomOut = register()(T.RandomZoomOut)
RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
Resize = register()(T.Resize)
# ToImageTensor = register()(T.ToImageTensor)
# ConvertDtype = register()(T.ConvertDtype)
# PILToTensor = register()(T.PILToTensor)
SanitizeBoundingBoxes = register(name='SanitizeBoundingBoxes')(SanitizeBoundingBoxes)
RandomCrop = register()(T.RandomCrop)
Normalize = register()(T.Normalize)


@register()
class EmptyTransform(T.Transform):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register()
class PadToSize(T.Pad):
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sp = F.get_spatial_size(flat_inputs[0])
        h, w = self.size[1] - sp[0], self.size[0] - sp[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def __init__(self, size, fill=0, padding_mode='constant') -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        fill = self._fill[type(inpt)]
        padding = params['padding']
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]['padding'] = torch.tensor(self.padding)
        return outputs


@register()
class PadToCanvas(T.Transform):
    """Pad right/bottom to a fixed canvas so a later fixed Resize applies one scale to every image.

    Boxes keep their coordinates (top-left aligned) and ``orig_size`` becomes the canvas
    size, so the postprocessor maps normalized predictions back to original-image pixels.
    """
    def __init__(self, size=1500, fill=0) -> None:
        super().__init__()
        self.size = (size, size) if isinstance(size, int) else tuple(size)  # (width, height)
        self.fill = fill

    def forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        image, target, *rest = sample
        width, height = image.size if isinstance(image, PIL.Image.Image) else image.shape[-1:-3:-1]
        canvas_w, canvas_h = self.size
        if width > canvas_w or height > canvas_h:
            raise ValueError(f'image {width}x{height} exceeds canvas {canvas_w}x{canvas_h}')
        padding = [0, 0, canvas_w - width, canvas_h - height]
        image = F.pad(image, padding=padding, fill=self.fill)
        target = dict(target)
        if 'boxes' in target:
            target['boxes'] = F.pad(target['boxes'], padding=padding)
        target['orig_size'] = torch.as_tensor([canvas_w, canvas_h])
        return (image, target, *rest)


@register()
class RandomRot90(T.Transform):
    """Rotate a PIL image and its XYXY boxes counterclockwise by a multiple of 90 degrees.

    Boxes stay tight and the scale does not change, unlike rotation by an arbitrary angle.
    With a horizontal flip, angles (90, 180, 270) reach all eight orientations; (180,) reaches
    only the flips, which keep the range and azimuth axes of the SAR image.
    """
    _TRANSPOSE = {90: PIL.Image.ROTATE_90, 180: PIL.Image.ROTATE_180, 270: PIL.Image.ROTATE_270}

    def __init__(self, p=.5, angles=(90, 180, 270)) -> None:
        super().__init__()
        self.p, self.angles = p, tuple(angles)

    def forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        if float(torch.rand(1)) >= self.p:
            return sample
        image, target, *rest = sample
        angle = self.angles[int(torch.randint(len(self.angles), (1,)))]
        width, height = image.size
        rotated = image.transpose(self._TRANSPOSE[angle])
        target = dict(target)
        x0, y0, x1, y1 = target['boxes'].as_subclass(torch.Tensor).unbind(-1)
        corners = {90: (y0, width - x1, y1, width - x0),
                   180: (width - x1, height - y1, width - x0, height - y0),
                   270: (height - y1, x0, height - y0, x1)}[angle]
        target['boxes'] = convert_to_tv_tensor(torch.stack(corners, -1), key='boxes', spatial_size=rotated.size[::-1])
        target['orig_size'] = torch.as_tensor(rotated.size)
        return (rotated, target, *rest)


@register()
class SpeckleResample(T.Transform):
    """Draw a new speckle realization for a stored SAR image; boxes and structure are unchanged.

    Overlapping tiles repeat a target with identical speckle, which a detector can memorize.
    The stored values behave like amplitude, so the speckle is a Nakagami field s of mean 1
    whose shape reproduces the image's median (block mean / block std)^2, the measured ENL.
    A Lee weight k keeps strong scatterers and resamples homogeneous clutter:
    new = R * (1 + (1 - k) * (s - 1)), R = m + k * (value - m), with m the local mean.
    s is a Gaussian field blurred to the speckle correlation of the stored images (noise_sigma,
    tools/check_speckle_resampling.py), mapped through the Nakagami quantiles. The plain Lee
    weight pulls a small bright target towards m; structure_gain scales it so such pixels stay intact.
    The inside of a saturated scatterer has little variance and would be read as clutter, leaving
    a hollow ring, so k also ramps to 1 as m goes from bright_ratio to 2 * bright_ratio times the
    median local mean of the image: such returns are not fully developed speckle.
    """
    def __init__(self, p=.5, window=7, block=16, noise_sigma=.8, structure_gain=2., bright_ratio=3.) -> None:
        super().__init__()
        self.p, self.window, self.block, self.noise_sigma = p, window, block, noise_sigma
        self.structure_gain, self.bright_ratio = structure_gain, bright_ratio

    def looks(self, values):
        h, w = (s - s % self.block for s in values.shape)
        if not h or not w:
            return None
        blocks = values[:h, :w].reshape(h // self.block, self.block, w // self.block, self.block)
        blocks = blocks.permute(0, 2, 1, 3).reshape(-1, self.block * self.block)
        mean, std = blocks.mean(dim=1), blocks.std(dim=1, unbiased=False)
        good = (mean > 0) & (std > 0)
        return float(((mean[good] / std[good]) ** 2).median()) if good.any() else None

    def noise(self, height, width):
        """Zero-mean, unit-variance Gaussian field with the configured spatial correlation."""
        if self.noise_sigma <= 0:
            return torch.randn(height, width)
        radius = int(torch.ceil(torch.tensor(3 * self.noise_sigma)))
        x = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel = torch.exp(-x ** 2 / (2 * self.noise_sigma ** 2))
        kernel /= kernel.sum()
        field = torch.randn(1, 1, height + 2 * radius, width + 2 * radius)
        field = NF.conv2d(field, kernel.view(1, 1, -1, 1))
        field = NF.conv2d(field, kernel.view(1, 1, 1, -1))
        return field[0, 0] / kernel.square().sum()

    @staticmethod
    def nakagami_shape(looks):
        """Shape m with (mean / std)^2 = looks; 3.66 is m = 1 (single-look Rayleigh amplitude)."""
        shapes = torch.logspace(-.3, 2.5, 512, dtype=torch.float64)
        ratio = torch.exp(2 * (torch.lgamma(shapes + .5) - torch.lgamma(shapes))) / shapes
        ratios = ratio / (1 - ratio)
        index = torch.searchsorted(ratios, torch.tensor(looks, dtype=torch.float64)).clamp_max(len(shapes) - 1)
        return float(shapes[index])

    def speckle_field(self, looks, height, width, table_size=4096):
        """Spatially correlated Nakagami field of mean 1, from the quantiles of a sampled table."""
        shape = self.nakagami_shape(looks)
        table = torch.distributions.Gamma(shape, shape).sample((table_size,)).sqrt().sort().values
        table /= table.mean()
        position = torch.special.ndtr(self.noise(height, width)) * (table_size - 1)
        lower = position.long().clamp_max(table_size - 2)
        return torch.lerp(table[lower], table[lower + 1], position - lower)

    def resample(self, values):
        """New values for a float H x W image in 0..255; None when the looks cannot be measured."""
        looks = self.looks(values)
        if looks is None:
            return None
        batch = values[None, None]
        pad = self.window // 2
        mean = NF.avg_pool2d(batch, self.window, stride=1, padding=pad, count_include_pad=False)[0, 0]
        power = NF.avg_pool2d(batch ** 2, self.window, stride=1, padding=pad, count_include_pad=False)[0, 0]
        variation = (power - mean ** 2).clamp_min(0) / mean.square().clamp_min(1e-6)
        speckle = 1 / looks
        weight = (self.structure_gain * (1 - speckle / variation.clamp_min(1e-12)) / (1 + speckle)).clamp(0, 1)
        clutter = mean[mean > 0].median() if (mean > 0).any() else torch.tensor(1.)
        weight = weight.maximum((mean / (self.bright_ratio * clutter) - 1).clamp(0, 1))
        structure = mean + weight * (values - mean)
        field = self.speckle_field(looks, *values.shape)
        return (structure * (1 + (1 - weight) * (field - 1))).clamp(0, 255)

    def forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        if float(torch.rand(1)) >= self.p:
            return sample
        image, *rest = sample
        stored = F.pil_to_tensor(image)
        values = self.resample(stored[0].float())
        if values is None:
            return sample
        result = values.round().to(torch.uint8)[None].expand_as(stored)
        return (F.to_pil_image(result, mode=image.mode), *rest)


@register()
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


@register()
class ConvertBoxes(T.Transform):
    _transformed_types = (
        BoundingBoxes,
    )
    def __init__(self, fmt='', normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        spatial_size = getattr(inpt, _boxes_keys[1])
        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            inpt = convert_to_tv_tensor(inpt, key='boxes', box_format=self.fmt.upper(), spatial_size=spatial_size)

        if self.normalize:
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        # New torchvision v2 calls transform; older releases call _transform.
        return self._transform(inpt, params)


@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (
        PIL.Image.Image,
    )
    def __init__(self, dtype='float32', scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == 'float32':
            inpt = inpt.float()

        if self.scale:
            inpt = inpt / 255.

        inpt = Image(inpt)

        return inpt

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        # Keep both v2 hook names without changing conversion semantics.
        return self._transform(inpt, params)
