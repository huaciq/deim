"""Imaging-annotation geometry decoupling for merged SAR detection data.

Official box = radar-response box + geometry domain translation + truncation extension.

* The detector's FDR, denoising and quality paths learn the response box. It is
  flip-equivariant, and targets stay inside the image.
* Translation is one 2-D vector per image, mixed from a few learned domains.
  Diagnostics on SARDet-100K val found a pure x translation that does not depend
  on box size or position, and per-image offsets that the image cannot predict,
  so a free per-image or per-edge regressor would only fit noise. The vector is
  defined in the unflipped frame, and its x sign follows the flip state.
* Extension is one nonnegative value per box edge, applied only to edges at the
  image border, for targets cut by tiling. It works in the current image frame,
  so flips need no special handling.

Both branches read detached features, so they cannot change the response path
through gradients. See docs/sardet100k_m1_imaging_geometry.md.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh

__all__ = ['ImagingGeometryHead', 'radiometric_statistics']


def radiometric_statistics(images):
    """Exactly flip-invariant log-intensity statistics, [B, 5]."""
    with torch.autocast(device_type=images.device.type, enabled=False):
        grey = images.float().mean(1)
        grey = F.avg_pool2d(grey.unsqueeze(1), 4, ceil_mode=True).flatten(1)
        logged = torch.log1p(grey.clamp_min(0) * 255.) / torch.log(torch.tensor(256., device=images.device))
        quantiles = torch.quantile(logged, logged.new_tensor([.1, .5, .9]), dim=1).T
        return torch.cat((logged.mean(1, keepdim=True), logged.std(1, keepdim=True, unbiased=False), quantiles), 1)


class ImagingGeometryHead(nn.Module):
    def __init__(self, hidden_dim, num_levels, num_domains=4, embed_dim=128, amodal_dim=128,
                 border_margin=.01, translation_unit=256., use_translation=True, use_amodal=True,
                 act=nn.SiLU):
        super().__init__()
        if not (use_translation or use_amodal):
            raise ValueError('Imaging geometry needs at least one of translation or extension.')
        if num_domains < 2:
            raise ValueError('Imaging geometry needs at least two domains.')
        self.use_translation = bool(use_translation)
        self.use_amodal = bool(use_amodal)
        self.border_margin = float(border_margin)
        self.translation_unit = float(translation_unit)
        if self.use_translation:
            self.domain_norm = nn.LayerNorm(hidden_dim * num_levels + 5)
            self.domain_mlp = nn.Sequential(nn.Linear(hidden_dim * num_levels + 5, embed_dim), act(),
                                            nn.Linear(embed_dim, num_domains))
            # Zero translation makes the head start exactly at the baseline.
            self.domain_translations = nn.Parameter(torch.zeros(num_domains, 2))
        if self.use_amodal:
            core = torch.linspace(-.375, .375, 3)
            yy, xx = torch.meshgrid(core, core, indexing='ij')
            inside = torch.stack((xx, yy), dim=-1).reshape(9, 2)
            along = torch.tensor([-1. / 3., 0., 1. / 3.]) * 1.05
            outer = torch.full_like(along, .525)
            ring = torch.cat((torch.stack((along, -outer), -1), torch.stack((along, outer), -1),
                              torch.stack((-outer, along), -1), torch.stack((outer, along), -1)))
            self.register_buffer('sample_offsets', torch.cat((inside, ring)), persistent=False)
            self.amodal_norm = nn.LayerNorm(3 * hidden_dim)
            self.amodal_mlp = nn.Sequential(nn.Linear(3 * hidden_dim + 8, amodal_dim), act(),
                                            nn.Linear(amodal_dim, amodal_dim), act(),
                                            nn.Linear(amodal_dim, 4))
            # softplus(-8) is about 3e-4 of the box size, so edges start in place.
            nn.init.zeros_(self.amodal_mlp[-1].weight)
            nn.init.constant_(self.amodal_mlp[-1].bias, -8.)

    def translation(self, memory, spatial_shapes, images):
        """Mixture translation in units of 1/translation_unit of the image, [B, 2]."""
        batch = memory.shape[0]
        if not self.use_translation:
            zero = memory.new_zeros(batch, 2, dtype=torch.float32)
            return zero, memory.new_zeros(batch, 1, dtype=torch.float32)
        with torch.autocast(device_type=memory.device.type, enabled=False):
            pooled, offset = [], 0
            source = memory.detach().float()
            for height, width in spatial_shapes:
                count = int(height) * int(width)
                pooled.append(source[:, offset:offset + count].mean(1))
                offset += count
            evidence = torch.cat(pooled + [radiometric_statistics(images.detach())], 1)
            probabilities = self.domain_mlp(self.domain_norm(evidence)).softmax(-1)
            return probabilities @ self.domain_translations.float(), probabilities

    @torch.no_grad()
    def response_targets(self, targets, translation):
        """Return targets whose boxes are the response box, plus the official box.

        Mosaic and MixUp samples mix several images, so they keep their boxes
        and do not join the geometry loss.
        """
        results = []
        for index, target in enumerate(targets):
            item = dict(target)
            clean = bool(target.get('geometry_clean', False)) and 'mixup' not in target
            item['geometry_clean'] = clean
            if clean:
                visible = box_cxcywh_to_xyxy(torch.as_tensor(target['boxes']).float())
                item['official_xyxy'] = visible + target['official_overflow'].float() * visible.new_tensor([-1., -1., 1., 1.])
                shift = translation[index].float() / self.translation_unit * visible.new_tensor([float(target['flip_state']), 1.])
                moved = (visible - shift.repeat(2)).clamp(0., 1.)
                size = (moved[:, 2:] - moved[:, :2]).clamp_min(1e-4)
                moved = torch.cat((moved[:, :2], moved[:, :2] + size), 1)
                item['boxes'] = box_xyxy_to_cxcywh(moved).to(dtype=target['boxes'].dtype)
            results.append(item)
        return results

    def edge_gate(self, boxes):
        """Edges within border_margin of the image border, [B, Q, 4]."""
        xyxy = box_cxcywh_to_xyxy(boxes.detach().float())
        distance = torch.cat((xyxy[..., :2], 1. - xyxy[..., 2:]), -1)
        return distance < self.border_margin

    def extension(self, queries, boxes, memory, spatial_shapes):
        """Nonnegative edge extension as a fraction of box size, gated to border edges."""
        if not self.use_amodal:
            return boxes.new_zeros(boxes.shape, dtype=torch.float32), torch.zeros_like(boxes, dtype=torch.bool)
        height, width = int(spatial_shapes[0][0]), int(spatial_shapes[0][1])
        with torch.autocast(device_type=memory.device.type, enabled=False):
            reference = boxes.detach().float()
            feature = memory[:, :height * width].detach().float().transpose(1, 2).reshape(
                memory.shape[0], memory.shape[-1], height, width)
            points = reference[..., None, :2] + reference[..., None, 2:] * self.sample_offsets.float()
            sampled = F.grid_sample(feature, 2 * points - 1, mode='bilinear', padding_mode='border', align_corners=False)
            inside = sampled[..., :9].mean(-1).transpose(1, 2)
            ring = sampled[..., 9:].mean(-1).transpose(1, 2)
            xyxy = box_cxcywh_to_xyxy(reference)
            distance = torch.cat((xyxy[..., :2], 1. - xyxy[..., 2:]), -1)
            gate = distance < self.border_margin
            content = self.amodal_norm(torch.cat((queries.detach().float(), inside, ring), -1))
            values = F.softplus(self.amodal_mlp(torch.cat((content, reference, distance), -1)))
            return values * gate.to(values.dtype), gate

    def compose(self, boxes, translation, extension):
        """Official-frame prediction from response boxes; inference is always unflipped."""
        with torch.autocast(device_type=boxes.device.type, enabled=False):
            source = boxes.float()
            xyxy = box_cxcywh_to_xyxy(source)
            size = source[..., 2:].repeat(1, 1, 2)
            xyxy = xyxy + extension.float() * size * size.new_tensor([-1., -1., 1., 1.])
            shift = (translation.float() / self.translation_unit)[:, None, :].repeat(1, 1, 2)
            return box_xyxy_to_cxcywh(xyxy + shift).to(dtype=boxes.dtype)
