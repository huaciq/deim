"""SCS: instance-wise scattering description and matched-query separation.

Descriptors represent statistics of displayed SAR intensities, not calibrated
radar cross sections. Extraction is deterministic and consumes no RNG state.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..misc.dist_utils import get_world_size, is_dist_available_and_initialized


class ScatteringSignatureHead(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 28)

    def forward(self, queries):
        return self.fc2(F.silu(self.fc1(self.norm(queries))))


@torch.no_grad()
def scattering_descriptors(images, targets, ring_scale=2.0, clip=5.0,
                           epsilon=1.0 / 255, min_background_pixels=16):
    """Return N x 28 descriptors/validity per image, before batch MixUp.

    Boxes are normalized cxcywh in the already augmented/resized image.
    The clipped ring excludes every GT box, including neighboring objects.
    P = 4x4 pooled Z / clip, H = 8-bin probability, G = 4 unsigned gradient
    orientation bins weighted by Sobel magnitude. Constant crops have G=0.
    """
    if images.device.type != 'cpu':
        raise ValueError('Extract SCS targets in the CPU collator before MixUp.')
    height, width = images.shape[-2:]
    results = []
    for image, target in zip(images, targets):
        gray = image.float().mean(0)
        boxes = target['boxes'].detach().to(dtype=torch.float32).clone()
        xyxy = torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,
                          boxes[:, :2] + boxes[:, 2:] / 2), dim=1)
        xyxy *= xyxy.new_tensor([width, height, width, height])
        descriptors = torch.zeros((len(boxes), 28), dtype=torch.float32)
        valid = torch.zeros(len(boxes), dtype=torch.bool)
        for i, (box, edges) in enumerate(zip(boxes, xyxy)):
            x1, y1, x2, y2 = edges.tolist()
            left, top = max(0, math.floor(x1)), max(0, math.floor(y1))
            right, bottom = min(width, math.ceil(x2)), min(height, math.ceil(y2))
            if right <= left or bottom <= top:
                continue
            cx, cy = float(box[0]) * width, float(box[1]) * height
            bw, bh = float(box[2]) * width, float(box[3]) * height
            rx1, ry1 = max(0, math.floor(cx - bw * ring_scale / 2)), max(0, math.floor(cy - bh * ring_scale / 2))
            rx2, ry2 = min(width, math.ceil(cx + bw * ring_scale / 2)), min(height, math.ceil(cy + bh * ring_scale / 2))
            yy, xx = torch.meshgrid(torch.arange(ry1, ry2) + .5,
                                    torch.arange(rx1, rx2) + .5, indexing='ij')
            ring = torch.ones_like(xx, dtype=torch.bool)
            for gx1, gy1, gx2, gy2 in xyxy.tolist():
                ring &= ~((xx >= gx1) & (xx < gx2) & (yy >= gy1) & (yy < gy2))
            background = gray[ry1:ry2, rx1:rx2][ring]
            if len(background) < min_background_pixels:
                continue
            median = background.median()
            scale = 1.4826 * (background - median).abs().median() + epsilon
            z = ((gray[top:bottom, left:right] - median) / scale).clamp(-clip, clip)
            spatial = F.adaptive_avg_pool2d(z[None, None], (4, 4)).flatten() / clip
            histogram = torch.histc(z, bins=8, min=-clip, max=clip) / z.numel()
            kx = z.new_tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8
            padded = F.pad(z[None, None], (1, 1, 1, 1), mode='replicate')
            gx = F.conv2d(padded, kx[None, None])[0, 0]
            gy = F.conv2d(padded, kx.t()[None, None])[0, 0]
            magnitude = torch.sqrt(gx.square() + gy.square())
            orientation = torch.remainder(torch.atan2(gy, gx), math.pi)
            bins = (orientation * (4 / math.pi)).long().clamp(0, 3)
            gradient = torch.bincount(bins.flatten(), weights=magnitude.flatten(), minlength=4)
            gradient /= gradient.sum().clamp_min(1e-8)
            descriptors[i] = torch.cat((spatial, histogram, gradient))
            valid[i] = True
        results.append((descriptors, valid))
    return results


def distributed_average_count(count):
    count = count.detach().float().clone()
    if is_dist_available_and_initialized():
        torch.distributed.all_reduce(count)
    return (count / get_world_size()).clamp_min(1)


class SCSLoss(nn.Module):
    def __init__(self, num_classes, pairs, margin=.5, beta=.1):
        super().__init__()
        weights = torch.zeros(num_classes, num_classes)
        for source, rival, weight in pairs:
            if source == rival or not 0 <= source < num_classes or not 0 <= rival < num_classes or weight <= 0:
                raise ValueError('SCS pairs require distinct valid classes and positive weights.')
            weights[source, rival] = weight
        self.register_buffer('pair_weights', weights)
        self.margin, self.beta = margin, beta

    def forward(self, outputs, targets, indices):
        predicted = outputs['pred_scs']
        logits = outputs['pred_logits']
        source_batch = torch.cat([torch.full_like(i, b) for b, (i, _) in enumerate(indices)])
        source_query = torch.cat([i for i, _ in indices])
        target_description = torch.cat([t['scs_signature'][j] for t, (_, j) in zip(targets, indices)])
        valid = torch.cat([t['scs_valid'][j] for t, (_, j) in zip(targets, indices)])
        classes = torch.cat([t['labels'][j] for t, (_, j) in zip(targets, indices)])
        selected = predicted[source_batch, source_query].float()
        sig_count = distributed_average_count(valid.sum())
        per_instance = F.smooth_l1_loss(selected, target_description.float(), beta=self.beta,
                                      reduction='none').mean(-1)
        signature_loss = (per_instance * valid).sum() / sig_count
        selected_logits = logits[source_batch, source_query].float()
        true_logits = selected_logits.gather(1, classes[:, None])
        violation = F.relu(self.margin - (true_logits - selected_logits))
        weights = self.pair_weights[classes]
        positive_count = distributed_average_count(logits.new_tensor(len(classes)))
        pair_loss = (violation * weights).sum() / positive_count
        # Empty/invalid matches keep a zero gradient path to every head parameter.
        return dict(loss_scs_sig=signature_loss + predicted.sum() * 0,
                    loss_scs_pair=pair_loss)
