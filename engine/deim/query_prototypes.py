"""Training-only query discrimination with one or more prototypes per class.

Uses soft prototype assignment inspired by SoftTriple (ICCV 2019). The head
never replaces the detector logits or contributes to the matching cost.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .box_ops import box_cxcywh_to_xyxy
from ..misc.dist_utils import get_world_size, is_dist_available_and_initialized


class QueryPrototypeHead(nn.Module):
    def __init__(self, input_dim, num_classes, projection_dim=128,
                 prototype_counts=None, prototype_temperature=0.1,
                 class_temperature=0.1):
        super().__init__()
        self.counts = tuple(prototype_counts or [1] * num_classes)
        if len(self.counts) != num_classes or any(k < 1 for k in self.counts):
            raise ValueError('MPQ requires a positive prototype count for every class.')
        if prototype_temperature <= 0 or class_temperature <= 0:
            raise ValueError('MPQ temperatures must be positive.')
        self.norm = nn.LayerNorm(input_dim)
        self.projection = nn.Linear(input_dim, projection_dim, bias=False)
        self.prototypes = nn.Parameter(torch.empty(sum(self.counts), projection_dim))
        nn.init.normal_(self.prototypes, std=projection_dim ** -0.5)
        self.register_buffer('prototype_classes', torch.repeat_interleave(
            torch.arange(num_classes), torch.tensor(self.counts)))
        self.prototype_temperature = prototype_temperature
        self.class_temperature = class_temperature

    def forward(self, queries):
        # Cosine normalization and soft assignment remain FP32 under FP16 AMP.
        with torch.autocast(device_type=queries.device.type, enabled=False):
            embedding = F.normalize(self.projection(self.norm(queries.float())), dim=-1)
            centers = F.normalize(self.prototypes.float(), dim=-1)
            similarities = F.linear(embedding, centers)
            assignments, scores = [], []
            for similarity in similarities.split(self.counts, dim=-1):
                assignment = (similarity / self.prototype_temperature).softmax(-1)
                assignments.append(assignment)
                scores.append((assignment * similarity).sum(-1))
            return dict(pred_mpq_logits=torch.stack(scores, dim=-1) / self.class_temperature,
                        mpq_assignments=torch.cat(assignments, dim=-1),
                        mpq_prototype_classes=self.prototype_classes)


class MatchedQueryPrototypeLoss(nn.Module):
    """Class-balanced means over the global DDP batch, with detached IoU weights."""
    def __init__(self, num_classes, min_iou=0.7, start_epoch=49, ramp_epochs=5):
        super().__init__()
        if not 0 <= min_iou <= 1 or start_epoch < 0 or ramp_epochs < 0:
            raise ValueError('Invalid MPQ quality threshold or epoch schedule.')
        self.num_classes = num_classes
        self.min_iou = min_iou
        self.start_epoch = start_epoch
        self.ramp_epochs = ramp_epochs
        self.last_metrics = {}

    def ramp(self, epoch, step=0, epoch_step=1):
        progress = float(epoch) + float(step) / max(int(epoch_step), 1) - self.start_epoch
        if progress < 0:
            return 0.0
        return min(progress / self.ramp_epochs, 1.0) if self.ramp_epochs else 1.0

    def forward(self, outputs, targets, indices, epoch=0, step=0, epoch_step=1, **kwargs):
        logits = outputs['pred_mpq_logits'].float()
        zero = logits.sum() * 0.0  # Connect every head parameter even before activation.
        ramp = self.ramp(epoch, step, epoch_step)
        prototype_classes = outputs['mpq_prototype_classes']
        metrics = dict(mpq_ramp=logits.new_tensor(ramp), mpq_ce=zero.detach(),
                       mpq_valid_fraction=zero.detach())
        for c in range(self.num_classes):
            metrics[f'mpq_valid_c{c}'] = zero.detach()
        for p in range(len(prototype_classes)):
            metrics[f'mpq_usage_p{p}'] = zero.detach()
        if ramp == 0:
            self.last_metrics = metrics
            return zero

        batch = torch.cat([torch.full_like(i, b) for b, (i, _) in enumerate(indices)])
        query = torch.cat([i for i, _ in indices])
        labels = torch.cat([t['labels'][j] for t, (_, j) in zip(targets, indices)])
        gt = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)]).float()
        with torch.no_grad():
            pred = box_cxcywh_to_xyxy(outputs['pred_boxes'][batch, query].detach().float())
            gt = box_cxcywh_to_xyxy(gt)
            intersection = (torch.minimum(pred[:, 2:], gt[:, 2:]) -
                            torch.maximum(pred[:, :2], gt[:, :2])).clamp_min(0).prod(-1)
            pred_area = (pred[:, 2:] - pred[:, :2]).clamp_min(0).prod(-1)
            gt_area = (gt[:, 2:] - gt[:, :2]).clamp_min(0).prod(-1)
            iou = intersection / (pred_area + gt_area - intersection).clamp_min(1e-8)
            valid = iou >= self.min_iou
            weights = iou.square() * valid
            denominators = logits.new_zeros(self.num_classes).scatter_add_(0, labels, weights)
            counts = logits.new_zeros(self.num_classes).scatter_add_(0, labels, valid.float())
            own_assignments = outputs['mpq_assignments'][batch, query].detach()
            usage = (own_assignments * (labels[:, None] == prototype_classes[None]) *
                     valid[:, None]).sum(0)
            matched_count = logits.new_tensor(float(len(labels)))
            global_stats = torch.cat((denominators, counts, usage, matched_count[None]))
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(global_stats)
            denominators = global_stats[:self.num_classes]
            counts = global_stats[self.num_classes:2 * self.num_classes]
            usage = global_stats[2 * self.num_classes:-1]
            present = denominators > 0

        ce = F.cross_entropy(logits[batch, query], labels, reduction='none')
        numerators = logits.new_zeros(self.num_classes).scatter_add(0, labels, ce * weights)
        # DDP averages parameter gradients: undo that average for the local
        # numerators, which share globally reduced, detached denominators.
        raw_loss = ((numerators / denominators.clamp_min(1e-8)) * present).sum()
        raw_loss *= get_world_size() / present.sum().clamp_min(1)
        metrics.update(mpq_ce=raw_loss.detach(),
                       mpq_valid_fraction=counts.sum() / global_stats[-1].clamp_min(1))
        for c in range(self.num_classes):
            metrics[f'mpq_valid_c{c}'] = counts[c]
        for p, c in enumerate(prototype_classes):
            metrics[f'mpq_usage_p{p}'] = usage[p] / counts[c].clamp_min(1)
        self.last_metrics = metrics
        return raw_loss * ramp + zero
