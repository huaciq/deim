"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
import torchvision

import copy

from .dfine_utils import bbox2distance, weighting_function
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou, wasserstein_distance_normalized
from .sar_boundary import generate_gt_boundary_map
from .background_rejection import SBRLoss
from .boundary_refinement import BRLCriterion
from .scattering_signature import SCSLoss
from .query_prototypes import MatchedQueryPrototypeLoss
from .wasserstein_fdr import WassersteinFDRLoss
from ..misc.dist_utils import get_world_size, is_dist_available_and_initialized
from ..core import register


def edge_tolerant_iou(src_boxes, target_boxes, tolerance_px, input_size):
    """Paired IoU after forgiving up to `tolerance_px` input pixels of error on each edge.

    Boxes are normalized cxcywh; input_size is (H, W) of the batch tensor. Each predicted edge
    moves toward its target edge by min(|error|, tolerance), so edges within the tolerance count
    as exact. Large boxes are barely affected; boxes near the pixel limit are no longer scored
    low only for being small.
    """
    source = box_cxcywh_to_xyxy(src_boxes.float())
    target = box_cxcywh_to_xyxy(target_boxes.float())
    height, width = float(input_size[0]), float(input_size[1])
    epsilon = source.new_tensor([tolerance_px / width, tolerance_px / height] * 2)
    error = source - target
    source = source - error.sign() * torch.minimum(error.abs(), epsilon)
    lt = torch.maximum(source[:, :2], target[:, :2])
    rb = torch.minimum(source[:, 2:], target[:, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area_s = (source[:, 2:] - source[:, :2]).clamp(min=0).prod(-1)
    area_t = (target[:, 2:] - target[:, :2]).clamp(min=0).prod(-1)
    return inter / (area_s + area_t - inter).clamp(min=1e-12)


@register()
class DEIMCriterion(nn.Module):
    """ This class computes the loss for DEIM.
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, \
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        num_classes=80,
        reg_max=32,
        boxes_weight_format=None,
        share_matched_indices=False,
        mal_alpha=None,
        use_uni_set=True,
        sbr_neg_iou_threshold=0.3,
        sbr_ignore_iou_threshold=0.5,
        sbr_bg_label_metric='iou',
        use_scs=False,
        scs_pairs=None,
        scs_margin=0.5,
        scs_smooth_l1_beta=0.1,
        use_mpq=False,
        mpq_min_iou=0.7,
        mpq_start_epoch=49,
        mpq_ramp_epochs=5,
        use_sdcp=False,
        sdcp_margin=0.2,
        sdcp_epsilon_px=1.0,
        sdcp_lambda_over=0.2,
        use_wfdr=False,
        wfdr_weight=0.5,
        use_bccq=False,
        mal_edge_tolerance_px=0.,
        log_matching_stats=False,
        matching_stats_class=None,
        ):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals.
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            num_classes: number of object categories, omitting the special no-object category.
            reg_max (int): Max number of the discrete bins in D-FINE.
            boxes_weight_format: format for boxes weight (iou, ).
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.reg_max = reg_max
        self.num_pos, self.num_neg = None, None
        self.mal_alpha = mal_alpha
        self.use_uni_set = use_uni_set

        # SBR background-label thresholds (must match the DFINETransformer values)
        self.sbr_neg_iou_threshold = sbr_neg_iou_threshold
        self.sbr_ignore_iou_threshold = sbr_ignore_iou_threshold
        self.sbr_bg_label_metric = sbr_bg_label_metric

        # SBR loss (optional)
        self.sbr_loss = SBRLoss(loss_weight=1.0, focal_alpha=0.25, focal_gamma=2.0)

        # BRL criterion (optional)
        self.brl_criterion = BRLCriterion(loss_weight=2.0)
        self.scs_loss = SCSLoss(num_classes, scs_pairs or [], scs_margin, scs_smooth_l1_beta) if use_scs else None
        self.mpq_loss = MatchedQueryPrototypeLoss(
            num_classes, mpq_min_iou, mpq_start_epoch, mpq_ramp_epochs) if use_mpq else None
        self.use_sdcp = use_sdcp
        self.sdcp_margin = float(sdcp_margin)
        self.sdcp_epsilon_px = float(sdcp_epsilon_px)
        self.sdcp_lambda_over = float(sdcp_lambda_over)
        if use_sdcp and (sdcp_epsilon_px <= 0 or sdcp_lambda_over < 0):
            raise ValueError('SDCP requires positive pixel tolerance and nonnegative over-width penalty.')
        self.use_wfdr = use_wfdr
        self.wfdr_weight = float(wfdr_weight)
        if use_wfdr and 'loss_wfdr' not in weight_dict:
            raise ValueError('W-FDR requires an explicit loss_wfdr entry in weight_dict (normally loss_fgl weight).')
        self.wfdr_loss = WassersteinFDRLoss(reg_max, wfdr_weight) if use_wfdr else None
        self.use_bccq = use_bccq
        # Q-MAL: forgive this many input pixels of error per box edge before the MAL quality target.
        # 0 keeps the original IoU target exactly.
        if mal_edge_tolerance_px < 0:
            raise ValueError('mal_edge_tolerance_px must be nonnegative.')
        self.mal_edge_tolerance_px = float(mal_edge_tolerance_px)
        self.input_size = None
        self.bccq_metrics = {}
        # Logging only: per-batch matching statistics of the actual training assignment (no gradients, no RNG).
        self.log_matching_stats = bool(log_matching_stats)
        self.matching_stats_class = None if matching_stats_class is None else int(matching_stats_class)
        self.matching_metrics = {}
        if use_bccq and (use_sdcp or 'mal' not in losses or
                         not {'loss_bccq_type', 'loss_bccq_quality'}.issubset(weight_dict)):
            raise ValueError('BCCQ requires mal dispatch, explicit type/quality weights and no SDCP.')

    def loss_grea_evidence(self, outputs, targets, indices, num_boxes):
        """Focal classification of target evidence on the final-layer matches."""
        with torch.autocast(device_type=outputs['pred_logits'].device.type, enabled=False):
            logits = outputs['grea_evidence_logits'].float()
            idx = self._get_src_permutation_idx(indices)
            labels = torch.cat([t['labels'][j] for t, (_, j) in zip(targets, indices)])
            wanted = torch.zeros_like(logits)
            wanted[idx[0], idx[1], labels] = 1.
            loss = torchvision.ops.sigmoid_focal_loss(logits, wanted, self.alpha, self.gamma, reduction='none')
            return loss.mean(1).sum() * logits.shape[1] / num_boxes

    @torch.no_grad()
    def _record_grea_metrics(self, outputs, targets, indices):
        """Per-class gate sums for matched queries; divide by the positive count for a mean."""
        idx = self._get_src_permutation_idx(indices)
        labels = torch.cat([t['labels'][j] for t, (_, j) in zip(targets, indices)])
        gates = outputs['grea_gate'][idx].float()
        stats = gates.new_zeros(self.num_classes, 4)
        stats.index_add_(0, labels, torch.cat((gates, torch.ones_like(gates[:, :1])), -1))
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(stats)
        metrics = {}
        for category in range(self.num_classes):
            metrics[f'grea_positive_c{category}'] = stats[category, 3]
            for position, name in enumerate(('target', 'clutter', 'contrast')):
                metrics[f'grea_gate_sum_c{category}_{name}'] = stats[category, position]
        for layer, value in enumerate(outputs['grea_alpha']):
            metrics[f'grea_alpha_{layer}'] = value
        self.bccq_metrics = metrics

    def loss_imaging_geometry(self, outputs, targets, indices):
        """Supervise translation and edge extension on final-layer matches of clean samples.

        Inputs are detached, so gradients reach only the geometry head. Smooth L1
        with a small beta gives a robust translation estimate under heavy-tailed
        label noise. Truncated and border boxes do not vote for the translation.
        """
        with torch.autocast(device_type=outputs['pred_boxes'].device.type, enabled=False):
            translation = outputs['geometry_translation'].float()
            extension = outputs['geometry_extension'].float()
            gate = outputs['geometry_edge_gate']
            unit = outputs['geometry_translation_unit']
            boxes = outputs['pred_boxes'].detach().float()
            translation_terms, extension_terms = [], []
            for image, (source, target_index) in enumerate(indices):
                target = targets[image]
                if not target.get('geometry_clean', False) or not len(source):
                    continue
                predicted = box_cxcywh_to_xyxy(boxes[image, source])
                response = box_cxcywh_to_xyxy(torch.as_tensor(target['boxes'])[target_index].float())
                official = target['official_xyxy'][target_index].float()
                overflow = target['official_overflow'][target_index].float()
                good = torch.diag(box_iou(predicted, response)[0]) >= .5
                edges = gate[image, source]
                interior = good & ~edges.any(-1) & (overflow.max(-1).values <= 1e-6)
                if interior.any():
                    residual = ((official[interior, :2] + official[interior, 2:]) -
                                (predicted[interior, :2] + predicted[interior, 2:])) / 2 * unit
                    residual = residual * residual.new_tensor([float(target['flip_state']), 1.])
                    translation_terms.append(F.smooth_l1_loss(
                        translation[image].expand_as(residual), residual, beta=.5, reduction='none').sum(-1))
                if good.any():
                    size = (predicted[good, 2:] - predicted[good, :2]).clamp_min(1e-4).repeat(1, 2)
                    wanted = (overflow[good] / size).clamp(0., 1.5)
                    selected = edges[good]
                    if selected.any():
                        extension_terms.append(F.l1_loss(extension[image, source][good][selected], wanted[selected], reduction='none'))
            connected = translation.sum() * 0. + extension.sum() * 0.
            translation_loss = torch.cat(translation_terms).mean() if translation_terms else translation.new_zeros(())
            extension_loss = torch.cat(extension_terms).mean() if extension_terms else extension.new_zeros(())
        return {'loss_geo_translation': translation_loss + connected, 'loss_geo_extension': extension_loss + connected}

    def loss_bccq(self, outputs, targets, indices, num_boxes, **kwargs):
        """Unweighted positive type CE and scalar MAL-style IoU-quality supervision.

        Each layer uses its own original classification matches, never the box
        matching union. DN negatives/padding receive quality=0 and no type CE.
        num_boxes is the global GT average per rank (times DN groups for DN).
        """
        types = outputs['bccq_type_logits'].float()
        quality = outputs['bccq_quality_logits'].float()
        idx = self._get_src_permutation_idx(indices)
        labels = torch.cat([t['labels'][j] for t, (_, j) in zip(targets, indices)])
        with torch.autocast(device_type=quality.device.type, enabled=False):
            with torch.no_grad():
                boxes = outputs['pred_boxes'][idx].detach().float()
                gt = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)]).float()
                iou = torch.diag(box_iou(box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(gt))[0]).clamp(0., 1.)
                target = torch.zeros_like(quality)
                target[idx] = iou.pow(self.gamma).unsqueeze(-1)
                probability = quality.detach().sigmoid()
                weight = probability.pow(self.gamma) * (1. if self.mal_alpha is None else self.mal_alpha)
                weight[idx] = 1.
            type_loss = F.cross_entropy(types[idx], labels, reduction='sum') if labels.numel() else types.sum() * 0.
            quality_loss = F.binary_cross_entropy_with_logits(quality, target, weight=weight, reduction='sum')
        return {'loss_bccq_type': type_loss / num_boxes, 'loss_bccq_quality': quality_loss / num_boxes}

    @torch.no_grad()
    def _record_matching_metrics(self, outputs, targets, indices, cached_indices, cached_indices_enc, indices_go):
        """Statistics of the assignment actually used in this training batch. Logging only.

        Stages: encoder Top-K, decoder-1 pre head, decoder layers 1..L (the last is the main output).
        Every value is a sum over the batch (all-reduced across ranks); dividing the epoch mean of a
        sum by the epoch mean of its count gives an instance-weighted ratio. Keys ending in _tank are
        restricted to GTs of matching_stats_class.
          gt / union             counts of GTs and of (query, GT) pairs in the union set used by box losses
          iou_<stage>, iou50_, iou75_   matched-query IoU (sum, and counts at >= .5 / >= .75)
          same_dec_all           GTs matched to one query at every decoder layer
          same_dec_<k>_<k+1>     GTs matched to the same query at consecutive decoder layers
          boxpos_clsbg_dec<k>    union queries not matched at decoder layer k: box supervision but MAL background
          clsbg_run2 / clsbg_all union queries that are background at >= 2 consecutive / all decoder layers
        """
        decoder = [ind for ind in cached_indices[:len(outputs['aux_outputs'])]] + [indices]
        stages = [('enc', outputs['enc_aux_outputs'][0]['pred_boxes'], cached_indices_enc[0])] if cached_indices_enc else []
        if 'pre_outputs' in outputs:
            stages.append(('pre', outputs['pre_outputs']['pred_boxes'], cached_indices[-1]))
        boxes = [aux['pred_boxes'] for aux in outputs['aux_outputs']] + [outputs['pred_boxes']]
        stages += [(f'dec{k + 1}', boxes[k], decoder[k]) for k in range(len(decoder))]
        suffixes = [''] if self.matching_stats_class is None else ['', '_tank']
        names = []
        for s in suffixes:
            names += [f'gt{s}', f'union{s}', f'same_dec_all{s}', f'clsbg_run2{s}', f'clsbg_all{s}']
            names += [f'same_dec_{k + 1}_{k + 2}{s}' for k in range(len(decoder) - 1)]
            names += [f'boxpos_clsbg_dec{k + 1}{s}' for k in range(len(decoder))]
            for stage, _, _ in stages:
                names += [f'iou_{stage}{s}', f'iou50_{stage}{s}', f'iou75_{stage}{s}']
        position = {name: n for n, name in enumerate(names)}
        stats = outputs['pred_boxes'].new_zeros(len(names), dtype=torch.float32)

        def add(name, masks, values):
            for s, mask in zip(suffixes, masks):
                stats[position[name + s]] += values[mask].float().sum() if torch.is_tensor(values) else float(values) * float(mask)

        for b, target in enumerate(targets):
            labels = target['labels']
            count = len(labels)
            if count == 0:
                continue
            groups = [torch.ones_like(labels, dtype=torch.bool)]
            if self.matching_stats_class is not None:
                groups.append(labels == self.matching_stats_class)
            ones = torch.ones(count, device=labels.device)
            add('gt', groups, ones)
            gt_boxes = box_cxcywh_to_xyxy(target['boxes'].float())
            query_of = {}
            for stage, pred, ind in stages:
                q = torch.full((count,), -1, dtype=torch.long, device=labels.device)
                q[ind[b][1].to(labels.device)] = ind[b][0].to(labels.device)
                query_of[stage] = q
                iou = torch.diag(box_iou(box_cxcywh_to_xyxy(pred[b][q].float()), gt_boxes)[0])
                add(f'iou_{stage}', groups, iou)
                add(f'iou50_{stage}', groups, (iou >= .5).float())
                add(f'iou75_{stage}', groups, (iou >= .75).float())
            layers = [query_of[f'dec{k + 1}'] for k in range(len(decoder))]
            same = torch.stack([layers[k] == layers[0] for k in range(len(layers))]).all(0)
            add('same_dec_all', groups, same.float())
            for k in range(len(layers) - 1):
                add(f'same_dec_{k + 1}_{k + 2}', groups, (layers[k] == layers[k + 1]).float())
            union_q, union_g = indices_go[b][0].to(labels.device), indices_go[b][1].to(labels.device)
            union_groups = [torch.ones_like(union_q, dtype=torch.bool)]
            if self.matching_stats_class is not None:
                union_groups.append(labels[union_g] == self.matching_stats_class)
            add('union', union_groups, torch.ones(len(union_q), device=labels.device))
            background = torch.stack([~torch.isin(union_q, decoder[k][b][0].to(labels.device)) for k in range(len(decoder))])
            for k in range(len(decoder)):
                add(f'boxpos_clsbg_dec{k + 1}', union_groups, background[k].float())
            run2 = (background[:-1] & background[1:]).any(0) if len(decoder) > 1 else background[0]
            add('clsbg_run2', union_groups, run2.float())
            add('clsbg_all', union_groups, background.all(0).float())
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(stats)
        # One device-to-host copy per step; the training loop reads plain floats.
        self.matching_metrics = {'match_' + name: value for name, value in zip(names, stats.tolist())}

    @torch.no_grad()
    def _record_bccq_metrics(self, outputs, targets, indices):
        from .band_conditioned_queries import SOURCE_NAMES
        idx = self._get_src_permutation_idx(indices)
        labels = torch.cat([t['labels'][j] for t, (_, j) in zip(targets, indices)])
        gates = outputs['bccq_source_weights'][idx][torch.arange(len(labels), device=labels.device), labels]
        entropy = outputs['bccq_source_entropy'][idx][torch.arange(len(labels), device=labels.device), labels]
        correct = outputs['bccq_type_logits'][idx].float().softmax(-1).gather(1, labels[:, None]).flatten()
        values = torch.cat((gates.float(), entropy[:, None].float(), correct[:, None],
                            torch.ones_like(correct[:, None])), -1)
        stats = values.new_zeros(self.num_classes, len(SOURCE_NAMES) + 3)
        stats.index_add_(0, labels, values)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(stats)
        metrics = {}
        for category in range(self.num_classes):
            count = stats[category, -1]
            metrics[f'bccq_positive_c{category}'] = count
            # Epoch means of these sums divided by the epoch mean count yield
            # instance-weighted statistics, including batches where a class is absent.
            for source, value in zip(SOURCE_NAMES, stats[category]):
                metrics[f'bccq_gate_sum_c{category}_{source}'] = value
            metrics[f'bccq_gate_entropy_sum_c{category}'] = stats[category, -3]
            metrics[f'bccq_correct_probability_sum_c{category}'] = stats[category, -2]
        self.bccq_metrics = metrics

    def loss_sdcp_margin(self, outputs, targets, indices):
        """Mean squared probability-margin hinge on final matched ordinary queries.

        Normalize by the global positive count (not GT count or union matches).
        The world-size factor compensates DDP's averaged gradients, including
        a rank with zero positives. This adds one collective per training step.
        """
        logits = outputs['sdcp_logits']
        idx = self._get_src_permutation_idx(indices)
        labels = torch.cat([target['labels'][matched] for target, (_, matched) in zip(targets, indices)])
        probabilities = logits[idx].float().sigmoid()
        correct = probabilities.gather(1, labels[:, None]).squeeze(1)
        mask = F.one_hot(labels, num_classes=logits.shape[-1]).bool()
        competitor = probabilities.masked_fill(mask, -1.).amax(dim=-1)
        penalty = F.relu(self.sdcp_margin - (correct - competitor)).square()
        count = logits.new_tensor(float(labels.numel()), dtype=torch.float32)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(count)
        return penalty.sum() * get_world_size() / count.clamp_min(1.)

    def sdcp_boundary_terms(self, src_boxes, target_boxes, image_size):
        """Return robust per-edge penalty, FGL weights, and a GIoU tolerance weight.

        The epsilon-insensitive Smooth-L1 has zero gradient within epsilon,
        a smooth transition over the following epsilon, then unit slope.
        Only horizontal edges receive 1+lambda_over when predicted width is
        too large. FGL/GIoU gates are detached supervision weights.
        """
        source, target = src_boxes.float(), target_boxes.float()
        size = torch.as_tensor(image_size, device=source.device, dtype=torch.float32)
        epsilon = self.sdcp_epsilon_px / size[[1, 0, 1, 0]]
        error = (box_cxcywh_to_xyxy(source) - box_cxcywh_to_xyxy(target)).abs()
        excess = F.relu(error - epsilon)
        smooth = torch.where(excess < epsilon, .5 * excess.square() / epsilon, excess - .5 * epsilon)
        over_width = (source[:, 2] > target[:, 2]).to(source.dtype)
        horizontal = 1. + self.sdcp_lambda_over * over_width
        asymmetric = torch.stack((horizontal, torch.ones_like(horizontal),
                                  horizontal, torch.ones_like(horizontal)), dim=-1)
        tolerance = (excess.detach() / epsilon).clamp(0., 1.)
        return smooth * asymmetric, tolerance * asymmetric, tolerance.mean(dim=-1)

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    def loss_labels_mal(self, outputs, targets, indices, num_boxes, values=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if self.mal_edge_tolerance_px > 0:
            if values is not None:
                raise ValueError('Q-MAL computes its own quality target; boxes_weight_format must be None.')
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious = edge_tolerant_iou(src_boxes, target_boxes, self.mal_edge_tolerance_px, self.input_size).detach()
        elif values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        target_score = target_score.pow(self.gamma)
        if self.mal_alpha != None:
            weight = self.mal_alpha * pred_score.pow(self.gamma) * (1 - target) + target
        else:
            weight = pred_score.pow(self.gamma) * (1 - target) + target

        # print(" ### DEIM-gamma{}-alpha{} ### ".format(self.gamma, self.mal_alpha))
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_mal': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}
        if self.use_sdcp:
            src_boxes, target_boxes = src_boxes.float(), target_boxes.float()
            penalty, _, giou_tolerance = self.sdcp_boundary_terms(
                src_boxes, target_boxes, outputs['sdcp_image_size'])
            losses['loss_asym_box'] = penalty.sum() / num_boxes
        else:
            loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
            losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(\
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        if self.use_sdcp:
            loss_giou = loss_giou * giou_tolerance
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        # Add NWD loss for small object detection (optional)
        if 'loss_nwd' in self.weight_dict:
            # Cast to float32 to avoid numerical issues under AMP
            src_boxes_f32 = src_boxes.float()
            target_boxes_f32 = target_boxes.float()
            nwd_matrix = wasserstein_distance_normalized(src_boxes_f32, target_boxes_f32)
            nwd_scores = torch.diag(nwd_matrix)
            loss_nwd = 1 - nwd_scores
            losses['loss_nwd'] = loss_nwd.sum() / num_boxes

        # Add Boundary Refinement Loss (BRL) for SAR small objects (optional, training only)
        if 'loss_brl' in self.weight_dict and 'pred_boundary_feats' in outputs and 'gt_boundary_feats' in outputs:
            pred_boundary = outputs['pred_boundary_feats'][idx]
            gt_boundary = outputs['gt_boundary_feats'][idx]
            loss_brl = F.mse_loss(pred_boundary, gt_boundary, reduction='mean')
            losses['loss_brl'] = loss_brl

        return losses

    def loss_local(self, outputs, targets, indices, num_boxes, T=5):
        """Compute Fine-Grained Localization (FGL) Loss
            and Decoupled Distillation Focal (DDF) Loss. """

        losses = {}
        if 'pred_corners' in outputs:
            idx = self._get_src_permutation_idx(indices)
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            pred_corners = outputs['pred_corners'][idx].reshape(-1, (self.reg_max+1))
            ref_points = outputs['ref_points'][idx].detach()
            with torch.no_grad():
                if self.fgl_targets_dn is None and 'is_dn' in outputs:
                        self.fgl_targets_dn= bbox2distance(ref_points, box_cxcywh_to_xyxy(target_boxes),
                                                        self.reg_max, outputs['reg_scale'], outputs['up'])
                if self.fgl_targets is None and 'is_dn' not in outputs:
                        self.fgl_targets = bbox2distance(ref_points, box_cxcywh_to_xyxy(target_boxes),
                                                        self.reg_max, outputs['reg_scale'], outputs['up'])

            target_corners, weight_right, weight_left = self.fgl_targets_dn if 'is_dn' in outputs else self.fgl_targets

            ious = torch.diag(box_iou(\
                        box_cxcywh_to_xyxy(outputs['pred_boxes'][idx]), box_cxcywh_to_xyxy(target_boxes))[0])
            weight_targets = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()
            if self.use_sdcp:
                _, edge_weights, _ = self.sdcp_boundary_terms(
                    outputs['pred_boxes'][idx], target_boxes, outputs['sdcp_image_size'])
                weight_targets = weight_targets * edge_weights.reshape(-1)

            losses['loss_fgl'] = self.unimodal_distribution_focal_loss(
                pred_corners, target_corners, weight_right, weight_left, weight_targets, avg_factor=num_boxes)
            if self.wfdr_loss is not None:
                with torch.autocast(device_type=pred_corners.device.type, enabled=False):
                    grid = weighting_function(self.reg_max, outputs['up'].float(), outputs['reg_scale'].float())
                losses['loss_wfdr'] = self.wfdr_loss(
                    pred_corners, target_corners, grid, weight_targets,
                    weight_left=weight_left, weight_right=weight_right, avg_factor=num_boxes)

            if 'teacher_corners' in outputs:
                pred_corners = outputs['pred_corners'].reshape(-1, (self.reg_max+1))
                target_corners = outputs['teacher_corners'].reshape(-1, (self.reg_max+1))
                if not torch.equal(pred_corners, target_corners):
                    weight_targets_local = outputs['teacher_logits'].sigmoid().max(dim=-1)[0]

                    mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                    mask[idx] = True
                    mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)

                    weight_targets_local[idx] = ious.reshape_as(weight_targets_local[idx]).to(weight_targets_local.dtype)
                    weight_targets_local = weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

                    loss_match_local = weight_targets_local * (T ** 2) * (nn.KLDivLoss(reduction='none')
                    (F.log_softmax(pred_corners / T, dim=1), F.softmax(target_corners.detach() / T, dim=1))).sum(-1)
                    if 'is_dn' not in outputs:
                        batch_scale = 8 / outputs['pred_boxes'].shape[0]  # Avoid the influence of batch size per GPU
                        self.num_pos, self.num_neg = (mask.sum() * batch_scale) ** 0.5, ((~mask).sum() * batch_scale) ** 0.5
                    loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0
                    loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0
                    losses['loss_ddf'] = (loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg) / (self.num_pos + self.num_neg)

        return losses

    def loss_boundary(self, outputs, targets, feat_stride=8):
        """Compute Boundary Consistency Map (BCM) auxiliary loss.

        This loss trains the encoder to extract boundary features,
        which indirectly improves localization quality.

        Args:
            outputs: dict containing 'boundary_map' [B, 1, H, W]
            targets: list of dicts with 'boxes' and 'labels'
            feat_stride: spatial stride of boundary map

        Returns:
            dict with 'loss_bcm'
        """
        if 'boundary_map' not in outputs:
            return {}

        pred_boundary = outputs['boundary_map']  # [B, 1, H, W]
        B, _, H, W = pred_boundary.shape

        # Generate GT boundary maps (on same device as pred_boundary)
        gt_boundary = generate_gt_boundary_map(
            targets, H, W, feat_stride=feat_stride, line_width=2
        )  # [B, 1, H, W]
        gt_boundary = gt_boundary.to(pred_boundary.device)  # Ensure same device

        # BCE loss with class rebalancing (boundaries are sparse)
        # Compute positive weight (balance foreground/background)
        num_pos = (gt_boundary > 0.5).sum().clamp(min=1.0)
        num_neg = (gt_boundary <= 0.5).sum().clamp(min=1.0)
        pos_weight = num_neg / num_pos

        loss_bcm = F.binary_cross_entropy_with_logits(
            pred_boundary, gt_boundary,
            pos_weight=pos_weight.clamp(max=10.0)  # Prevent extreme imbalance
        )

        return {'loss_bcm': loss_bcm}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_go_indices(self, indices, indices_aux_list):
        """Get a matching union set across all decoder layers. """
        results = []
        for indices_aux in indices_aux_list:
            indices = [(torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                        for idx1, idx2 in zip(indices.copy(), indices_aux.copy())]

        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}
            for idx in unique_sorted:
                row_idx, col_idx = idx[0].item(), idx[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        return results

    def _clear_cache(self):
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.num_pos, self.num_neg = None, None

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        if self.use_bccq and loss == 'mal' and 'bccq_type_logits' in outputs:
            return self.loss_bccq(outputs, targets, indices, num_boxes, **kwargs)
        loss_map = {
            'boxes': self.loss_boxes,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
            'mal': self.loss_labels_mal,
            'local': self.loss_local,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Batch input (H, W) from the training loop; only Q-MAL needs it.
        self.input_size = kwargs.pop('input_size', None)
        if self.mal_edge_tolerance_px > 0 and self.input_size is None:
            raise ValueError('Q-MAL needs the batch input_size (H, W) from the training loop.')
        if self.use_sdcp:
            # All GT-supervised box/FGL branches use the actual batch input size.
            image_size = outputs['sdcp_image_size']
            for group in ['aux_outputs', 'enc_aux_outputs', 'dn_outputs']:
                for item in outputs.get(group, []):
                    item['sdcp_image_size'] = image_size
            for name in ['pre_outputs', 'dn_pre_outputs']:
                if name in outputs:
                    outputs[name]['sdcp_image_size'] = image_size
        # Imaging geometry supervises every detection loss with the response box.
        targets = outputs.get('geometry_response_targets', targets)
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)['indices']
        self._clear_cache()
        if self.use_bccq:
            if 'bccq_type_logits' not in outputs or 'bccq_quality_logits' not in outputs:
                raise ValueError('BCCQ criterion requires the complete BCCQ model outputs.')
            self._record_bccq_metrics(outputs, targets, indices)

        # Get the matching union set across all decoder layers.
        if 'aux_outputs' in outputs:
            indices_aux_list, cached_indices, cached_indices_enc = [], [], []
            aux_outputs_list = outputs['aux_outputs']
            if 'pre_outputs' in outputs:
                aux_outputs_list = outputs['aux_outputs'] + [outputs['pre_outputs']]
            for i, aux_outputs in enumerate(aux_outputs_list):
                indices_aux = self.matcher(aux_outputs, targets)['indices']
                cached_indices.append(indices_aux)
                indices_aux_list.append(indices_aux)
            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                indices_enc = self.matcher(aux_outputs, targets)['indices']
                cached_indices_enc.append(indices_enc)
                indices_aux_list.append(indices_enc)
            indices_go = self._get_go_indices(indices, indices_aux_list)
            if self.log_matching_stats:
                self._record_matching_metrics(outputs, targets, indices, cached_indices, cached_indices_enc, indices_go)

            num_boxes_go = sum(len(x[0]) for x in indices_go)
            num_boxes_go = torch.as_tensor([num_boxes_go], dtype=torch.float, device=next(iter(outputs.values())).device)
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(num_boxes_go)
            num_boxes_go = torch.clamp(num_boxes_go / get_world_size(), min=1).item()
        else:
            assert 'aux_outputs' in outputs, ''

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses, main loss
        losses = {}
        if self.use_sdcp:
            losses['loss_margin'] = self.loss_sdcp_margin(outputs, targets, indices) * self.weight_dict['loss_margin']
        for loss in self.losses:
            # TODO, indices and num_box are different from RT-DETRv2
            use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
            indices_in = indices_go if use_uni_set else indices
            num_boxes_in = num_boxes_go if use_uni_set else num_boxes
            meta = self.get_loss_meta_info(loss, outputs, targets, indices_in)
            l_dict = self.get_loss(loss, outputs, targets, indices_in, num_boxes_in, **meta)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if 'local' in self.losses:      # only work for local loss
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    # TODO, indices and num_box are different from RT-DETRv2
                    use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
                    indices_in = indices_go if use_uni_set else cached_indices[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)

                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of auxiliary traditional head output at first decoder layer. just for dfine
        if 'pre_outputs' in outputs:
            aux_outputs = outputs['pre_outputs']
            for loss in self.losses:
                # TODO, indices and num_box are different from RT-DETRv2
                use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
                indices_in = indices_go if use_uni_set else cached_indices[-1]
                num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)

                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                l_dict = {k + '_pre': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # In case of encoder auxiliary losses.
        if 'enc_aux_outputs' in outputs:
            assert 'enc_meta' in outputs, ''
            class_agnostic = outputs['enc_meta']['class_agnostic']
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t['labels'] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                for loss in self.losses:
                    # TODO, indices and num_box are different from RT-DETRv2
                    use_uni_set = self.use_uni_set and (loss == 'boxes')
                    indices_in = indices_go if use_uni_set else cached_indices_enc[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices_in)
                    l_dict = self.get_loss(loss, aux_outputs, enc_targets, indices_in, num_boxes_in, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            if class_agnostic:
                self.num_classes = orig_num_classes

        # In case of cdn auxiliary losses.
        if 'dn_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices_dn = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']

            for i, aux_outputs in enumerate(outputs['dn_outputs']):
                if 'local' in self.losses:      # only work for local loss
                    aux_outputs['is_dn'] = True
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            # In case of auxiliary traditional head output at first decoder layer, just for dfine
            if 'dn_pre_outputs' in outputs:
                aux_outputs = outputs['dn_pre_outputs']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + '_dn_pre': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # Boundary Consistency Map (BCM) auxiliary loss (training only)
        if 'loss_bcm' in self.weight_dict and 'boundary_map' in outputs:
            bcm_loss_dict = self.loss_boundary(outputs, targets, feat_stride=8)
            if bcm_loss_dict:
                bcm_loss_dict = {k: v * self.weight_dict[k] for k, v in bcm_loss_dict.items() if k in self.weight_dict}
                losses.update(bcm_loss_dict)

        # Simple Background Rejection (SBR) auxiliary loss (training only)
        if 'loss_sbr' in self.weight_dict and 'bg_scores' in outputs:
            from .background_rejection import SimpleBackgroundRejection
            # Get the model's SBR module to access get_supervision_labels
            # (This is a bit hacky, ideally we'd pass the model reference)
            # For now, compute labels directly here

            bg_scores = outputs['bg_scores']  # [B, num_queries]
            pred_boxes = outputs['pred_boxes']  # [B, num_queries, 4]

            # Generate supervision labels
            device = bg_scores.device
            dtype = bg_scores.dtype  # Match dtype with bg_scores (FP16 in AMP)
            B, num_queries = bg_scores.shape
            labels = torch.full((B, num_queries), -1, dtype=torch.long, device=device)
            weights = torch.zeros((B, num_queries), dtype=dtype, device=device)  # Use same dtype as bg_scores

            for batch_idx in range(B):
                # Positive samples: matched queries
                pred_idx = indices[batch_idx][0]
                labels[batch_idx, pred_idx] = 0  # Target
                weights[batch_idx, pred_idx] = 1.0

                # Negative samples: far from all GT (IoU < 0.3)
                if len(targets[batch_idx]['boxes']) > 0:
                    pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes[batch_idx])
                    gt_boxes = targets[batch_idx]['boxes'].to(device)
                    gt_boxes_xyxy = box_cxcywh_to_xyxy(gt_boxes)

                    # IoU matrix. Default is plain IoU; 'giou' reproduces the
                    # historical v1 labelling, where GIoU mapped by (g+1)/2 is
                    # systematically low for tiny boxes and mislabels genuine
                    # small targets as background.
                    if self.sbr_bg_label_metric == 'giou':
                        iou_matrix = generalized_box_iou(pred_boxes_xyxy, gt_boxes_xyxy)
                        iou_matrix = (iou_matrix + 1) / 2  # GIoU to IoU approx
                    else:
                        iou_matrix, _ = box_iou(pred_boxes_xyxy, gt_boxes_xyxy)

                    max_iou, _ = iou_matrix.max(dim=1)

                    # Background: IoU below the configured negative threshold
                    bg_mask = max_iou < self.sbr_neg_iou_threshold
                    labels[batch_idx, bg_mask] = 1
                    weights[batch_idx, bg_mask] = 1.0

                    # Ignore: between the negative and ignore thresholds
                    ignore_mask = (max_iou >= self.sbr_neg_iou_threshold) & \
                                  (max_iou < self.sbr_ignore_iou_threshold)
                    labels[batch_idx, ignore_mask] = -1
                    weights[batch_idx, ignore_mask] = 0.0

            # Compute SBR loss
            sbr_loss_value = self.sbr_loss(bg_scores, labels, weights)
            losses['loss_sbr'] = sbr_loss_value * self.weight_dict.get('loss_sbr', 1.0)

        # Boundary Refinement Loss (BRL) auxiliary loss (training only)
        if 'pred_iou' in outputs:
            brl_loss_dict = self.brl_criterion(outputs, targets, indices)
            # Only apply loss if the key exists in weight_dict
            brl_loss_dict = {k: v * self.weight_dict[k] for k, v in brl_loss_dict.items() if k in self.weight_dict}
            losses.update(brl_loss_dict)

        if self.scs_loss is not None:
            # Only final-layer original Hungarian matches; no union, aux, DN,
            # or encoder labels enter the instance-description supervision.
            scs_losses = self.scs_loss(outputs, targets, indices)
            losses.update({name:value * self.weight_dict[name] for name, value in scs_losses.items()})

        if self.mpq_loss is not None:
            # Original final-layer matches only. The auxiliary prototype logits
            # never replace pred_logits in matching or any detection loss.
            losses['loss_mpq'] = self.mpq_loss(outputs, targets, indices, **kwargs) * self.weight_dict['loss_mpq']

        if 'mdcc_ortho' in outputs:
            if 'loss_mdcc_ortho' not in self.weight_dict:
                raise ValueError('MDCC requires weight_dict.loss_mdcc_ortho.')
            losses['loss_mdcc_ortho'] = outputs['mdcc_ortho'] * self.weight_dict['loss_mdcc_ortho']

        if 'grea_evidence_logits' in outputs:
            losses['loss_grea_evidence'] = self.loss_grea_evidence(outputs, targets, indices, num_boxes) * self.weight_dict['loss_grea_evidence']
            self._record_grea_metrics(outputs, targets, indices)

        if 'geometry_translation' in outputs:
            geometry = self.loss_imaging_geometry(outputs, targets, indices)
            losses.update({name: value * self.weight_dict[name] for name, value in geometry.items()})

        # Keep numerical failures visible to the engine before backward.
        # Replacing a NaN loss with zero does not repair its backward graph.
        return losses

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}

        src_boxes = outputs['pred_boxes'][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if self.boxes_weight_format == 'iou':
            iou, _ = box_iou(box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes))
            iou = torch.diag(iou)
        elif self.boxes_weight_format == 'giou':
            iou = torch.diag(generalized_box_iou(\
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)))
        else:
            raise AttributeError()

        if loss in ('boxes', ):
            meta = {'boxes_weight': iou}
        elif loss in ('vfl', 'mal'):
            meta = {'values': iou}
        else:
            meta = {}

        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices
        """
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                    torch.zeros(0, dtype=torch.int64,  device=device)))

        return dn_match_indices


    def feature_loss_function(self, fea, target_fea):
        loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()
        return torch.abs(loss)


    def unimodal_distribution_focal_loss(self, pred, label, weight_right, weight_left, weight=None, reduction='sum', avg_factor=None):
        dis_left = label.long()
        dis_right = dis_left + 1

        loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left.reshape(-1) \
             + F.cross_entropy(pred, dis_right, reduction='none') * weight_right.reshape(-1)

        if weight is not None:
            weight = weight.float()
            loss = loss * weight

        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()

        return loss

    def get_gradual_steps(self, outputs):
        num_layers = len(outputs['aux_outputs']) + 1 if 'aux_outputs' in outputs else 1
        step = .5 / (num_layers - 1)
        opt_list = [.5  + step * i for i in range(num_layers)] if num_layers > 1 else [1]
        return opt_list
