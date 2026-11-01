"""Inference-only readouts and GT-aligned score decomposition for REM diagnosis."""

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from tools.sar_analysis_core import classify_image, iou_matrix


BANDS = ('ll', 'lh', 'hl', 'hh')
RESPONSE_NAMES = ('input', 'total', *BANDS)
VARIANTS = ('baseline', 'rem', 'rem_half', 'rem_ll_half', 'rem_hf_half')
SCALES = dict(baseline=(1., 1., 1., 1.), rem=(1., 1., 1., 1.),
              rem_half=(.5, .5, .5, .5), rem_ll_half=(.5, 1., 1., 1.), rem_hf_half=(1., .5, .5, .5))
ARRAY_KEYS = ('image_ids', 'orig_sizes', 'boxes', 'raw_logits', 'quality', 'final_logits',
              'core_abs', 'ring_abs', 'valid_fractions')


class BandReadout:
    """Apply fixed branch multipliers without changing weights or the REM class."""

    def __init__(self, rem, scales, projection=None):
        self.rem, self.scales, self.handles = rem, tuple(scales), []
        self.clear()
        if rem is None:
            self.handles.append(projection.register_forward_hook(self._projection))
        else:
            if rem.variant != 'haar_directional' or rem.residual_mode != 'shared':
                raise ValueError('This probe targets the reported shared-alpha Haar REM.')
            self.handles.append(rem.register_forward_pre_hook(self._input))
            self.handles.append(rem.register_forward_hook(self._output))
            for index, name in enumerate(BANDS):
                self.handles.append(getattr(rem, name).register_forward_hook(self._band(index)))

    def clear(self):
        self.source, self.source_abs, self.total_abs, self.band_abs = None, None, None, {}

    def _projection(self, module, args, output):
        self.source_abs = output.detach().abs().mean(1, keepdim=True)

    def _input(self, module, args):
        self.source = args[0].detach()
        self.source_abs = args[0].detach().abs().mean(1, keepdim=True)

    def _output(self, module, args, output):
        self.total_abs = (output.detach() - self.source).abs().mean(1, keepdim=True)
        self.source = None

    def _band(self, index):
        def observe(module, args, output):
            value = output if self.scales[index] == 1. else output * self.scales[index]
            self.band_abs[index] = value.detach().abs().mean(1, keepdim=True)
            return value
        return observe

    def maps(self):
        if self.rem is None:
            return torch.cat((self.source_abs, *[torch.zeros_like(self.source_abs) for _ in range(5)]), 1)
        # A Haar synthesis component is +/- coefficient/2 on its 2x2 cell.
        # Its mean absolute channel response therefore needs no extra IDWT.
        h, w = self.source_abs.shape[-2:]
        alpha = self.rem.alpha.detach().abs()
        bands = [F.interpolate(self.band_abs[i], scale_factor=2, mode='nearest')[..., :h, :w] * alpha * .5
                 for i in range(4)]
        return torch.cat((self.source_abs, self.total_abs, *bands), 1)

    def close(self):
        for handle in self.handles:
            handle.remove()


class EvidenceReadout:
    def __init__(self, model, variant):
        decoder = model.decoder.decoder
        lqe = decoder.lqe_layers[decoder.eval_idx]
        if lqe.clcr is not None or decoder.use_clcr_lite:
            raise ValueError('Score decomposition expects original LQE without CLCR corrections.')
        self.bands = BandReadout(model.encoder.rem, SCALES[variant], model.encoder.input_proj[0])
        self.handles = [lqe.register_forward_pre_hook(self._raw),
                        lqe.reg_conf.register_forward_hook(self._quality)]
        self.clear()

    def clear(self):
        self.raw, self.quality = None, None
        self.bands.clear()

    def _raw(self, module, args):
        self.raw = args[0].detach()

    def _quality(self, module, args, output):
        self.quality = output.detach()

    def close(self):
        self.bands.close()
        for handle in self.handles:
            handle.remove()


def pool_responses(maps, boxes):
    """Pool diagnostic response magnitudes; never feed them back to the model."""
    h, w = maps.shape[-2:]
    line = torch.tensor([-.375, 0., .375], device=boxes.device, dtype=boxes.dtype)
    yy, xx = torch.meshgrid(line, line, indexing='ij')
    core = torch.stack((xx.flatten(), yy.flatten()), -1)
    side = torch.tensor([-.35, 0., .35], device=boxes.device, dtype=boxes.dtype)
    ring = torch.cat((torch.stack((side, torch.full_like(side, -.525)), -1),
                      torch.stack((side, torch.full_like(side, .525)), -1),
                      torch.stack((torch.full_like(side, -.525), side), -1),
                      torch.stack((torch.full_like(side, .525), side), -1)), 0)
    offsets = torch.cat((core, ring), 0)
    points = boxes[..., None, :2] + boxes[..., None, 2:] * offsets
    valid = ((points[..., 0] >= .5 / w) & (points[..., 0] <= 1 - .5 / w) &
             (points[..., 1] >= .5 / h) & (points[..., 1] <= 1 - .5 / h))
    sampled = F.grid_sample(maps, points * 2 - 1, align_corners=False).permute(0, 2, 3, 1)

    def pool(start, end):
        mask = valid[..., start:end].to(sampled.dtype)
        return (sampled[..., start:end, :] * mask[..., None]).sum(-2) / mask.sum(-1, keepdim=True).clamp_min(1)

    return pool(0, 9), pool(9, 21), torch.stack((valid[..., :9].float().mean(-1), valid[..., 9:].float().mean(-1)), -1)


def postprocess_with_queries(postprocessor, outputs, sizes, image_ids):
    results = postprocessor(outputs, sizes)
    if not postprocessor.use_focal_loss or postprocessor.remap_mscoco_category:
        raise ValueError('This SAR probe uses sigmoid scores and zero-based category IDs.')
    logits = outputs['pred_logits']
    indices = logits.sigmoid().flatten(1).topk(postprocessor.num_top_queries, dim=-1).indices
    records = []
    for batch, result in enumerate(results):
        boxes = result['boxes'].detach().cpu().clone()
        boxes[:, 2:] -= boxes[:, :2]
        for index, box, score, label in zip(indices[batch].cpu().tolist(), boxes.tolist(),
                                           result['scores'].cpu().tolist(), result['labels'].cpu().tolist()):
            records.append(dict(image_id=int(image_ids[batch]), category_id=int(label),
                                bbox=box, score=float(score), query_index=index // logits.shape[-1]))
    return records


def raw_only_predictions(arrays, topk=300):
    from engine.deim.postprocessor import PostProcessor
    logits = torch.from_numpy(arrays['raw_logits'])
    processor = PostProcessor(num_classes=logits.shape[-1], num_top_queries=topk)
    return postprocess_with_queries(processor, dict(pred_logits=logits, pred_boxes=torch.from_numpy(arrays['boxes'])),
                                    torch.from_numpy(arrays['orig_sizes']), arrays['image_ids'])


def merge_query_shards(shards):
    merged = {key: np.concatenate([shard[key] for shard in shards], 0) for key in ARRAY_KEYS}
    order = np.argsort(merged['image_ids'])
    return {key: value[order] for key, value in merged.items()}


def make_anchors(dataset, reference_predictions):
    gts, dts = defaultdict(list), {role: defaultdict(list) for role in reference_predictions}
    for gt in dataset['annotations']:
        gts[gt['image_id']].append(gt)
    for role in dts:
        for prediction in reference_predictions[role]:
            dts[role][prediction['image_id']].append(prediction)
    anchors, wrong, backgrounds = [], {}, defaultdict(list)
    for gt in dataset['annotations']:
        if gt['category_id'] in (0, 2):
            group = 'a220_true' if gt['category_id'] == 0 else 'a330_true'
            anchors.append(dict(anchor_id=f'{group}/{gt["id"]}', group=group, image_id=gt['image_id'],
                                gt_id=gt['id'], gt_category=gt['category_id'], category=gt['category_id'],
                                reference_box=gt['bbox'], minimum_iou=.5 if gt['category_id'] == 0 else .75))
    for role in reference_predictions:
        for image in dataset['images']:
            records, _ = classify_image(gts[image['id']], dts[role][image['id']], .3, .5)
            for record in records:
                if record['category_id'] != 0:
                    continue
                if record['kind'] == 'classification':
                    wrong[(record['image_id'], record['gt_id'])] = record
                elif record['kind'] == 'background':
                    regions = backgrounds[image['id']]
                    if not regions or iou_matrix([record['bbox']], regions).max() < .5:
                        regions.append(record['bbox'])
    by_id = {gt['id']: gt for gt in dataset['annotations']}
    for (image_id, gt_id), record in sorted(wrong.items()):
        anchors.append(dict(anchor_id=f'a220_wrongclass/{gt_id}', group='a220_wrongclass', image_id=image_id,
                            gt_id=gt_id, gt_category=record['gt_category_id'], category=0,
                            reference_box=by_id[gt_id]['bbox'], minimum_iou=.5))
    for image_id, regions in sorted(backgrounds.items()):
        for index, box in enumerate(regions):
            anchors.append(dict(anchor_id=f'a220_background/{image_id}/{index}', group='a220_background',
                                image_id=image_id, gt_id=None, gt_category=None, category=0,
                                reference_box=box, minimum_iou=.5))
    return anchors


def anchor_evidence(anchors, arrays, variant):
    image_rows = {int(image_id): i for i, image_id in enumerate(arrays['image_ids'])}
    records = []
    for anchor in anchors:
        i = image_rows[anchor['image_id']]
        w, h = arrays['orig_sizes'][i]
        boxes = arrays['boxes'][i]
        xywh = np.stack(((boxes[:, 0] - boxes[:, 2] / 2) * w, (boxes[:, 1] - boxes[:, 3] / 2) * h,
                         boxes[:, 2] * w, boxes[:, 3] * h), -1)
        overlap = iou_matrix([anchor['reference_box']], xywh)[0]
        candidates = np.flatnonzero(overlap >= anchor['minimum_iou'])
        row = dict(anchor_id=anchor['anchor_id'], group=anchor['group'], image_id=anchor['image_id'],
                   gt_id=anchor['gt_id'], gt_category=anchor['gt_category'], target_category=anchor['category'],
                   variant=variant, present=bool(candidates.size))
        if candidates.size:
            c = anchor['category']
            q = int(candidates[np.argmax(arrays['final_logits'][i, candidates, c])])
            raw = arrays['raw_logits'][i, q]
            final = float(arrays['final_logits'][i, q, c])
            row.update(query_index=q, iou=float(overlap[q]), raw_logit=float(raw[c]),
                       quality_logit=float(arrays['quality'][i, q, 0]), final_logit=final,
                       score=float(1 / (1 + np.exp(-np.clip(final, -80, 80)))),
                       raw_margin=float(raw[c] - np.max(np.delete(raw, c))),
                       raw_top_class=int(np.argmax(raw)), predicted_box=xywh[q].tolist())
            if anchor['gt_category'] is not None:
                row['gt_raw_logit'] = float(raw[anchor['gt_category']])
            for location in ('core', 'ring'):
                for index, band in enumerate(RESPONSE_NAMES):
                    row[f'{location}_{band}_abs'] = float(arrays[f'{location}_abs'][i, q, index])
            row.update(core_valid_fraction=float(arrays['valid_fractions'][i, q, 0]),
                       ring_valid_fraction=float(arrays['valid_fractions'][i, q, 1]))
        records.append(row)
    return records


def paired_decomposition(records):
    lookup = {(row['variant'], row['anchor_id']): row for row in records}
    rows = []
    for row in records:
        if row['variant'] == 'baseline':
            continue
        reference = 'baseline' if row['variant'] == 'rem' else 'rem'
        base = lookup[(reference, row['anchor_id'])]
        if row['present'] and base['present']:
            item = dict(anchor_id=row['anchor_id'], group=row['group'], variant=row['variant'], reference=reference)
            for key in ('raw_logit', 'quality_logit', 'final_logit', 'score', 'raw_margin', 'iou'):
                item[f'delta_{key}'] = row[key] - base[key]
            rows.append(item)
    return rows
