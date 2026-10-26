"""Shared, deterministic geometry and error accounting for SAR diagnostics.

This is an explicitly defined count decomposition, not a TIDE dAP analysis.
FP types are mutually exclusive. Unmatched GT (FN) is a separate denominator.
"""
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

import numpy as np

ERROR_NAMES = ('background', 'localization', 'duplicate', 'classification', 'classification_localization')
SCALE_EDGES = (0, 16, 32, 64, 96, float('inf'))
SCALE_LABELS = ('<16', '16-32', '32-64', '64-96', '>=96')


def load_json(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def iou_matrix(boxes_a, boxes_b):
    a = np.asarray(boxes_a, dtype=float).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=float).reshape(-1, 4)
    intersection = np.maximum(0, np.minimum(a[:, None, :2] + a[:, None, 2:],
                                           b[None, :, :2] + b[None, :, 2:]) -
                              np.maximum(a[:, None, :2], b[None, :, :2])).prod(axis=2)
    union = a[:, 2:].prod(axis=1)[:, None] + b[:, 2:].prod(axis=1)[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def scale_bin(side):
    for i, (low, high) in enumerate(zip(SCALE_EDGES[:-1], SCALE_EDGES[1:])):
        if low <= side < high:
            return SCALE_LABELS[i]
    raise ValueError(side)


def object_size(annotation, image, input_size=640):
    w, h = annotation['bbox'][2:]
    iw, ih = image['width'], image['height']
    return dict(width=w, height=h, side=float(np.sqrt(w * h)),
                input_width=w * input_size / iw, input_height=h * input_size / ih,
                input_side=float(np.sqrt(w * h * input_size ** 2 / (iw * ih))))


def classify_image(ground_truth, predictions, score_threshold=0.3, iou_threshold=0.5,
                   background_iou=0.1):
    """Score-greedy, one-to-one same-class TP; then exclusive FP precedence.

    TP uses the best *unmatched* same-class GT (so a matched neighbor cannot
    steal a valid match). Remaining predictions: duplicate, classification,
    localization, joint class+localization, background, in that precedence.
    Each record stores the actual annotation ID and max/associated overlap.
    """
    if any(g.get('iscrowd', 0) or g.get('ignore', 0) for g in ground_truth):
        raise ValueError('This diagnostic decomposition requires non-crowd, non-ignored GT.')
    ordered = [(i, p) for i, p in enumerate(predictions) if p['score'] >= score_threshold]
    ordered.sort(key=lambda pair: (-pair[1]['score'], pair[0]))
    overlap = iou_matrix([p['bbox'] for _, p in ordered], [g['bbox'] for g in ground_truth])
    used = set()
    records = []
    for row, (prediction_index, prediction) in enumerate(ordered):
        same = [i for i, g in enumerate(ground_truth) if g['category_id'] == prediction['category_id']]
        candidates = [i for i in same if i not in used and overlap[row, i] >= iou_threshold]
        gt_index = None
        if candidates:
            gt_index = max(candidates, key=lambda i: overlap[row, i])
            used.add(gt_index)
            kind = 'tp'
        elif any(overlap[row, i] >= iou_threshold for i in same):
            gt_index = max(same, key=lambda i: overlap[row, i])
            kind = 'duplicate'
        elif len(ground_truth) and overlap[row].max() >= iou_threshold:
            gt_index = int(overlap[row].argmax())
            kind = 'classification'
        elif same and max(overlap[row, i] for i in same) >= background_iou:
            gt_index = max(same, key=lambda i: overlap[row, i])
            kind = 'localization'
        elif len(ground_truth) and overlap[row].max() >= background_iou:
            gt_index = int(overlap[row].argmax())
            kind = 'classification_localization'
        else:
            kind = 'background'
        gt = ground_truth[gt_index] if gt_index is not None else None
        records.append(dict(kind=kind, prediction_index=prediction_index,
                            image_id=prediction['image_id'], category_id=prediction['category_id'],
                            score=prediction['score'], bbox=prediction['bbox'],
                            gt_id=gt['id'] if gt else None,
                            gt_category_id=gt['category_id'] if gt else None,
                            iou=float(overlap[row, gt_index]) if gt is not None else
                            float(overlap[row].max()) if len(ground_truth) else 0.0))
    misses = [dict(kind='miss', image_id=g['image_id'], gt_id=g['id'],
                   category_id=g['category_id'], bbox=g['bbox'])
              for i, g in enumerate(ground_truth) if i not in used]
    return records, misses


def error_analysis(dataset, predictions, threshold=0.3, iou=0.5):
    gts, dts = defaultdict(list), defaultdict(list)
    image_map = {im['id']: im for im in dataset['images']}
    for g in dataset['annotations']:
        gts[g['image_id']].append(g)
    for d in predictions:
        if d['image_id'] not in image_map:
            raise ValueError('Prediction contains an image outside this split.')
        dts[d['image_id']].append(d)
    records, misses, objects = [], [], []
    for image in dataset['images']:
        image_id = image['id']
        found, missing = classify_image(gts[image_id], dts[image_id], threshold, iou)
        records.extend(found)
        misses.extend(missing)
        matched = {r['gt_id']: r for r in found if r['kind'] == 'tp'}
        for g in gts[image_id]:
            size = object_size(g, image)
            match = matched.get(g['id'])
            objects.append(dict(gt_id=g['id'], image_id=image_id, category_id=g['category_id'],
                                **size, scale_bin=scale_bin(size['input_side']),
                                detected=int(match is not None),
                                matched_iou=match['iou'] if match else None))
    count = Counter(r['kind'] for r in records)
    tp, fp, fn = count['tp'], len(records) - count['tp'], len(misses)
    summary = dict(score_threshold=threshold, iou_threshold=iou, background_iou=0.1,
                   gt=len(objects), predictions=len(records), tp=tp, fp=fp, fn=fn,
                   precision=tp / max(1, tp + fp), recall=tp / max(1, tp + fn),
                   f1=2 * tp / max(1, 2 * tp + fp + fn),
                   fp_types={name:dict(count=count[name], fraction_of_fp=count[name] / max(1, fp),
                                       fraction_of_predictions=count[name] / max(1, len(records)))
                             for name in ERROR_NAMES},
                   miss_fraction_of_gt=fn / max(1, len(objects)))
    if tp + fn != len(objects) or sum(count[name] for name in ERROR_NAMES) != fp:
        raise ValueError('Error accounting does not partition predictions/GT.')
    return summary, records, misses, objects


def summarize_values(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return dict(n=0)
    q = np.quantile(values, [0, .1, .25, .5, .75, .9, 1])
    return dict(n=len(values), mean=float(values.mean()),
                **{name:float(v) for name, v in zip(['min', 'p10', 'p25', 'p50', 'p75', 'p90', 'max'], q)})


def training_summary(log):
    best = max(log, key=lambda r: r['test_coco_eval_bbox'][0])
    best_ap = best['test_coco_eval_bbox'][0]
    near = next(r['epoch'] for r in log if r['test_coco_eval_bbox'][0] >= best_ap - .005)
    last12 = [r['test_coco_eval_bbox'][0] for r in log[-12:]]
    groups = []
    for start, end in [(0, 3), (4, 48), (49, 89), (90, 101)]:
        rows = [r for r in log if start <= r['epoch'] <= end]
        if rows:
            best_group = max(rows, key=lambda r: r['test_coco_eval_bbox'][0])
            groups.append(dict(start=start, end=end, best_epoch=best_group['epoch'],
                               best_ap=best_group['test_coco_eval_bbox'][0],
                               last_ap=rows[-1]['test_coco_eval_bbox'][0]))
    return dict(completed_epochs=len(log), first_epoch=log[0]['epoch'], last_epoch=log[-1]['epoch'],
                best_epoch=best['epoch'], best_metrics=best['test_coco_eval_bbox'],
                last_metrics=log[-1]['test_coco_eval_bbox'],
                first_epoch_within_0_5_ap_point=near,
                last12_ap_mean=float(np.mean(last12)), last12_ap_std=float(np.std(last12)),
                last12_ap_range=float(np.ptp(last12)),
                retry_epochs=sum(r.get('train_fp32_retry_rate', 0) > 0 for r in log),
                initial_loss=log[0]['train_loss'], final_loss=log[-1]['train_loss'], stages=groups)
