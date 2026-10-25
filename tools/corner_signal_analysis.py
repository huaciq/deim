#!/usr/bin/env python3
"""Test whether FDR projected edge standard deviation predicts edge error."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

EDGE = ['left', 'top', 'right', 'bottom']
SIZE_BINS = [('tiny<32', 0, 32), ('small32-96', 32, 96),
             ('med96-256', 96, 256), ('large>=256', 256, float('inf'))]


def iou_matrix(a, b):
    a = np.asarray(a, np.float64).reshape(-1, 4)
    b = np.asarray(b, np.float64).reshape(-1, 4)
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, :2] + a[:, None, 2:], b[None, :, :2] + b[None, :, 2:])
    inter = np.maximum(hi - lo, 0).prod(-1)
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def edge_dists(ref, boxes, reg_scale):
    """Continuous signed projection coordinates before translate_gt binning."""
    rs = abs(reg_scale)
    lo, hi = boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2
    left_top = (ref[:, :2] - lo) / (ref[:, 2:] / rs + 1e-16) - rs / 2
    right_bottom = (hi - ref[:, :2]) / (ref[:, 2:] / rs + 1e-16) - rs / 2
    return np.concatenate([left_top, right_bottom], axis=-1)


def corr(a, b):
    if len(a) < 30 or a.std() < 1e-12 or b.std() < 1e-12:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def match_edges(z, gt, iou_thr=0.5, border=0.02):
    images = {im['id']: im for im in gt['images']}
    gts, detections = defaultdict(list), defaultdict(list)
    for ann in gt['annotations']:
        if not ann.get('iscrowd', 0):
            gts[ann['image_id']].append(ann)
    for i, image_id in enumerate(z['image_id']):
        detections[int(image_id)].append(i)
    lower, upper = float(z['project'].min()), float(z['project'].max())
    counts = Counter()
    widths, sigmas, errors, overlaps, scores = [], [], [], [], []
    for image_id, indices in detections.items():
        candidates = gts.get(image_id, [])
        if not candidates:
            continue
        image = images[image_id]
        width, height = image['width'], image['height']
        if width <= 0 or height <= 0:
            raise ValueError(f'Invalid image dimensions for {image_id}')
        gt_boxes = np.asarray([g['bbox'] for g in candidates], np.float64)
        labels = np.asarray([g['category_id'] for g in candidates])
        used = np.zeros(len(candidates), bool)
        for k in sorted(indices, key=lambda i: -float(z['score'][i])):
            box = z['box'][k].astype(np.float64)
            ref = z['ref'][k].astype(np.float64)
            pixel_box = np.concatenate([(box[:2] - box[2:] / 2) * [width, height],
                                        box[2:] * [width, height]])
            candidate_indices = np.flatnonzero(labels == z['label'][k])
            if not len(candidate_indices):
                continue
            ious = iou_matrix(pixel_box, gt_boxes[candidate_indices])[0]
            best = int(np.argmax(ious))
            j = candidate_indices[best]
            if ious[best] < iou_thr or used[j]:
                continue
            used[j] = True
            counts['matched_before_exclusions'] += 1
            g = gt_boxes[j]
            if border > 0 and (g[0] / width < border or g[1] / height < border or
                               (g[0] + g[2]) / width > 1 - border or
                               (g[1] + g[3]) / height > 1 - border):
                counts['excluded_border'] += 1
                continue
            if np.any(ref[2:] <= 0):
                counts['excluded_invalid_reference'] += 1
                continue
            gt_box = np.concatenate([(g[:2] + g[2:] / 2) / [width, height],
                                     g[2:] / [width, height]])
            predicted = edge_dists(ref[None], box[None], float(z['reg_scale']))[0]
            target = edge_dists(ref[None], gt_box[None], float(z['reg_scale']))[0]
            if np.any(target < lower) or np.any(target > upper):
                counts['excluded_outside_project_support'] += 1
                continue
            if np.any(target < 0) or np.any(target > int(z['reg_max']) - 0.1):
                counts['valid_pairs_rejected_by_document_index_filter'] += 1
            widths.append(g[2])
            sigmas.append(z['std'][k])
            errors.append(predicted - target)
            overlaps.append(ious[best])
            scores.append(z['score'][k])
    counts['retained'] = len(sigmas)
    return (np.asarray(widths), np.asarray(sigmas).reshape(-1, 4),
            np.asarray(errors).reshape(-1, 4), np.asarray(overlaps),
            np.asarray(scores), dict(counts))


def main(args):
    # NpzFile does not cache arrays: repeated per-query indexing would unzip
    # complete arrays thousands of times. Load the small statistics once.
    with np.load(args.npz) as archive:
        z = {key: archive[key] for key in archive.files}
    reconstruction_error = float(z['recon_err'])
    if not np.isfinite(reconstruction_error) or reconstruction_error >= 1e-4:
        raise ValueError('Distribution reconstruction failed; do not analyze signal')
    gt = json.loads(Path(args.gt).read_text())
    label_ids = {int(c['id']) for c in gt['categories']}
    if not set(map(int, np.unique(z['label']))).issubset(label_ids):
        raise ValueError('Prediction labels and annotation category IDs differ')
    width, sigma, error, overlap, score, counts = match_edges(z, gt, args.iou_thr, args.border)
    lines = []

    def emit(text=''):
        print(text)
        lines.append(text)

    emit(f'reconstruction error: {reconstruction_error:.9e}')
    emit(f'project support: [{z["project"].min()}, {z["project"].max()}]')
    emit('selection counts: ' + json.dumps(counts, sort_keys=True))
    emit(f'matched pairs: {len(sigma)}')
    if not len(sigma):
        if args.output:
            Path(args.output).write_text('\n'.join(lines) + '\n')
        raise ValueError('No matched detections after exclusions')
    absolute_error = np.abs(error)
    correlations = [corr(sigma[:, e], absolute_error[:, e]) for e in range(4)]
    median_corr = float(np.median(correlations))
    result = dict(reconstruction_error=reconstruction_error, selection=counts,
                  projected_support=z['project'][[0, -1]].tolist(),
                  overall_correlations=correlations, median_correlation=median_corr)
    emit('\n== 1. corr(sigma, absolute edge error), projected units ==')
    emit(f'{"edge":8s} {"n":>7s} {"corr":>9s} {"mean_sigma":>11s} {"mean_abs_error":>15s}')
    for e, name in enumerate(EDGE):
        emit(f'{name:8s} {len(sigma):7d} {correlations[e]:9.5f} '
             f'{sigma[:, e].mean():11.5f} {absolute_error[:, e].mean():15.5f}')
    emit(f'median correlation: {median_corr:.6f}')

    emit('\n== 1b. by GT width in source pixels (not COCO area bins) ==')
    sizes = {}
    for name, lo, hi in SIZE_BINS:
        selected = (width >= lo) & (width < hi)
        cs = [corr(sigma[selected, e], absolute_error[selected, e]) for e in range(4)]
        sizes[name] = dict(n=int(selected.sum()), correlations=cs,
                           median_correlation=float(np.median(cs)))
        emit(f'{name:12s} n={selected.sum():6d} corr L/T/R/B = ' + ' '.join(f'{v:8.5f}' for v in cs))
    result['by_gt_width'] = sizes

    mu = sigma.mean(0)
    lr, tb, spread = float(mu[0] / mu[2]), float(mu[1] / mu[3]), float(mu.max() / mu.min())
    emit('\n== 2. edge anisotropy after border exclusion ==')
    emit('mean sigma L/T/R/B: ' + ' '.join(f'{v:.6f}' for v in mu))
    emit(f'L/R={lr:.6f} T/B={tb:.6f} max/min={spread:.6f}')
    result['anisotropy'] = dict(mean_sigma=mu.tolist(), left_right_ratio=lr,
                                top_bottom_ratio=tb, max_min_ratio=spread)

    emit('\n== 3. non-overlapping sigma quintiles ==')
    calibration = {}
    for e, name in enumerate(EDGE):
        s, a = sigma[:, e], absolute_error[:, e]
        quantiles = np.quantile(s, np.linspace(0, 1, 6))
        rows = []
        emit(name + ':')
        for i in range(5):
            selected = (s >= quantiles[i]) & ((s <= quantiles[i + 1]) if i == 4 else (s < quantiles[i + 1]))
            n = int(selected.sum())
            mean_error = float(a[selected].mean()) if n else float('nan')
            rows.append(dict(quintile=i + 1, n=n, lower=float(quantiles[i]), upper=float(quantiles[i + 1]),
                             mean_absolute_error=mean_error, correlation=corr(s[selected], a[selected])))
            emit(f'  q{i+1} [{quantiles[i]:.6f},{quantiles[i+1]:.6f}] n={n:6d} mean_abs_error={mean_error:.6f}')
        nondecreasing = bool(all(rows[i + 1]['mean_absolute_error'] >= rows[i]['mean_absolute_error'] for i in range(4)))
        calibration[name] = dict(rows=rows, nondecreasing=nondecreasing)
        emit(f'  monotone nondecreasing: {nondecreasing}')
    result['calibration'] = calibration

    mean_sigma = sigma.mean(1)
    emit('\n== 4. exploratory scalar signal on matched detections only ==')
    combined = dict(corr_mean_sigma_iou=corr(mean_sigma, overlap),
                    corr_mean_sigma_one_minus_iou=corr(mean_sigma, 1 - overlap),
                    corr_mean_sigma_score=corr(mean_sigma, score))
    emit(json.dumps(combined, sort_keys=True))
    emit('This selection excludes unmatched detections and does not establish AP ranking gain.')
    result['combined_signal'] = combined

    first = 'none within distribution-only criteria; AP_IQS criterion evaluated separately'
    if median_corr < 0.15:
        first = 'criterion 1: median correlation below 0.15; terminate this direction'
    elif not np.isfinite(median_corr) or median_corr < 0.30:
        first = 'criterion 1: insufficient/uncertain median correlation; do not design a module'
    elif sizes['tiny<32']['median_correlation'] < median_corr:
        first = 'criterion 2: tiny-width median correlation lower than overall; terminate per protocol'
    elif not np.isfinite(sizes['tiny<32']['median_correlation']):
        first = 'criterion 2: insufficient tiny-width matches; do not design a module'
    elif spread < 1.05:
        first = 'criterion 3: nearly isotropic edge spread; terminate per protocol'
    elif spread < 1.15:
        first = 'criterion 3: uncertain edge anisotropy; do not design a module'
    elif not all(x['nondecreasing'] for x in calibration.values()):
        first = 'criterion 4: calibration not monotone on every edge; do not design a module'
    result['first_distribution_gate_issue'] = first
    emit('\nfirst distribution gate issue: ' + first)
    if args.output:
        path = Path(args.output)
        path.write_text('\n'.join(lines) + '\n')
        path.with_suffix('.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--npz', required=True)
    parser.add_argument('--gt', required=True)
    parser.add_argument('--iou-thr', type=float, default=0.5)
    parser.add_argument('--border', type=float, default=0.02)
    parser.add_argument('--output')
    main(parser.parse_args())
