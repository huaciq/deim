"""Val-only zero-training test: does a label-free dark-water map add information beyond the model score?"""
import json
import multiprocessing as mp
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

RUN = Path(sys.argv[1])
ROOT = Path('/root/zxy/TCRA-SAR/datasets/SARDet_100K')
val = json.load(open(ROOT / 'Annotations/val.json'))
saved = np.load(RUN / 'independent_verify_20261106/val_predictions.npz')
PI, PC, PS, PB = saved['image_id'], saved['label'].astype(int), saved['score'].astype(np.float64), saved['box']
names = {c['id']: c['name'] for c in val['categories']}
images = {i['id']: i for i in val['images']}
gts = defaultdict(list)
for a in val['annotations']:
    x, y, w, h = a['bbox']
    gts[a['image_id']].append((a['category_id'], np.array([x, y, x + w, y + h], np.float32), max(w, h)))
order = np.argsort(PI, kind='stable')
vals, first = np.unique(PI[order], return_index=True)
SLICES = dict(zip(vals.tolist(), np.split(order, first[1:])))
THR = np.linspace(.5, .95, 10)
REC = np.linspace(0, 1, 101)
MIN_SCORE = .01


def water_map(values):
    """Connected low-backscatter regions: dark after smoothing, compared with the image's own median."""
    smooth = ndi.uniform_filter(values, 5)
    median = np.median(smooth)
    dark = smooth < .5 * max(median, 1.)
    labels, count = ndi.label(dark)
    if count:
        sizes = ndi.sum(dark, labels, range(1, count + 1))
        keep = np.flatnonzero(sizes >= .001 * values.size) + 1
        dark = np.isin(labels, keep)
    return dark


def features(box, integral, dark, shape):
    height, width = shape
    x0, y0, x1, y1 = box
    w, h = max(x1 - x0, 1), max(y1 - y0, 1)
    pad = max(w, h)

    def frac(a0, b0, a1, b1):
        a0, b0 = int(np.clip(a0, 0, width)), int(np.clip(b0, 0, height))
        a1, b1 = int(np.clip(np.ceil(a1), 0, width)), int(np.clip(np.ceil(b1), 0, height))
        if a1 <= a0 or b1 <= b0:
            return 0., 0
        area = (a1 - a0) * (b1 - b0)
        return float(integral[b1, a1] - integral[b0, a1] - integral[b1, a0] + integral[b0, a0]), area

    inner, inner_area = frac(x0, y0, x1, y1)
    outer, outer_area = frac(x0 - pad, y0 - pad, x1 + pad, y1 + pad)
    ring_area = max(outer_area - inner_area, 1)
    # Two-sided evidence: dark on opposite sides along either axis.
    left, la = frac(x0 - pad, y0, x0, y1)
    right, ra = frac(x1, y0, x1 + pad, y1)
    top, ta = frac(x0, y0 - pad, x1, y0)
    bottom, ba = frac(x0, y1, x1, y1 + pad)
    side = lambda v, a: v / a if a else 0.
    opposite = max(min(side(left, la), side(right, ra)), min(side(top, ta), side(bottom, ba)))
    return (outer - inner) / ring_area, inner / max(inner_area, 1), opposite


def process(image_id):
    im = images[image_id]
    values = np.asarray(Image.open(ROOT / 'JPEGImages/val' / im['file_name']).convert('L'), np.float32)
    dark = water_map(values)
    integral = np.zeros((dark.shape[0] + 1, dark.shape[1] + 1), np.float64)
    integral[1:, 1:] = dark.cumsum(0).cumsum(1)
    pos = SLICES.get(image_id, np.array([], int))
    pos = pos[PS[pos] >= MIN_SCORE]
    out = [features(PB[p], integral, dark, dark.shape) for p in pos]
    gt_out = [(c, s) + features(b, integral, dark, dark.shape) for c, b, s in gts.get(image_id, [])]
    # Channel width estimate near each bridge GT.
    distance = ndi.distance_transform_edt(dark) if dark.any() else np.zeros_like(values)
    widths = []
    for c, b, s in gts.get(image_id, []):
        if c == 4:
            pad = int(max(b[2] - b[0], b[3] - b[1]))
            x0, y0, x1, y1 = int(max(b[0] - pad, 0)), int(max(b[1] - pad, 0)), int(b[2] + pad + 1), int(b[3] + pad + 1)
            region = distance[y0:y1, x0:x1]
            widths.append((float(2 * region.max()) if region.size else 0., float(s)))
    return image_id, pos, np.array(out, np.float32).reshape(-1, 3), gt_out, widths, float(dark.mean())


def iou(a, b):
    x1, y1 = np.maximum(a[:, None, 0], b[:, 0]), np.maximum(a[:, None, 1], b[:, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[:, 2]), np.minimum(a[:, None, 3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = (a[:, 2] - a[:, 0])[:, None] * (a[:, 3] - a[:, 1])[:, None] + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return inter / np.maximum(union, 1e-9)


def auc(pos, neg):
    if not len(pos) or not len(neg):
        return None
    joined = np.concatenate([pos, neg])
    rank = np.argsort(np.argsort(joined, kind='mergesort'), kind='mergesort') + 1.
    # Average ties.
    unique, inverse, counts = np.unique(joined, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, rank)
    rank = (sums / counts)[inverse]
    return round(float((rank[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))), 3)


def ap_curve(s, h, n):
    r = np.argsort(-s, kind='mergesort')
    h = h[r]
    tp, fp = np.cumsum(h), np.cumsum(~h)
    rc = tp / n
    pr = tp / np.maximum(tp + fp, 1e-12)
    pr = np.maximum.accumulate(pr[::-1])[::-1] if len(pr) else pr
    w = np.searchsorted(rc, REC, side='left')
    o = np.zeros(101)
    k = w < len(pr)
    o[k] = pr[w[k]]
    return o.mean()


def evaluate(scores, c):
    lines = {t: ([], []) for t in range(10)}
    n = 0
    for image_id in images:
        rows = [g for g in gts.get(image_id, []) if g[0] == c]
        n += len(rows)
        pos = SLICES.get(image_id, np.array([], int))
        keep = pos[PC[pos] == c]
        if not len(keep):
            continue
        keep = keep[np.argsort(-scores[keep], kind='mergesort')][:100]
        m = iou(PB[keep], np.stack([g[1] for g in rows])) if rows else None
        for t, th in enumerate(THR):
            used = set()
            hit = np.zeros(len(keep), bool)
            if rows:
                for d in range(len(keep)):
                    best, chosen = th, -1
                    for g in range(len(rows)):
                        if g not in used and m[d, g] >= best:
                            best, chosen = m[d, g], g
                    if chosen >= 0:
                        used.add(chosen)
                        hit[d] = True
            lines[t][0].append(scores[keep])
            lines[t][1].append(hit)
    v = [ap_curve(np.concatenate(lines[t][0]), np.concatenate(lines[t][1]), n) for t in range(10)]
    return dict(AP=round(float(np.mean(v)) * 100, 2), AP50=round(v[0] * 100, 2), AP75=round(v[5] * 100, 2))


if __name__ == '__main__':
    with mp.Pool(32) as pool:
        results = pool.map(process, list(images), chunksize=32)
    ring = np.zeros(len(PS), np.float32)
    inside = np.zeros(len(PS), np.float32)
    opposite = np.zeros(len(PS), np.float32)
    has = np.zeros(len(PS), bool)
    gt_rows, widths, dark_share = [], [], []
    for image_id, pos, feats, gt_out, w, share in results:
        ring[pos], inside[pos], opposite[pos] = feats[:, 0], feats[:, 1], feats[:, 2]
        has[pos] = True
        gt_rows += gt_out
        widths += w
        dark_share.append(share)
    report = dict(images_with_water_over_1pct=round(float(np.mean(np.array(dark_share) > .01)), 3))
    gt_rows = np.array(gt_rows, np.float32)
    report['gt_water_evidence'] = {}
    for c, name in names.items():
        m = gt_rows[:, 0] == c
        report['gt_water_evidence'][name] = dict(n=int(m.sum()), ring_median=round(float(np.median(gt_rows[m, 2])), 3), ring_ge_0_1=round(float((gt_rows[m, 2] >= .1).mean()), 3), inside_median=round(float(np.median(gt_rows[m, 3])), 3), opposite_ge_0_3=round(float((gt_rows[m, 4] >= .3).mean()), 3))
    if widths:
        arr = np.array(widths)
        valid = arr[:, 0] > 0
        report['bridge_channel_width'] = dict(n=int(len(arr)), with_channel=round(float(valid.mean()), 3), pearson_log=round(float(np.corrcoef(np.log(arr[valid, 0]), np.log(arr[valid, 1]))[0, 1]), 3), median_channel=round(float(np.median(arr[valid, 0])), 1), median_long_side=round(float(np.median(arr[valid, 1])), 1))
    # Per-image greedy matching at IoU .5 of detections above MIN_SCORE.
    label = np.full(len(PS), -1, np.int8)  # 1 TP, 0 background FP (IoU < .1 with same class), 2 other
    for image_id, pos in SLICES.items():
        for c in names:
            keep = pos[(PC[pos] == c) & (PS[pos] >= MIN_SCORE)]
            if not len(keep):
                continue
            keep = keep[np.argsort(-PS[keep], kind='mergesort')]
            rows = [g for g in gts.get(image_id, []) if g[0] == c]
            if not rows:
                label[keep] = 0
                continue
            m = iou(PB[keep], np.stack([g[1] for g in rows]))
            used = set()
            for d, p in enumerate(keep):
                best, chosen = .5, -1
                for g in range(len(rows)):
                    if g not in used and m[d, g] >= best:
                        best, chosen = m[d, g], g
                if chosen >= 0:
                    used.add(chosen)
                    label[p] = 1
                else:
                    label[p] = 0 if m[d].max() < .1 else 2
    report['separation'] = {}
    for c, name in names.items():
        entry = {}
        for band, low, high in [('all', MIN_SCORE, 1.01), ('0.05-0.5', .05, .5), ('0.1-0.3', .1, .3)]:
            m = (PC == c) & (PS >= low) & (PS < high)
            tp, fp = m & (label == 1), m & (label == 0)
            entry[band] = dict(tp=int(tp.sum()), fp=int(fp.sum()), auc_score=auc(PS[tp], PS[fp]), auc_ring=auc(ring[tp], ring[fp]), auc_opposite=auc(opposite[tp], opposite[fp]), auc_inside=auc(inside[tp], inside[fp]), ring_tp=round(float(np.median(ring[tp])), 3) if tp.any() else None, ring_fp=round(float(np.median(ring[fp])), 3) if fp.any() else None)
        report['separation'][name] = entry
    report['rescore'] = {}
    logit = np.log(np.clip(PS, 1e-6, 1 - 1e-6) / (1 - np.clip(PS, 1e-6, 1 - 1e-6)))
    for c, name in names.items():
        table = {'base': evaluate(PS, c)}
        mask = (PC == c) & has
        for feature, values in [('ring', ring), ('opposite', opposite)]:
            for sign in (1, -1):
                for beta in (.5, 1., 2.):
                    scores = PS.copy()
                    scores[mask] = 1 / (1 + np.exp(-(logit[mask] + sign * beta * (values[mask] - np.median(values[mask & (label == 1)])))))
                    table[f'{feature}_{"+" if sign > 0 else "-"}{beta}'] = evaluate(scores, c)
        report['rescore'][name] = table
    Path(RUN / 'independent_verify_20261106/explore_tiny/water.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, ensure_ascii=False))
