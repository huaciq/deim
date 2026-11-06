"""Zero-training checks on val: annotation precision limits and out-of-image boxes."""
import json
import multiprocessing as mp
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

RUN = Path(sys.argv[1])
ROOT = Path('/root/zxy/TCRA-SAR/datasets/SARDet_100K')
val = json.load(open(ROOT / 'Annotations/val.json'))
train = json.load(open(ROOT / 'Annotations/train.json'))
saved = np.load(RUN / 'independent_verify_20261106/val_predictions.npz')
pi, pc, ps, pb = saved['image_id'], saved['label'].astype(int), saved['score'], saved['box']
names = {c['id']: c['name'] for c in val['categories']}
images = {i['id']: i for i in val['images']}
THRESHOLDS = np.linspace(.5, .95, 10)
RECALLS = np.linspace(0, 1, 101)
report = {}


def corners(a):
    x, y, w, h = a['bbox']
    return np.array([x, y, x + w, y + h], np.float32)


def iou(a, b):
    x1, y1 = np.maximum(a[:, None, 0], b[:, 0]), np.maximum(a[:, None, 1], b[:, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[:, 2]), np.minimum(a[:, None, 3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = (a[:, 2] - a[:, 0])[:, None] * (a[:, 3] - a[:, 1])[:, None] + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return inter / np.maximum(union, 1e-9)


def ap(scores, hit, n):
    rank = np.argsort(-scores, kind='mergesort')
    h = hit[rank]
    tp, fp = np.cumsum(h), np.cumsum(~h)
    r = tp / n
    p = tp / np.maximum(tp + fp, 1e-12)
    p = np.maximum.accumulate(p[::-1])[::-1] if len(p) else p
    where = np.searchsorted(r, RECALLS, side='left')
    out = np.zeros(101)
    ok = where < len(p)
    out[ok] = p[where[ok]]
    return out.mean()


gts = defaultdict(list)
for a in val['annotations']:
    im = images[a['image_id']]
    b = corners(a)
    margin = max(-b[0], -b[1], b[2] - im['width'], b[3] - im['height'])
    gts[a['image_id']].append(dict(id=a['id'], cls=a['category_id'], box=b, side=max(a['bbox'][2], a['bbox'][3]), w=a['bbox'][2], h=a['bbox'][3], out=float(margin), raw=a['bbox']))
order = np.argsort(pi, kind='stable')
values, first = np.unique(pi[order], return_index=True)
slices = dict(zip(values.tolist(), np.split(order, first[1:])))

# 1. coordinate precision
allg = [g for rows in gts.values() for g in rows]
integer = {}
for c, name in names.items():
    rows = [g for g in allg if g['cls'] == c]
    integer[name] = dict(gt=len(rows), all_integer=round(float(np.mean([all(abs(v - round(v)) < 1e-3 for v in g['raw']) for g in rows])), 3),
                         out_of_image_gt_1px=round(float(np.mean([g['out'] > 1 for g in rows])), 4))
report['coordinates'] = integer

# 2. IoU ceiling implied by integer rounding alone, per long side (square box, independent +-0.5 per edge)
rng = np.random.default_rng(0)
ceiling = {}
for side in [4, 6, 8, 10, 12, 16, 24, 32, 64, 128]:
    a = np.array([0, 0, side, side], float)
    e1, e2 = rng.uniform(-.5, .5, (200000, 4)), rng.uniform(-.5, .5, (200000, 4))
    b1, b2 = a + e1, a + e2
    inter = (np.minimum(b1[:, 2], b2[:, 2]) - np.maximum(b1[:, 0], b2[:, 0])) * (np.minimum(b1[:, 3], b2[:, 3]) - np.maximum(b1[:, 1], b2[:, 1]))
    union = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1]) + (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1]) - inter
    v = inter / union
    ceiling[side] = dict(median=round(float(np.median(v)), 3), ge_0_75=round(float((v >= .75).mean()), 3), ge_0_9=round(float((v >= .9).mean()), 3))
report['rounding_only_iou_ceiling'] = ceiling

# 3. per-edge absolute error of confident true positives, in source pixels and in 640-input pixels
edge = defaultdict(list)
out_stats = {'inside': [], 'outside': []}
for image_id, rows in gts.items():
    positions = slices.get(image_id)
    if positions is None:
        continue
    keep = positions[ps[positions] >= .3]
    if not len(keep):
        continue
    im = images[image_id]
    boxes = np.stack([g['box'] for g in rows])
    matrix = iou(pb[keep], boxes)
    labels = np.array([g['cls'] for g in rows])
    matrix[pc[keep][:, None] != labels[None, :]] = 0
    for g_index, g in enumerate(rows):
        d = int(matrix[:, g_index].argmax())
        best = float(matrix[d, g_index])
        if best >= .5:
            error = np.abs(pb[keep][d] - boxes[g_index])
            bin_ = next(b for b in [8, 16, 32, 64, 128, 1e9] if g['side'] < b)
            edge[bin_].append((float(np.median(error)), float(np.median(error) * 640 / im['width']), best, im['width']))
            out_stats['outside' if g['out'] > 1 else 'inside'].append((g['cls'], best))
report['edge_error_px'] = {str(k): dict(n=len(v), median_edge_error_source_px=round(float(np.median([x[0] for x in v])), 2), median_edge_error_input_px=round(float(np.median([x[1] for x in v])), 2), median_iou=round(float(np.median([x[2] for x in v])), 3)) for k, v in sorted(edge.items())}
report['tp_iou_inside_vs_outside'] = {}
for c, name in names.items():
    report['tp_iou_inside_vs_outside'][name] = {k: dict(n=sum(1 for r in v if r[0] == c), median=round(float(np.median([r[1] for r in v if r[0] == c] or [0])), 3), ge_0_75=round(float(np.mean([r[1] >= .75 for r in v if r[0] == c] or [0])), 3)) for k, v in out_stats.items()}


# 4. AP with out-of-image GT ignored, and with those GT clipped to the image
def full_ap(mode):
    result = {}
    for c, name in names.items():
        lines = {t: ([], []) for t in range(10)}
        n = 0
        for image_id, im in images.items():
            rows = [g for g in gts.get(image_id, []) if g['cls'] == c]
            if mode == 'ignore':
                rows.sort(key=lambda g: g['out'] > 1)
                ignored = np.array([g['out'] > 1 for g in rows], bool)
            else:
                ignored = np.zeros(len(rows), bool)
            n += int((~ignored).sum())
            positions = slices.get(image_id, np.array([], int))
            keep = positions[pc[positions] == c]
            if not len(keep):
                continue
            keep = keep[np.argsort(-ps[keep], kind='mergesort')][:100]
            if rows:
                boxes = np.stack([g['box'] for g in rows]).copy()
                if mode == 'clip':
                    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, im['width'])
                    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, im['height'])
                matrix = iou(pb[keep], boxes)
            for t, threshold in enumerate(THRESHOLDS):
                used = np.zeros(len(rows), bool)
                hit, drop = np.zeros(len(keep), bool), np.zeros(len(keep), bool)
                for d in range(len(keep)):
                    if rows and matrix[d].max() >= threshold:
                        best, chosen = threshold, -1
                        for g in range(len(rows)):
                            if used[g]:
                                continue
                            if chosen > -1 and not ignored[chosen] and ignored[g]:
                                break
                            if matrix[d, g] >= best:
                                best, chosen = matrix[d, g], g
                        if chosen > -1:
                            used[chosen] = True
                            hit[d], drop[d] = not ignored[chosen], ignored[chosen]
                lines[t][0].append(ps[keep][~drop])
                lines[t][1].append(hit[~drop])
        result[name] = [float(ap(np.concatenate(lines[t][0]), np.concatenate(lines[t][1]), n)) for t in range(10)]
    matrix = np.array(list(result.values()))
    return dict(AP=round(float(matrix.mean() * 100), 2), AP50=round(float(matrix[:, 0].mean() * 100), 2), AP75=round(float(matrix[:, 5].mean() * 100), 2),
                per_class_AP={k: round(float(np.mean(v)) * 100, 2) for k, v in result.items()})


for mode in ['official', 'clip', 'ignore']:
    report['ap_' + mode] = full_ap(mode)


# 5. labels of pixel-identical val/train images
def thumb(task):
    folder, name = task
    v = np.asarray(Image.open(f'{folder}/{name}'), np.float32)
    v = v.mean(2) if v.ndim == 3 else v
    t = np.asarray(Image.fromarray(v).resize((32, 32), Image.BILINEAR), np.float32).ravel()
    t = t - t.mean()
    n = np.linalg.norm(t)
    return t / n if n > 0 else t


if __name__ == '__main__':
    with mp.Pool(32) as pool:
        tv = np.stack(pool.map(thumb, [(str(ROOT / 'JPEGImages/val'), i['file_name']) for i in val['images']], chunksize=64))
        tt = np.stack(pool.map(thumb, [(str(ROOT / 'JPEGImages/train'), i['file_name']) for i in train['images']], chunksize=64))
    import torch
    a, b = torch.as_tensor(tv).cuda(), torch.as_tensor(tt).cuda()
    best_s, best_i = [], []
    for start in range(0, len(a), 1024):
        s, i = (a[start:start + 1024] @ b.T).max(1)
        best_s.append(s.cpu().numpy())
        best_i.append(i.cpu().numpy())
    best_s, best_i = np.concatenate(best_s), np.concatenate(best_i)
    by_train = defaultdict(list)
    for x in train['annotations']:
        by_train[x['image_id']].append(x)
    pairs, ious = 0, defaultdict(list)
    for index in np.flatnonzero(best_s >= .98):
        vi, ti = val['images'][index], train['images'][best_i[index]]
        if (vi['width'], vi['height']) != (ti['width'], ti['height']):
            continue
        A = np.asarray(Image.open(ROOT / 'JPEGImages/val' / vi['file_name']).convert('L'), np.float32)
        B = np.asarray(Image.open(ROOT / 'JPEGImages/train' / ti['file_name']).convert('L'), np.float32)
        if np.abs(A - B).mean() >= 1:
            continue
        pairs += 1
        rows, other = gts.get(vi['id'], []), by_train[ti['id']]
        if not rows or not other:
            continue
        boxes = np.stack([corners(x) for x in other])
        labels = np.array([x['category_id'] for x in other])
        for g in rows:
            m = iou(g['box'][None], boxes)[0]
            m[labels != g['cls']] = 0
            ious[g['cls']].append(float(m.max()))
    report['pixel_identical_val_train'] = dict(pairs=pairs, classes={names[c]: dict(n=len(v), median_iou=round(float(np.median(v)), 3), exact=round(float(np.mean([x >= .999 for x in v])), 3), ge_0_75=round(float(np.mean([x >= .75 for x in v])), 3)) for c, v in ious.items()})
    (RUN / 'independent_verify_20261106/val_limits.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, ensure_ascii=False))
