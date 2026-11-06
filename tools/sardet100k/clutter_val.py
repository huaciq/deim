"""Val-only: how much AP depends on high-score background false positives, and can they be told from weak targets?"""
import json
import multiprocessing as mp
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

RUN = Path(sys.argv[1])
OUT = RUN / 'independent_verify_20261106/clutter'
OUT.mkdir(parents=True, exist_ok=True)
ROOT = Path('/root/zxy/TCRA-SAR/datasets/SARDet_100K')
val = json.load(open(ROOT / 'Annotations/val.json'))
saved = np.load(RUN / 'independent_verify_20261106/val_predictions.npz')
PI, PC, PS, PB = saved['image_id'], saved['label'].astype(int), saved['score'], saved['box']
names = {c['id']: c['name'] for c in val['categories']}
images = {i['id']: i for i in val['images']}
gts = defaultdict(list)
for a in val['annotations']:
    x, y, w, h = a['bbox']
    gts[a['image_id']].append((a['category_id'], np.array([x, y, x + w, y + h], np.float32)))
order = np.argsort(PI, kind='stable')
vals, first = np.unique(PI[order], return_index=True)
SLICES = dict(zip(vals.tolist(), np.split(order, first[1:])))
THR = np.linspace(.5, .95, 10)
REC = np.linspace(0, 1, 101)


def iou(a, b):
    x1, y1 = np.maximum(a[:, None, 0], b[:, 0]), np.maximum(a[:, None, 1], b[:, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[:, 2]), np.minimum(a[:, None, 3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = (a[:, 2] - a[:, 0])[:, None] * (a[:, 3] - a[:, 1])[:, None] + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return inter / np.maximum(union, 1e-9)


# Background = IoU < .1 with every GT of any class. Weak target = same-class IoU >= .5 and score below .3.
background = np.zeros(len(PS), bool)
weak_tp = np.zeros(len(PS), bool)
strong_tp = np.zeros(len(PS), bool)
for image_id, pos in SLICES.items():
    rows = gts.get(image_id, [])
    if not rows:
        background[pos] = True
        continue
    boxes = np.stack([r[1] for r in rows])
    labels = np.array([r[0] for r in rows])
    m = iou(PB[pos], boxes)
    background[pos] = m.max(1) < .1
    same = m * (PC[pos][:, None] == labels[None, :])
    # Best-scoring box for each GT at IoU >= .5.
    for g in range(len(rows)):
        candidates = np.flatnonzero(same[:, g] >= .5)
        if len(candidates):
            top = candidates[np.argmax(PS[pos[candidates]])]
            (strong_tp if PS[pos[top]] >= .3 else weak_tp)[pos[top]] = True


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


def evaluate(removed):
    table = {}
    for c, name in names.items():
        lines = {t: ([], []) for t in range(10)}
        n = 0
        for image_id in images:
            rows = [r for r in gts.get(image_id, []) if r[0] == c]
            n += len(rows)
            pos = SLICES.get(image_id, np.array([], int))
            keep = pos[(PC[pos] == c) & ~removed[pos]]
            if not len(keep):
                continue
            keep = keep[np.argsort(-PS[keep], kind='mergesort')][:100]
            m = iou(PB[keep], np.stack([r[1] for r in rows])) if rows else None
            for t, th in enumerate(THR):
                used = set()
                hit = np.zeros(len(keep), bool)
                if rows:
                    for d in range(len(keep)):
                        if m[d].max() < th:
                            continue
                        best, chosen = th, -1
                        for g in range(len(rows)):
                            if g not in used and m[d, g] >= best:
                                best, chosen = m[d, g], g
                        if chosen >= 0:
                            used.add(chosen)
                            hit[d] = True
                lines[t][0].append(PS[keep])
                lines[t][1].append(hit)
        v = [ap_curve(np.concatenate(lines[t][0]), np.concatenate(lines[t][1]), n) for t in range(10)]
        table[name] = dict(AP=round(float(np.mean(v)) * 100, 2), AP50=round(v[0] * 100, 2), AP75=round(v[5] * 100, 2))
    table['mean'] = {k: round(float(np.mean([table[n][k] for n in names.values()])), 2) for k in ('AP', 'AP50', 'AP75')}
    return table


def top_share_mask(share):
    mask = np.zeros(len(PS), bool)
    for c in names:
        idx = np.flatnonzero(background & (PC == c))
        idx = idx[np.argsort(-PS[idx])]
        mask[idx[:int(len(idx) * share)]] = True
    return mask


def score_mask(threshold):
    return background & (PS >= threshold)


def task(item):
    label, mask = item
    return label, evaluate(mask)


def features(args):
    image_id, positions = args
    im = images[image_id]
    values = np.asarray(Image.open(ROOT / 'JPEGImages/val' / im['file_name']).convert('L'), np.float32)
    h, w = values.shape
    out = []
    for p in positions:
        x0, y0, x1, y1 = PB[p]
        x0, y0 = int(np.clip(x0, 0, w - 1)), int(np.clip(y0, 0, h - 1))
        x1, y1 = int(np.clip(np.ceil(x1), x0 + 1, w)), int(np.clip(np.ceil(y1), y0 + 1, h))
        pad = max(x1 - x0, y1 - y0)
        ax0, ay0, ax1, ay1 = max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, w), min(y1 + pad, h)
        inner = values[y0:y1, x0:x1]
        mask = np.ones((ay1 - ay0, ax1 - ax0), bool)
        mask[y0 - ay0:y1 - ay0, x0 - ax0:x1 - ax0] = False
        ring = values[ay0:ay1, ax0:ax1][mask]
        base = ring.mean() if ring.size else 1.
        spread = ring.std() if ring.size else 1.
        out.append((p, float(inner.mean() / max(base, 1.)), float(inner.max() / max(base, 1.)), float((inner.mean() - base) / max(spread, 1.)), float(base)))
    return out


def auc(pos, neg):
    if not len(pos) or not len(neg):
        return None
    joined = np.concatenate([pos, neg])
    unique, inverse, counts = np.unique(joined, return_inverse=True, return_counts=True)
    rank = np.argsort(np.argsort(joined, kind='mergesort'), kind='mergesort') + 1.
    rank = (np.bincount(inverse, rank) / counts)[inverse]
    return round(float((rank[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))), 3)


if __name__ == '__main__':
    report = {}
    report['counts'] = {}
    for c, name in names.items():
        cls = PC == c
        report['counts'][name] = {f'bkg_ge_{t}': int((background & cls & (PS >= t)).sum()) for t in (.1, .3, .5, .7)}
        report['counts'][name].update(strong_tp=int((strong_tp & cls).sum()), weak_tp=int((weak_tp & cls).sum()))
    masks = [('base', np.zeros(len(PS), bool)), ('remove_all_bkg', background)]
    masks += [(f'remove_top_{int(s * 100)}pct_bkg', top_share_mask(s)) for s in (.001, .01, .05)]
    masks += [(f'remove_bkg_ge_{t}', score_mask(t)) for t in (.5, .3, .1, .05)]
    with mp.Pool(10) as pool:
        report['ap'] = dict(pool.imap_unordered(task, masks))
        # Appearance overlap: high-score clutter against weak targets, same class.
        selected = defaultdict(list)
        chosen = np.flatnonzero((background & (PS >= .3)) | weak_tp)
        for p in chosen:
            selected[int(PI[p])].append(int(p))
        rows = [r for part in pool.map(features, list(selected.items()), chunksize=32) for r in part]
    feature = np.array(rows, np.float64)
    index = feature[:, 0].astype(int)
    report['weak_target_vs_high_score_clutter'] = {}
    for c, name in names.items():
        cls = PC[index] == c
        w = cls & weak_tp[index]
        f = cls & background[index]
        report['weak_target_vs_high_score_clutter'][name] = dict(
            weak=int(w.sum()), clutter=int(f.sum()),
            auc_mean_ratio=auc(feature[w, 1], feature[f, 1]), auc_peak_ratio=auc(feature[w, 2], feature[f, 2]), auc_contrast_z=auc(feature[w, 3], feature[f, 3]), auc_ring_level=auc(feature[w, 4], feature[f, 4]),
            median_ratio_weak=round(float(np.median(feature[w, 1])), 3) if w.any() else None, median_ratio_clutter=round(float(np.median(feature[f, 1])), 3) if f.any() else None)
    (OUT / 'report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, ensure_ascii=False))
    for c, name in names.items():
        idx = np.flatnonzero(background & (PC == c))
        idx = idx[np.argsort(-PS[idx])][:48]
        cell = 240
        sheet = Image.new('RGB', (cell * 8, cell * 6))
        for i, p in enumerate(idx):
            im = images[int(PI[p])]
            image = Image.open(ROOT / 'JPEGImages/val' / im['file_name']).convert('RGB')
            x0, y0, x1, y1 = PB[p]
            pad = max(max(x1 - x0, y1 - y0) * 3, 40)
            box = (int(max(x0 - pad, 0)), int(max(y0 - pad, 0)), int(min(x1 + pad, im['width'])), int(min(y1 + pad, im['height'])))
            scale = cell / max(box[2] - box[0], box[3] - box[1])
            tile = image.crop(box).resize((int((box[2] - box[0]) * scale), int((box[3] - box[1]) * scale)), Image.BILINEAR)
            draw = ImageDraw.Draw(tile)
            draw.rectangle([(x0 - box[0]) * scale, (y0 - box[1]) * scale, (x1 - box[0]) * scale, (y1 - box[1]) * scale], outline=(255, 255, 0))
            for _, g in gts.get(int(PI[p]), []):
                draw.rectangle([(g[0] - box[0]) * scale, (g[1] - box[1]) * scale, (g[2] - box[0]) * scale, (g[3] - box[1]) * scale], outline=(0, 255, 0))
            draw.text((2, 2), f'{PS[p]:.2f}', fill=(255, 64, 64))
            sheet.paste(tile, (i % 8 * cell, i // 8 * cell))
        sheet.save(OUT / f'top_fp_{name}.png')
    print('DONE', flush=True)
