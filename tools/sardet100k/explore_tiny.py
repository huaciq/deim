"""Val-only exploration: density, within-image size consistency, label-free post-processing, and sample montages."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

RUN = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
ROOT = Path('/root/zxy/TCRA-SAR/datasets/SARDet_100K')
val = json.load(open(ROOT / 'Annotations/val.json'))
saved = np.load(RUN / 'independent_verify_20261106/val_predictions.npz')
PI, PC, PS, PB = saved['image_id'], saved['label'].astype(int), saved['score'], saved['box']
names = {c['id']: c['name'] for c in val['categories']}
images = {i['id']: i for i in val['images']}
THR = np.linspace(.5, .95, 10)
REC = np.linspace(0, 1, 101)
gts = defaultdict(list)
for a in val['annotations']:
    x, y, w, h = a['bbox']
    gts[a['image_id']].append(dict(id=a['id'], cls=a['category_id'], box=np.array([x, y, x + w, y + h], np.float32), w=w, h=h, side=max(w, h)))
order = np.argsort(PI, kind='stable')
vals, first = np.unique(PI[order], return_index=True)
SLICES = dict(zip(vals.tolist(), np.split(order, first[1:])))
report = {}


def iou(a, b):
    x1, y1 = np.maximum(a[:, None, 0], b[:, 0]), np.maximum(a[:, None, 1], b[:, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[:, 2]), np.minimum(a[:, None, 3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = (a[:, 2] - a[:, 0])[:, None] * (a[:, 3] - a[:, 1])[:, None] + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return inter / np.maximum(union, 1e-9)


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


def evaluate(boxes, scores, classes_only=None):
    """COCO AP (area all, top 100 per image and class) for arbitrary boxes and scores."""
    table = {}
    for c, name in names.items():
        if classes_only and c not in classes_only:
            continue
        lines = {t: ([], []) for t in range(10)}
        n = 0
        for image_id in images:
            rows = [g for g in gts.get(image_id, []) if g['cls'] == c]
            n += len(rows)
            pos = SLICES.get(image_id, np.array([], int))
            keep = pos[PC[pos] == c]
            if not len(keep):
                continue
            keep = keep[np.argsort(-scores[keep], kind='mergesort')][:100]
            m = iou(boxes[keep], np.stack([g['box'] for g in rows])) if rows else None
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
        table[name] = dict(AP=round(float(np.mean(v)) * 100, 2), AP50=round(v[0] * 100, 2), AP75=round(v[5] * 100, 2))
    return table


# --- density and images per class
density = {}
for c, name in names.items():
    counts = np.array([sum(g['cls'] == c for g in rows) for rows in gts.values()])
    counts = counts[counts > 0]
    per_gt = np.repeat(counts, counts)
    density[name] = dict(images=int(len(counts)), gt=int(counts.sum()), median_per_image=float(np.median(counts)), gt_weighted_median=float(np.median(per_gt)), p90_gt_weighted=float(np.percentile(per_gt, 90)), max=int(counts.max()), gt_in_images_over_100=int(counts[counts > 100].sum()), gt_in_images_over_50=int(counts[counts > 50].sum()))
report['density'] = density

# --- recall vs density at IoU .5 for tank/bridge/ship
recall_density = {}
for c, name in names.items():
    bins = defaultdict(lambda: [0, 0, 0])
    for image_id, rows in gts.items():
        cur = [g for g in rows if g['cls'] == c]
        if not cur:
            continue
        pos = SLICES.get(image_id, np.array([], int))
        keep = pos[(PC[pos] == c)]
        keep = keep[np.argsort(-PS[keep])][:100]
        high = keep[PS[keep] >= .3]
        boxes = np.stack([g['box'] for g in cur])
        best_all = iou(PB[keep], boxes).max(0) if len(keep) else np.zeros(len(cur))
        best_high = iou(PB[high], boxes).max(0) if len(high) else np.zeros(len(cur))
        k = next(b for b in [1, 2, 5, 10, 20, 50, 100, 10**6] if len(cur) <= b)
        bins[k][0] += len(cur)
        bins[k][1] += int((best_all >= .5).sum())
        bins[k][2] += int((best_high >= .5).sum())
    recall_density[name] = {str(k): dict(gt=v[0], recall_any=round(v[1] / v[0], 3), recall_0_3=round(v[2] / v[0], 3)) for k, v in sorted(bins.items())}
report['recall_by_density'] = recall_density

# --- within-image size consistency of GT and of predictions
consistency = {}
for c, name in names.items():
    gt_w, gt_h, pred = [], [], []
    for image_id, rows in gts.items():
        cur = [g for g in rows if g['cls'] == c]
        if len(cur) >= 3:
            w, h = np.array([g['w'] for g in cur]), np.array([g['h'] for g in cur])
            gt_w.append(w.std() / max(w.mean(), 1e-3))
            gt_h.append(h.std() / max(h.mean(), 1e-3))
    if gt_w:
        consistency[name] = dict(images=len(gt_w), median_cv_w=round(float(np.median(gt_w)), 3), median_cv_h=round(float(np.median(gt_h)), 3))
report['gt_size_cv_within_image'] = consistency


# --- label-free post-processing candidates (val only)
def size_consensus(strength, classes, seed_score=.5, min_seeds=3):
    boxes = PB.copy()
    for image_id, pos in SLICES.items():
        for c in classes:
            keep = pos[PC[pos] == c]
            seeds = keep[PS[keep] >= seed_score]
            if len(seeds) < min_seeds:
                continue
            ww = PB[seeds, 2] - PB[seeds, 0]
            hh = PB[seeds, 3] - PB[seeds, 1]
            mw, mh = np.median(ww), np.median(hh)
            cx = (PB[keep, 0] + PB[keep, 2]) / 2
            cy = (PB[keep, 1] + PB[keep, 3]) / 2
            w = PB[keep, 2] - PB[keep, 0]
            h = PB[keep, 3] - PB[keep, 1]
            nw = w * (1 - strength) + mw * strength
            nh = h * (1 - strength) + mh * strength
            boxes[keep] = np.stack([cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2], 1)
    return boxes


def context_rescore(strength, classes, seed_score=.5, min_seeds=2):
    scores = PS.copy()
    for image_id, pos in SLICES.items():
        for c in classes:
            keep = pos[PC[pos] == c]
            seeds = keep[PS[keep] >= seed_score]
            if len(seeds) < min_seeds:
                continue
            ww = PB[seeds, 2] - PB[seeds, 0]
            hh = PB[seeds, 3] - PB[seeds, 1]
            mw, mh = np.median(ww), np.median(hh)
            w = PB[keep, 2] - PB[keep, 0]
            h = PB[keep, 3] - PB[keep, 1]
            ratio = np.minimum(w / max(mw, 1e-3), mw / np.maximum(w, 1e-3)) * np.minimum(h / max(mh, 1e-3), mh / np.maximum(h, 1e-3))
            scores[keep] = PS[keep] ** (1 - strength * ratio)
    return scores


base = evaluate(PB, PS)
report['base'] = base
report['size_consensus'] = {}
report['context_rescore'] = {}
for strength in (.5, 1.):
    for label, cls_set in [('tank', [3]), ('car', [2]), ('aircraft', [1]), ('ship', [0])]:
        report['size_consensus'][f'{label}_{strength}'] = evaluate(size_consensus(strength, cls_set), PS, classes_only=cls_set)
for strength in (.3, .6):
    for label, cls_set in [('tank', [3]), ('car', [2]), ('aircraft', [1]), ('ship', [0]), ('bridge', [4])]:
        report['context_rescore'][f'{label}_{strength}'] = evaluate(PB, context_rescore(strength, cls_set), classes_only=cls_set)

# --- within-image score vs class of source
source = {}
for c, name in names.items():
    rows = Counter()
    for image_id, cur in gts.items():
        for g in cur:
            if g['cls'] == c:
                rows[images[image_id]['width']] += 1
    source[name] = rows.most_common(4)
report['class_source_width'] = source
(OUT / 'explore.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))


# --- montages
def montage(name, cls, chooser, title_count=40, cell=240):
    picked = []
    for image_id, rows in gts.items():
        cur = [g for g in rows if g['cls'] == cls and g['side'] < 16]
        if cur:
            picked.append(image_id)
    rng = np.random.default_rng(0)
    rng.shuffle(picked)
    tiles = []
    for image_id in picked:
        im = images[image_id]
        image = Image.open(ROOT / 'JPEGImages/val' / im['file_name']).convert('RGB')
        scale = cell / max(im['width'], im['height'])
        image = image.resize((int(im['width'] * scale), int(im['height'] * scale)), Image.BILINEAR)
        draw = ImageDraw.Draw(image)
        rows = gts[image_id]
        pos = SLICES.get(image_id, np.array([], int))
        keep = pos[(PC[pos] == cls) & (PS[pos] >= .3)]
        boxes = np.stack([g['box'] for g in rows])
        matched = iou(PB[keep], boxes).max(0) if len(keep) else np.zeros(len(rows))
        miss = 0
        for g, m in zip(rows, matched):
            if g['cls'] != cls:
                continue
            color = (0, 255, 0) if m >= .5 else (255, 0, 0)
            miss += m < .5
            draw.rectangle(list(g['box'] * scale), outline=color)
        for b in PB[keep]:
            draw.rectangle(list(b * scale), outline=(255, 255, 0))
        if chooser(miss, rows, cls):
            tiles.append(image)
        if len(tiles) == title_count:
            break
    sheet = Image.new('RGB', (cell * 8, cell * 5))
    for i, t in enumerate(tiles):
        sheet.paste(t, (i % 8 * cell, i // 8 * cell))
    sheet.save(OUT / name)


montage('tank_miss.png', 3, lambda miss, rows, cls: miss >= 2)
montage('tank_ok.png', 3, lambda miss, rows, cls: miss == 0)
montage('bridge_miss.png', 4, lambda miss, rows, cls: miss >= 1)
montage('bridge_ok.png', 4, lambda miss, rows, cls: miss == 0)
print(json.dumps(report, ensure_ascii=False))
