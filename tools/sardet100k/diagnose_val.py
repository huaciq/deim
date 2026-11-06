"""CPU diagnosis of saved val predictions: error decomposition, localization, recall and groups."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

RUN = Path(sys.argv[1])
ROOT = Path('/root/zxy/TCRA-SAR/datasets/SARDet_100K')
data = json.load(open(ROOT / 'Annotations/val.json'))
saved = np.load(RUN / 'independent_verify_20261106/val_predictions.npz')
pred_img, pred_cls, pred_score, pred_box = saved['image_id'], saved['label'].astype(int), saved['score'], saved['box']
names = {c['id']: c['name'] for c in data['categories']}
classes = sorted(names)
images = {im['id']: im for im in data['images']}
RECALLS = np.linspace(0, 1, 101)
THRESHOLDS = np.linspace(.5, .95, 10)

gts_by_image = defaultdict(list)
for a in data['annotations']:
    x, y, w, h = a['bbox']
    gts_by_image[a['image_id']].append(dict(id=a['id'], cls=a['category_id'], box=np.array([x, y, x + w, y + h], np.float32), side=max(w, h), short=min(w, h), aspect=max(w, h) / max(min(w, h), 1e-3)))
order = np.argsort(pred_img, kind='stable')
start = {}
values, first = np.unique(pred_img[order], return_index=True)
slices = dict(zip(values.tolist(), np.split(order, first[1:])))


def iou(a, b):
    x1, y1 = np.maximum(a[:, None, 0], b[:, 0]), np.maximum(a[:, None, 1], b[:, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[:, 2]), np.minimum(a[:, None, 3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = (a[:, 2] - a[:, 0])[:, None] * (a[:, 3] - a[:, 1])[:, None] + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return inter / np.maximum(union, 1e-9)


def ap(scores, tp, positives):
    if positives == 0:
        return float('nan')
    rank = np.argsort(-scores, kind='mergesort')
    hit = tp[rank]
    tps, fps = np.cumsum(hit), np.cumsum(~hit)
    recall = tps / positives
    precision = tps / np.maximum(tps + fps, 1e-12)
    precision = np.maximum.accumulate(precision[::-1])[::-1] if len(precision) else precision
    where = np.searchsorted(recall, RECALLS, side='left')
    sampled = np.zeros(101)
    inside = where < len(precision)
    sampled[inside] = precision[where[inside]]
    return float(sampled.mean())


# ---- per-image matching ----
records = []  # image, class, score, kind, tp, target gt id, other class, best same-class IoU
gt_state = {}
per_threshold = {c: {t: [] for t in range(10)} for c in classes}
positives = Counter(a['category_id'] for a in data['annotations'])
gt_best_iou_any_score = {}
gt_best_score_localized = {}
for image_id, im in images.items():
    gts = gts_by_image.get(image_id, [])
    positions = slices.get(image_id, np.array([], int))
    gt_boxes = np.stack([g['box'] for g in gts]) if gts else np.zeros((0, 4), np.float32)
    gt_cls = np.array([g['cls'] for g in gts], int)
    for g in gts:
        gt_best_iou_any_score[g['id']] = 0.
        gt_best_score_localized[g['id']] = 0.
    for c in classes:
        keep = positions[pred_cls[positions] == c]
        keep = keep[np.argsort(-pred_score[keep], kind='mergesort')][:100]
        if not len(keep):
            continue
        boxes, scores = pred_box[keep], pred_score[keep]
        same = np.flatnonzero(gt_cls == c)
        other = np.flatnonzero(gt_cls != c)
        matrix = iou(boxes, gt_boxes) if len(gts) else np.zeros((len(keep), 0))
        for t, threshold in enumerate(THRESHOLDS):
            used = set()
            flags = np.zeros(len(keep), bool)
            for d in range(len(keep)):
                best, chosen = threshold, -1
                for g in same:
                    if g not in used and matrix[d, g] >= best:
                        best, chosen = matrix[d, g], g
                if chosen >= 0:
                    used.add(chosen)
                    flags[d] = True
            per_threshold[c][t].append((scores, flags))
        used = set()
        for d in range(len(keep)):
            row = matrix[d] if len(gts) else np.zeros(0)
            best_same = row[same].max() if len(same) else 0.
            best_other = row[other].max() if len(other) else 0.
            chosen, best = -1, .5
            for g in same:
                if g not in used and row[g] >= best:
                    best, chosen = row[g], g
            target = int(same[row[same].argmax()]) if len(same) else -1
            other_target = int(other[row[other].argmax()]) if len(other) else -1
            if chosen >= 0:
                used.add(chosen)
                records.append((image_id, c, scores[d], 'TP', gts[chosen]['id'], -1, float(row[chosen])))
            elif best_same >= .5:
                records.append((image_id, c, scores[d], 'Dupe', gts[target]['id'], -1, float(best_same)))
            elif best_other >= .5:
                records.append((image_id, c, scores[d], 'Cls', gts[other_target]['id'], int(gts[other_target]['cls']), float(best_other)))
            elif best_same >= .1:
                records.append((image_id, c, scores[d], 'Loc', gts[target]['id'], -1, float(best_same)))
            elif best_other >= .1:
                records.append((image_id, c, scores[d], 'Both', gts[other_target]['id'], int(gts[other_target]['cls']), float(best_other)))
            else:
                records.append((image_id, c, scores[d], 'Bkg', -1, -1, float(max(best_same, best_other))))
        if len(same):
            for g in same:
                gid = gts[g]['id']
                gt_best_iou_any_score[gid] = max(gt_best_iou_any_score[gid], float(matrix[:, g].max()))
                localized = matrix[:, g] >= .5
                if localized.any():
                    gt_best_score_localized[gid] = max(gt_best_score_localized[gid], float(scores[localized].max()))

score = np.array([r[2] for r in records], np.float32)
kind = np.array([r[3] for r in records])
cls = np.array([r[1] for r in records], int)
target = np.array([r[4] for r in records], np.int64)
other_cls = np.array([r[5] for r in records], int)
match_iou = np.array([r[6] for r in records], np.float32)
report = {}

# ---- AP by threshold ----
table = {}
for c in classes:
    table[names[c]] = [round(ap(np.concatenate([s for s, _ in per_threshold[c][t]]), np.concatenate([f for _, f in per_threshold[c][t]]), positives[c]) * 100, 2) for t in range(10)]
matrix = np.array(list(table.values()))
report['ap_by_iou'] = dict(thresholds=[round(float(t), 2) for t in THRESHOLDS], per_class=table,
                           mean=[round(float(v), 2) for v in matrix.mean(0)],
                           AP=round(float(matrix.mean()), 2),
                           per_class_AP={k: round(float(np.mean(v)), 2) for k, v in table.items()})


# ---- TIDE-style decomposition at IoU .5 ----
def ap50(fix=None):
    out = {}
    for c in classes:
        mask = cls == c
        s, k, g = score[mask], kind[mask], target[mask]
        n = positives[c]
        tp = k == 'TP'
        keep = np.ones(len(s), bool)
        if fix in ('Bkg', 'Dupe', 'Both'):
            keep = k != fix
        elif fix == 'Loc':
            taken = set(g[tp].tolist())
            rank = np.argsort(-s, kind='mergesort')
            tp = tp.copy()
            for i in rank:
                if k[i] == 'Loc':
                    if g[i] in taken:
                        keep[i] = False
                    else:
                        taken.add(g[i])
                        tp[i] = True
        elif fix == 'Cls':
            keep = k != 'Cls'
            incoming = (kind == 'Cls') & (other_cls == c)
            taken = set(g[tp].tolist())
            extra = np.flatnonzero(incoming)
            extra = extra[np.argsort(-score[extra], kind='mergesort')]
            add_s, add_t = [], []
            for i in extra:
                if target[i] not in taken:
                    taken.add(target[i])
                    add_s.append(score[i])
                    add_t.append(True)
            s2 = np.concatenate([s[keep], np.array(add_s, np.float32)])
            t2 = np.concatenate([tp[keep], np.array(add_t, bool)])
            out[c] = ap(s2, t2, n)
            continue
        elif fix == 'AllFP':
            keep = tp
        elif fix == 'Miss':
            n = int(tp.sum())
        out[c] = ap(s[keep], tp[keep], n)
    return out


base = ap50()
report['ap50_base'] = {names[c]: round(v * 100, 2) for c, v in base.items()}
report['ap50_base_mean'] = round(float(np.mean(list(base.values()))) * 100, 2)
report['delta_ap50_if_fixed'] = {}
for fix in ['Cls', 'Loc', 'Both', 'Dupe', 'Bkg', 'Miss', 'AllFP']:
    result = ap50(fix)
    report['delta_ap50_if_fixed'][fix] = dict(mean=round((np.mean(list(result.values())) - np.mean(list(base.values()))) * 100, 2),
                                              **{names[c]: round((result[c] - base[c]) * 100, 2) for c in classes})

# ---- false positive counts at usable scores ----
report['fp_counts'] = {}
for threshold in (.3, .5):
    mask = score >= threshold
    report['fp_counts'][str(threshold)] = {names[c]: dict(TP=int(((kind == 'TP') & (cls == c) & mask).sum()), **{k: int(((kind == k) & (cls == c) & mask).sum()) for k in ['Loc', 'Cls', 'Both', 'Dupe', 'Bkg']}) for c in classes}
confusion = Counter()
for c, o in zip(cls[(kind == 'Cls') & (score >= .3)], other_cls[(kind == 'Cls') & (score >= .3)]):
    confusion[f'{names[int(o)]}->{names[int(c)]}'] += 1
report['confusion_true_to_pred_score_0.3'] = confusion.most_common(12)

# ---- recall ----
all_gt = [dict(g, image=i) for i, rows in gts_by_image.items() for g in rows]
tp_mask = kind == 'TP'
tp_best = defaultdict(float)
for g, s in zip(target[tp_mask], score[tp_mask]):
    tp_best[int(g)] = max(tp_best[int(g)], float(s))
rows = []
for g in all_gt:
    score_loc = gt_best_score_localized[g['id']]
    best = gt_best_iou_any_score[g['id']]
    rows.append((g['cls'], g['side'], best, score_loc))
rows = np.array(rows, np.float32)
recall = {}
for c in classes + ['all']:
    mask = rows[:, 0] == c if c != 'all' else np.ones(len(rows), bool)
    part = rows[mask]
    missed = part[:, 3] < .3
    localized_low = missed & (part[:, 2] >= .5)
    recall[names.get(c, 'all')] = dict(gt=int(mask.sum()), missed_at_0_3=round(float(missed.mean()), 4), of_missed_has_iou50_box=round(float(localized_low.sum() / max(missed.sum(), 1)), 4),
                                       median_score_when_localized_missed=round(float(np.median(part[localized_low, 3])) if localized_low.any() else 0, 4),
                                       no_box_iou50=round(float((part[:, 2] < .5).mean()), 4))
report['recall'] = recall

# ---- localization quality of true positives ----
sides = {g['id']: g['side'] for g in all_gt}
gt_cls = {g['id']: g['cls'] for g in all_gt}
shorts = {g['id']: g['short'] for g in all_gt}
aspects = {g['id']: g['aspect'] for g in all_gt}
source = {g['id']: images[g['image']]['width'] for g in all_gt}
mask = (kind == 'TP') & (score >= .3)
tp_ids = target[mask]
tp_iou = match_iou[mask]
quality = {}
for label, select in [('all', np.ones(len(tp_ids), bool))] + [(names[c], np.array([gt_cls[int(i)] == c for i in tp_ids])) for c in classes]:
    values = tp_iou[select]
    quality[label] = dict(n=int(select.sum()), median_iou=round(float(np.median(values)), 3), ge_0_75=round(float((values >= .75).mean()), 3), ge_0_85=round(float((values >= .85).mean()), 3), ge_0_95=round(float((values >= .95).mean()), 3))
report['tp_iou_score_0.3'] = quality
tp_sides = np.array([sides[int(i)] for i in tp_ids])
tp_short = np.array([shorts[int(i)] for i in tp_ids])
tp_aspect = np.array([aspects[int(i)] for i in tp_ids])
tp_source = np.array([source[int(i)] for i in tp_ids])
report['tp_iou_by_long_side_px'] = {f'{a}-{b}': dict(n=int(((tp_sides >= a) & (tp_sides < b)).sum()), median=round(float(np.median(tp_iou[(tp_sides >= a) & (tp_sides < b)])), 3), ge_0_75=round(float((tp_iou[(tp_sides >= a) & (tp_sides < b)] >= .75).mean()), 3)) for a, b in [(0, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 9999)]}
report['tp_iou_by_aspect'] = {f'{a}-{b}': dict(n=int(((tp_aspect >= a) & (tp_aspect < b)).sum()), median=round(float(np.median(tp_iou[(tp_aspect >= a) & (tp_aspect < b)])), 3), ge_0_75=round(float((tp_iou[(tp_aspect >= a) & (tp_aspect < b)] >= .75).mean()), 3)) for a, b in [(1, 1.5), (1.5, 3), (3, 6), (6, 999)]}


# ---- AP within groups ----
def group_ap(select_gt=None, select_image=None):
    """COCO AP over 10 thresholds; GT outside select_gt ignored, detections matching them excluded."""
    outcome = {}
    for c in classes:
        lines, n = {t: ([], []) for t in range(10)}, 0
        for image_id in images:
            if select_image is not None and not select_image(images[image_id]):
                continue
            rows_gt = [g for g in gts_by_image.get(image_id, []) if g['cls'] == c]
            rows_gt.sort(key=lambda g: 0 if (select_gt is None or select_gt(g)) else 1)
            ignored = np.array([not (select_gt is None or select_gt(g)) for g in rows_gt], bool)
            n += int((~ignored).sum())
            positions = slices.get(image_id, np.array([], int))
            keep = positions[pred_cls[positions] == c]
            if not len(keep):
                continue
            keep = keep[np.argsort(-pred_score[keep], kind='mergesort')][:100]
            matrix_g = iou(pred_box[keep], np.stack([g['box'] for g in rows_gt])) if rows_gt else np.zeros((len(keep), 0))
            for t, threshold in enumerate(THRESHOLDS):
                used = np.zeros(len(rows_gt), bool)
                hit, drop = np.zeros(len(keep), bool), np.zeros(len(keep), bool)
                for d in range(len(keep)):
                    if rows_gt and matrix_g[d].max() >= threshold:
                        best, chosen = threshold, -1
                        for g in range(len(rows_gt)):
                            if used[g]:
                                continue
                            if chosen > -1 and not ignored[chosen] and ignored[g]:
                                break
                            if matrix_g[d, g] >= best:
                                best, chosen = matrix_g[d, g], g
                        if chosen > -1:
                            used[chosen] = True
                            hit[d], drop[d] = not ignored[chosen], ignored[chosen]
                lines[t][0].append(pred_score[keep][~drop])
                lines[t][1].append(hit[~drop])
        if n:
            values = [ap(np.concatenate(lines[t][0]), np.concatenate(lines[t][1]), n) for t in range(10)]
            outcome[names[c]] = dict(gt=n, AP=round(float(np.mean(values)) * 100, 2), AP50=round(values[0] * 100, 2), AP75=round(values[5] * 100, 2))
    return outcome


report['by_source_image_width'] = {}
for label, test in [('224-256', lambda im: im['width'] <= 256), ('257-600', lambda im: 256 < im['width'] <= 600), ('601+', lambda im: im['width'] > 600)]:
    report['by_source_image_width'][label] = dict(images=sum(test(im) for im in images.values()), classes=group_ap(select_image=test))
report['by_long_side_px'] = {}
for a, b in [(0, 16), (16, 32), (32, 64), (64, 128), (128, 99999)]:
    report['by_long_side_px'][f'{a}-{b}'] = group_ap(select_gt=lambda g, a=a, b=b: a <= g['side'] < b)
report['by_aspect'] = {}
for a, b in [(1, 3), (3, 6), (6, 99999)]:
    report['by_aspect'][f'{a}-{b}'] = group_ap(select_gt=lambda g, a=a, b=b: a <= g['aspect'] < b)

(RUN / 'independent_verify_20261106/val_diagnosis.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
print(json.dumps(report, ensure_ascii=False))
