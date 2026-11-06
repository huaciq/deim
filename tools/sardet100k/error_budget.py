"""Val-only budget from saved E000 predictions: what does the same 20% improvement in each error type buy in AP?

These are simulations with ground truth, so they are ceilings for comparing levers, not reachable values.
Usage: error_budget.py RUN
"""
import json
import multiprocessing as mp
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RUN = Path(sys.argv[1])
OUT = RUN / 'independent_verify_20261106/budget'
OUT.mkdir(parents=True, exist_ok=True)
ROOT = Path('/root/zxy/TCRA-SAR/datasets/SARDet_100K')
val = json.load(open(ROOT / 'Annotations/val.json'))
names = {c['id']: c['name'] for c in val['categories']}
saved = np.load(RUN / 'independent_verify_20261106/val_predictions.npz')
PI, PC, PS, PB = saved['image_id'], saved['label'].astype(int), saved['score'].astype(np.float32), saved['box'].astype(np.float32)
keep = PS >= .01
PI, PC, PS, PB = PI[keep], PC[keep], PS[keep], PB[keep]
gts = defaultdict(list)
for a in val['annotations']:
    x, y, w, h = a['bbox']
    gts[(a['image_id'], a['category_id'])].append([x, y, x + w, y + h])
gts = {k: np.array(v, np.float32) for k, v in gts.items()}
IMAGE_IDS = [i['id'] for i in val['images']]
THR = np.linspace(.5, .95, 10)
REC = np.linspace(0, 1, 101)
BINS = {'lt16': (0, 16), '16_64': (16, 64), 'ge64': (64, 1e9)}


def iou(a, b):
    x1, y1 = np.maximum(a[:, None, 0], b[:, 0]), np.maximum(a[:, None, 1], b[:, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[:, 2]), np.minimum(a[:, None, 3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = (a[:, 2] - a[:, 0])[:, None] * (a[:, 3] - a[:, 1])[:, None] + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return inter / np.maximum(union, 1e-9)


def ap_curve(score, hit, n):
    order = np.argsort(-score, kind='mergesort')
    hit = hit[order]
    tp, fp = np.cumsum(hit), np.cumsum(~hit)
    recall = tp / n
    precision = tp / np.maximum(tp + fp, 1e-12)
    precision = np.maximum.accumulate(precision[::-1])[::-1] if len(precision) else precision
    where = np.searchsorted(recall, REC, side='left')
    out = np.zeros(101)
    ok = where < len(precision)
    out[ok] = precision[where[ok]]
    return out.mean()


def slices(pi):
    order = np.argsort(pi, kind='stable')
    values, first = np.unique(pi[order], return_index=True)
    return dict(zip(values.tolist(), np.split(order, first[1:])))


def evaluate(pi, pc, ps, pb):
    where = slices(pi)
    per_class = {}
    curves = []
    for c, name in names.items():
        lines = {t: ([], []) for t in range(10)}
        n = 0
        for image_id in IMAGE_IDS:
            cur = gts.get((image_id, c))
            n += 0 if cur is None else len(cur)
            pos = where.get(image_id)
            if pos is None:
                continue
            mine = pos[pc[pos] == c]
            if not len(mine):
                continue
            mine = mine[np.argsort(-ps[mine], kind='mergesort')][:100]
            m = iou(pb[mine], cur) if cur is not None else None
            for t, th in enumerate(THR):
                used = set()
                hit = np.zeros(len(mine), bool)
                if cur is not None:
                    for d in range(len(mine)):
                        if m[d].max() < th:
                            continue
                        best, chosen = th, -1
                        for g in range(len(cur)):
                            if g not in used and m[d, g] >= best:
                                best, chosen = m[d, g], g
                        if chosen >= 0:
                            used.add(chosen)
                            hit[d] = True
                lines[t][0].append(ps[mine])
                lines[t][1].append(hit)
        v = [ap_curve(np.concatenate(lines[t][0]), np.concatenate(lines[t][1]), n) for t in range(10)]
        per_class[name] = round(float(np.mean(v)) * 100, 2)
        curves.append(v)
    m = np.array(curves).mean(0) * 100
    return dict(AP=round(float(m.mean()), 2), AP50=round(float(m[0]), 2), AP75=round(float(m[5]), 2), AP90=round(float(m[8]), 2), classes=per_class)


def assignment():
    """Best same-class GT of every detection, and GT with no detection at IoU >= 0.5."""
    target = np.full((len(PI), 4), np.nan, np.float32)
    best_iou = np.zeros(len(PI), np.float32)
    where = slices(PI)
    missed = []
    for (image_id, c), cur in gts.items():
        pos = where.get(image_id)
        mine = pos[PC[pos] == c] if pos is not None else np.array([], int)
        mine = mine[PS[mine] >= .05] if len(mine) else mine
        if len(mine):
            m = iou(PB[mine], cur)
            target[mine] = cur[m.argmax(1)]
            best_iou[mine] = m.max(1)
            covered = m.max(0) >= .5
        else:
            covered = np.zeros(len(cur), bool)
        for g in np.flatnonzero(~covered):
            missed.append((image_id, c, cur[g]))
    return target, best_iou, missed


TARGET, BEST, MISSED = assignment()
LONG = np.nan_to_num(np.maximum(TARGET[:, 2] - TARGET[:, 0], TARGET[:, 3] - TARGET[:, 1]))


def shrink(factor, selector=None):
    pick = BEST >= .3
    if selector is not None:
        pick &= selector
    boxes = PB.copy()
    boxes[pick] = PB[pick] + factor * (TARGET[pick] - PB[pick])
    return evaluate(PI, PC, PS, boxes)


def job(name):
    rng = np.random.default_rng(0)
    if name == 'baseline':
        return name, evaluate(PI, PC, PS, PB)
    if name == 'loc_20':
        return name, shrink(.2)
    if name == 'loc_10':
        return name, shrink(.1)
    if name == 'loc_50':
        return name, shrink(.5)
    if name.startswith('loc_20_'):
        low, high = BINS[name[len('loc_20_'):]]
        return name, shrink(.2, (LONG >= low) & (LONG < high))
    if name == 'fn_20':
        take = rng.permutation(len(MISSED))[:len(MISSED) // 5]
        extra = [MISSED[k] for k in take]
        jitter = rng.normal(0, .05, (len(extra), 4))
        boxes = np.stack([b + jitter[k] * max(b[2] - b[0], b[3] - b[1]) for k, (_, _, b) in enumerate(extra)])
        return name, evaluate(np.r_[PI, [e[0] for e in extra]], np.r_[PC, [e[1] for e in extra]], np.r_[PS, np.full(len(extra), .5, np.float32)], np.r_[PB, boxes.astype(np.float32)])
    if name == 'bg_20':
        background = np.flatnonzero((BEST < .1) & (PS >= .05))
        drop = rng.permutation(background)[:len(background) // 5]
        mask = np.ones(len(PI), bool)
        mask[drop] = False
        return name, evaluate(PI[mask], PC[mask], PS[mask], PB[mask])
    if name == 'rank_iou':
        return name, evaluate(PI, PC, np.sqrt(PS * np.maximum(BEST, 1e-3)), PB)
    raise ValueError(name)


if __name__ == '__main__':
    jobs = ['baseline', 'loc_10', 'loc_20', 'loc_50', 'loc_20_lt16', 'loc_20_16_64', 'loc_20_ge64', 'fn_20', 'bg_20', 'rank_iou']
    with mp.Pool(10) as pool:
        result = dict(pool.map(job, jobs, chunksize=1))
    result['counts'] = dict(detections=int(len(PI)), missed=len(MISSED), matched_03=int((BEST >= .3).sum()))
    # Edge errors of matched detections by true size.
    matched = BEST >= .5
    error = np.abs(PB[matched] - TARGET[matched])
    result['edge_error_median'] = {k: round(float(np.median(error[(LONG[matched] >= a) & (LONG[matched] < b)])), 3) for k, (a, b) in BINS.items()}
    (OUT / 'report.json').write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps(result, ensure_ascii=False))
