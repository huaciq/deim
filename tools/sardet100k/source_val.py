"""Val-only: identify OGSOD/MSAR tiles by pixels; AP per source; label offset coherence; class-set gating; montages."""
import glob
import json
import multiprocessing as mp
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

RUN = Path(sys.argv[1])
OUT = RUN / 'independent_verify_20261106/source'
OUT.mkdir(parents=True, exist_ok=True)
BASE = Path('/root/zxy/TCRA-SAR/datasets')
ROOT = BASE / 'SARDet_100K'
train = json.load(open(ROOT / 'Annotations/train.json'))
val = json.load(open(ROOT / 'Annotations/val.json'))
names = {c['id']: c['name'] for c in val['categories']}
images = {i['id']: i for i in val['images']}
saved = np.load(RUN / 'independent_verify_20261106/val_predictions.npz')
PI, PC, PS, PB = saved['image_id'], saved['label'].astype(int), saved['score'], saved['box']
THR = np.linspace(.5, .95, 10)
REC = np.linspace(0, 1, 101)
gts = defaultdict(list)
for a in val['annotations']:
    x, y, w, h = a['bbox']
    gts[a['image_id']].append((a['category_id'], np.array([x, y, x + w, y + h], np.float32)))
order = np.argsort(PI, kind='stable')
vals, first = np.unique(PI[order], return_index=True)
SLICES = dict(zip(vals.tolist(), np.split(order, first[1:])))
report = {}


def thumb(path):
    image = Image.open(path).convert('L')
    t = np.asarray(image.resize((32, 32), Image.BILINEAR), np.float32).ravel()
    t = t - t.mean()
    n = np.linalg.norm(t)
    return (t / n if n > 0 else t), image.size


def same(task):
    a, b = task
    A = np.asarray(Image.open(a).convert('L'), np.float32)
    B = np.asarray(Image.open(b).convert('L'), np.float32)
    return A.shape == B.shape and np.abs(A - B).mean() < 2


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


def evaluate(boxes, scores, ids, removed=None):
    out = {}
    for c, name in names.items():
        lines = {t: ([], []) for t in range(10)}
        n = 0
        for image_id in ids:
            cur = [g[1] for g in gts.get(image_id, []) if g[0] == c]
            n += len(cur)
            pos = SLICES.get(image_id, np.array([], int))
            mask = PC[pos] == c
            if removed is not None:
                mask &= ~removed[pos]
            keep = pos[mask]
            if not len(keep):
                continue
            keep = keep[np.argsort(-scores[keep], kind='mergesort')][:100]
            m = iou(boxes[keep], np.stack(cur)) if cur else None
            for t, th in enumerate(THR):
                used = set()
                hit = np.zeros(len(keep), bool)
                if cur:
                    for d in range(len(keep)):
                        if m[d].max() < th:
                            continue
                        best, chosen = th, -1
                        for g in range(len(cur)):
                            if g not in used and m[d, g] >= best:
                                best, chosen = m[d, g], g
                        if chosen >= 0:
                            used.add(chosen)
                            hit[d] = True
                lines[t][0].append(scores[keep])
                lines[t][1].append(hit)
        if n and lines[0][0]:
            v = [ap_curve(np.concatenate(lines[t][0]), np.concatenate(lines[t][1]), n) for t in range(10)]
            out[name] = dict(gt=n, AP=round(float(np.mean(v)) * 100, 2), AP50=round(v[0] * 100, 2), AP75=round(v[5] * 100, 2))
    if out:
        out['mean'] = round(float(np.mean([v['AP'] for v in out.values()])), 2)
    return out


TASKS = {}


def run_task(name):
    return name, TASKS[name]()


def main():
    import torch
    ogsod = sorted(glob.glob(str(BASE / 'OGSOD-1.0/sar/images/*/*')))
    msar = sorted(glob.glob(str(BASE / 'MSAR-1.0 dataset/JPEGImages/*')))
    t_paths = [str(ROOT / 'JPEGImages/train' / i['file_name']) for i in train['images']]
    v_paths = [str(ROOT / 'JPEGImages/val' / i['file_name']) for i in val['images']]
    with mp.Pool(32) as pool:
        groups = {}
        for label, paths in [('ogsod', ogsod), ('msar', msar), ('train', t_paths), ('val', v_paths)]:
            rows = pool.map(thumb, paths, chunksize=64)
            groups[label] = (np.stack([r[0] for r in rows]), [r[1] for r in rows], paths)
        print('thumbs', {k: len(v[2]) for k, v in groups.items()}, flush=True)
        source = {}
        for split in ('train', 'val'):
            query = torch.as_tensor(groups[split][0]).cuda()
            labels = np.array(['other'] * len(query), dtype=object)
            for reference in ('ogsod', 'msar'):
                R = torch.as_tensor(groups[reference][0]).cuda()
                best_s, best_i = [], []
                for start in range(0, len(query), 1024):
                    s, i = (query[start:start + 1024] @ R.T).max(1)
                    best_s.append(s.cpu().numpy())
                    best_i.append(i.cpu().numpy())
                best_s, best_i = np.concatenate(best_s), np.concatenate(best_i)
                cand = [(q, int(best_i[q])) for q in np.flatnonzero(best_s >= .98) if groups[split][1][q] == groups[reference][1][best_i[q]]]
                ok = pool.map(same, [(groups[split][2][q], groups[reference][2][r]) for q, r in cand], chunksize=64)
                matched = [q for (q, r), good in zip(cand, ok) if good]
                labels[matched] = reference
                print(split, reference, 'matched', len(matched), 'of', len(query), flush=True)
                del R
            source[split] = labels
        del query
        torch.cuda.empty_cache()
    report['source_counts'] = {k: dict(Counter(v.tolist())) for k, v in source.items()}
    src_val = dict(zip([i['id'] for i in val['images']], source['val']))
    per = defaultdict(Counter)
    for a in train['annotations']:
        per[a['image_id']][a['category_id']] += 1
    table = defaultdict(Counter)
    single = defaultdict(lambda: [0, 0])
    for i, s in zip(train['images'], source['train']):
        table[s].update(per[i['id']])
        single[s][0] += 1
        single[s][1] += len(per[i['id']]) == 1
    report['train_classes_by_source'] = {s: {names[c]: n for c, n in v.most_common()} for s, v in table.items()}
    report['single_class_share_by_source'] = {s: round(v[1] / v[0], 4) for s, v in single.items()}

    # Compare native OGSOD labels against SARDet labels on identical tiles.
    ids_by_source = defaultdict(list)
    for image_id, s in src_val.items():
        ids_by_source[s].append(image_id)
    ids_by_source['all'] = list(images)

    # Offsets of confident true positives.
    offsets = defaultdict(list)
    tp_of = {}
    for image_id, cur in gts.items():
        pos = SLICES.get(image_id, np.array([], int))
        high = pos[PS[pos] >= .3]
        for c in {g[0] for g in cur}:
            mine = [g[1] for g in cur if g[0] == c]
            keep = high[PC[high] == c]
            if not len(keep):
                continue
            m = iou(PB[keep], np.stack(mine))
            for g, box in enumerate(mine):
                d = int(m[:, g].argmax())
                if m[d, g] >= .3:
                    p = PB[keep[d]]
                    w, h = box[2] - box[0], box[3] - box[1]
                    pw, ph = p[2] - p[0], p[3] - p[1]
                    row = ((p[0] + p[2] - box[0] - box[2]) / 2, (p[1] + p[3] - box[1] - box[3]) / 2, pw / max(w, 1e-3), ph / max(h, 1e-3), int(keep[d]), float(max(w, h)))
                    offsets[(image_id, c)].append(row)
    stats = {}
    for s in ['ogsod', 'msar', 'other']:
        for c, name in names.items():
            rows = [r for (i, k), v in offsets.items() if k == c and src_val[i] == s for r in v]
            if len(rows) < 30:
                continue
            arr = np.array([r[:4] for r in rows])
            groups_ = [np.array([r[:2] for r in v]) for (i, k), v in offsets.items() if k == c and src_val[i] == s and len(v) >= 3]
            icc = {}
            if len(groups_) >= 10:
                for axis, label in enumerate(('dx', 'dy')):
                    total = np.concatenate([g[:, axis] for g in groups_])
                    within = np.mean([g[:, axis].var(ddof=1) for g in groups_])
                    between = np.var([g[:, axis].mean() for g in groups_], ddof=1)
                    n0 = np.mean([len(g) for g in groups_])
                    icc[label] = round(float(max(between - within / n0, 0) / max(max(between - within / n0, 0) + within, 1e-9)), 3)
            stats[f'{s}/{name}'] = dict(n=len(rows), mean_dx=round(float(arr[:, 0].mean()), 3), mean_dy=round(float(arr[:, 1].mean()), 3), median_abs_dx=round(float(np.median(np.abs(arr[:, 0]))), 3), median_abs_dy=round(float(np.median(np.abs(arr[:, 1]))), 3), median_w_ratio=round(float(np.median(arr[:, 2])), 3), median_h_ratio=round(float(np.median(arr[:, 3])), 3), images_ge3=len(groups_), icc=icc)
    report['offset_stats'] = stats

    # Corrections: constant source/class bias (no GT at test image, estimated from val), leave-one-out same-image oracle, shuffled control.
    constant = PB.copy()
    for s, ids in ids_by_source.items():
        if s == 'all':
            continue
        members = set(ids)
        for c in names:
            rows = [r for (i, k), v in offsets.items() if k == c and i in members for r in v]
            if len(rows) < 30:
                continue
            arr = np.array([r[:2] for r in rows])
            shift = np.median(arr, 0)
            idx = np.concatenate([SLICES.get(i, np.array([], int)) for i in ids])
            idx = idx[PC[idx] == c]
            constant[idx] = PB[idx] - np.array([shift[0], shift[1], shift[0], shift[1]], np.float32)

    def loo(shuffle):
        boxes = PB.copy()
        rng = np.random.default_rng(0)
        keys = [k for k, v in offsets.items() if len(v) >= 3]
        donors = defaultdict(list)
        for k in keys:
            donors[(src_val[k[0]], k[1])].append(k)
        for key in keys:
            image_id, c = key
            donor = key
            if shuffle:
                choices = donors[(src_val[image_id], c)]
                donor = choices[rng.integers(len(choices))]
            rows = offsets[donor]
            own = {r[4] for r in offsets[key]}
            pos = SLICES[image_id]
            for p in pos[PC[pos] == c]:
                others = [r for r in rows if r[4] != p]
                if len(others) < 2:
                    continue
                dx, dy = np.median([r[0] for r in others]), np.median([r[1] for r in others])
                boxes[p] = PB[p] - np.array([dx, dy, dx, dy], np.float32)
        return boxes

    oracle_boxes, shuffled_boxes = loo(False), loo(True)

    # Class-set gating.
    present = {i: {g[0] for g in gts.get(i, [])} for i in images}
    cross = np.zeros(len(PS), bool)
    for image_id, pos in SLICES.items():
        cross[pos] = ~np.isin(PC[pos], list(present[image_id]))
    source_sets = {s: {c for c, n in table[s].items() if n >= .01 * sum(table[s].values())} for s in table}
    outside = np.zeros(len(PS), bool)
    for image_id, pos in SLICES.items():
        outside[pos] = ~np.isin(PC[pos], list(source_sets[src_val[image_id]]))
    top_gate = np.zeros(len(PS), bool)
    soft = PS.copy()
    for image_id, pos in SLICES.items():
        best = {c: PS[pos][PC[pos] == c].max() for c in np.unique(PC[pos])}
        top = max(best.values())
        winner = max(best, key=best.get)
        top_gate[pos] = PC[pos] != winner
        soft[pos] = PS[pos] * np.array([best[c] / top for c in PC[pos]], np.float32)
    report['cross_class_high_score'] = {}
    background = np.zeros(len(PS), bool)
    for image_id, pos in SLICES.items():
        rows = gts.get(image_id, [])
        background[pos] = iou(PB[pos], np.stack([g[1] for g in rows])).max(1) < .1 if rows else True
    for t in (.3, .5):
        mask = background & (PS >= t)
        report['cross_class_high_score'][str(t)] = dict(bkg_fp=int(mask.sum()), class_not_in_image=int((mask & cross).sum()), by_class={names[c]: [int((mask & (PC == c)).sum()), int((mask & cross & (PC == c)).sum())] for c in names})
    S = {s: ids for s, ids in ids_by_source.items()}
    for s, ids in S.items():
        TASKS[f'base/{s}'] = lambda ids=ids: evaluate(PB, PS, ids)
        TASKS[f'constant_shift/{s}'] = lambda ids=ids: evaluate(constant, PS, ids)
        TASKS[f'loo_oracle/{s}'] = lambda ids=ids: evaluate(oracle_boxes, PS, ids)
        TASKS[f'loo_shuffled/{s}'] = lambda ids=ids: evaluate(shuffled_boxes, PS, ids)
    TASKS['gate_oracle_image_classes/all'] = lambda: evaluate(PB, PS, list(images), removed=cross)
    TASKS['gate_oracle_source_classes/all'] = lambda: evaluate(PB, PS, list(images), removed=outside)
    TASKS['gate_top_class_hard/all'] = lambda: evaluate(PB, PS, list(images), removed=top_gate)
    TASKS['gate_top_class_soft/all'] = lambda: evaluate(PB, soft, list(images))
    with mp.Pool(16) as pool:
        report['ap'] = dict(pool.imap_unordered(run_task, list(TASKS)))
    (OUT / 'report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print('RESULT', json.dumps(report, ensure_ascii=False), flush=True)

    # Montages for inspection.
    def sheet(indices, name, cell=240):
        tiles = Image.new('RGB', (cell * 8, cell * 6))
        for n, p in enumerate(indices[:48]):
            im = images[int(PI[p])]
            image = Image.open(ROOT / 'JPEGImages/val' / im['file_name']).convert('RGB')
            scale = cell / max(im['width'], im['height'])
            image = image.resize((int(im['width'] * scale), int(im['height'] * scale)), Image.BILINEAR)
            draw = ImageDraw.Draw(image)
            draw.rectangle(list(PB[p] * scale), outline=(255, 255, 0))
            for c, g in gts.get(int(PI[p]), []):
                draw.rectangle(list(g * scale), outline=(0, 255, 0))
            draw.text((2, 2), f'{names[PC[p]]} {PS[p]:.2f}', fill=(255, 64, 64))
            tiles.paste(image, (n % 8 * cell, n // 8 * cell))
        tiles.save(OUT / name)

    idx = np.flatnonzero(background & cross)
    sheet(idx[np.argsort(-PS[idx])], 'cross_class_top.png')
    is_ogsod = np.array([src_val[int(i)] == 'ogsod' for i in PI])
    tanks = [i for i, s in src_val.items() if s == 'ogsod' and any(g[0] == 3 for g in gts.get(i, []))]
    rows = []
    for image_id in tanks[:48]:
        pos = SLICES[image_id]
        keep = pos[(PC[pos] == 3) & (PS[pos] >= .3)]
        rows.append(int(keep[0]) if len(keep) else int(pos[0]))
    sheet(rows, 'ogsod_tank.png')
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
