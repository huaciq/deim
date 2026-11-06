"""Val-only, CPU only: is the detector constant-false-alarm with respect to clutter, and does it use the target or the context?"""
import json
import multiprocessing as mp
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

RUN = Path(sys.argv[1])
OUT = RUN / 'independent_verify_20261106/behaviour'
OUT.mkdir(parents=True, exist_ok=True)
ROOT = Path('/root/zxy/TCRA-SAR/datasets/SARDet_100K')
val = json.load(open(ROOT / 'Annotations/val.json'))
names = {c['id']: c['name'] for c in val['categories']}
images = {i['id']: i for i in val['images']}
saved = np.load(RUN / 'independent_verify_20261106/val_predictions.npz')
PI, PC, PS, PB = saved['image_id'], saved['label'].astype(int), saved['score'], saved['box']
order = np.argsort(PI, kind='stable')
vals, first = np.unique(PI[order], return_index=True)
SLICES = dict(zip(vals.tolist(), np.split(order, first[1:])))
gts = defaultdict(list)
for a in val['annotations']:
    x, y, w, h = a['bbox']
    gts[a['image_id']].append(dict(id=a['id'], cls=a['category_id'], box=np.array([x, y, x + w, y + h], np.float32), side=max(w, h)))
report = {}


def iou(a, b):
    x1, y1 = np.maximum(a[:, None, 0], b[:, 0]), np.maximum(a[:, None, 1], b[:, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[:, 2]), np.minimum(a[:, None, 3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = (a[:, 2] - a[:, 0])[:, None] * (a[:, 3] - a[:, 1])[:, None] + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return inter / np.maximum(union, 1e-9)


def local_stats(values, box):
    """Target and guard-ring reference statistics on log intensity."""
    height, width = values.shape
    x0, y0 = int(np.clip(box[0], 0, width - 1)), int(np.clip(box[1], 0, height - 1))
    x1, y1 = int(np.clip(np.ceil(box[2]), x0 + 1, width)), int(np.clip(np.ceil(box[3]), y0 + 1, height))
    side = max(x1 - x0, y1 - y0)
    guard = max(side // 4, 1)
    a0, b0, a1, b1 = max(x0 - side - guard, 0), max(y0 - side - guard, 0), min(x1 + side + guard, width), min(y1 + side + guard, height)
    g0, h0, g1, h1 = max(x0 - guard, 0), max(y0 - guard, 0), min(x1 + guard, width), min(y1 + guard, height)
    mask = np.ones((b1 - b0, a1 - a0), bool)
    mask[h0 - b0:h1 - b0, g0 - a0:g1 - a0] = False
    ring = values[b0:b1, a0:a1][mask]
    if ring.size < 8:
        return None
    inner = values[y0:y1, x0:x1]
    log_in, log_ring = np.log1p(inner), np.log1p(ring)
    return float(log_in.mean()), float(log_ring.mean()), float(log_ring.std()), float(log_in.max()), float(ring.std() / max(ring.mean(), 1.))


def image_task(image_id):
    values = np.asarray(Image.open(ROOT / 'JPEGImages/val' / images[image_id]['file_name']).convert('L'), np.float32)
    rows = []
    for g in gts.get(image_id, []):
        stats = local_stats(values, g['box'])
        if stats is not None:
            rows.append((g['id'], *stats))
    # Background false positives and area histogram over clutter level.
    logged = np.log1p(values)
    h, w = values.shape
    cell = max(min(h, w) // 8, 8)
    levels = np.array([logged[y:y + cell, x:x + cell].mean() for y in range(0, h - cell + 1, cell) for x in range(0, w - cell + 1, cell)])
    pos = SLICES.get(image_id, np.array([], int))
    pos = pos[PS[pos] >= .3]
    fp_rows = []
    if len(pos):
        boxes = np.stack([g['box'] for g in gts[image_id]]) if gts.get(image_id) else np.zeros((0, 4), np.float32)
        far = iou(PB[pos], boxes).max(1) < .1 if len(boxes) else np.ones(len(pos), bool)
        for p in pos[far]:
            stats = local_stats(values, PB[p])
            if stats is not None:
                fp_rows.append((int(PC[p]), float(PS[p]), *stats))
    return rows, levels.tolist(), fp_rows


def fit(x, y, steps=400):
    import torch
    X = torch.as_tensor(x, dtype=torch.float32)
    Y = torch.as_tensor(y, dtype=torch.float32)
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, Y) + 1e-4 * w.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return w.detach().numpy()


def crop_task(task):
    path, boxes = task
    v = np.asarray(Image.open(path).convert('L'), np.float32)
    h, w = v.shape
    rows = []
    for b in boxes:
        x0, y0 = min(max(int(b[0]), 0), w - 1), min(max(int(b[1]), 0), h - 1)
        x1, y1 = min(max(int(np.ceil(b[2])), x0 + 1), w), min(max(int(np.ceil(b[3])), y0 + 1), h)
        t = np.asarray(Image.fromarray(v[y0:y1, x0:x1]).resize((24, 24), Image.BILINEAR), np.float32).ravel()
        t = t - t.mean()
        n = np.linalg.norm(t)
        rows.append(t / n if n > 0 else t)
    return np.stack(rows) if rows else np.zeros((0, 576), np.float32)


def seen_ids(pool):
    train = json.load(open(ROOT / 'Annotations/train.json'))
    by = defaultdict(list)
    for a in train['annotations']:
        by[a['image_id']].append(a)
    corner = lambda a: [a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]]
    reference = np.concatenate(pool.map(crop_task, [(str(ROOT / 'JPEGImages/train' / i['file_name']), [corner(a) for a in by[i['id']]]) for i in train['images']], chunksize=64))
    sizes_r = np.array([max(a['bbox'][2], a['bbox'][3]) for i in train['images'] for a in by[i['id']]], np.float32)
    query = np.concatenate(pool.map(crop_task, [(str(ROOT / 'JPEGImages/val' / i['file_name']), [g['box'] for g in gts.get(i['id'], [])]) for i in val['images']], chunksize=64))
    ids = [g['id'] for i in val['images'] for g in gts.get(i['id'], [])]
    sizes_q = np.array([g['side'] for i in val['images'] for g in gts.get(i['id'], [])], np.float32)
    best = np.zeros(len(query), np.float32)
    # Size-limited comparison keeps this practical on CPU.
    order_r = np.argsort(sizes_r)
    sorted_r = sizes_r[order_r]
    for start in range(len(query)):
        low, high = np.searchsorted(sorted_r, [sizes_q[start] / 1.1, sizes_q[start] / .9])
        if high > low:
            subset = order_r[low:high]
            if len(subset) > 40000:
                subset = subset[:: len(subset) // 40000 + 1]
            best[start] = (reference[subset] @ query[start]).max()
    return {i for i, b in zip(ids, best) if b >= .9}


class Masked:
    def __init__(self, rows, mode):
        self.rows, self.mode = rows, mode

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torchvision.transforms.functional as F
        row = self.rows[index]
        values = np.asarray(Image.open(ROOT / 'JPEGImages/val' / row['file_name']).convert('L'), np.float32).copy()
        h, w = values.shape
        rng = np.random.default_rng(index)
        boxes = [g['box'] for g in gts[row['id']]]
        keep = np.zeros((h, w), bool)
        for b in boxes:
            bw, bh = b[2] - b[0], b[3] - b[1]
            x0, y0 = max(int(b[0] - .25 * bw), 0), max(int(b[1] - .25 * bh), 0)
            x1, y1 = min(int(np.ceil(b[2] + .25 * bw)), w), min(int(np.ceil(b[3] + .25 * bh)), h)
            keep[y0:y1, x0:x1] = True
        if self.mode == 'context_replaced':
            # Outside the targets, keep only the image's own clutter statistics: reshuffled pixels.
            outside = values[~keep]
            if outside.size:
                values[~keep] = rng.choice(outside, outside.size, replace=True)
        elif self.mode == 'target_removed':
            for b in boxes:
                bw, bh = b[2] - b[0], b[3] - b[1]
                x0, y0 = max(int(b[0]), 0), max(int(b[1]), 0)
                x1, y1 = min(int(np.ceil(b[2])), w), min(int(np.ceil(b[3])), h)
                pad = int(max(bw, bh)) + 2
                a0, c0, a1, c1 = max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, w), min(y1 + pad, h)
                mask = ~keep[c0:c1, a0:a1]
                ring = values[c0:c1, a0:a1][mask]
                if ring.size and y1 > y0 and x1 > x0:
                    values[y0:y1, x0:x1] = rng.choice(ring, (y1 - y0, x1 - x0), replace=True)
        image = Image.fromarray(np.clip(values, 0, 255).astype(np.uint8)).convert('RGB')
        return F.to_tensor(F.resize(image, [640, 640])), index, w, h


def main():
    with mp.Pool(16) as pool:
        parts = pool.map(image_task, list(images), chunksize=32)
        seen = seen_ids(pool)
    report['seen_share'] = round(len(seen) / sum(len(v) for v in gts.values()), 4)
    stats = {r[0]: r[1:] for part in parts for r in part[0]}
    levels = np.concatenate([np.array(p[1]) for p in parts])
    fps = [r for p in parts for r in p[2]]
    # Best score of a same-class localized box for each GT.
    best_score = {}
    for image_id, rows in gts.items():
        pos = SLICES.get(image_id, np.array([], int))
        for g in rows:
            keep = pos[PC[pos] == g['cls']]
            best_score[g['id']] = 0.
            if len(keep):
                m = iou(PB[keep], g['box'][None])[:, 0]
                if (m >= .5).any():
                    best_score[g['id']] = float(PS[keep][m >= .5].max())
    rows = []
    for image_rows in gts.values():
        for g in image_rows:
            if g['id'] in stats:
                s = stats[g['id']]
                rows.append((g['cls'], g['id'] in seen, g['side'], s[0], s[1], s[2], s[3], s[4], best_score[g['id']]))
    A = np.array(rows, np.float64)
    cls, is_seen, side, log_in, log_ring, ring_std, log_peak, ring_cv, score = A.T
    scr = log_in - log_ring
    peak_scr = log_peak - log_ring
    detected = score >= .3
    report['n'] = len(A)

    def table(mask, by, quantiles=(0, 33, 66, 100)):
        edges = np.percentile(by[mask], quantiles)
        return [dict(range=[round(float(a), 2), round(float(b), 2)], n=int((mask & (by >= a) & (by <= b)).sum()), recall=round(float(detected[mask & (by >= a) & (by <= b)].mean()), 3)) for a, b in zip(edges[:-1], edges[1:])]

    report['recall_by_scr'] = {names[c]: table(cls == c, scr) for c in names}
    report['recall_by_clutter_level'] = {names[c]: table(cls == c, log_ring) for c in names}
    report['recall_by_clutter_cv'] = {names[c]: table(cls == c, ring_cv) for c in names}
    # Fixed SCR tertile, varying clutter level: a CFAR detector would be flat.
    report['recall_scr_x_clutter'] = {}
    for c, name in names.items():
        mask = cls == c
        scr_edges = np.percentile(scr[mask], [0, 33, 66, 100])
        level_edges = np.percentile(log_ring[mask], [0, 33, 66, 100])
        grid = []
        for a, b in zip(scr_edges[:-1], scr_edges[1:]):
            line = []
            for u, v in zip(level_edges[:-1], level_edges[1:]):
                cell = mask & (scr >= a) & (scr <= b) & (log_ring >= u) & (log_ring <= v)
                line.append([int(cell.sum()), round(float(detected[cell].mean()), 3) if cell.sum() >= 15 else None])
            grid.append(line)
        report['recall_scr_x_clutter'][name] = grid
    # Standardized logistic model.
    z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
    feature_names = ['scr', 'peak_scr', 'clutter_level', 'clutter_cv', 'log_side', 'seen']
    base = np.stack([z(scr), z(peak_scr), z(log_ring), z(np.log1p(ring_cv)), z(np.log(side + 1)), z(is_seen.astype(float))], 1)
    onehot = np.eye(len(names))[cls.astype(int)]
    coefficients = fit(np.c_[base, onehot], detected.astype(float))
    report['logistic_all'] = dict(zip(feature_names, np.round(coefficients[:6], 3).tolist()))
    report['logistic_by_class'] = {}
    for c, name in names.items():
        m = cls == c
        if detected[m].mean() < .995:
            report['logistic_by_class'][name] = dict(zip(feature_names, np.round(fit(base[m], detected[m].astype(float))[:6], 3).tolist()))
    # False positives relative to area, by clutter level.
    fp = np.array([r for r in fps], np.float64)
    if len(fp):
        edges = np.percentile(levels, [0, 20, 40, 60, 80, 100])
        area = np.histogram(levels, edges)[0] / len(levels)
        hist = np.histogram(fp[:, 3], edges)[0] / len(fp)
        report['fp_density_vs_clutter'] = dict(edges=np.round(edges, 2).tolist(), area_share=np.round(area, 3).tolist(), fp_share=np.round(hist, 3).tolist(), ratio=np.round(hist / np.maximum(area, 1e-6), 2).tolist(), fp=int(len(fp)))
        report['fp_scr'] = dict(median=round(float(np.median(fp[:, 2] - fp[:, 3])), 3), weak_gt_median=round(float(np.median(scr[(score < .3)])), 3), detected_gt_median=round(float(np.median(scr[detected])), 3))
    report['seen_vs_unseen_explained'] = {}
    for c, name in names.items():
        m = cls == c
        report['seen_vs_unseen_explained'][name] = dict(seen=[int((m & is_seen.astype(bool)).sum()), round(float(detected[m & is_seen.astype(bool)].mean()), 3), round(float(np.median(scr[m & is_seen.astype(bool)])), 3)], unseen=[int((m & ~is_seen.astype(bool)).sum()), round(float(detected[m & ~is_seen.astype(bool)].mean()), 3), round(float(np.median(scr[m & ~is_seen.astype(bool)])), 3)])
    (OUT / 'report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print('PART1', json.dumps(report, ensure_ascii=False), flush=True)

    # Part 2: inference with masked context or targets on a small sample, CPU.
    import torch
    from engine.core import YAMLConfig
    torch.set_num_threads(16)
    cfg = YAMLConfig(str(RUN / 'resolved_config.yml'))
    model, post = cfg.model, cfg.postprocessor
    model.load_state_dict(torch.load(RUN / 'best_stg2.pth', map_location='cpu')['ema']['module'])
    model, post = model.eval(), post.eval()
    rng = np.random.default_rng(0)
    chosen = []
    for c in names:
        ids = [i for i, rows_g in gts.items() if any(g['cls'] == c for g in rows_g)]
        chosen += rng.permutation(ids)[:150].tolist()
    chosen = list(dict.fromkeys(chosen))
    selected = [images[i] for i in chosen]
    result = {}
    for mode in ('original', 'context_replaced', 'target_removed'):
        loader = torch.utils.data.DataLoader(Masked(selected, mode), batch_size=16, num_workers=4)
        scores = {}
        fp_counts = []
        with torch.no_grad():
            for batch, order, w, h in loader:
                outputs = post(model(batch), torch.stack([w, h], 1))
                for i, r in zip(order.tolist(), outputs):
                    image_id = selected[i]['id']
                    pc, ps, pb = r['labels'].numpy().astype(int), r['scores'].numpy(), r['boxes'].numpy()
                    for g in gts[image_id]:
                        same = pc == g['cls']
                        m = iou(pb[same], g['box'][None])[:, 0] if same.any() else np.zeros(0)
                        ok = m >= .5
                        scores[g['id']] = float(ps[same][ok].max()) if ok.any() else 0.
                    boxes = np.stack([g['box'] for g in gts[image_id]])
                    high = ps >= .3
                    outside = iou(pb[high], boxes).max(1) < .1 if high.any() else np.zeros(0, bool)
                    fp_counts.append(int(outside.sum()))
        result[mode] = dict(scores=scores, mean_bkg_fp_per_image=round(float(np.mean(fp_counts)), 3))
        print('mode', mode, flush=True)
    original = result['original']['scores']
    meta = {g['id']: g for rows_g in gts.values() for g in rows_g}
    behaviour = {}
    for mode in ('context_replaced', 'target_removed'):
        now = result[mode]['scores']
        groups = {'all': lambda g: True, 'seen': lambda g: g['id'] in seen, 'unseen': lambda g: g['id'] not in seen}
        groups.update({n: (lambda g, c=c: g['cls'] == c) for c, n in names.items()})
        groups.update({f'{n}_unseen': (lambda g, c=c: g['cls'] == c and g['id'] not in seen) for c, n in names.items()})
        entry = {}
        for label, test in groups.items():
            ids = [i for i in original if test(meta[i]) and original[i] >= .3]
            if len(ids) >= 15:
                entry[label] = dict(n=len(ids), still_detected=round(float(np.mean([now[i] >= .3 for i in ids])), 3), median_score_original=round(float(np.median([original[i] for i in ids])), 3), median_score_now=round(float(np.median([now[i] for i in ids])), 3))
        behaviour[mode] = dict(groups=entry, mean_bkg_fp_per_image=result[mode]['mean_bkg_fp_per_image'])
    behaviour['original_mean_bkg_fp_per_image'] = result['original']['mean_bkg_fp_per_image']
    report['masking'] = behaviour
    (OUT / 'report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print('PART2', json.dumps(behaviour, ensure_ascii=False), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
