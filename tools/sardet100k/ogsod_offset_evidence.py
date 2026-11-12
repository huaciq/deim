"""Is the OGSOD x offset in the labels, the model, or neither? SARDet train and val only; test is never read.

1. Prediction minus official box, with image-level bootstrap intervals.
2. Model-free: stack image profiles around box centres and measure where the content sits.
3. SARDet boxes against native OGSOD boxes on pixel-identical tiles.

Usage: ogsod_offset_evidence.py RUN_DIR
"""
import glob
import json
import multiprocessing as mp
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

RUN = Path(sys.argv[1])
OUT = RUN / 'independent_verify_20261106/offset_evidence'
OUT.mkdir(parents=True, exist_ok=True)
BASE = Path('/root/zxy/TCRA-SAR/datasets')
ROOT = BASE / 'SARDet_100K'
HALF, STEP, REACH = 16., .25, 8.
GRID = np.arange(-HALF, HALF + STEP / 2, STEP)
MAX_TARGETS, TRAIN_SAMPLE, BOOT = 3000, 25000, 300


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


def match_sources(pool, split_paths, ogsod, msar):
    rows = pool.map(thumb, split_paths, chunksize=64)
    query = np.stack([r[0] for r in rows])
    labels = np.array(['other'] * len(query), dtype=object)
    reference_of = {}
    for name, paths in (('ogsod', ogsod), ('msar', msar)):
        ref = pool.map(thumb, paths, chunksize=64)
        R = np.stack([r[0] for r in ref])
        best_s, best_i = [], []
        for start in range(0, len(query), 512):
            M = query[start:start + 512] @ R.T
            best_s.append(M.max(1))
            best_i.append(M.argmax(1))
        best_s, best_i = np.concatenate(best_s), np.concatenate(best_i)
        cand = [(q, int(best_i[q])) for q in np.flatnonzero(best_s >= .98) if rows[q][1] == ref[best_i[q]][1]]
        ok = pool.map(same, [(split_paths[q], paths[r]) for q, r in cand], chunksize=64)
        for (q, r), good in zip(cand, ok):
            if good:
                labels[q] = name
                if name == 'ogsod':
                    reference_of[q] = paths[r]
    return labels, reference_of, [r[1] for r in rows]


def profile_task(task):
    """task: path, [(key, cx, cy, w, h)]. COCO pixel i covers [i, i+1), so array index = coordinate - 0.5."""
    path, items = task
    image = np.asarray(Image.open(path).convert('L'), np.float32)
    out = []
    for key, cx, cy, w, h in items:
        half_h, half_w = max(h / 2, 1.), max(w / 2, 1.)
        ys = np.arange(-half_h, half_h + .01, .5)
        xs = np.arange(-half_w, half_w + .01, .5)
        px = map_coordinates(image, np.stack(np.meshgrid(cy - .5 + ys, cx - .5 + GRID, indexing='ij')), order=1, mode='nearest').mean(0)
        py = map_coordinates(image, np.stack(np.meshgrid(cy - .5 + GRID, cx - .5 + xs, indexing='ij')), order=1, mode='nearest').mean(1)
        patch = map_coordinates(image, np.stack(np.meshgrid(cy - .5 + GRID[::2], cx - .5 + GRID[::2], indexing='ij')), order=1, mode='nearest')
        scale = max(float(patch.std()), 1e-3)
        base = float(np.median(patch))
        out.append((key, (px - base) / scale, (py - base) / scale, (patch - base) / scale))
    return out


def mirror_shift(profile):
    """Sub-pixel centre of the profile by mirror correlation; candidates every STEP/2 px."""
    centre = len(profile) // 2
    reach = int(REACH / STEP)
    best, chosen = -2., 0.
    for half_steps in range(-int(4 / (STEP / 2)), int(4 / (STEP / 2)) + 1):
        pos = centre + half_steps / 2
        t = np.arange(.5 if half_steps % 2 else 1, reach + 1)
        a = np.interp(pos + t, np.arange(len(profile)), profile)
        b = np.interp(pos - t, np.arange(len(profile)), profile)
        if a.std() < 1e-6 or b.std() < 1e-6:
            continue
        c = float(np.corrcoef(a, b)[0, 1])
        if c > best:
            best, chosen = c, half_steps * STEP / 2
    return chosen


def centroid(profile):
    k = np.abs(GRID) <= REACH
    weight = np.clip(profile[k] - np.median(profile[np.abs(GRID) >= REACH]), 0, None)
    return float((GRID[k] * weight).sum() / max(weight.sum(), 1e-6))


def boot(values, groups, stat, count=BOOT, seed=0):
    rng = np.random.default_rng(seed)
    unique, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(unique))]
    stats = []
    for _ in range(count):
        take = np.concatenate([members[g] for g in rng.integers(len(unique), size=len(unique))])
        stats.append(stat(values[take]))
    return [round(float(np.percentile(stats, 2.5)), 3), round(float(np.percentile(stats, 97.5)), 3)]


def main():
    train = json.load(open(ROOT / 'Annotations/train.json'))
    val = json.load(open(ROOT / 'Annotations/val.json'))
    names = {c['id']: c['name'] for c in val['categories']}
    ogsod = sorted(glob.glob(str(BASE / 'OGSOD-1.0/sar/images/*/*')))
    msar = sorted(glob.glob(str(BASE / 'MSAR-1.0 dataset/JPEGImages/*')))
    rng = np.random.default_rng(0)
    picked = sorted(rng.choice(len(train['images']), TRAIN_SAMPLE, replace=False).tolist())
    train_images = [train['images'][i] for i in picked]
    report = {}
    with mp.Pool(12) as pool:
        v_paths = [str(ROOT / 'JPEGImages/val' / i['file_name']) for i in val['images']]
        t_paths = [str(ROOT / 'JPEGImages/train' / i['file_name']) for i in train_images]
        v_src, v_ref, v_size = match_sources(pool, v_paths, ogsod, msar)
        t_src, t_ref, t_size = match_sources(pool, t_paths, ogsod, msar)
        report['tile_sizes_val'] = {s: Counter(f'{a}x{b}' for a, b in np.array(v_size, dtype=object)[v_src == s].tolist()).most_common(3) for s in ('ogsod', 'msar', 'other')}
        report['source_counts'] = dict(val=dict(Counter(v_src.tolist())), train_sample=dict(Counter(t_src.tolist())))
        print(report['tile_sizes_val'], report['source_counts'], flush=True)

        gt_val, gt_train = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
        for a in val['annotations']:
            x, y, w, h = a['bbox']
            gt_val[a['image_id']][a['category_id']].append([x, y, x + w, y + h])
        chosen = {i['id'] for i in train_images}
        for a in train['annotations']:
            if a['image_id'] in chosen:
                x, y, w, h = a['bbox']
                gt_train[a['image_id']][a['category_id']].append([x, y, x + w, y + h])

        saved = np.load(RUN / 'independent_verify_20261106/val_predictions.npz')
        keep = saved['score'] >= .3
        PI, PC, PB = saved['image_id'][keep], saved['label'][keep].astype(int), saved['box'][keep]
        pred = defaultdict(lambda: defaultdict(list))
        for i, c, b in zip(PI.tolist(), PC.tolist(), PB):
            pred[i][c].append(b)

        # 1. Prediction minus official box, one row per target.
        offsets = defaultdict(list)
        tasks = defaultdict(list)
        for n, image in enumerate(val['images']):
            s = v_src[n]
            for c, boxes in gt_val[image['id']].items():
                G = np.array(boxes, float)
                P = np.array(pred[image['id']].get(c, []), float).reshape(-1, 4)
                M = iou(G, P) if len(P) else None
                for g, b in enumerate(G):
                    w, h = b[2] - b[0], b[3] - b[1]
                    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                    if max(w, h) > 64 or b[0] < 2 or b[1] < 2 or b[2] > image['width'] - 2 or b[3] > image['height'] - 2:
                        continue
                    tasks[n].append(((s, names[c], 'val_gt'), float(cx), float(cy), float(w), float(h)))
                    if M is not None and M[g].max() >= .5:
                        p = P[M[g].argmax()]
                        pcx, pcy = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
                        offsets[(s, names[c])].append((image['id'], pcx - cx, pcy - cy))
                        tasks[n].append(((s, names[c], 'val_pred'), float(pcx), float(pcy), float(p[2] - p[0]), float(p[3] - p[1])))
        for key in sorted(offsets):
            r = np.array(offsets[key])
            if len(r) < 30:
                continue
            row = dict(n=len(r), images=len(np.unique(r[:, 0])), dx=round(float(r[:, 1].mean()), 3), dx_ci=boot(r[:, 1], r[:, 0], np.mean),
                       dy=round(float(r[:, 2].mean()), 3), dy_ci=boot(r[:, 2], r[:, 0], np.mean), dx_median=round(float(np.median(r[:, 1])), 3),
                       share_dx_pos=round(float((r[:, 1] > 0).mean()), 3), share_dx_neg=round(float((r[:, 1] < 0).mean()), 3))
            report.setdefault('prediction_minus_official', {})['/'.join(key)] = row
            print('PRED', key, row, flush=True)

        # 2. Model-free stacks. Train boxes never go through the detector here.
        path_of = {('val', n): p for n, p in enumerate(v_paths)}
        jobs = [(path_of[('val', n)], items) for n, items in tasks.items()]
        for n, image in enumerate(train_images):
            items = []
            for c, boxes in gt_train[image['id']].items():
                for b in boxes:
                    w, h = b[2] - b[0], b[3] - b[1]
                    if max(w, h) > 64 or b[0] < 2 or b[1] < 2 or b[2] > image['width'] - 2 or b[3] > image['height'] - 2:
                        continue
                    items.append(((t_src[n], names[c], 'train_gt'), float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2), float(w), float(h)))
            if items:
                jobs.append((t_paths[n], items))
        collected = defaultdict(list)
        counts = Counter()
        rng.shuffle(jobs)
        for result in pool.imap_unordered(profile_task, jobs, chunksize=16):
            for key, px, py, patch in result:
                if counts[key] < MAX_TARGETS:
                    counts[key] += 1
                    collected[key].append((px, py, patch))
    images_out = {}
    for key in sorted(collected):
        if len(collected[key]) < 30:
            continue
        PX = np.stack([r[0] for r in collected[key]])
        PY = np.stack([r[1] for r in collected[key]])
        ids = np.arange(len(PX))
        row = dict(n=len(PX), mirror_x=mirror_shift(PX.mean(0)), mirror_y=mirror_shift(PY.mean(0)),
                   centroid_x=round(centroid(PX.mean(0)), 3), centroid_y=round(centroid(PY.mean(0)), 3),
                   mirror_x_ci=boot(PX, ids, lambda a: mirror_shift(a.mean(0)), count=100),
                   centroid_x_ci=boot(PX, ids, lambda a: centroid(a.mean(0)), count=100))
        report.setdefault('content_minus_box', {})['/'.join(key)] = row
        print('STACK', key, row, flush=True)
        images_out['/'.join(key)] = np.stack([r[2] for r in collected[key]]).mean(0)
    np.savez_compressed(OUT / 'mean_patches.npz', **images_out)

    # 3. SARDet against native OGSOD labels on identical tiles.
    native = {}
    for file in ('train.json', 'val.json'):
        data = json.load(open(BASE / 'OGSOD-1.0/sar/RTDETR_COCO' / file))
        ids = {i['id']: os.path.basename(i['file_name']) for i in data['images']}
        cats = {c['id']: c['name'] for c in data['categories']}
        report.setdefault('native_categories', {})[file] = list(cats.values())
        for a in data['annotations']:
            x, y, w, h = a['bbox']
            native.setdefault(ids[a['image_id']], []).append([x, y, x + w, y + h])
    diffs = []
    for image_list, refs in ((val['images'], v_ref), (train_images, t_ref)):
        gts = gt_val if image_list is val['images'] else gt_train
        for n, path in refs.items():
            A = np.array([b for boxes in gts[image_list[n]['id']].values() for b in boxes], float).reshape(-1, 4)
            B = np.array(native.get(os.path.basename(path), []), float).reshape(-1, 4)
            if len(A) and len(B):
                M = iou(A, B)
                for g in range(len(A)):
                    j = M[g].argmax()
                    if M[g, j] >= .3:
                        diffs.append(np.r_[(B[j] - A[g]), M[g, j]])
    diffs = np.array(diffs)
    report['native_minus_sardet'] = dict(n=len(diffs), mean_edges=np.round(diffs[:, :4].mean(0), 3).tolist(), iou_median=round(float(np.median(diffs[:, 4])), 4),
                                         share_identical=round(float((np.abs(diffs[:, :4]).max(1) < .5).mean()), 4))
    print('NATIVE', report['native_minus_sardet'], report['native_categories'], flush=True)
    (OUT / 'report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
