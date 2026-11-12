"""Is the 2 px gap between mirrored OGSOD copies one pixel per image, or two pixels in some copies?

1. All OGSOD train labels: copies by orientation, label minus correct geometric mapping.
2. Val with E000 predictions: image-level dx, and dx in both members of each mirrored pair.
   Constant in frame predicts equal residuals in a pair; an error in some copies predicts about 0 and 2.
3. Native YOLO labels against the COCO labels, to see whether a conversion step added the shift.

Test labels are not read. Native test images are only used to identify val tiles by pixels.
Usage: ogsod_mirror_offset_check.py RUN_DIR
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

BASE = Path('/root/zxy/TCRA-SAR/datasets')
ROOT = BASE / 'SARDet_100K'
NATIVE = BASE / 'OGSOD-1.0/sar'
W = 256
ORIENT = ('identity', 'flip_x', 'flip_y', 'rot180')


def transform(a, orient):
    if orient in ('flip_x', 'rot180'):
        a = a[..., ::-1]
    if orient in ('flip_y', 'rot180'):
        a = a[..., ::-1, :]
    return a


def thumb(path):
    image = Image.open(path).convert('L')
    t = np.asarray(image.resize((32, 32), Image.BILINEAR), np.float32)
    t = t - t.mean()
    n = np.linalg.norm(t)
    return t / n if n > 0 else t


def pixel_gap(task):
    a, b, orient = task
    A = transform(np.asarray(Image.open(a).convert('L'), np.float32), orient)
    B = np.asarray(Image.open(b).convert('L'), np.float32)
    return float(np.abs(A - B).mean()) if A.shape == B.shape else 1e9


def map_boxes(boxes, orient):
    out = boxes.copy()
    if orient in ('flip_x', 'rot180'):
        out[:, [0, 2]] = W - boxes[:, [2, 0]]
    if orient in ('flip_y', 'rot180'):
        out[:, [1, 3]] = W - boxes[:, [3, 1]]
    return out


def iou(a, b):
    x1, y1 = np.maximum(a[:, None, 0], b[:, 0]), np.maximum(a[:, None, 1], b[:, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[:, 2]), np.minimum(a[:, None, 3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = (a[:, 2] - a[:, 0])[:, None] * (a[:, 3] - a[:, 1])[:, None] + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return inter / np.maximum(union, 1e-9)


def find_pairs(pool, paths, gap_limit=8., per_orient=8000):
    thumbs = np.stack(pool.map(thumb, paths, chunksize=64))
    flat = thumbs.reshape(len(thumbs), -1)
    found = {}
    for orient in ORIENT:
        moved = transform(thumbs, orient).reshape(len(thumbs), -1)
        cand = []
        for start in range(0, len(flat), 512):
            rows, cols = np.nonzero(flat[start:start + 512] @ moved.T >= .98)
            cand += [(int(c), int(r) + start) for r, c in zip(rows, cols) if c < r + start]
        cand = cand[:per_orient]
        gaps = pool.map(pixel_gap, [(paths[a], paths[b], orient) for a, b in cand], chunksize=32)
        found[orient] = [(a, b, g) for (a, b), g in zip(cand, gaps) if g < gap_limit]
        print(orient, 'candidates', len(cand), 'kept', len(found[orient]), flush=True)
    return found


def label_gaps(found, boxes_of, paths):
    """B's boxes minus A's boxes mapped into B's frame, median per pair."""
    report = {}
    for orient, pairs in found.items():
        rows = []
        for a, b, _ in pairs:
            A, B = boxes_of.get(paths[a]), boxes_of.get(paths[b])
            if A is None or B is None:
                continue
            diffs = []
            for c in set(A) & set(B):
                P, Q = map_boxes(np.array(A[c], float), orient), np.array(B[c], float)
                m = iou(P, Q)
                while m.size and m.max() >= .3:
                    i, j = np.unravel_index(m.argmax(), m.shape)
                    diffs.append((Q[j] - P[i]))
                    m[i, :] = -1
                    m[:, j] = -1
            if diffs:
                d = np.median(np.array(diffs), 0)
                rows.append(((d[0] + d[2]) / 2, (d[1] + d[3]) / 2, d[2] - d[0], d[3] - d[1]))
        if rows:
            r = np.array(rows)
            report[orient] = dict(pairs=len(r), median=np.round(np.median(r, 0), 3).tolist(),
                                  dx_counts=Counter(np.round(r[:, 0] * 2).astype(int).tolist()).most_common(6),
                                  dy_counts=Counter(np.round(r[:, 1] * 2).astype(int).tolist()).most_common(6))
            print('LABEL', orient, report[orient], flush=True)
    return report


def main():
    run = Path(sys.argv[1])
    out = run / 'independent_verify_20261106/mirror_offset'
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    train_coco = json.load(open(NATIVE / 'RTDETR_COCO/train.json'))
    name_of = {i['id']: i['file_name'] for i in train_coco['images']}
    coco_boxes = defaultdict(lambda: defaultdict(list))
    for a in train_coco['annotations']:
        x, y, w, h = a['bbox']
        coco_boxes[str(NATIVE / name_of[a['image_id']])][a['category_id']].append([x, y, x + w, y + h])
    train_paths = sorted(glob.glob(str(NATIVE / 'images/train/*')))
    print('native train', len(train_paths), 'labelled', sum(p in coco_boxes for p in train_paths), 'example', next(iter(coco_boxes)), flush=True)

    val = json.load(open(ROOT / 'Annotations/val.json'))
    with mp.Pool(12) as pool:
        # 1. Train labels.
        found = find_pairs(pool, train_paths)
        report['train_label_gaps'] = label_gaps(found, coco_boxes, train_paths)

        # 2. Val residuals.
        native_all = sorted(glob.glob(str(NATIVE / 'images/*/*')))
        native_thumbs = np.stack(pool.map(thumb, native_all, chunksize=64)).reshape(len(native_all), -1)
        val_paths = [str(ROOT / 'JPEGImages/val' / i['file_name']) for i in val['images']]
        val_thumbs = np.stack(pool.map(thumb, val_paths, chunksize=64)).reshape(len(val_paths), -1)
        best, where = [], []
        for start in range(0, len(val_thumbs), 512):
            M = val_thumbs[start:start + 512] @ native_thumbs.T
            best.append(M.max(1))
            where.append(M.argmax(1))
        best, where = np.concatenate(best), np.concatenate(where)
        cand = np.flatnonzero(best >= .98).tolist()
        gaps = pool.map(pixel_gap, [(val_paths[q], native_all[where[q]], 'identity') for q in cand], chunksize=32)
        ogsod = [q for q, g in zip(cand, gaps) if g < 2]
        print('val ogsod', len(ogsod), flush=True)
        o_paths = [val_paths[q] for q in ogsod]
        o_images = [val['images'][q] for q in ogsod]
        val_pairs = find_pairs(pool, o_paths)

    gt = defaultdict(lambda: defaultdict(list))
    for a in val['annotations']:
        x, y, w, h = a['bbox']
        gt[a['image_id']][a['category_id']].append([x, y, x + w, y + h])
    report['val_label_gaps'] = label_gaps(val_pairs, {p: gt[i['id']] for p, i in zip(o_paths, o_images)}, o_paths)

    saved = np.load(run / 'independent_verify_20261106/val_predictions.npz')
    keep = saved['score'] >= .3
    PI, PC, PB = saved['image_id'][keep], saved['label'][keep].astype(int), saved['box'][keep]
    pred = defaultdict(lambda: defaultdict(list))
    for i, c, b in zip(PI.tolist(), PC.tolist(), PB):
        pred[i][c].append(b)
    residual = {}
    for n, image in enumerate(o_images):
        rows = []
        for c, boxes in gt[image['id']].items():
            P = np.array(pred[image['id']].get(c, []), float).reshape(-1, 4)
            if not len(P):
                continue
            G = np.array(boxes, float)
            m = iou(G, P)
            for g in range(len(G)):
                j = m[g].argmax()
                if m[g, j] >= .5:
                    d = P[j] - G[g]
                    rows.append(((d[0] + d[2]) / 2, (d[1] + d[3]) / 2))
        if len(rows) >= 2:
            residual[n] = np.mean(rows, 0)
    values = np.array(list(residual.values()))
    bins = [-9, -1, -.5, 0, .5, 1, 1.5, 2, 2.5, 9]
    report['image_level'] = dict(images=len(values), dx_hist=dict(bins=bins, counts=np.histogram(values[:, 0], bins)[0].tolist()),
                                 dy_hist=dict(bins=bins, counts=np.histogram(values[:, 1], bins)[0].tolist()),
                                 dx_median=round(float(np.median(values[:, 0])), 3), dy_median=round(float(np.median(values[:, 1])), 3))
    print('IMAGE', report['image_level'], flush=True)

    rng = np.random.default_rng(0)
    pair_report = {}
    for orient, pairs in val_pairs.items():
        rows = [(residual[a][0], residual[b][0], residual[a][1], residual[b][1]) for a, b, _ in pairs if a in residual and b in residual]
        if rows:
            r = np.array(rows)
            pair_report[orient] = dict(pairs=len(r), median_sum_dx=round(float(np.median(r[:, 0] + r[:, 1])), 3), median_abs_diff_dx=round(float(np.median(np.abs(r[:, 0] - r[:, 1]))), 3),
                                       median_sum_dy=round(float(np.median(r[:, 2] + r[:, 3])), 3), median_abs_diff_dy=round(float(np.median(np.abs(r[:, 2] - r[:, 3]))), 3),
                                       share_one_near_0_other_near_2=round(float(np.mean((np.minimum(r[:, 0], r[:, 1]) < .5) & (np.maximum(r[:, 0], r[:, 1]) > 1.5))), 3))
            print('PAIR', orient, pair_report[orient], flush=True)
    shuffled = rng.permutation(values[:, 0])
    pair_report['random_pairs_control_abs_diff_dx'] = round(float(np.median(np.abs(values[:, 0] - shuffled))), 3)
    report['val_pair_residuals'] = pair_report
    print('CONTROL', pair_report['random_pairs_control_abs_diff_dx'], flush=True)

    # 3. Native labels against COCO.
    label_dir = NATIVE / 'labels'
    print('labels dir', os.listdir(label_dir)[:6], flush=True)
    files = sorted(glob.glob(str(label_dir / 'train/*')))[:3000] or sorted(glob.glob(str(label_dir / '*')))[:3000]
    print('sample', files[0] if files else None, open(files[0]).read()[:160] if files and os.path.isfile(files[0]) else None, flush=True)
    diffs = []
    stem_to_path = {Path(p).stem: p for p in coco_boxes}
    for f in files:
        path = stem_to_path.get(Path(f).stem)
        if path is None or not os.path.isfile(f):
            continue
        A = np.array([b for v in coco_boxes[path].values() for b in v], float).reshape(-1, 4)
        rows = [line.split() for line in open(f) if line.strip()]
        if not len(A) or not rows or len(rows[0]) < 5:
            continue
        v = np.array([r[1:5] for r in rows], float) * W
        B = np.stack((v[:, 0] - v[:, 2] / 2, v[:, 1] - v[:, 3] / 2, v[:, 0] + v[:, 2] / 2, v[:, 1] + v[:, 3] / 2), 1)
        m = iou(A, B)
        for g in range(len(A)):
            j = m[g].argmax()
            if m[g, j] >= .3:
                diffs.append(B[j] - A[g])
    if diffs:
        diffs = np.array(diffs)
        report['yolo_minus_coco'] = dict(n=len(diffs), mean_edges=np.round(diffs.mean(0), 3).tolist(), median_edges=np.round(np.median(diffs, 0), 3).tolist())
        print('NATIVE', report['yolo_minus_coco'], flush=True)
    (out / 'report.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
