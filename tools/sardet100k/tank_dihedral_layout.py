"""Do OGSOD tank tiles repeat under rotation or flip? Checks pixels and box layouts against val and train in all 8 orientations.

Usage: python tools/sardet100k/tank_dihedral_layout.py OUT

The earlier layout check in tank_feature_diagnosis.py allows integer shifts only, so a rotated copy of a scene looks unseen.
"""
import json
import multiprocessing as mp
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from error_gallery import ROOT  # noqa: E402

OUT = Path(sys.argv[1])
SIZE = 256
ORIENTATIONS = ('identity', 'rot90', 'rot180', 'rot270', 'flip_x', 'flip_y', 'transpose', 'anti_transpose')


def orient_boxes(boxes, name):
    """Boxes are x1 y1 x2 y2 in a SIZE x SIZE tile."""
    x1, y1, x2, y2 = boxes.T
    s = SIZE
    table = dict(identity=(x1, y1, x2, y2), rot90=(y1, s - x2, y2, s - x1), rot180=(s - x2, s - y2, s - x1, s - y1), rot270=(s - y2, x1, s - y1, x2),
                 flip_x=(s - x2, y1, s - x1, y2), flip_y=(x1, s - y2, x2, s - y1), transpose=(y1, x1, y2, x2), anti_transpose=(s - y2, s - x2, s - y1, s - x1))
    return np.stack(table[name], 1)


def orient_array(a, name):
    table = dict(identity=a, rot90=np.rot90(a, -1), rot180=np.rot90(a, 2), rot270=np.rot90(a, 1), flip_x=a[:, ::-1], flip_y=a[::-1], transpose=a.T, anti_transpose=a[::-1, ::-1].T)
    return table[name]


def thumb_array(path):
    image = Image.open(path).convert('L')
    return np.asarray(image.resize((32, 32), Image.BILINEAR), np.float32)


def unit(a):
    a = a.ravel() - a.mean()
    n = np.linalg.norm(a)
    return a / n if n > 0 else a


def best_match(boxes, others):
    """Share of boxes reproduced by one other layout under one orientation and one integer shift. Returns share, orientation, index, flags."""
    best = (0., 'identity', -1, np.zeros(len(boxes), bool))
    for name in ORIENTATIONS:
        turned = orient_boxes(boxes, name)
        centers, sizes = (turned[:, :2] + turned[:, 2:]) / 2, turned[:, 2:] - turned[:, :2]
        for index, other in enumerate(others):
            oc, os_ = (other[:, :2] + other[:, 2:]) / 2, other[:, 2:] - other[:, :2]
            shifts = np.unique(np.round((centers[:12, None] - oc[None]).reshape(-1, 2)), axis=0)
            for shift in shifts:
                distance = np.abs(centers[:, None] - shift - oc[None]).max(-1)
                near = distance.argmin(1)
                hit = (distance.min(1) <= 1.5) & (np.abs(sizes - os_[near]).max(1) <= 2)
                if hit.mean() > best[0]:
                    best = (float(hit.mean()), name, index, hit)
    return best


def layout_job(task):
    boxes, others = task
    share, name, index, flags = best_match(boxes, others)
    return share, name, index, flags.tolist()


def main():
    rows = [r for r in json.load(open(OUT / 'gt_rows.json')) if r['source'] == 'ogsod' and r['native_gw'] > 0 and r['native_gh'] > 0]
    by = defaultdict(list)
    for r in rows:
        by[r['image_id']].append(r)
    ids = [i for i in sorted(by) if len(by[i]) >= 3]
    val_layouts = {i: np.array([[r['gt_x1'], r['gt_y1'], r['gt_x2'], r['gt_y2']] for r in by[i]], np.float32) for i in ids}

    train = json.load(open(ROOT / 'Annotations/train.json'))
    names = {i['id']: i['file_name'] for i in train['images']}
    grouped = defaultdict(list)
    for a in train['annotations']:
        if a['category_id'] == 3 and a['bbox'][2] > 0 and a['bbox'][3] > 0:
            grouped[a['image_id']].append([a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]])
    # OGSOD-sized tiles only: their tanks are small, which also keeps the search tractable.
    train_ids = [k for k, v in grouped.items() if len(v) >= 3 and np.median([max(b[2] - b[0], b[3] - b[1]) for b in v]) < 20]
    train_layouts = [np.array(grouped[k], np.float32) for k in train_ids]
    print('val images', len(ids), 'train layouts', len(train_layouts), flush=True)

    with mp.Pool(32) as pool:
        val_results = pool.map(layout_job, [(val_layouts[i], [val_layouts[j] for j in ids if j != i]) for i in ids], chunksize=1)
        train_results = pool.map(layout_job, [(val_layouts[i], train_layouts) for i in ids], chunksize=1)
        val_thumbs = pool.map(thumb_array, [str(ROOT / 'JPEGImages/val' / by[i][0]['file_name']) for i in ids], chunksize=4)
        train_files = [names[k] for k in train_ids]
        train_thumbs = pool.map(thumb_array, [str(ROOT / 'JPEGImages/train' / f) for f in train_files], chunksize=32)

    train_matrix = np.stack([unit(t) for t in train_thumbs]).astype(np.float32)
    val_matrix = {name: np.stack([unit(orient_array(t, name)) for t in val_thumbs]).astype(np.float32) for name in ORIENTATIONS}
    report = []
    flags = {}
    for position, image_id in enumerate(ids):
        others = [j for j in ids if j != image_id]
        val_share, val_name, val_index, _ = val_results[position]
        share, name, index, hit = train_results[position]
        for r, h in zip(by[image_id], hit):
            flags[r['gt_id']] = bool(h)
        pixel_best = {orientation: float((train_matrix @ val_matrix[orientation][position]).max()) for orientation in ORIENTATIONS}
        pixel_name = max(pixel_best, key=pixel_best.get)
        val_pixel = {orientation: float(np.delete(np.stack([unit(t) for t in val_thumbs]), position, axis=0).dot(val_matrix[orientation][position]).max()) for orientation in ORIENTATIONS}
        val_pixel_name = max(val_pixel, key=val_pixel.get)
        report.append(dict(file=by[image_id][0]['file_name'], gt=len(by[image_id]), recall=round(float(np.mean([r['status'] == 'TP' for r in by[image_id]])), 3),
                           train_layout_share=round(share, 3), train_layout_orientation=name, train_layout_file=names[train_ids[index]] if index >= 0 else None, train_pixel_corr=round(pixel_best[pixel_name], 3), train_pixel_orientation=pixel_name,
                           identity_pixel_corr=round(pixel_best['identity'], 3), val_layout_share=round(val_share, 3), val_layout_orientation=val_name, val_layout_file=by[others[val_index]][0]['file_name'] if val_index >= 0 else None, val_pixel_corr=round(val_pixel[val_pixel_name], 3), val_pixel_orientation=val_pixel_name))
    json.dump(dict(images=report, gt_seen={str(k): v for k, v in flags.items()}), open(OUT / 'dihedral_layout.json', 'w'), indent=1)
    print('orientation of best train layout', Counter(r['train_layout_orientation'] for r in report if r['train_layout_share'] >= .8))
    print('orientation of best val layout', Counter(r['val_layout_orientation'] for r in report if r['val_layout_share'] >= .8))
    print('train layout share >= .8:', sum(r['train_layout_share'] >= .8 for r in report), '/', len(report))
    print('train pixel corr >= .98, identity:', sum(r['identity_pixel_corr'] >= .98 for r in report), ' any orientation:', sum(r['train_pixel_corr'] >= .98 for r in report))
    for r in sorted(report, key=lambda r: r['train_layout_share'])[:12]:
        print(r)
    print('--- named scenes')
    for r in report:
        if r['file'] in ('0091189.png', '0036526.png', '0112914.png', '0075919.png', '0009161.png', '0102342.png'):
            print(r)
    print('--- val pairs under any orientation (share >= .8), recall on each side')
    recall = {r['file']: r['recall'] for r in report}
    seen = set()
    for r in report:
        if r['val_layout_share'] >= .8:
            key = tuple(sorted((r['file'], r['val_layout_file'])))
            if key not in seen:
                seen.add(key)
                print(key, r['val_layout_orientation'], round(r['val_layout_share'], 2), 'pixel', r['val_pixel_corr'], r['val_pixel_orientation'], 'recall', recall.get(key[0]), recall.get(key[1]), 'gt', r['gt'])
    for label, test in (('train share >= .8', lambda s: s >= .8), ('train share < .8', lambda s: s < .8)):
        part = [r for r in report if test(r['train_layout_share'])]
        gt = sum(r['gt'] for r in part)
        print(label, 'images', len(part), 'gt', gt, 'recall', round(sum(r['gt'] * r['recall'] for r in part) / max(gt, 1), 3))


if __name__ == '__main__':
    main()
