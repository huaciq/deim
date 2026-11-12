"""Mirror-copy label gaps using only the official SARDet-100K train and val annotations; test is not read.

Pairs: same pixels up to a flip or 180-degree rotation, any source, within and across train and val.
Gap: label in copy B minus copy A's label mapped by the exact geometric transform, original pixels.
Usage: CUDA_VISIBLE_DEVICES=<idle gpu> official_mirror_label_check.py SARDET_ROOT OUT_JSON
"""
import json
import multiprocessing as mp
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
ORIENT = ('identity', 'flip_x', 'flip_y', 'rot180')


def transform(a, orient):
    if orient in ('flip_x', 'rot180'):
        a = a[..., ::-1]
    if orient in ('flip_y', 'rot180'):
        a = a[..., ::-1, :]
    return np.ascontiguousarray(a)


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


def map_boxes(boxes, orient, w, h):
    out = boxes.copy()
    if orient in ('flip_x', 'rot180'):
        out[:, [0, 2]] = w - boxes[:, [2, 0]]
    if orient in ('flip_y', 'rot180'):
        out[:, [1, 3]] = h - boxes[:, [3, 1]]
    return out


def iou(a, b):
    x1, y1 = np.maximum(a[:, None, 0], b[:, 0]), np.maximum(a[:, None, 1], b[:, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[:, 2]), np.minimum(a[:, None, 3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = (a[:, 2] - a[:, 0])[:, None] * (a[:, 3] - a[:, 1])[:, None] + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return inter / np.maximum(union, 1e-9)


def main():
    images, boxes = [], defaultdict(lambda: defaultdict(list))
    names = None
    for split in ('train', 'val'):
        data = json.load(open(ROOT / f'Annotations/{split}.json'))
        names = {c['id']: c['name'] for c in data['categories']}
        for i in data['images']:
            images.append(dict(key=(split, i['id']), path=str(ROOT / 'JPEGImages' / split / i['file_name']), w=i['width'], h=i['height'], split=split))
        for a in data['annotations']:
            x, y, w, h = a['bbox']
            boxes[(split, a['image_id'])][a['category_id']].append([x, y, x + w, y + h])
    groups = defaultdict(list)
    for n, i in enumerate(images):
        groups[(i['w'], i['h'])].append(n)
    print({k: len(v) for k, v in groups.items()}, flush=True)
    device = torch.device('cuda:0')
    pairs = defaultdict(list)
    with mp.Pool(48) as pool:
        for (w, h), members in groups.items():
            thumbs = np.stack(pool.map(thumb, [images[n]['path'] for n in members], chunksize=128))
            base = torch.as_tensor(thumbs.reshape(len(members), -1), device=device, dtype=torch.float16)
            for orient in ORIENT if w == h else ('identity', 'flip_x', 'flip_y', 'rot180'):
                moved = torch.as_tensor(transform(thumbs, orient).reshape(len(members), -1), device=device, dtype=torch.float16)
                cand = []
                for start in range(0, len(members), 4096):
                    r, c = torch.nonzero(base[start:start + 4096] @ moved.T >= .98, as_tuple=True)
                    r = r.cpu().numpy() + start
                    c = c.cpu().numpy()
                    keep = c < r
                    cand += list(zip(c[keep].tolist(), r[keep].tolist()))
                gaps = pool.map(pixel_gap, [(images[members[a]]['path'], images[members[b]]['path'], orient) for a, b in cand], chunksize=64)
                kept = [(members[a], members[b]) for (a, b), g in zip(cand, gaps) if g < 8]
                pairs[orient] += kept
                print((w, h), orient, 'candidates', len(cand), 'kept', len(kept), flush=True)
    report = {}
    for orient, rows in pairs.items():
        stats = defaultdict(list)
        for a, b in rows:
            A, B = images[a], images[b]
            link = '-'.join(sorted((A['split'], B['split'])))
            diffs, classes = [], Counter()
            for c in set(boxes[A['key']]) & set(boxes[B['key']]):
                P = map_boxes(np.array(boxes[A['key']][c], float), orient, A['w'], A['h'])
                Q = np.array(boxes[B['key']][c], float)
                m = iou(P, Q)
                while m.size and m.max() >= .3:
                    i, j = np.unravel_index(m.argmax(), m.shape)
                    diffs.append(Q[j] - P[i])
                    classes[names[c]] += 1
                    m[i, :] = -1
                    m[:, j] = -1
            if not diffs:
                continue
            d = np.median(np.array(diffs), 0)
            row = ((d[0] + d[2]) / 2, (d[1] + d[3]) / 2, d[2] - d[0], d[3] - d[1])
            cls = classes.most_common(1)[0][0]
            stats[('all', link)].append(row)
            stats[(cls, 'any')].append(row)
        for (cls, link), rows_ in sorted(stats.items()):
            r = np.array(rows_)
            report.setdefault(orient, {})[f'{cls}/{link}'] = dict(
                pairs=len(r), median=np.round(np.median(r, 0), 3).tolist(),
                dx_counts=Counter(np.round(r[:, 0] * 2).astype(int).tolist()).most_common(5),
                dy_counts=Counter(np.round(r[:, 1] * 2).astype(int).tolist()).most_common(5))
            print('LABEL', orient, cls, link, report[orient][f'{cls}/{link}'], flush=True)
    OUT.write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
