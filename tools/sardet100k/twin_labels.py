"""Aircraft on val images with a pixel-identical train twin: does the correction also approach the twin's own labels?"""
import json, multiprocessing as mp, sys, tarfile
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

RUN = Path(sys.argv[1])
OUT = RUN / 'independent_verify_20261106/edge_probe'
ROOT = Path('/root/zxy/TCRA-SAR/datasets/SARDet_100K')
val = json.load(open(ROOT / 'Annotations/val.json'))
train = json.load(open(ROOT / 'Annotations/train.json'))
air = next(c['id'] for c in val['categories'] if c['name'] == 'aircraft')


def boxes(data):
    out = defaultdict(list)
    for a in data['annotations']:
        if a['category_id'] == air:
            x, y, w, h = a['bbox']
            out[a['image_id']].append([x, y, x + w, y + h])
    return {k: np.array(v, np.float32) for k, v in out.items()}


def thumb(path):
    image = Image.open(path).convert('L')
    t = np.asarray(image.resize((32, 32), Image.BILINEAR), np.float32).ravel()
    t = t - t.mean()
    n = np.linalg.norm(t)
    return (t / n if n > 0 else t), image.size


def same(task):
    a, b = task
    A, B = np.asarray(Image.open(a).convert('L'), np.float32), np.asarray(Image.open(b).convert('L'), np.float32)
    return A.shape == B.shape and np.abs(A - B).mean() < 2


def iou(a, b):
    x1, y1 = np.maximum(a[:, None, 0], b[:, 0]), np.maximum(a[:, None, 1], b[:, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[:, 2]), np.minimum(a[:, None, 3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = (a[:, 2] - a[:, 0])[:, None] * (a[:, 3] - a[:, 1])[:, None] + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return inter / np.maximum(union, 1e-9)


def main():
    gt_val, gt_train = boxes(val), boxes(train)
    rows_val = [i for i in val['images'] if i['id'] in gt_val]
    rows_train = [i for i in train['images'] if i['id'] in gt_train]
    vp = [str(ROOT / 'JPEGImages/val' / i['file_name']) for i in rows_val]
    tp = [str(ROOT / 'JPEGImages/train' / i['file_name']) for i in rows_train]
    with mp.Pool(16) as pool:
        V = pool.map(thumb, vp, chunksize=32)
        T = pool.map(thumb, tp, chunksize=64)
        Q, R = np.stack([x[0] for x in V]), np.stack([x[0] for x in T])
        M = Q @ R.T
        best, arg = M.max(1), M.argmax(1)
        cand = [(q, int(arg[q])) for q in np.flatnonzero(best >= .98) if V[q][1] == T[arg[q]][1]]
        ok = pool.map(same, [(vp[q], tp[r]) for q, r in cand], chunksize=16)
    twins = {rows_val[q]['id']: gt_train[rows_train[r]['id']] for (q, r), g in zip(cand, ok) if g}
    print('twin images', len(twins), flush=True)
    s = np.load(OUT / 'boxes.npz')
    PI, PC, PS, base, aligned = s['image_id'], s['label'], s['score'], s['base'], s['aligned']
    keep = (PC == air) & (PS >= .3)
    PI, base, aligned = PI[keep], base[keep], aligned[keep]
    rows = []
    for image_id, twin in twins.items():
        mine = np.flatnonzero(PI == image_id)
        if not len(mine):
            continue
        gt = gt_val[image_id]
        m = iou(base[mine], gt)
        for d, k in enumerate(mine):
            g = int(m[d].argmax())
            if m[d, g] < .5:
                continue
            link = iou(gt[g][None], twin)[0]
            if link.max() < .3:
                continue
            o = twin[int(link.argmax())]
            a, b = aligned[k][None], base[k][None]
            rows.append((image_id, max(gt[g][2] - gt[g][0], gt[g][3] - gt[g][1]), link.max(), iou(b, gt[g][None])[0, 0], iou(a, gt[g][None])[0, 0], iou(b, o[None])[0, 0], iou(a, o[None])[0, 0]))
    r = np.array(rows)
    report = dict(images=len(np.unique(r[:, 0])), boxes=len(r))

    def summarize(mask, name):
        x = r[mask]
        ids = x[:, 0]
        unique = np.unique(ids)
        where = {u: np.flatnonzero(ids == u) for u in unique}
        rng = np.random.default_rng(0)
        draws = [np.concatenate([where[u] for u in rng.choice(unique, len(unique))]) for _ in range(2000)]
        gain_val, gain_twin = x[:, 4] - x[:, 3], x[:, 6] - x[:, 5]
        report[name] = dict(
            boxes=len(x), images=len(unique), side_median=round(float(np.median(x[:, 1])), 1), val_twin_iou_median=round(float(np.median(x[:, 2])), 3),
            base_to_val=round(float(x[:, 3].mean()), 4), aligned_to_val=round(float(x[:, 4].mean()), 4), base_to_twin=round(float(x[:, 5].mean()), 4), aligned_to_twin=round(float(x[:, 6].mean()), 4),
            gain_val=round(float(gain_val.mean()), 4), gain_val_ci95=np.round(np.percentile([gain_val[d].mean() for d in draws], [2.5, 97.5]), 4).tolist(),
            gain_twin=round(float(gain_twin.mean()), 4), gain_twin_ci95=np.round(np.percentile([gain_twin[d].mean() for d in draws], [2.5, 97.5]), 4).tolist())

    summarize(np.ones(len(r), bool), 'all')
    summarize(r[:, 1] >= 64, 'ge64')
    summarize((r[:, 2] < .95) & (r[:, 1] >= 64), 'ge64_labels_differ')
    # RSAR aircraft: size of the population.
    want = {Path(i['file_name']).stem: i['id'] for i in val['images']}
    sides = []
    with tarfile.open('/root/zxy/TCRA-SAR/datasets/RSAR.tar', 'r:') as archive:
        for member in archive:
            if member.name.startswith('val/annfiles/') and member.name.endswith('.txt') and Path(member.name).stem in want:
                for line in archive.extractfile(member).read().decode().splitlines():
                    p = line.split()
                    if len(p) >= 9 and p[8] == 'aircraft':
                        c = np.array(p[:8], float)
                        sides.append(max(np.ptp(c[0::2]), np.ptp(c[1::2])))
    report['rsar_aircraft_side_percentiles'] = np.round(np.percentile(sides, [10, 50, 90]), 1).tolist()
    (OUT / 'twin.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
