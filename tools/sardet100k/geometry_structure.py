"""Val-only structure of label geometry: what does the offset depend on, and what can be predicted?"""
import glob
import json
import multiprocessing as mp
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

RUN = Path(sys.argv[1])
OUT = RUN / 'independent_verify_20261106/geometry'
OUT.mkdir(parents=True, exist_ok=True)
BASE = Path('/root/zxy/TCRA-SAR/datasets')
ROOT = BASE / 'SARDet_100K'
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
    gts[a['image_id']].append((a['category_id'], np.array([x, y, x + w, y + h], np.float32)))
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


def r2_cv(x, y, groups, ridge=1e-3):
    """Two-fold ridge R2 by groups."""
    rng = np.random.default_rng(0)
    uniq = np.unique(groups)
    fold = dict(zip(uniq.tolist(), rng.integers(0, 2, len(uniq)).tolist()))
    f = np.array([fold[g] for g in groups.tolist()])
    pred = np.zeros_like(y, dtype=np.float64)
    for k in (0, 1):
        a, b = f != k, f == k
        xa = np.c_[x[a], np.ones(a.sum())]
        w = np.linalg.solve(xa.T @ xa + ridge * np.eye(xa.shape[1]), xa.T @ y[a])
        pred[b] = np.c_[x[b], np.ones(b.sum())] @ w
    return round(float(1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9)), 4)


def crop_pair(task):
    path, boxes, side = task
    v = np.asarray(Image.open(path).convert('L'), np.float32)
    h, w = v.shape
    out = []
    for gt_box, pred_box in boxes:
        row = []
        for box in (gt_box, pred_box):
            x0, y0, x1, y1 = box
            bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)
            sx = np.linspace(x0 - bw, x1 + bw, side)
            sy = np.linspace(y0 - bh, y1 + bh, side)
            ix = np.clip(np.round(sx - .5).astype(int), 0, w - 1)
            iy = np.clip(np.round(sy - .5).astype(int), 0, h - 1)
            row.append(np.log1p(v[np.ix_(iy, ix)]))
        out.append((row[0], row[1], float(bw), float(bh)))
    return out


def main():
    import torch
    ogsod = sorted(glob.glob(str(BASE / 'OGSOD-1.0/sar/images/*/*')))
    msar = sorted(glob.glob(str(BASE / 'MSAR-1.0 dataset/JPEGImages/*')))
    paths = [str(ROOT / 'JPEGImages/val' / i['file_name']) for i in val['images']]
    source = np.array(['other'] * len(paths), dtype=object)
    parent = {}
    with mp.Pool(32) as pool:
        thumbs = pool.map(thumb, paths, chunksize=64)
        Q = torch.as_tensor(np.stack([t[0] for t in thumbs])).cuda()
        for name, refs in (('ogsod', ogsod), ('msar', msar)):
            rt = pool.map(thumb, refs, chunksize=64)
            R = torch.as_tensor(np.stack([t[0] for t in rt])).cuda()
            bs, bi = [], []
            for s in range(0, len(Q), 1024):
                a, b = (Q[s:s + 1024] @ R.T).max(1)
                bs.append(a.cpu().numpy())
                bi.append(b.cpu().numpy())
            bs, bi = np.concatenate(bs), np.concatenate(bi)
            cand = [(q, int(bi[q])) for q in np.flatnonzero(bs >= .98) if thumbs[q][1] == rt[bi[q]][1]]
            ok = pool.map(same, [(paths[q], refs[r]) for q, r in cand], chunksize=64)
            for (q, r), g in zip(cand, ok):
                if g:
                    source[q] = name
                    parent[val['images'][q]['id']] = os.path.basename(refs[r])
            del R
        del Q
        torch.cuda.empty_cache()
        src = {i['id']: s for i, s in zip(val['images'], source)}
        json.dump({str(k): v for k, v in src.items()}, open(OUT / 'val_source.json', 'w'))
        json.dump({str(k): v for k, v in parent.items()}, open(OUT / 'val_parent.json', 'w'))
        names_sample = Counter(re.sub(r'^\d+', 'N', v) for k, v in parent.items() if src[k] == 'ogsod').most_common(6)
        report['ogsod_filename_patterns'] = names_sample

        # Matched confident detections.
        rows = []
        for image_id, cur in gts.items():
            pos = SLICES.get(image_id, np.array([], int))
            high = pos[PS[pos] >= .3]
            im = images[image_id]
            for c in {g[0] for g in cur}:
                target = np.stack([g[1] for g in cur if g[0] == c])
                keep = high[PC[high] == c]
                if not len(keep):
                    continue
                m = iou(PB[keep], target)
                for g in range(len(target)):
                    d = int(m[:, g].argmax())
                    if m[d, g] >= .3:
                        p, t = PB[keep[d]], target[g]
                        rows.append(dict(image=image_id, c=c, src=src[image_id], W=im['width'], H=im['height'], p=p, t=t, iou=float(m[d, g]), score=float(PS[keep[d]])))
        report['matched'] = len(rows)

        def arr(selector):
            sel = [r for r in rows if selector(r)]
            p = np.array([r['p'] for r in sel])
            t = np.array([r['t'] for r in sel])
            return sel, p, t

        # A. Edge errors and dependence.
        report['edge_errors'] = {}
        report['dependence'] = {}
        for s in ('ogsod', 'msar', 'other'):
            for c, name in names.items():
                sel, p, t = arr(lambda r: r['src'] == s and r['c'] == c)
                if len(sel) < 100:
                    continue
                e = p - t
                dx, dy = (e[:, 0] + e[:, 2]) / 2, (e[:, 1] + e[:, 3]) / 2
                report['edge_errors'][f'{s}/{name}'] = dict(n=len(sel), left=round(float(e[:, 0].mean()), 3), top=round(float(e[:, 1].mean()), 3), right=round(float(e[:, 2].mean()), 3), bottom=round(float(e[:, 3].mean()), 3), dx_sd=round(float(dx.std()), 3), dy_sd=round(float(dy.std()), 3))
                w, h = t[:, 2] - t[:, 0], t[:, 3] - t[:, 1]
                cx = (t[:, 0] + t[:, 2]) / 2 / np.array([r['W'] for r in sel])
                cy = (t[:, 1] + t[:, 3]) / 2 / np.array([r['H'] for r in sel])
                tertile = lambda v, q: [round(float(q[(v >= a) & (v <= b)].mean()), 3) for a, b in zip(np.percentile(v, [0, 33, 66]), np.percentile(v, [33, 66, 100]))]
                report['dependence'][f'{s}/{name}'] = dict(dx_by_width_tertile=tertile(w, dx), dx_by_height_tertile=tertile(h, dx), dx_by_cx_tertile=tertile(cx, dx), dx_by_cy_tertile=tertile(cy, dx), dx_by_score_tertile=tertile(np.array([r['score'] for r in sel]), dx), dy_by_height_tertile=tertile(h, dy), dy_by_cy_tertile=tertile(cy, dy), slope_dx_per_width=round(float(np.polyfit(w, dx, 1)[0]), 4), slope_dx_per_cx=round(float(np.polyfit(cx, dx, 1)[0]), 3))

        # B. Per-image offsets.
        report['per_image'] = {}
        for s, cs in (('ogsod', [3, 4, 5]), ('other', [1, 0]), ('msar', [0, 3])):
            for c in cs:
                by = defaultdict(list)
                for r in rows:
                    if r['src'] == s and r['c'] == c:
                        by[r['image']].append(((r['p'][0] + r['p'][2] - r['t'][0] - r['t'][2]) / 2, (r['p'][1] + r['p'][3] - r['t'][1] - r['t'][3]) / 2))
                medians = np.array([np.median(v, 0) for v in by.values() if len(v) >= 3])
                if len(medians) < 20:
                    continue
                report['per_image'][f'{s}/{names[c]}'] = dict(images=len(medians), dx_p10_50_90=[round(float(x), 2) for x in np.percentile(medians[:, 0], [10, 50, 90])], dy_p10_50_90=[round(float(x), 2) for x in np.percentile(medians[:, 1], [10, 50, 90])], share_dx_negative=round(float((medians[:, 0] < 0).mean()), 3), histogram_dx=np.histogram(medians[:, 0], bins=[-4, -2, -1, -.5, 0, .5, 1, 1.5, 2, 3, 6])[0].tolist(), corr_dx_dy=round(float(np.corrcoef(medians[:, 0], medians[:, 1])[0, 1]), 3))

        # C. Averaged appearance: where is the radar response relative to label and prediction?
        report['average_crop'] = {}
        side = 49
        arrays = {}
        for s, c in (('ogsod', 3), ('ogsod', 4), ('ogsod', 5), ('msar', 3), ('msar', 0), ('other', 0), ('other', 1), ('other', 2)):
            sel = [r for r in rows if r['src'] == s and r['c'] == c and r['iou'] >= .5]
            groups = defaultdict(list)
            for r in sel:
                groups[r['image']].append((r['t'], r['p']))
            tasks = [(str(ROOT / 'JPEGImages/val' / images[i]['file_name']), v, side) for i, v in groups.items()]
            parts = pool.map(crop_pair, tasks, chunksize=32)
            samples = [x for part in parts for x in part]
            if len(samples) < 50:
                continue
            gt_mean = np.mean([x[0] - np.median(x[0]) for x in samples], 0)
            pred_mean = np.mean([x[1] - np.median(x[1]) for x in samples], 0)
            mean_w = float(np.mean([x[2] for x in samples]))
            mean_h = float(np.mean([x[3] for x in samples]))

            def centroid(a):
                a = np.clip(a - np.percentile(a, 60), 0, None)
                ys, xs = np.mgrid[0:side, 0:side]
                to_box = 3 / (side - 1)  # image spans three box sizes.
                return float(((xs - (side - 1) / 2) * a).sum() / a.sum() * to_box), float(((ys - (side - 1) / 2) * a).sum() / a.sum() * to_box)

            def edge_profile(a):
                # Location of the steepest rise and fall along x, in box-width units from centre.
                prof = a[side // 3:2 * side // 3 + 1].mean(0)
                grad = np.gradient(prof)
                return float((grad.argmax() - (side - 1) / 2) * 3 / (side - 1)), float((grad.argmin() - (side - 1) / 2) * 3 / (side - 1))

            cg, cp = centroid(gt_mean), centroid(pred_mean)
            report['average_crop'][f'{s}/{names[c]}'] = dict(n=len(samples), mean_w=round(mean_w, 1), mean_h=round(mean_h, 1), gt_centroid_dx_px=round(cg[0] * mean_w, 3), gt_centroid_dy_px=round(cg[1] * mean_h, 3), pred_centroid_dx_px=round(cp[0] * mean_w, 3), pred_centroid_dy_px=round(cp[1] * mean_h, 3), gt_edges_x_box_units=[round(v, 3) for v in edge_profile(gt_mean)], pred_edges_x_box_units=[round(v, 3) for v in edge_profile(pred_mean)])
            arrays[f'{s}_{names[c]}'] = np.stack([gt_mean, pred_mean])
        np.savez(OUT / 'average_crops.npz', **arrays)

    # D. Explained variance of OGSOD offsets.
    sel, p, t = arr(lambda r: r['src'] == 'ogsod' and r['iou'] >= .5)
    dx = ((p[:, 0] + p[:, 2] - t[:, 0] - t[:, 2]) / 2).astype(np.float64)
    groups = np.array([r['image'] for r in sel])
    cls = np.eye(6)[[r['c'] for r in sel]]
    w, h = p[:, 2] - p[:, 0], p[:, 3] - p[:, 1]
    size = np.c_[np.log(w), np.log(h), w, h]
    pos = np.c_[(p[:, 0] + p[:, 2]) / 512, (p[:, 1] + p[:, 3]) / 512]
    score = np.array([r['score'] for r in sel])[:, None]
    report['ogsod_dx_r2'] = dict(n=len(dx), variance=round(float(dx.var()), 3), mean=round(float(dx.mean()), 3), mean_squared_vs_variance=round(float(dx.mean() ** 2 / (dx.var() + dx.mean() ** 2)), 3), class_only=r2_cv(cls, dx, groups), class_size=r2_cv(np.c_[cls, size], dx, groups), class_size_pos=r2_cv(np.c_[cls, size, pos], dx, groups), class_size_pos_score=r2_cv(np.c_[cls, size, pos, score], dx, groups))
    # Oracle: how much would the per-image mean explain?
    means = defaultdict(list)
    for g, v in zip(groups, dx):
        means[g].append(v)
    loo = np.array([(sum(means[g]) - v) / (len(means[g]) - 1) if len(means[g]) >= 3 else dx.mean() for g, v in zip(groups, dx)])
    report['ogsod_dx_r2']['leave_one_out_same_image_mean'] = round(float(1 - ((dx - loo) ** 2).sum() / ((dx - dx.mean()) ** 2).sum()), 4)

    # Predict per-image offset from the image itself with a small CNN, cross-fitted.
    per = defaultdict(list)
    for r, v in zip(sel, dx):
        per[r['image']].append(v)
    ids = [i for i, v in per.items() if len(v) >= 2]
    target = np.array([np.median(per[i]) for i in ids], np.float32)
    data = np.stack([np.asarray(Image.open(ROOT / 'JPEGImages/val' / images[i]['file_name']).convert('L').resize((96, 96), Image.BILINEAR), np.float32) / 255 for i in ids])
    X, Y = torch.as_tensor(data).unsqueeze(1), torch.as_tensor(target)
    pred = np.zeros(len(ids), np.float32)
    rng = np.random.default_rng(0)
    fold = rng.integers(0, 2, len(ids))
    for k in (0, 1):
        torch.manual_seed(0)
        net = torch.nn.Sequential(*[m for c0, c1 in ((1, 16), (16, 32), (32, 64), (64, 96)) for m in (torch.nn.Conv2d(c0, c1, 3, padding=1), torch.nn.BatchNorm2d(c1), torch.nn.SiLU(), torch.nn.MaxPool2d(2))], torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(96, 1)).cuda()
        opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
        tr = torch.as_tensor(np.flatnonzero(fold != k))
        te = torch.as_tensor(np.flatnonzero(fold == k))
        for epoch in range(40):
            net.train()
            perm = tr[torch.randperm(len(tr))]
            for s in range(0, len(perm), 64):
                b = perm[s:s + 64]
                loss = torch.nn.functional.smooth_l1_loss(net(X[b].cuda()).squeeze(1), Y[b].cuda())
                opt.zero_grad()
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            pred[te.numpy()] = net(X[te].cuda()).squeeze(1).cpu().numpy()
    report['ogsod_image_level_dx'] = dict(images=len(ids), sd=round(float(target.std()), 3), mean=round(float(target.mean()), 3), cnn_r2=round(float(1 - ((target - pred) ** 2).sum() / ((target - target.mean()) ** 2).sum()), 4))

    # E. Truncated aircraft: can the out-of-image extent be inferred from sizes elsewhere in the image?
    trunc = []
    by_img = defaultdict(list)
    for g in val['annotations']:
        if g['category_id'] == 1:
            by_img[g['image_id']].append(g['bbox'])
    for r in rows:
        if r['c'] != 1:
            continue
        t, W, H = r['t'], r['W'], r['H']
        over = np.array([max(-t[0], 0), max(-t[1], 0), max(t[2] - W, 0), max(t[3] - H, 0)])
        if over.max() <= 1:
            continue
        inside = [b for b in by_img[r['image']] if b[0] >= -1 and b[1] >= -1 and b[0] + b[2] <= W + 1 and b[1] + b[3] <= H + 1]
        p = r['p']
        clipped = np.array([max(t[0], 0), max(t[1], 0), min(t[2], W), min(t[3], H)], np.float32)
        full = float(iou(p[None], t[None])[0, 0])
        vis = float(iou(p[None], clipped[None])[0, 0])
        amodal = None
        if inside:
            mw, mh = np.median([b[2] for b in inside]), np.median([b[3] for b in inside])
            q = p.copy()
            if t[0] < 0:
                q[0] = min(q[0], q[2] - mw)
            if t[2] > W:
                q[2] = max(q[2], q[0] + mw)
            if t[1] < 0:
                q[1] = min(q[1], q[3] - mh)
            if t[3] > H:
                q[3] = max(q[3], q[1] + mh)
            amodal = float(iou(q[None], t[None])[0, 0])
        trunc.append((full, vis, amodal, over.max(), max(t[2] - t[0], t[3] - t[1]), len(inside)))
    arrt = np.array([(a, b, c if c is not None else np.nan, d, e, f) for a, b, c, d, e, f in trunc], np.float64)
    ok = ~np.isnan(arrt[:, 2])
    report['truncated_aircraft'] = dict(n=len(trunc), median_iou_official=round(float(np.median(arrt[:, 0])), 3), median_iou_vs_visible=round(float(np.median(arrt[:, 1])), 3), share_with_full_aircraft_in_image=round(float(ok.mean()), 3), median_iou_with_size_consensus=round(float(np.median(arrt[ok, 2])), 3), median_iou_official_same_subset=round(float(np.median(arrt[ok, 0])), 3), median_overshoot_px=round(float(np.median(arrt[:, 3])), 2), overshoot_vs_long_side=round(float(np.median(arrt[:, 3] / arrt[:, 4])), 3))
    (OUT / 'report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str))
    print('RESULT', json.dumps(report, ensure_ascii=False, default=str), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
