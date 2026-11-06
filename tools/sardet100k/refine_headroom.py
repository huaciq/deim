"""Val-only, detector frozen: cross-fitted small head on raw-pixel crops. How much AP is reachable by edge refinement and quality?

Folds are split by near-duplicate groups so overlapping tiles do not cross folds. Also tests per-brightness gamma choice.
"""
import json
import multiprocessing as mp
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

RUN = Path(sys.argv[1])
OUT = RUN / 'independent_verify_20261106/refine'
OUT.mkdir(parents=True, exist_ok=True)
ROOT = Path('/root/zxy/TCRA-SAR/datasets/SARDet_100K')
val = json.load(open(ROOT / 'Annotations/val.json'))
names = {c['id']: c['name'] for c in val['categories']}
images = {i['id']: i for i in val['images']}
saved = np.load(RUN / 'independent_verify_20261106/val_predictions.npz')
PI, PC, PS, PB = saved['image_id'], saved['label'].astype(int), saved['score'].astype(np.float32), saved['box'].astype(np.float32)
gts = defaultdict(list)
for a in val['annotations']:
    x, y, w, h = a['bbox']
    gts[a['image_id']].append((a['category_id'], np.array([x, y, x + w, y + h], np.float32)))
order = np.argsort(PI, kind='stable')
vals, first = np.unique(PI[order], return_index=True)
SLICES = dict(zip(vals.tolist(), np.split(order, first[1:])))
THR = np.linspace(.5, .95, 10)
REC = np.linspace(0, 1, 101)
MIN_SCORE = .05
CROP = 48
report = {}


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


def evaluate(pc, ps, pb, slices, ids=None):
    out, lines_mean = {}, []
    ids = list(images) if ids is None else ids
    for c, name in names.items():
        lines = {t: ([], []) for t in range(10)}
        n = 0
        for image_id in ids:
            cur = [g[1] for g in gts.get(image_id, []) if g[0] == c]
            n += len(cur)
            pos = slices.get(image_id, np.array([], int))
            keep = pos[pc[pos] == c]
            if not len(keep):
                continue
            keep = keep[np.argsort(-ps[keep], kind='mergesort')][:100]
            m = iou(pb[keep], np.stack(cur)) if cur else None
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
                lines[t][0].append(ps[keep])
                lines[t][1].append(hit)
        if n >= 30 and lines[0][0]:
            v = [ap_curve(np.concatenate(lines[t][0]), np.concatenate(lines[t][1]), n) for t in range(10)]
            out[name] = dict(gt=n, AP=round(float(np.mean(v)) * 100, 2), AP50=round(v[0] * 100, 2), AP75=round(v[5] * 100, 2), AP90=round(v[8] * 100, 2))
            lines_mean.append(v)
    if lines_mean:
        m = np.array(lines_mean)
        out['mean'] = dict(AP=round(float(m.mean()) * 100, 2), AP50=round(float(m[:, 0].mean()) * 100, 2), AP75=round(float(m[:, 5].mean()) * 100, 2), AP85=round(float(m[:, 7].mean()) * 100, 2), AP90=round(float(m[:, 8].mean()) * 100, 2))
    return out


def crops_for(image_id):
    im = images[image_id]
    values = np.asarray(Image.open(ROOT / 'JPEGImages/val' / im['file_name']).convert('L'), np.float32)
    H, W = values.shape
    pos = SLICES.get(image_id, np.array([], int))
    pos = pos[PS[pos] >= MIN_SCORE]
    rows = []
    for p in pos:
        x0, y0, x1, y1 = PB[p]
        w, h = max(x1 - x0, 1.), max(y1 - y0, 1.)
        ax0, ay0 = x0 - max(.5 * w, 4), y0 - max(.5 * h, 4)
        ax1, ay1 = x1 + max(.5 * w, 4), y1 + max(.5 * h, 4)
        ix0, iy0, ix1, iy1 = int(np.floor(ax0)), int(np.floor(ay0)), int(np.ceil(ax1)), int(np.ceil(ay1))
        canvas = np.zeros((iy1 - iy0, ix1 - ix0), np.float32)
        sx0, sy0, sx1, sy1 = max(ix0, 0), max(iy0, 0), min(ix1, W), min(iy1, H)
        if sx1 > sx0 and sy1 > sy0:
            canvas[sy0 - iy0:sy1 - iy0, sx0 - ix0:sx1 - ix0] = values[sy0:sy1, sx0:sx1]
        patch = np.asarray(Image.fromarray(canvas).resize((CROP, CROP), Image.BILINEAR), np.float32)
        rows.append((int(p), patch.astype(np.uint8), float(W)))
    return rows


def thumb(row):
    image = Image.open(ROOT / 'JPEGImages/val' / row['file_name']).convert('L')
    t = np.asarray(image.resize((32, 32), Image.BILINEAR), np.float32).ravel()
    t = t - t.mean()
    n = np.linalg.norm(t)
    return t / n if n > 0 else t


def folds():
    import torch
    with mp.Pool(32) as pool:
        thumbs = pool.map(thumb, val['images'], chunksize=64)
    T = torch.as_tensor(np.stack(thumbs)).cuda()
    parent = list(range(len(T)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for start in range(0, len(T), 1024):
        s, i = (T[start:start + 1024] @ T.T).topk(10, dim=1)
        s, i = s.cpu().numpy(), i.cpu().numpy()
        for r in range(len(s)):
            for score, b in zip(s[r], i[r]):
                if score >= .9:
                    parent[find(start + r)] = find(int(b))
    del T
    torch.cuda.empty_cache()
    groups = np.array([find(i) for i in range(len(thumbs))])
    uniq, sizes = np.unique(groups, return_counts=True)
    rng = np.random.default_rng(0)
    assign = dict(zip(uniq.tolist(), rng.integers(0, 2, len(uniq)).tolist()))
    report['folds'] = dict(groups=len(uniq), largest_group=int(sizes.max()))
    return {i['id']: assign[int(g)] for i, g in zip(val['images'], groups)}


def main():
    import torch
    import torch.nn as nn
    fold_of = folds()
    with mp.Pool(32) as pool:
        parts = pool.map(crops_for, list(images), chunksize=32)
    rows = [r for part in parts for r in part]
    index = np.array([r[0] for r in rows])
    patches = np.stack([r[1] for r in rows])
    source_width = np.array([r[2] for r in rows], np.float32)
    del parts, rows
    fold = np.array([fold_of[int(PI[p])] for p in index])
    report['detections'] = int(len(index))
    # Targets.
    delta, quality = np.zeros((len(index), 4), np.float32), np.zeros(len(index), np.float32)
    per_image = defaultdict(list)
    for k, p in enumerate(index):
        per_image[int(PI[p])].append(k)
    for image_id, members in per_image.items():
        members = np.array(members)
        for c in names:
            mine = members[PC[index[members]] == c]
            target = [g[1] for g in gts.get(image_id, []) if g[0] == c]
            if not len(mine) or not target:
                continue
            target = np.stack(target)
            box = PB[index[mine]]
            m = iou(box, target)
            best = m.argmax(1)
            quality[mine] = m.max(1)
            size = np.maximum(box[:, [2, 3, 2, 3]] - box[:, [0, 1, 0, 1]], 1)
            delta[mine] = np.clip((target[best] - box) / size, -.5, .5)
    has = quality >= .5
    box = PB[index]
    w, h = np.maximum(box[:, 2] - box[:, 0], 1), np.maximum(box[:, 3] - box[:, 1], 1)
    score = np.clip(PS[index], 1e-4, 1 - 1e-4)
    extras = np.concatenate([np.eye(len(names), dtype=np.float32)[PC[index]], np.stack([np.log(w) / 5, np.log(h) / 5, np.log(source_width) / 7, np.log(score / (1 - score)) / 5], 1)], 1)
    images_t = torch.as_tensor(patches, dtype=torch.float32)
    del patches

    def channels(batch, mode):
        grey = batch.unsqueeze(1) / 255
        if mode == 'grey':
            return grey
        logged = torch.log1p(batch).unsqueeze(1) / np.log(256)
        size = batch.shape[-1]
        mask = torch.ones_like(batch)
        a, b = size // 4, size - size // 4
        mask[:, a:b, a:b] = 0
        ring = (batch * mask).sum((1, 2), keepdim=True) / mask.sum((1, 2), keepdim=True)
        ratio = torch.log((batch + 1) / (ring + 1)).unsqueeze(1) / 3
        return torch.cat([grey, logged, ratio], 1)

    class Head(nn.Module):
        def __init__(self, inputs, use_image):
            super().__init__()
            self.use_image = use_image
            layers, c = [], inputs
            for out in (32, 64, 128, 192):
                layers += [nn.Conv2d(c, out, 3, padding=1, bias=False), nn.BatchNorm2d(out), nn.SiLU(), nn.Conv2d(out, out, 3, padding=1, bias=False), nn.BatchNorm2d(out), nn.SiLU(), nn.MaxPool2d(2)]
                c = out
            self.net = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.mlp = nn.Sequential(nn.Linear((192 if use_image else 0) + extras.shape[1], 256), nn.SiLU(), nn.Linear(256, 5))

        def forward(self, x, e):
            feat = torch.cat([self.net(x), e], 1) if self.use_image else e
            return self.mlp(feat)

    extras_t = torch.as_tensor(extras)
    delta_t = torch.as_tensor(delta)
    quality_t = torch.as_tensor(quality)
    has_t = torch.as_tensor(has)

    def crossfit(mode):
        pred = np.zeros((len(index), 5), np.float32)
        use_image = mode != 'no_image'
        inputs = 1 if mode == 'grey' else 3
        for test_fold in (0, 1):
            train_idx = torch.as_tensor(np.flatnonzero(fold != test_fold))
            test_idx = torch.as_tensor(np.flatnonzero(fold == test_fold))
            torch.manual_seed(0)
            model = Head(inputs, use_image).cuda()
            epochs = 14
            opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
            steps = epochs * (len(train_idx) // 512)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
            for epoch in range(epochs):
                model.train()
                perm = train_idx[torch.randperm(len(train_idx))]
                for s in range(0, len(perm) - 511, 512):
                    b = perm[s:s + 512]
                    x = channels(images_t[b].cuda(), 'phys' if inputs == 3 else 'grey')
                    out = model(x, extras_t[b].cuda())
                    mask = has_t[b].cuda()
                    reg = nn.functional.smooth_l1_loss(out[mask, :4], delta_t[b].cuda()[mask], beta=.05) if mask.any() else out.sum() * 0
                    qual = nn.functional.binary_cross_entropy_with_logits(out[:, 4], quality_t[b].cuda())
                    loss = 4 * reg + qual
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    sched.step()
            model.eval()
            with torch.no_grad():
                for s in range(0, len(test_idx), 2048):
                    b = test_idx[s:s + 2048]
                    x = channels(images_t[b].cuda(), 'phys' if inputs == 3 else 'grey')
                    pred[b.numpy()] = model(x, extras_t[b].cuda()).cpu().numpy()
            del model
            torch.cuda.empty_cache()
            print('fold', mode, test_fold, flush=True)
        return pred

    report['variants'] = {}
    report['base'] = evaluate(PC, PS, PB, SLICES)
    print('base', report['base']['mean'], flush=True)
    for mode in ('no_image', 'grey', 'phys'):
        pred = crossfit(mode)
        size = np.stack([w, h, w, h], 1)
        refined = box + pred[:, :4] * size
        q = 1 / (1 + np.exp(-pred[:, 4]))
        # Accuracy of edges on matched detections.
        before = np.abs(delta[has]) * size[has]
        after = np.abs(delta[has] - pred[has, :4]) * size[has]
        entry = dict(median_edge_error_before=round(float(np.median(before)), 3), median_edge_error_after=round(float(np.median(after)), 3))
        boxes_new = PB.copy()
        boxes_new[index] = refined
        scores_new = PS.copy()
        scores_new[index] = np.sqrt(np.clip(score, 1e-6, 1) * np.clip(q, 1e-6, 1))
        entry['refine_only'] = evaluate(PC, PS, boxes_new, SLICES)
        entry['rescore_only'] = evaluate(PC, scores_new, PB, SLICES)
        entry['both'] = evaluate(PC, scores_new, boxes_new, SLICES)
        report['variants'][mode] = entry
        print(mode, entry['median_edge_error_before'], entry['median_edge_error_after'], entry['refine_only']['mean'], entry['rescore_only']['mean'], entry['both']['mean'], flush=True)
    (OUT / 'report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))

    # Per-brightness choice among gamma conditions from saved predictions.
    gamma = {}
    with mp.Pool(16) as pool:
        medians = dict(pool.map(median_of, list(images), chunksize=64))
    bins = {'p50<5': [i for i, v in medians.items() if v < 5], '5-30': [i for i, v in medians.items() if 5 <= v < 30], '>=30': [i for i, v in medians.items() if v >= 30]}
    for condition in ('clean', 'gamma_0.7', 'gamma_1.4'):
        data = np.load(RUN / f'independent_verify_20261106/robustness/{condition}.npz')
        pi, pc, ps, pb = data['image_id'], data['label'].astype(int), data['score'], data['box']
        o = np.argsort(pi, kind='stable')
        v, f = np.unique(pi[o], return_index=True)
        sl = dict(zip(v.tolist(), np.split(o, f[1:])))
        gamma[condition] = {k: evaluate(pc, ps, pb, sl, ids=ids) for k, ids in bins.items()}
        print(condition, {k: x.get('mean') for k, x in gamma[condition].items()}, flush=True)
    report['gamma_by_brightness'] = dict(sizes={k: len(v) for k, v in bins.items()}, results=gamma)
    (OUT / 'report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print('RESULT', json.dumps(report, ensure_ascii=False), flush=True)
    print('DONE', flush=True)


def median_of(image_id):
    v = np.asarray(Image.open(ROOT / 'JPEGImages/val' / images[image_id]['file_name']).convert('L'), np.float32)
    return image_id, float(np.median(v))


if __name__ == '__main__':
    main()
