"""Val-only: run the model on horizontally flipped images and flip boxes back. Does the OGSOD dx keep its sign?"""
import glob
import json
import multiprocessing as mp
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

RUN = Path(sys.argv[1])
BASE = Path('/root/zxy/TCRA-SAR/datasets')
ROOT = BASE / 'SARDet_100K'
OUT = RUN / 'independent_verify_20261106/source'
val = json.load(open(ROOT / 'Annotations/val.json'))
names = {c['id']: c['name'] for c in val['categories']}
images = {i['id']: i for i in val['images']}
gts = defaultdict(list)
for a in val['annotations']:
    x, y, w, h = a['bbox']
    gts[a['image_id']].append((a['category_id'], np.array([x, y, x + w, y + h], np.float32)))
THR = np.linspace(.5, .95, 10)
REC = np.linspace(0, 1, 101)


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


class Images:
    def __init__(self, flip):
        self.flip = flip

    def __len__(self):
        return len(val['images'])

    def __getitem__(self, i):
        import torchvision.transforms.functional as F
        image = Image.open(ROOT / 'JPEGImages/val' / val['images'][i]['file_name']).convert('RGB')
        size = image.size
        if self.flip == 'h':
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif self.flip == 'v':
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        return F.to_tensor(F.resize(image, [640, 640])), i, size[0], size[1]


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


def evaluate(pred, ids):
    out = {}
    for c, name in names.items():
        lines = {t: ([], []) for t in range(10)}
        n = 0
        for image_id in ids:
            cur = [g[1] for g in gts.get(image_id, []) if g[0] == c]
            n += len(cur)
            labels, scores, boxes = pred[image_id]
            keep = np.flatnonzero(labels == c)
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


def offsets(pred, src):
    rows = defaultdict(list)
    for image_id, cur in gts.items():
        labels, scores, boxes = pred[image_id]
        keep = scores >= .3
        labels, boxes = labels[keep], boxes[keep]
        for c in {g[0] for g in cur}:
            mine = boxes[labels == c]
            target = np.stack([g[1] for g in cur if g[0] == c])
            if not len(mine):
                continue
            m = iou(mine, target)
            for g in range(len(target)):
                d = int(m[:, g].argmax())
                if m[d, g] >= .3:
                    rows[(src[image_id], names[c])].append(((mine[d, 0] + mine[d, 2] - target[g, 0] - target[g, 2]) / 2, (mine[d, 1] + mine[d, 3] - target[g, 1] - target[g, 3]) / 2))
    return {f'{s}/{c}': dict(n=len(v), mean_dx=round(float(np.mean([r[0] for r in v])), 3), mean_dy=round(float(np.mean([r[1] for r in v])), 3)) for (s, c), v in rows.items() if len(v) >= 30}


def main():
    import torch
    from engine.core import YAMLConfig
    ogsod = sorted(glob.glob(str(BASE / 'OGSOD-1.0/sar/images/*/*')))
    msar = sorted(glob.glob(str(BASE / 'MSAR-1.0 dataset/JPEGImages/*')))
    paths = [str(ROOT / 'JPEGImages/val' / i['file_name']) for i in val['images']]
    with mp.Pool(32) as pool:
        thumbs = pool.map(thumb, paths, chunksize=64)
        labels = np.array(['other'] * len(paths), dtype=object)
        query = torch.as_tensor(np.stack([t[0] for t in thumbs])).cuda()
        for name, ref_paths in (('ogsod', ogsod), ('msar', msar)):
            refs = pool.map(thumb, ref_paths, chunksize=64)
            R = torch.as_tensor(np.stack([t[0] for t in refs])).cuda()
            bs, bi = [], []
            for s in range(0, len(query), 1024):
                a, b = (query[s:s + 1024] @ R.T).max(1)
                bs.append(a.cpu().numpy())
                bi.append(b.cpu().numpy())
            bs, bi = np.concatenate(bs), np.concatenate(bi)
            cand = [(q, int(bi[q])) for q in np.flatnonzero(bs >= .98) if thumbs[q][1] == refs[bi[q]][1]]
            ok = pool.map(same, [(paths[q], ref_paths[r]) for q, r in cand], chunksize=64)
            labels[[q for (q, r), g in zip(cand, ok) if g]] = name
            del R
        del query
        torch.cuda.empty_cache()
    src = {i['id']: s for i, s in zip(val['images'], labels)}
    cfg = YAMLConfig(str(RUN / 'resolved_config.yml'))
    model, post = cfg.model, cfg.postprocessor
    model.load_state_dict(torch.load(RUN / 'best_stg2.pth', map_location='cpu')['ema']['module'])
    model, post = model.cuda().eval(), post.cuda().eval()
    report = {}
    subsets = {s: [i for i, v in src.items() if v == s] for s in ('ogsod', 'msar', 'other')}
    subsets['all'] = list(images)
    for flip in ('none', 'h', 'v'):
        pred = {}
        loader = torch.utils.data.DataLoader(Images(flip), batch_size=64, num_workers=8)
        with torch.no_grad():
            for batch, order, w, h in loader:
                for i, r, W, H in zip(order.tolist(), post(model(batch.cuda()), torch.stack([w, h], 1).cuda()), w.tolist(), h.tolist()):
                    b = r['boxes'].cpu().numpy().copy()
                    if flip == 'h':
                        b[:, [0, 2]] = W - b[:, [2, 0]]
                    elif flip == 'v':
                        b[:, [1, 3]] = H - b[:, [3, 1]]
                    pred[val['images'][i]['id']] = (r['labels'].cpu().numpy().astype(int), r['scores'].cpu().numpy(), b)
        report[flip] = dict(offsets=offsets(pred, src), ap={k: evaluate(pred, v) for k, v in subsets.items()})
        print(flip, json.dumps(report[flip]['offsets'], ensure_ascii=False), flush=True)
    (OUT / 'flip_test.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print('RESULT', json.dumps({k: dict(offsets={a: b for a, b in v['offsets'].items() if a.startswith('ogsod') or a in ('msar/ship', 'other/ship', 'other/aircraft')}, ap={a: b.get('mean') for a, b in v['ap'].items()}, ogsod=v['ap']['ogsod']) for k, v in report.items()}, ensure_ascii=False), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
