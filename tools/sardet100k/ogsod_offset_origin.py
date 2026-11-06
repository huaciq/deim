"""Is the OGSOD x offset in the model or in the labels? Train images and val only; no test predictions."""
import glob
import json
import multiprocessing as mp
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

RUN = Path(sys.argv[1])
BASE = Path('/root/zxy/TCRA-SAR/datasets')
ROOT = BASE / 'SARDet_100K'
OUT = RUN / 'independent_verify_20261106/source'
report = {}
native = BASE / 'OGSOD-1.0/sar'


# 1. Raw label formats, native OGSOD. Test labels are read for format only.
def sample(split):
    files = sorted(glob.glob(str(native / f'labels/{split}/*')))[:4000]
    fractions, lines, examples = Counter(), 0, []
    for f in files:
        for line in open(f):
            parts = line.split()
            if len(examples) < 4:
                examples.append(line.strip())
            lines += 1
            if len(parts) >= 5:
                for value in parts[1:5]:
                    v = float(value)
                    fractions[round(v * 256 % 1, 1)] += 1
    return dict(files=len(files), lines=lines, examples=examples, frac_of_256xvalue=fractions.most_common(6))


report['native_label_format'] = {split: sample(split) for split in ('train', 'test')}
for split in ('train', 'val', 'test'):
    data = json.load(open(native / f'RTDETR_COCO/{split}.json'))
    boxes = np.array([a['bbox'] for a in data['annotations']], np.float64)
    report[f'native_coco_{split}'] = dict(images=len(data['images']), anns=len(boxes), first=data['annotations'][0]['bbox'], x_frac=Counter(np.round(boxes[:, 0] % 1, 2).tolist()).most_common(4), w_frac=Counter(np.round(boxes[:, 2] % 1, 2).tolist()).most_common(4), mean_x=round(float(boxes[:, 0].mean()), 3), mean_cx=round(float((boxes[:, 0] + boxes[:, 2] / 2).mean()), 3), mean_cy=round(float((boxes[:, 1] + boxes[:, 3] / 2).mean()), 3), mean_w=round(float(boxes[:, 2].mean()), 3), mean_h=round(float(boxes[:, 3].mean()), 3))


# 2. Label-vs-image alignment that does not involve the model: edge of the bright target against the box, train vs val.
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


def centroid(task):
    """Intensity-weighted centroid of a tank box region minus the box centre, using a pad to allow for offsets."""
    path, boxes = task
    v = np.asarray(Image.open(path).convert('L'), np.float32)
    rows = []
    h, w = v.shape
    for x0, y0, x1, y1 in boxes:
        pad = 4
        a0, b0, a1, b1 = max(int(x0) - pad, 0), max(int(y0) - pad, 0), min(int(np.ceil(x1)) + pad, w), min(int(np.ceil(y1)) + pad, h)
        if a1 - a0 < 3 or b1 - b0 < 3:
            continue
        patch = v[b0:b1, a0:a1]
        patch = np.clip(patch - np.percentile(patch, 50), 0, None)
        if patch.sum() <= 0:
            continue
        ys, xs = np.mgrid[b0:b1, a0:a1]
        rows.append(((xs + .5) * patch).sum() / patch.sum() - (x0 + x1) / 2)
        rows[-1] = (rows[-1], ((ys + .5) * patch).sum() / patch.sum() - (y0 + y1) / 2)
    return rows


class Images:
    def __init__(self, rows, folder):
        self.rows, self.folder = rows, folder

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import torchvision.transforms.functional as F
        image = Image.open(self.folder / self.rows[i]['file_name']).convert('RGB')
        return F.to_tensor(F.resize(image, [640, 640])), i, image.size[0], image.size[1]


def iou(a, b):
    x1, y1 = np.maximum(a[:, None, 0], b[:, 0]), np.maximum(a[:, None, 1], b[:, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[:, 2]), np.minimum(a[:, None, 3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = (a[:, 2] - a[:, 0])[:, None] * (a[:, 3] - a[:, 1])[:, None] + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return inter / np.maximum(union, 1e-9)


def main():
    import torch
    from engine.core import YAMLConfig
    train = json.load(open(ROOT / 'Annotations/train.json'))
    val = json.load(open(ROOT / 'Annotations/val.json'))
    names = {c['id']: c['name'] for c in val['categories']}
    ogsod = sorted(glob.glob(str(native / 'images/*/*')))
    msar = sorted(glob.glob(str(BASE / 'MSAR-1.0 dataset/JPEGImages/*')))
    paths = {'train': [str(ROOT / 'JPEGImages/train' / i['file_name']) for i in train['images']], 'val': [str(ROOT / 'JPEGImages/val' / i['file_name']) for i in val['images']]}
    with mp.Pool(32) as pool:
        ref_o = pool.map(thumb, ogsod, chunksize=64)
        ref_m = pool.map(thumb, msar, chunksize=64)
        thumbs = {k: pool.map(thumb, v, chunksize=64) for k, v in paths.items()}
        source = {}
        for split in ('train', 'val'):
            query = torch.as_tensor(np.stack([t[0] for t in thumbs[split]])).cuda()
            labels = np.array(['other'] * len(query), dtype=object)
            for name, refs, ref_paths in (('ogsod', ref_o, ogsod), ('msar', ref_m, msar)):
                R = torch.as_tensor(np.stack([t[0] for t in refs])).cuda()
                bs, bi = [], []
                for s in range(0, len(query), 1024):
                    a, b = (query[s:s + 1024] @ R.T).max(1)
                    bs.append(a.cpu().numpy())
                    bi.append(b.cpu().numpy())
                bs, bi = np.concatenate(bs), np.concatenate(bi)
                cand = [(q, int(bi[q])) for q in np.flatnonzero(bs >= .98) if thumbs[split][q][1] == refs[bi[q]][1]]
                ok = pool.map(same, [(paths[split][q], ref_paths[r]) for q, r in cand], chunksize=64)
                labels[[q for (q, r), g in zip(cand, ok) if g]] = name
                del R
            source[split] = labels
        del query
        torch.cuda.empty_cache()
        by = {k: defaultdict(list) for k in ('train', 'val')}
        for k, d in (('train', train), ('val', val)):
            for a in d['annotations']:
                x, y, w, h = a['bbox']
                by[k][a['image_id']].append((a['category_id'], [x, y, x + w, y + h]))
        # Image-only centroid against labels; no model involved.
        report['label_vs_intensity_centroid'] = {}
        for split, d in (('train', train), ('val', val)):
            for s in ('ogsod', 'msar', 'other'):
                for c, name in names.items():
                    tasks = [(paths[split][i], [b for k, b in by[split][im['id']] if k == c]) for i, im in enumerate(d['images']) if source[split][i] == s and any(k == c for k, _ in by[split][im['id']])]
                    if len(tasks) < 20:
                        continue
                    rows = [r for part in pool.map(centroid, tasks[:6000], chunksize=32) for r in part]
                    if rows:
                        arr = np.array(rows)
                        report['label_vs_intensity_centroid'][f'{split}/{s}/{name}'] = dict(n=len(arr), median_dx=round(float(np.median(arr[:, 0])), 3), median_dy=round(float(np.median(arr[:, 1])), 3), mean_dx=round(float(arr[:, 0].mean()), 3), mean_dy=round(float(arr[:, 1].mean()), 3))

    # Model on train and val, by source.
    cfg = YAMLConfig(str(RUN / 'resolved_config.yml'))
    model, post = cfg.model, cfg.postprocessor
    model.load_state_dict(torch.load(RUN / 'best_stg2.pth', map_location='cpu')['ema']['module'])
    model, post = model.cuda().eval(), post.cuda().eval()
    rng = np.random.default_rng(0)
    selected = []
    for s, limit in (('ogsod', 14664), ('msar', 4000), ('other', 6000)):
        idx = np.flatnonzero(source['train'] == s)
        selected += rng.permutation(idx)[:limit].tolist()
    rows = [train['images'][i] for i in selected]
    src_rows = [source['train'][i] for i in selected]
    loader = torch.utils.data.DataLoader(Images(rows, ROOT / 'JPEGImages/train'), batch_size=64, num_workers=8)
    stats = defaultdict(list)
    with torch.no_grad():
        for batch, order, w, h in loader:
            for i, r in zip(order.tolist(), post(model(batch.cuda()), torch.stack([w, h], 1).cuda())):
                keep = r['scores'] >= .3
                pc, pb = r['labels'][keep].cpu().numpy().astype(int), r['boxes'][keep].cpu().numpy()
                image_id = rows[i]['id']
                for c in {k for k, _ in by['train'][image_id]}:
                    gt = np.array([b for k, b in by['train'][image_id] if k == c], np.float32)
                    mine = pb[pc == c]
                    if not len(mine):
                        continue
                    m = iou(mine, gt)
                    for g in range(len(gt)):
                        d = int(m[:, g].argmax())
                        if m[d, g] >= .3:
                            stats[(src_rows[i], names[c])].append(((mine[d, 0] + mine[d, 2] - gt[g, 0] - gt[g, 2]) / 2, (mine[d, 1] + mine[d, 3] - gt[g, 1] - gt[g, 3]) / 2, float(m[d, g])))
    report['model_offset_on_train'] = {f'{s}/{c}': dict(n=len(v), mean_dx=round(float(np.mean([r[0] for r in v])), 3), mean_dy=round(float(np.mean([r[1] for r in v])), 3), median_dx=round(float(np.median([r[0] for r in v])), 3), median_iou=round(float(np.median([r[2] for r in v])), 3)) for (s, c), v in stats.items() if len(v) >= 30}
    (OUT / 'offset_origin.json').write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str))
    print('RESULT', json.dumps(report, ensure_ascii=False, default=str), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
