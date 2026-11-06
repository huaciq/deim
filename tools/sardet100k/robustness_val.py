"""Val-only sensitivity of E000 to imaging perturbations. Usage: robustness_val.py RUN {infer|eval}."""
import json
import multiprocessing as mp
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

RUN = Path(sys.argv[1])
STAGE = sys.argv[2]
OUT = RUN / 'independent_verify_20261106/robustness'
ROOT = Path('/root/zxy/TCRA-SAR/datasets/SARDet_100K')
val = json.load(open(ROOT / 'Annotations/val.json'))
names = {c['id']: c['name'] for c in val['categories']}
images = {i['id']: i for i in val['images']}
THR = np.linspace(.5, .95, 10)
REC = np.linspace(0, 1, 101)
CONDITIONS = {
    'clean': None,
    'gain_0.85': ('gain', .85), 'gain_1.15': ('gain', 1.15),
    'gamma_0.7': ('gamma', .7), 'gamma_1.4': ('gamma', 1.4),
    'speckle_L16': ('speckle', 16), 'speckle_L4': ('speckle', 4), 'speckle_L2': ('speckle', 2),
    'blur_s1': ('blur', 1.), 'blur_s2': ('blur', 2.),
    'down_0.5': ('down', .5), 'down_0.25': ('down', .25),
}
BINS = [('all', 0, 1e9), ('lt16', 0, 16), ('16_32', 16, 32), ('32_64', 32, 64), ('ge64', 64, 1e9)]


def perturb(image, condition, seed):
    if condition is None:
        return image
    kind, value = condition
    if kind == 'blur':
        return image.filter(ImageFilter.GaussianBlur(value))
    if kind == 'down':
        w, h = image.size
        small = image.resize((max(1, int(w * value)), max(1, int(h * value))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)
    values = np.asarray(image, np.float32)
    if kind == 'gain':
        values = values * value
    elif kind == 'gamma':
        values = 255 * (values / 255) ** value
    elif kind == 'speckle':
        rng = np.random.default_rng(seed)
        noise = np.sqrt(rng.gamma(value, 1 / value, values.shape[:2]))
        values = values * noise[..., None]
    return Image.fromarray(np.clip(values, 0, 255).astype(np.uint8))


class Images:
    def __init__(self, condition):
        self.condition = condition

    def __len__(self):
        return len(val['images'])

    def __getitem__(self, index):
        import torchvision.transforms.functional as F
        row = val['images'][index]
        image = Image.open(ROOT / 'JPEGImages/val' / row['file_name']).convert('RGB')
        size = image.size
        image = perturb(image, self.condition, index)
        return F.to_tensor(F.resize(image, [640, 640])), index, size[0], size[1]


def infer():
    import torch
    from engine.core import YAMLConfig
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = YAMLConfig(str(RUN / 'resolved_config.yml'))
    model, post = cfg.model, cfg.postprocessor
    model.load_state_dict(torch.load(RUN / 'best_stg2.pth', map_location='cpu')['ema']['module'])
    model, post = model.cuda().eval(), post.cuda().eval()
    for name, condition in CONDITIONS.items():
        loader = torch.utils.data.DataLoader(Images(condition), batch_size=64, num_workers=8)
        rows = []
        with torch.no_grad():
            for batch, indices, widths, heights in loader:
                outputs = model(batch.cuda())
                sizes = torch.stack([widths, heights], 1).cuda()
                for index, result in zip(indices.tolist(), post(outputs, sizes)):
                    scores = result['scores'].cpu().numpy()
                    keep = scores >= .001
                    rows.append((np.full(int(keep.sum()), val['images'][index]['id']), result['labels'].cpu().numpy()[keep], scores[keep], result['boxes'].cpu().numpy()[keep]))
        np.savez(OUT / f'{name}.npz', image_id=np.concatenate([r[0] for r in rows]), label=np.concatenate([r[1] for r in rows]).astype(np.int8),
                 score=np.concatenate([r[2] for r in rows]), box=np.concatenate([r[3] for r in rows]))
        print('inferred', name, flush=True)


gts = defaultdict(list)
for a in val['annotations']:
    x, y, w, h = a['bbox']
    gts[a['image_id']].append(dict(id=a['id'], cls=a['category_id'], box=np.array([x, y, x + w, y + h], np.float32), side=max(w, h)))


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


def load(name):
    data = np.load(OUT / f'{name}.npz')
    pi, pc, ps, pb = data['image_id'], data['label'].astype(int), data['score'], data['box']
    order = np.argsort(pi, kind='stable')
    values, first = np.unique(pi[order], return_index=True)
    return pc, ps, pb, dict(zip(values.tolist(), np.split(order, first[1:])))


def group_task(task):
    name, label, low, high = task
    pc, ps, pb, slices = load(name)
    out = {}
    for c, cname in names.items():
        lines = {t: ([], []) for t in range(10)}
        n = 0
        for image_id in images:
            rows = [g for g in gts.get(image_id, []) if g['cls'] == c]
            ignored = np.array([not (low <= g['side'] < high) for g in rows], bool)
            if len(rows):
                order = np.argsort(ignored, kind='stable')
                rows, ignored = [rows[i] for i in order], ignored[order]
            n += int((~ignored).sum())
            pos = slices.get(image_id, np.array([], int))
            keep = pos[pc[pos] == c]
            if not len(keep):
                continue
            keep = keep[np.argsort(-ps[keep], kind='mergesort')][:100]
            m = iou(pb[keep], np.stack([g['box'] for g in rows])) if rows else None
            for t, th in enumerate(THR):
                used = np.zeros(len(rows), bool)
                hit, drop = np.zeros(len(keep), bool), np.zeros(len(keep), bool)
                if rows:
                    for d in range(len(keep)):
                        if m[d].max() < th:
                            continue
                        best, chosen = th, -1
                        for g in range(len(rows)):
                            if used[g]:
                                continue
                            if chosen > -1 and not ignored[chosen] and ignored[g]:
                                break
                            if m[d, g] >= best:
                                best, chosen = m[d, g], g
                        if chosen > -1:
                            used[chosen] = True
                            hit[d], drop[d] = not ignored[chosen], ignored[chosen]
                lines[t][0].append(ps[keep][~drop])
                lines[t][1].append(hit[~drop])
        if n >= 30:
            v = [ap_curve(np.concatenate(lines[t][0]), np.concatenate(lines[t][1]), n) for t in range(10)]
            out[cname] = dict(gt=n, AP=round(float(np.mean(v)) * 100, 2), AP50=round(v[0] * 100, 2), AP75=round(v[5] * 100, 2))
    return name, label, out


def instance_table(name):
    """For each GT: same-class best IoU, score of the best localized box, and best any-class box."""
    pc, ps, pb, slices = load(name)
    table = {}
    for image_id, rows in gts.items():
        pos = slices.get(image_id, np.array([], int))
        if not len(pos):
            for g in rows:
                table[g['id']] = (0., 0., 0.)
            continue
        boxes = np.stack([g['box'] for g in rows])
        m = iou(pb[pos], boxes)
        for gi, g in enumerate(rows):
            same = pc[pos] == g['cls']
            line = m[:, gi]
            best_same = float(line[same].max()) if same.any() else 0.
            localized = same & (line >= .5)
            score = float(ps[pos][localized].max()) if localized.any() else 0.
            other = float(line[~same].max()) if (~same).any() else 0.
            high = same & (ps[pos] >= .3)
            tp_iou = float(line[high].max()) if high.any() else 0.
            table[g['id']] = (best_same, score, other if other >= .5 else 0., tp_iou)
    return table


def crop(values, box):
    h, w = values.shape
    x0, y0 = min(max(int(box[0]), 0), w - 1), min(max(int(box[1]), 0), h - 1)
    x1, y1 = min(max(int(np.ceil(box[2])), x0 + 1), w), min(max(int(np.ceil(box[3])), y0 + 1), h)
    v = np.asarray(Image.fromarray(values[y0:y1, x0:x1]).resize((24, 24), Image.BILINEAR), np.float32).ravel()
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def encode(task):
    folder, name, boxes = task
    values = np.asarray(Image.open(f'{folder}/{name}').convert('L'), np.float32)
    return np.stack([crop(values, b) for b in boxes]) if boxes else np.zeros((0, 576), np.float32)


def seen_ids(pool):
    train = json.load(open(ROOT / 'Annotations/train.json'))
    by = defaultdict(list)
    for a in train['annotations']:
        by[a['image_id']].append(a)
    corner = lambda a: [a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]]
    reference = np.concatenate(pool.map(encode, [(str(ROOT / 'JPEGImages/train'), i['file_name'], [corner(a) for a in by[i['id']]]) for i in train['images']], chunksize=64))
    sizes_r = np.array([max(a['bbox'][2], a['bbox'][3]) for i in train['images'] for a in by[i['id']]], np.float32)
    query = np.concatenate(pool.map(encode, [(str(ROOT / 'JPEGImages/val'), i['file_name'], [g['box'] for g in gts.get(i['id'], [])]) for i in val['images']], chunksize=64))
    ids = [g['id'] for i in val['images'] for g in gts.get(i['id'], [])]
    sizes_q = np.array([g['side'] for i in val['images'] for g in gts.get(i['id'], [])], np.float32)
    import torch
    R, S = torch.as_tensor(reference).cuda(), torch.as_tensor(sizes_r).cuda()
    best = []
    for start in range(0, len(query), 1024):
        q = torch.as_tensor(query[start:start + 1024]).cuda()
        scores = q @ R.T
        ratio = torch.as_tensor(sizes_q[start:start + 1024]).cuda()[:, None] / S[None, :].clamp(min=1e-3)
        scores[(ratio - 1).abs() > .1] = -1
        best.append(scores.max(1).values.cpu().numpy())
    best = np.concatenate(best)
    return {i for i, b in zip(ids, best) if b >= .9}


def evaluate():
    report = dict(conditions=list(CONDITIONS), groups={})
    tasks = [(name, label, low, high) for name in CONDITIONS for label, low, high in BINS]
    with mp.Pool(16) as pool:
        for name, label, out in pool.imap_unordered(group_task, tasks):
            report['groups'].setdefault(name, {})[label] = out
            print('group', name, label, flush=True)
        tables = dict(zip(CONDITIONS, pool.map(instance_table, list(CONDITIONS))))
        seen = seen_ids(pool)
    clean = tables['clean']
    meta = {g['id']: g for rows in gts.values() for g in rows}
    report['seen_share'] = round(len(seen) / len(meta), 4)
    report['stability'] = {}
    for name in CONDITIONS:
        if name == 'clean':
            continue
        current = tables[name]
        entry = {}
        groups = {'all': lambda g: True, 'seen': lambda g: g['id'] in seen, 'unseen': lambda g: g['id'] not in seen,
                  'lt16': lambda g: g['side'] < 16, 'ge16': lambda g: g['side'] >= 16}
        groups.update({n: (lambda g, c=c: g['cls'] == c) for c, n in names.items()})
        for label, test in groups.items():
            ids = [i for i, g in meta.items() if test(g) and clean[i][1] >= .3]
            if not ids:
                continue
            kept = np.array([current[i][1] >= .3 for i in ids])
            has_box = np.array([current[i][0] >= .5 for i in ids])
            flipped = np.array([current[i][2] >= .5 and current[i][0] < .5 for i in ids])
            score_ratio = np.array([current[i][1] / clean[i][1] for i in ids])
            both = [i for i in ids if current[i][3] >= .5 and clean[i][3] >= .5]
            d_iou = np.array([current[i][3] - clean[i][3] for i in both]) if both else np.zeros(1)
            entry[label] = dict(n=len(ids), retained_0_3=round(float(kept.mean()), 4),
                                lost_score_box_ok=round(float((~kept & has_box).mean()), 4),
                                lost_no_box=round(float((~has_box).mean()), 4),
                                class_flip=round(float(flipped.mean()), 4),
                                median_score_ratio=round(float(np.median(score_ratio)), 3),
                                median_delta_iou=round(float(np.median(d_iou)), 4),
                                p10_delta_iou=round(float(np.percentile(d_iou, 10)), 4))
        report['stability'][name] = entry
    (OUT / 'report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print('DONE', flush=True)


if __name__ == '__main__':
    infer() if STAGE == 'infer' else evaluate()
