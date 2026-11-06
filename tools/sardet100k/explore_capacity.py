"""Val-only, no training: more queries or larger input at inference with E000 weights; anchor-cell collisions."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

RUN = Path(sys.argv[1])
ROOT = Path('/root/zxy/TCRA-SAR/datasets/SARDet_100K')
val = json.load(open(ROOT / 'Annotations/val.json'))
names = {c['id']: c['name'] for c in val['categories']}
images = {i['id']: i for i in val['images']}
gts = defaultdict(list)
for a in val['annotations']:
    x, y, w, h = a['bbox']
    gts[a['image_id']].append((a['category_id'], np.array([x, y, x + w, y + h], np.float32), max(w, h)))
THR = np.linspace(.5, .95, 10)
REC = np.linspace(0, 1, 101)


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


class Images(torch.utils.data.Dataset):
    def __init__(self, size):
        self.size, self.rows = size, val['images']

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image = Image.open(ROOT / 'JPEGImages/val' / self.rows[index]['file_name']).convert('RGB')
        w, h = image.size
        return F.to_tensor(F.resize(image, [self.size, self.size])), index, w, h


def infer(size, queries):
    from engine.core import YAMLConfig
    cfg = YAMLConfig(str(RUN / 'resolved_config.yml'), eval_spatial_size=[size, size],
                     DFINETransformer={'num_queries': queries}, PostProcessor={'num_top_queries': queries})
    model, post = cfg.model, cfg.postprocessor
    state = torch.load(RUN / 'best_stg2.pth', map_location='cpu')['ema']['module']
    current = model.state_dict()
    state = {k: v for k, v in state.items() if k in current and current[k].shape == v.shape}
    print('loaded', len(state), 'of', len(current), 'missing', [k for k in current if k not in state][:6], flush=True)
    model.load_state_dict(state, strict=False)
    model, post = model.cuda().eval(), post.cuda().eval()
    loader = torch.utils.data.DataLoader(Images(size), batch_size=32 if size > 640 else 64, num_workers=8)
    rows = defaultdict(list)
    with torch.no_grad():
        for batch, indices, widths, heights in loader:
            outputs = model(batch.cuda())
            for index, result in zip(indices.tolist(), post(outputs, torch.stack([widths, heights], 1).cuda())):
                rows[val['images'][index]['id']] = (result['labels'].cpu().numpy().astype(int), result['scores'].cpu().numpy(), result['boxes'].cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return rows


def evaluate(rows):
    out = {}
    for c, name in names.items():
        lines = {t: ([], []) for t in range(10)}
        n = 0
        for image_id in images:
            cur = [g for g in gts.get(image_id, []) if g[0] == c]
            n += len(cur)
            labels, scores, boxes = rows[image_id]
            keep = np.flatnonzero(labels == c)
            if not len(keep):
                continue
            keep = keep[np.argsort(-scores[keep], kind='mergesort')][:100]
            m = iou(boxes[keep], np.stack([g[1] for g in cur])) if cur else None
            for t, th in enumerate(THR):
                used = set()
                hit = np.zeros(len(keep), bool)
                if cur:
                    for d in range(len(keep)):
                        best, chosen = th, -1
                        for g in range(len(cur)):
                            if g not in used and m[d, g] >= best:
                                best, chosen = m[d, g], g
                        if chosen >= 0:
                            used.add(chosen)
                            hit[d] = True
                lines[t][0].append(scores[keep])
                lines[t][1].append(hit)
        v = [ap_curve(np.concatenate(lines[t][0]), np.concatenate(lines[t][1]), n) for t in range(10)]
        out[name] = dict(AP=round(float(np.mean(v)) * 100, 2), AP50=round(v[0] * 100, 2), AP75=round(v[5] * 100, 2))
    out['mean'] = dict(AP=round(float(np.mean([out[n]['AP'] for n in names.values()])), 2), AP50=round(float(np.mean([out[n]['AP50'] for n in names.values()])), 2), AP75=round(float(np.mean([out[n]['AP75'] for n in names.values()])), 2))
    # Recall groups at IoU .5, any score and score >= .3
    groups = defaultdict(lambda: [0, 0, 0])
    for image_id, cur_all in gts.items():
        labels, scores, boxes = rows[image_id]
        for c, name in names.items():
            cur = [g for g in cur_all if g[0] == c]
            if not cur:
                continue
            keep = np.flatnonzero(labels == c)
            keep = keep[np.argsort(-scores[keep], kind='mergesort')][:100]
            high = keep[scores[keep] >= .3]
            target = np.stack([g[1] for g in cur])
            any_best = iou(boxes[keep], target).max(0) if len(keep) else np.zeros(len(cur))
            high_best = iou(boxes[high], target).max(0) if len(high) else np.zeros(len(cur))
            for g, a, h in zip(cur, any_best, high_best):
                keys = [name]
                if g[2] < 16:
                    keys.append(name + '_side<16')
                if len(cur) > 20:
                    keys.append(name + '_dense>20')
                for key in keys:
                    groups[key][0] += 1
                    groups[key][1] += int(a >= .5)
                    groups[key][2] += int(h >= .5)
    out['recall'] = {k: dict(gt=v[0], any=round(v[1] / v[0], 3), s03=round(v[2] / v[0], 3)) for k, v in sorted(groups.items()) if k.startswith(('tank', 'bridge'))}
    return out


report = {}
# Same stride-8 cell at 640 input: objects that compete for one anchor location.
collisions = Counter()
totals = Counter()
for image_id, rows_gt in gts.items():
    im = images[image_id]
    per = defaultdict(list)
    for c, b, s in rows_gt:
        cx, cy = (b[0] + b[2]) / 2 * 640 / im['width'], (b[1] + b[3]) / 2 * 640 / im['height']
        per[(c, int(cx // 8), int(cy // 8))].append(c)
    for key, members in per.items():
        totals[names[key[0]]] += len(members)
        if len(members) > 1:
            collisions[names[key[0]]] += len(members)
report['same_stride8_cell_share'] = {k: round(collisions[k] / totals[k], 4) for k in totals}
for label, size, queries in [('640_q300', 640, 300), ('640_q600', 640, 600), ('640_q1000', 640, 1000), ('800_q300', 800, 300), ('800_q600', 800, 600)]:
    report[label] = evaluate(infer(size, queries))
    print(label, json.dumps(report[label], ensure_ascii=False), flush=True)
(RUN / 'independent_verify_20261106/explore_tiny/capacity.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
print('RESULT', json.dumps(report, ensure_ascii=False))
