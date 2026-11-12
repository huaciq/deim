"""Group OGSOD tank images that share a GT layout, on the server. Same layout + different recall is a natural control."""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
OUT = Path(sys.argv[1])
ROOT = Path('/root/zxy/SARDet_100K/JPEGImages/val')
rows = [r for r in json.load(open(OUT / 'gt_rows.json')) if r['source'] == 'ogsod']
stages = json.load(open(OUT / 'stage_rows.json'))
by = defaultdict(list)
for r in rows:
    by[r['image_id']].append(r)
ids = sorted(by)
boxes = {i: np.array([[r['gt_x1'], r['gt_y1'], r['gt_x2'], r['gt_y2']] for r in by[i]], np.float32) for i in ids}


def match(a, b):
    """Share of boxes in a that have a same-position box in b after the best integer shift (the tiles may be offset)."""
    best = (0, (0, 0))
    ca, cb = (a[:, :2] + a[:, 2:]) / 2, (b[:, :2] + b[:, 2:]) / 2
    shifts = {tuple(np.round(p - q)) for p in ca[:12] for q in cb}
    for s in shifts:
        d = np.abs(ca[:, None] + np.array(s) - cb[None]).max(-1).min(1)
        share = float((d <= 2).mean())
        if share > best[0]:
            best = (share, s)
    return best


parent = {i: i for i in ids}
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x
links = []
for i, a in enumerate(ids):
    for b in ids[i + 1:]:
        if len(boxes[a]) < 5 or len(boxes[b]) < 5 or abs(len(boxes[a]) - len(boxes[b])) > .2 * max(len(boxes[a]), len(boxes[b])):
            continue
        share, shift = match(boxes[a], boxes[b])
        if share >= .8:
            links.append((a, b, round(share, 2), shift))
            parent[find(a)] = find(b)
groups = defaultdict(list)
for i in ids:
    groups[find(i)].append(i)
image_stats = {}
for i in ids:
    v = np.asarray(Image.open(ROOT / by[i][0]['file_name']).convert('L'), np.float32)
    lap = v[1:-1, 1:-1] * 4 - v[:-2, 1:-1] - v[2:, 1:-1] - v[1:-1, :-2] - v[1:-1, 2:]
    image_stats[i] = dict(mean=round(float(v.mean()), 1), std=round(float(v.std()), 1), sat=round(float((v >= 250).mean()), 3), lap=round(float(lap.std()), 2))
by_stage = defaultdict(lambda: defaultdict(list))
for s in stages:
    if s['kind'] == 'gt' and s['image_id'] in by:
        by_stage[s['image_id']][s['stage']].append(s)
report = []
for root, members in groups.items():
    if len(members) < 2:
        continue
    entry = []
    for i in sorted(members, key=lambda i: -np.mean([r['status'] == 'TP' for r in by[i]])):
        entry.append(dict(file=by[i][0]['file_name'], image_id=i, gt=len(by[i]), recall=round(float(np.mean([r['status'] == 'TP' for r in by[i]])), 3), status=dict(Counter(r['status'] for r in by[i])), side=float(np.median([r['input_side'] for r in by[i]])),
                          enc_prob=round(float(np.median([r['enc_prob_max_in_gt'] for r in by[i]])), 3), tp_iou=round(float(np.nanmedian([r['tp_iou'] for r in by[i]])), 3) if any(r['status'] == 'TP' for r in by[i]) else None, **image_stats[i],
                          **{f'z_{k}': round(float(np.nanmedian([s['z'] for s in by_stage[i][k]])), 2) for k in ('input', 'S2', 'P3', 'P4', 'P5', 'F3', 'F4', 'F5')}, **{f'pk_{k}': round(float(np.nanmedian([s['peak_pct'] for s in by_stage[i][k]])), 3) for k in ('input', 'P3', 'F3')}, **{f'cos_{k}': round(float(np.nanmedian([s['cos_dist'] for s in by_stage[i][k]])), 3) for k in ('P3', 'P5', 'F3', 'F5')}))
    report.append(entry)
report.sort(key=lambda e: -sum(x['gt'] for x in e))
json.dump(dict(groups=report, links=[(int(a), int(b), s, [float(v) for v in t]) for a, b, s, t in links]), open(OUT / 'layout_groups.json', 'w'), indent=1)
print('images', len(ids), 'in groups', sum(len(e) for e in report), 'groups', len(report), 'gt in groups', sum(x['gt'] for e in report for x in e), 'miss in groups', sum(round(x['gt'] * (1 - x['recall'])) for e in report for x in e))
for e in report[:8]:
    print('---')
    for x in e:
        print({k: x[k] for k in x if k not in ('image_id',)})
print(links[:12])
# Same physical target, paired: where in the network does the difference between images in a group appear?
pair = defaultdict(list)
for e in report:
    if len(e) < 2:
        continue
    high, low = e[0], e[-1]
    if high['recall'] - low['recall'] < .3:
        continue
    for k in ('input', 'S2', 'P3', 'P4', 'P5', 'F3', 'F4', 'F5'):
        pair[k].append((high['file'], low['file'], high[f'z_{k}'], low[f'z_{k}']))
for k, v in pair.items():
    print(k, 'n', len(v), 'median z high', np.median([x[2] for x in v]), 'low', np.median([x[3] for x in v]), 'high>low', sum(x[2] > x[3] for x in v))
