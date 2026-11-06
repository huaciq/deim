"""Val-only, CPU only, E000 frozen: do features sampled at the physical position of each FDR bin improve the box?

FDR gives every box edge 33 bins, each with a position along the edge normal, yet computes their logits from one query vector.
Heads add a residual to the final logits. Two folds over near-duplicate groups; every number comes from the held-out fold.

  query        query and current distribution only (control: also picks up constant bias)
  roi          query plus features pooled over the box (control: spatial features, not aligned to bins)
  aligned      query plus S2 and P3 features sampled at the position of each bin
  aligned_raw  aligned plus log grey level at the position of each bin

Criteria written before the run: aligned minus query >= +0.5 AP outside aircraft goes to full training; <= 0 drops the idea.

Usage: edge_probe.py RUN {pca | extract SHARD NSHARDS | fit}
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

RUN = Path(sys.argv[1])
STAGE = sys.argv[2]
OUT = RUN / 'independent_verify_20261106/edge_probe'
OUT.mkdir(parents=True, exist_ok=True)
ROOT = Path('/root/zxy/TCRA-SAR/datasets/SARDet_100K')
val = json.load(open(ROOT / 'Annotations/val.json'))
names = {c['id']: c['name'] for c in val['categories']}
images = {i['id']: i for i in val['images']}
TANGENT = (-.3, 0., .3)
MIN_STORE = .01
MIN_MATCH = .4


def load_model():
    import torch
    from engine.core import YAMLConfig
    from engine.deim.dfine_utils import weighting_function
    cfg = YAMLConfig(str(RUN / 'resolved_config.yml'))
    model = cfg.model
    model.load_state_dict(torch.load(RUN / 'best_stg2.pth', map_location='cpu')['ema']['module'], strict=False)
    model.eval()
    captured = {}
    model.backbone.stages[0].register_forward_hook(lambda m, i, o: captured.update(s2=o))
    model.encoder.register_forward_hook(lambda m, i, o: captured.update(p3=o[0] if torch.is_tensor(o[0]) else o[0][0]))
    model.decoder.dec_bbox_head[-1].register_forward_pre_hook(lambda m, i: captured.update(query=i[0]))
    model.decoder.decoder.register_forward_hook(lambda m, i, o: captured.update(box=o[0][-1], logit=o[1][-1], corners=o[2][-1], ref=o[3][-1]))
    bins = weighting_function(model.decoder.reg_max, model.decoder.up, model.decoder.reg_scale)
    return model, captured, bins.detach(), float(abs(model.decoder.reg_scale))


def read_image(row):
    import torchvision.transforms.functional as F
    image = Image.open(ROOT / 'JPEGImages/val' / row['file_name']).convert('RGB')
    return F.to_tensor(F.resize(image, [640, 640])).unsqueeze(0)


def stage_pca():
    import torch
    torch.set_num_threads(12)
    model, captured, _, _ = load_model()
    rows = np.random.default_rng(0).permutation(len(val['images']))[:150]
    pixels = {'s2': [], 'p3': []}
    with torch.no_grad():
        for k in rows:
            model(read_image(val['images'][k]))
            for key in pixels:
                f = captured[key][0].flatten(1).T
                pixels[key].append(f[torch.randperm(len(f))[:800]])
    result = {}
    for key, values in pixels.items():
        v = torch.cat(values).double()
        mean, std = v.mean(0), v.std(0).clamp(min=1e-6)
        _, vectors = torch.linalg.eigh(torch.cov(((v - mean) / std).T))
        result[key] = dict(mean=mean.float(), std=std.float(), vectors=vectors[:, -48:].float())
        print(key, v.shape, flush=True)
    torch.save(result, OUT / 'pca.pth')


def sample(feature, points):
    """feature 1,C,H,W; points ...,2 normalized; returns ...,C."""
    import torch.nn.functional as F
    shape = points.shape[:-1]
    grid = 2 * points.reshape(1, -1, 1, 2) - 1
    return F.grid_sample(feature, grid, mode='bilinear', padding_mode='zeros', align_corners=False)[0, :, :, 0].T.reshape(*shape, feature.shape[1])


def stage_extract(shard, total):
    import torch
    torch.set_num_threads(4)
    model, captured, bins, scale = load_model()
    pca = torch.load(OUT / 'pca.pth')

    def project(feature, key):
        p = pca[key]
        f = feature[0].permute(1, 2, 0)
        return (((f - p['mean']) / p['std']) @ p['vectors']).permute(2, 0, 1)[None]

    rows = val['images'][shard::total]
    keep = defaultdict(list)
    with torch.no_grad():
        for count, row in enumerate(rows):
            tensor = read_image(row)
            model(tensor)
            prob = captured['logit'][0].sigmoid()
            score, label = prob.max(-1)
            pick = score >= MIN_STORE
            if not pick.any():
                continue
            ref = captured['ref'][0][pick]
            s2, p3 = project(captured['s2'], 's2'), project(captured['p3'], 'p3')
            grey = torch.log1p(tensor.mean(1, keepdim=True) * 255) / np.log(256)
            cx, cy, w, h = ref.unbind(-1)
            reach = (scale / 2 + bins)[None] / scale  # n,33 in box units.
            t = torch.tensor(TANGENT)
            n = len(ref)
            # Bin positions of left, top, right and bottom edges: n,4,33,3,2. Bin order is always outward.
            along = torch.stack([cx[:, None] - reach * w[:, None], cy[:, None] - reach * h[:, None], cx[:, None] + reach * w[:, None], cy[:, None] + reach * h[:, None]], 1)
            points = torch.zeros(n, 4, 33, 3, 2)
            for e in range(4):
                normal, tangent = (0, 1) if e in (0, 2) else (1, 0)
                center, extent = (cy, h) if e in (0, 2) else (cx, w)
                points[:, e, :, :, normal] = along[:, e, :, None]
                points[:, e, :, :, tangent] = (center[:, None] + t[None] * extent[:, None])[:, None]
            profile = torch.cat([sample(s2, points), sample(p3, points), sample(grey, points)], -1).mean(3)
            u = torch.linspace(-.4, .4, 5)
            box = captured['box'][0][pick]
            gy, gx = torch.meshgrid(u, u, indexing='ij')
            inside = torch.stack([box[:, 0, None, None] + gx * box[:, 2, None, None], box[:, 1, None, None] + gy * box[:, 3, None, None]], -1)
            roi = torch.cat([sample(s2, inside), sample(p3, inside)], -1).mean((1, 2))
            keep['image_id'].append(np.full(n, row['id']))
            keep['label'].append(label[pick].numpy() + 1 if min(names) == 1 else label[pick].numpy())
            keep['score'].append(score[pick].numpy())
            keep['ref'].append(ref.numpy())
            keep['corners'].append(captured['corners'][0][pick].numpy())
            keep['query'].append(captured['query'][0][pick].numpy().astype(np.float16))
            keep['roi'].append(roi.numpy().astype(np.float16))
            whole = torch.cat([s2.mean((2, 3)), p3.mean((2, 3))], 1).expand(n, -1)
            keep['whole'].append(whole.numpy().astype(np.float16))
            keep['profile'].append(profile.numpy().astype(np.float16))
            if count % 200 == 0:
                print(shard, count, len(rows), flush=True)
    np.savez(OUT / f'shard_{shard}.npz', bins=bins.numpy(), scale=scale, **{k: np.concatenate(v) for k, v in keep.items()})
    print('done', shard, flush=True)


def iou(a, b):
    x1, y1 = np.maximum(a[:, None, 0], b[:, 0]), np.maximum(a[:, None, 1], b[:, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[:, 2]), np.minimum(a[:, None, 3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = (a[:, 2] - a[:, 0])[:, None] * (a[:, 3] - a[:, 1])[:, None] + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return inter / np.maximum(union, 1e-9)


THR = np.linspace(.5, .95, 10)
REC = np.linspace(0, 1, 101)


def ap_curve(score, hit, n):
    order = np.argsort(-score, kind='mergesort')
    hit = hit[order]
    tp, fp = np.cumsum(hit), np.cumsum(~hit)
    recall = tp / n
    precision = tp / np.maximum(tp + fp, 1e-12)
    precision = np.maximum.accumulate(precision[::-1])[::-1] if len(precision) else precision
    where = np.searchsorted(recall, REC, side='left')
    out = np.zeros(101)
    ok = where < len(precision)
    out[ok] = precision[where[ok]]
    return out.mean()


def evaluate(pi, pc, ps, pb, gts):
    order = np.argsort(pi, kind='stable')
    values, first = np.unique(pi[order], return_index=True)
    where = dict(zip(values.tolist(), np.split(order, first[1:])))
    per_class, curves = {}, {}
    for c, name in names.items():
        lines = {t: ([], []) for t in range(10)}
        n = 0
        for image_id in images:
            cur = gts.get((image_id, c))
            n += 0 if cur is None else len(cur)
            pos = where.get(image_id)
            if pos is None:
                continue
            mine = pos[pc[pos] == c]
            if not len(mine):
                continue
            mine = mine[np.argsort(-ps[mine], kind='mergesort')][:100]
            m = iou(pb[mine], cur) if cur is not None else None
            for t, th in enumerate(THR):
                used = set()
                hit = np.zeros(len(mine), bool)
                if cur is not None:
                    for d in range(len(mine)):
                        if m[d].max() < th:
                            continue
                        best, chosen = th, -1
                        for g in range(len(cur)):
                            if g not in used and m[d, g] >= best:
                                best, chosen = m[d, g], g
                        if chosen >= 0:
                            used.add(chosen)
                            hit[d] = True
                lines[t][0].append(ps[mine])
                lines[t][1].append(hit)
        curve = [ap_curve(np.concatenate(lines[t][0]), np.concatenate(lines[t][1]), n) for t in range(10)]
        curves[name] = curve
        per_class[name] = round(float(np.mean(curve)) * 100, 2)
    m = np.array(list(curves.values())).mean(0) * 100
    rest = np.array([v for k, v in curves.items() if k != 'aircraft']).mean() * 100
    return dict(AP=round(float(m.mean()), 2), AP50=round(float(m[0]), 2), AP75=round(float(m[5]), 2), AP90=round(float(m[8]), 2), AP_no_aircraft=round(float(rest), 2), classes=per_class)


def thumb(row):
    image = Image.open(ROOT / 'JPEGImages/val' / row['file_name']).convert('L')
    t = np.asarray(image.resize((32, 32), Image.BILINEAR), np.float32).ravel()
    t = t - t.mean()
    norm = np.linalg.norm(t)
    return t / norm if norm > 0 else t


def make_folds():
    import multiprocessing as mp
    import torch
    with mp.Pool(16) as pool:
        thumbs = pool.map(thumb, val['images'], chunksize=64)
    T = torch.as_tensor(np.stack(thumbs))
    parent = list(range(len(T)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for start in range(0, len(T), 1024):
        s, i = (T[start:start + 1024] @ T.T).topk(50, dim=1)
        for r in range(len(s)):
            for value, b in zip(s[r].tolist(), i[r].tolist()):
                if value >= float(os.environ.get('PROBE_GROUP_CORR', .9)):
                    parent[find(start + r)] = find(b)
    groups = np.array([find(i) for i in range(len(T))])
    unique, sizes = np.unique(groups, return_counts=True)
    print('folds', len(unique), 'groups, largest', int(sizes.max()), flush=True)
    assign = dict(zip(unique.tolist(), np.random.default_rng(0).integers(0, 2, len(unique)).tolist()))
    return {row['id']: assign[int(g)] for row, g in zip(val['images'], groups)}


def stage_fit():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.set_num_threads(20)
    shards = [np.load(p) for p in sorted(OUT.glob('shard_*.npz'))]
    bins = torch.as_tensor(shards[0]['bins'])
    scale = float(shards[0]['scale'])
    data = {k: np.concatenate([s[k] for s in shards]) for k in ('image_id', 'label', 'score', 'ref', 'corners', 'query', 'roi', 'whole', 'profile')}
    print('detections', len(data['score']), flush=True)
    PI, PC, PS = data['image_id'], data['label'].astype(int), data['score']
    size = np.array([[images[i]['width'], images[i]['height']] for i in PI], np.float32)
    ref = torch.as_tensor(data['ref'])
    corners = torch.as_tensor(data['corners']).reshape(-1, 4, 33)
    query = torch.as_tensor(data['query'].astype(np.float32))
    roi = torch.as_tensor(data['roi'].astype(np.float32))
    profile = torch.as_tensor(data['profile'].astype(np.float32)).nan_to_num()
    query = (query - query.mean(0)) / query.std(0).clamp(min=1e-6)
    roi = (roi - roi.mean(0)) / roi.std(0).clamp(min=1e-6)
    whole = torch.as_tensor(data['whole'].astype(np.float32))
    whole = (whole - whole.mean(0)) / whole.std(0).clamp(min=1e-6)
    center = profile.mean((0, 1, 2))
    spread = profile.std((0, 1, 2)).clamp(min=1e-6)
    profile = (profile - center) / spread

    def decode(logits):
        d = (logits.softmax(-1) * bins).sum(-1)
        cx, cy, w, h = ref.unbind(-1)
        return torch.stack([cx - (scale / 2 + d[:, 0]) * w / scale, cy - (scale / 2 + d[:, 1]) * h / scale, cx + (scale / 2 + d[:, 2]) * w / scale, cy + (scale / 2 + d[:, 3]) * h / scale], -1)

    def pixels(normalized):
        return normalized.numpy() * np.concatenate([size, size], 1)

    gts = defaultdict(list)
    for a in val['annotations']:
        x, y, w, h = a['bbox']
        gts[(a['image_id'], a['category_id'])].append([x, y, x + w, y + h])
    gts = {k: np.array(v, np.float32) for k, v in gts.items()}
    base_boxes = pixels(decode(corners))
    # Best GT per detection.
    target = np.zeros_like(base_boxes)
    best = np.zeros(len(PI), np.float32)
    order = np.argsort(PI, kind='stable')
    values, first = np.unique(PI[order], return_index=True)
    where = dict(zip(values.tolist(), np.split(order, first[1:])))
    for (image_id, c), cur in gts.items():
        pos = where.get(image_id)
        if pos is None:
            continue
        mine = pos[PC[pos] == c]
        if len(mine):
            m = iou(base_boxes[mine], cur)
            target[mine] = cur[m.argmax(1)]
            best[mine] = m.max(1)
    matched = torch.as_tensor(best >= MIN_MATCH)
    normalized = torch.as_tensor(target / np.concatenate([size, size], 1))
    cx, cy, w, h = ref.unbind(-1)
    distance = torch.stack([(cx - normalized[:, 0]) / (w / scale), (cy - normalized[:, 1]) / (h / scale), (normalized[:, 2] - cx) / (w / scale), (normalized[:, 3] - cy) / (h / scale)], -1) - scale / 2
    distance = distance.clamp(float(bins[0]), float(bins[-1]))
    left = (torch.bucketize(distance, bins, right=True) - 1).clamp(0, 31)
    weight_right = ((distance - bins[left]) / (bins[left + 1] - bins[left])).clamp(0, 1)

    class QueryHead(nn.Module):
        def __init__(self, extra):
            super().__init__()
            self.edge = nn.Embedding(4, 32)
            self.net = nn.Sequential(nn.Linear(256 + extra + 32 + 33, 384), nn.SiLU(), nn.Linear(384, 384), nn.SiLU(), nn.Linear(384, 33))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        def forward(self, q, r, p, c):
            n = len(q)
            parts = [q[:, None].expand(n, 4, -1)]
            if r is not None:
                parts.append(r[:, None].expand(n, 4, -1))
            parts += [self.edge.weight[None].expand(n, -1, -1), c.softmax(-1)]
            return self.net(torch.cat(parts, -1))

    class AlignedHead(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.edge = nn.Embedding(4, 32)
            self.context = nn.Sequential(nn.Linear(256 + 32, 128), nn.SiLU(), nn.Linear(128, 128))
            self.enter = nn.Linear(channels + 3, 128)
            self.convs = nn.ModuleList(nn.Sequential(nn.Conv1d(128, 128, 5, padding=2 * d, dilation=d), nn.GroupNorm(8, 128), nn.SiLU(), nn.Conv1d(128, 128, 1)) for d in (1, 2, 4))
            self.out = nn.Linear(128, 1)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

        def forward(self, q, r, p, c):
            n = len(q)
            context = self.context(torch.cat([q[:, None].expand(n, 4, -1), self.edge.weight[None].expand(n, -1, -1)], -1))
            position = (bins / 4)[None, None, :, None].expand(n, 4, 33, 1)
            token = self.enter(torch.cat([p, c.softmax(-1)[..., None], (c / 10)[..., None], position], -1)) + context[:, :, None]
            x = token.reshape(n * 4, 33, 128).transpose(1, 2)
            for conv in self.convs:
                x = x + conv(x)
            return self.out(x.transpose(1, 2)).reshape(n, 4, 33)

    fold_of = make_folds()
    fold = np.array([fold_of[int(i)] for i in PI])
    report = dict(detections=len(PI), matched=int(matched.sum()))
    report['base'] = evaluate(PI, PC, PS, base_boxes, gts)
    print('base', report['base'], flush=True)
    eval_mask = (PS >= .05) & (best >= .5)
    raw_edge = np.abs(base_boxes - target)[eval_mask]
    long_side = np.maximum(target[:, 2] - target[:, 0], target[:, 3] - target[:, 1])[eval_mask]
    bands = {'lt16': (0, 16), '16_64': (16, 64), 'ge64': (64, 1e9)}
    report['edge_base'] = {k: round(float(np.median(raw_edge[(long_side >= a) & (long_side < b)])), 3) for k, (a, b) in bands.items()}

    # Aircraft images with a GT reaching past the image border, against all other aircraft images.
    aircraft = next(c for c, name in names.items() if name == 'aircraft')
    overflow, inside = set(), set()
    for (image_id, c), cur in gts.items():
        if c == aircraft:
            row = images[image_id]
            over = (cur[:, 0] < -1).any() or (cur[:, 1] < -1).any() or (cur[:, 2] > row['width'] + 1).any() or (cur[:, 3] > row['height'] + 1).any()
            (overflow if over else inside).add(image_id)

    def aircraft_split(boxes):
        out = {}
        for key, ids in (('overflow_images', overflow), ('other_images', inside)):
            subset = {k: v for k, v in gts.items() if k[0] in ids and k[1] == aircraft}
            lines = []
            n = sum(len(v) for v in subset.values())
            for th in THR:
                scores, hits = [], []
                for image_id in ids:
                    pos = where.get(image_id, np.array([], int))
                    pos = pos[PC[pos] == aircraft]
                    if not len(pos):
                        continue
                    pos = pos[np.argsort(-PS[pos], kind='mergesort')][:100]
                    cur = subset[(image_id, aircraft)]
                    m = iou(boxes[pos], cur)
                    used, hit = set(), np.zeros(len(pos), bool)
                    for d in range(len(pos)):
                        chosen, value = -1, th
                        for g in range(len(cur)):
                            if g not in used and m[d, g] >= value:
                                value, chosen = m[d, g], g
                        if chosen >= 0:
                            used.add(chosen)
                            hit[d] = True
                    scores.append(PS[pos])
                    hits.append(hit)
                lines.append(ap_curve(np.concatenate(scores), np.concatenate(hits), n))
            out[key] = dict(gt=n, images=len(ids), AP=round(float(np.mean(lines)) * 100, 2), AP50=round(lines[0] * 100, 2), AP75=round(lines[5] * 100, 2), AP90=round(lines[8] * 100, 2))
        return out

    saved_boxes = {}
    report['aircraft_base'] = aircraft_split(base_boxes)
    print('aircraft base', report['aircraft_base'], flush=True)

    for variant in os.environ.get('PROBE_VARIANTS', 'query,roi,aligned,aligned_raw').split(','):
        delta = torch.zeros_like(corners)
        slices = {'aligned_raw': (0, 97), 'aligned_s2': (0, 48), 'aligned_p3': (48, 96)}.get(variant, (0, 96))
        channels = slices[1] - slices[0]
        classwise = os.environ.get('PROBE_CLASSWISE') == '1'
        wanted = [c for c, name in names.items() if name in os.environ.get('PROBE_CLASSES', ','.join(names.values())).split(',')]
        for test_fold, class_id in [(f, c) for f in (0, 1) for c in (wanted if classwise else [None])]:
            torch.manual_seed(0)
            model = {'query': lambda: QueryHead(0), 'roi': lambda: QueryHead(96), 'whole': lambda: QueryHead(96)}.get(variant, lambda: AlignedHead(channels))()
            chosen = PC == class_id if classwise else np.ones(len(PC), bool)
            train = torch.as_tensor(np.flatnonzero((fold != test_fold) & matched.numpy() & chosen))
            test = torch.as_tensor(np.flatnonzero((fold == test_fold) & chosen))
            epochs, batch = (24, 256) if classwise else (12, 512)
            optimizer = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
            schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-3, total_steps=epochs * (len(train) // batch))

            def run(b):
                r = {'roi': roi, 'whole': whole}[variant][b] if variant in ('roi', 'whole') else None
                p = profile[b, :, :, slices[0]:slices[1]] if variant.startswith('aligned') else None
                return corners[b] + model(query[b], r, p, corners[b])

            for epoch in range(epochs):
                model.train()
                perm = train[torch.randperm(len(train))]
                total = 0.
                for s in range(0, len(perm) - batch + 1, batch):
                    b = perm[s:s + batch]
                    logits = run(b)
                    log_p = F.log_softmax(logits, -1)
                    l, wr = left[b], weight_right[b]
                    ce = -(log_p.gather(-1, l[..., None])[..., 0] * (1 - wr) + log_p.gather(-1, l[..., None] + 1)[..., 0] * wr).mean()
                    expected = (logits.softmax(-1) * bins).sum(-1)
                    reg = F.smooth_l1_loss(expected / scale, distance[b] / scale, beta=.05)
                    loss = ce + 5 * reg
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    schedule.step()
                    total += float(loss)
                if epoch == epochs - 1:
                    print(variant, test_fold, class_id, round(total / max(len(perm) // batch, 1), 4), flush=True)
            model.eval()
            with torch.no_grad():
                for s in range(0, len(test), 2048):
                    b = test[s:s + 2048]
                    delta[b] = run(b) - corners[b]
        boxes = pixels(decode(corners + delta))
        saved_boxes[variant] = boxes
        np.savez(OUT / 'boxes.npz', image_id=PI, label=PC, score=PS, base=base_boxes, **saved_boxes)
        entry = evaluate(PI, PC, PS, boxes, gts)
        edge = np.abs(boxes - target)[eval_mask]
        entry['edge'] = {k: round(float(np.median(edge[(long_side >= a) & (long_side < b)])), 3) for k, (a, b) in bands.items()}
        entry['aircraft'] = aircraft_split(boxes)
        is_aircraft = (PC[eval_mask] == aircraft)
        entry['edge_aircraft'] = {k: round(float(np.median(edge[is_aircraft & (long_side >= a) & (long_side < b)])), 3) for k, (a, b) in bands.items()}
        entry['edge_aircraft_base'] = {k: round(float(np.median(raw_edge[is_aircraft & (long_side >= a) & (long_side < b)])), 3) for k, (a, b) in bands.items()}
        report[variant] = entry
        print(variant, entry, flush=True)
        (OUT / os.environ.get('PROBE_REPORT', 'report.json')).write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print('DONE', flush=True)


def stage_check():
    """Is the aircraft correction real? Independent RSAR labels, error modes and a gallery of the largest corrections.

    Criteria written before the run:
      1. Mean IoU gain against RSAR >= half of the gain against official boxes, with a 95% bootstrap interval above 0.
         Void when most RSAR boxes equal the official ones.
      2. The head that only sees whole-image features gains less than half of the aligned head (checked from report_check.json).
    """
    import tarfile
    from PIL import ImageDraw
    saved = np.load(OUT / 'boxes.npz')
    PI, PC, PS, base, aligned = saved['image_id'], saved['label'], saved['score'], saved['base'], saved['aligned']
    aircraft = next(c for c, name in names.items() if name == 'aircraft')
    keep = (PC == aircraft) & (PS >= .3)
    PI, PS, base, aligned = PI[keep], PS[keep], base[keep], aligned[keep]
    official = defaultdict(list)
    for a in val['annotations']:
        if a['category_id'] == aircraft:
            x, y, w, h = a['bbox']
            official[a['image_id']].append([x, y, x + w, y + h])
    official = {k: np.array(v, np.float32) for k, v in official.items()}
    want = {Path(i['file_name']).stem: i['id'] for i in val['images']}
    rsar = defaultdict(list)
    seen = set()
    with tarfile.open('/root/zxy/TCRA-SAR/datasets/RSAR.tar', 'r:') as archive:
        for member in archive:
            if not member.name.startswith('val/annfiles/') or not member.name.endswith('.txt'):
                continue
            stem = Path(member.name).stem
            if stem not in want:
                continue
            seen.add(want[stem])
            for line in archive.extractfile(member).read().decode().splitlines():
                p = line.split()
                if len(p) >= 9 and p[8] == 'aircraft':
                    c = np.array(p[:8], float)
                    rsar[want[stem]].append([c[0::2].min(), c[1::2].min(), c[0::2].max(), c[1::2].max()])
    rsar = {k: np.array(v, np.float32) for k, v in rsar.items()}
    rows = []
    for image_id in np.unique(PI):
        gt = official.get(int(image_id))
        if gt is None:
            continue
        mine = np.flatnonzero(PI == image_id)
        m = iou(base[mine], gt)
        for d, k in enumerate(mine):
            g = int(m[d].argmax())
            if m[d, g] < .5:
                continue
            row = dict(k=int(k), image=int(image_id), official=gt[g])
            other = rsar.get(int(image_id))
            if other is not None:
                link = iou(gt[g][None], other)[0]
                if link.max() >= .3:
                    row['rsar'] = other[int(link.argmax())]
            rows.append(row)
    K = np.array([r['k'] for r in rows])
    G = np.stack([r['official'] for r in rows])
    B, A = base[K], aligned[K]

    def paired(x, y):
        return iou(x[:, None].reshape(-1, 4), y)[np.arange(len(x)), np.arange(len(x))]

    base_iou, new_iou = paired(B, G), paired(A, G)
    size = np.stack([G[:, 2] - G[:, 0], G[:, 3] - G[:, 1], G[:, 2] - G[:, 0], G[:, 3] - G[:, 1]], 1).clip(1)
    outward = np.array([-1, -1, 1, 1])
    report = dict(matched=len(rows), official=dict(base=round(float(base_iou.mean()), 4), aligned=round(float(new_iou.mean()), 4)))
    report['modes'] = dict(
        share_below_75_fixed=round(float(((base_iou < .75) & (new_iou >= .75)).sum() / max((base_iou < .75).sum(), 1)), 3),
        share_above_75_broken=round(float(((base_iou >= .75) & (new_iou < .75)).sum() / max((base_iou >= .75).sum(), 1)), 3),
        base_outward_error_by_edge=np.round(np.median((B - G) * outward / size, 0), 4).tolist(),
        aligned_outward_error_by_edge=np.round(np.median((A - G) * outward / size, 0), 4).tolist(),
        mean_outward_move=np.round(np.mean((A - B) * outward / size, 0), 4).tolist(),
        abs_move_median=round(float(np.median(np.abs(A - B) / size)), 4),
    )
    side = np.maximum(G[:, 2] - G[:, 0], G[:, 3] - G[:, 1])
    report['by_size'] = {key: dict(n=int(((side >= a) & (side < b)).sum()), base=round(float(base_iou[(side >= a) & (side < b)].mean()), 4), aligned=round(float(new_iou[(side >= a) & (side < b)].mean()), 4)) for key, (a, b) in dict(lt16=(0, 16), s16_64=(16, 64), s64_128=(64, 128), ge128=(128, 1e9)).items() if ((side >= a) & (side < b)).any()}
    has = np.array(['rsar' in r for r in rows])
    report['rsar_images_seen'] = len(seen)
    report['rsar_matched'] = int(has.sum())
    if has.any():
        R = np.stack([r['rsar'] for r, h in zip(rows, has) if h])
        link = paired(G[has], R)
        same = float((np.abs(G[has] - R).max(1) <= .5).mean())
        gain_official = paired(A[has], G[has]) - paired(B[has], G[has])
        gain_rsar = paired(A[has], R) - paired(B[has], R)
        images_of = np.array([r['image'] for r, h in zip(rows, has) if h])
        rng = np.random.default_rng(0)
        unique = np.unique(images_of)
        where = {u: np.flatnonzero(images_of == u) for u in unique}
        draws = [np.concatenate([where[u] for u in rng.choice(unique, len(unique))]) for _ in range(1000)]
        report['rsar'] = dict(
            images=len(unique), same_box_share=round(same, 3), official_rsar_iou_median=round(float(np.median(link)), 3),
            base_iou=round(float(paired(B[has], R).mean()), 4), aligned_iou=round(float(paired(A[has], R).mean()), 4),
            gain_official=round(float(gain_official.mean()), 4), gain_rsar=round(float(gain_rsar.mean()), 4),
            gain_rsar_ci95=np.round(np.percentile([gain_rsar[d].mean() for d in draws], [2.5, 97.5]), 4).tolist(),
            gain_official_ci95=np.round(np.percentile([gain_official[d].mean() for d in draws], [2.5, 97.5]), 4).tolist(),
        )
    (OUT / 'check.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, ensure_ascii=False), flush=True)

    # Gallery: official green, base red, aligned cyan.
    def tile(i):
        row = rows[i]
        image = Image.open(ROOT / 'JPEGImages/val' / images[row['image']]['file_name']).convert('RGB')
        g = G[i]
        pad = max(g[2] - g[0], g[3] - g[1]) * .5 + 6
        x0, y0 = g[0] - pad, g[1] - pad
        extent = max(g[2] - g[0], g[3] - g[1]) + 2 * pad
        scale = 300 / extent
        crop = image.crop((int(x0), int(y0), int(x0 + extent), int(y0 + extent))).resize((300, 300), Image.BILINEAR)
        draw = ImageDraw.Draw(crop)
        for box, colour in ((g, (0, 255, 0)), (B[i], (255, 60, 60)), (A[i], (0, 220, 255))):
            draw.rectangle([(box[0] - int(x0)) * scale, (box[1] - int(y0)) * scale, (box[2] - int(x0)) * scale, (box[3] - int(y0)) * scale], outline=colour, width=2)
        draw.text((4, 4), f'{base_iou[i]:.2f}->{new_iou[i]:.2f}', fill=(255, 255, 0))
        return crop

    def sheet(order, name):
        canvas = Image.new('RGB', (6 * 304, 3 * 304))
        for n, i in enumerate(order[:18]):
            canvas.paste(tile(int(i)), (n % 6 * 304, n // 6 * 304))
        canvas.save(OUT / name, quality=90)

    gain = new_iou - base_iou
    sheet(np.argsort(-gain), 'aircraft_most_improved.jpg')
    sheet(np.argsort(gain), 'aircraft_most_harmed.jpg')
    sheet(np.random.default_rng(1).permutation(np.flatnonzero(gain > .05)), 'aircraft_random_improved.jpg')


def stage_calibrate():
    """Cross-fitted score calibration per (class, image format and size). Scores only change order across image groups."""
    saved = np.load(RUN / 'independent_verify_20261106/val_predictions.npz')
    PI, PC, PS, PB = saved['image_id'], saved['label'].astype(int), saved['score'].astype(np.float32), saved['box'].astype(np.float32)
    keep = PS >= .01
    PI, PC, PS, PB = PI[keep], PC[keep], PS[keep], PB[keep]
    gts = defaultdict(list)
    for a in val['annotations']:
        x, y, w, h = a['bbox']
        gts[(a['image_id'], a['category_id'])].append([x, y, x + w, y + h])
    gts = {k: np.array(v, np.float32) for k, v in gts.items()}
    order = np.argsort(PI, kind='stable')
    values, first = np.unique(PI[order], return_index=True)
    where = dict(zip(values.tolist(), np.split(order, first[1:])))
    best = np.zeros(len(PI), np.float32)
    for (image_id, c), cur in gts.items():
        pos = where.get(image_id)
        if pos is not None:
            mine = pos[PC[pos] == c]
            if len(mine):
                best[mine] = iou(PB[mine], cur).max(1)

    def group_of(row):
        return Path(row['file_name']).suffix.lower(), row['width'], row['height']

    groups = {row['id']: group_of(row) for row in val['images']}
    common = {g for g, n in zip(*np.unique([str(v) for v in groups.values()], return_counts=True)) if n >= 100}
    label = np.array([str(groups[i]) if str(groups[i]) in common else 'rest' for i in PI])
    fold_of = make_folds()
    fold = np.array([fold_of[int(i)] for i in PI])

    def fit_map(mask):
        s, q = PS[mask], best[mask]
        if len(s) < 50:
            return None
        edges = np.quantile(s, np.linspace(0, 1, 21))
        edges = np.unique(edges)
        centers, means = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            inside = (s >= a) & (s <= b)
            if inside.any():
                centers.append(s[inside].mean())
                means.append(q[inside].mean())
        return np.array(centers), np.maximum.accumulate(np.array(means))

    def calibrated(cross):
        out = PS.copy()
        for c in names:
            for g in np.unique(label):
                cell = (PC == c) & (label == g)
                if not cell.any():
                    continue
                for test_fold in (0, 1):
                    target = cell & (fold == test_fold)
                    source = cell & (fold != test_fold) if cross else target
                    mapping = fit_map(source)
                    if mapping is not None and target.any():
                        out[target] = np.interp(PS[target], mapping[0], mapping[1]) + 1e-3 * PS[target]
        return out

    report = dict(groups={g: int((label == g).sum()) for g in np.unique(label)})
    report['base'] = evaluate(PI, PC, PS, PB, gts)
    report['group_cross_fit'] = evaluate(PI, PC, calibrated(True), PB, gts)
    report['group_same_fold_ceiling'] = evaluate(PI, PC, calibrated(False), PB, gts)
    (OUT / 'calibration.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    if STAGE == 'pca':
        stage_pca()
    elif STAGE == 'calibrate':
        stage_calibrate()
    elif STAGE == 'check':
        stage_check()
    elif STAGE == 'extract':
        stage_extract(int(sys.argv[3]), int(sys.argv[4]))
    else:
        stage_fit()
