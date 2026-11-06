"""E000 frozen, CPU only: does a bin-aligned head trained on TRAIN labels improve aircraft boxes on val?

edge_probe.py trains on val labels, so its gain may only reflect the labelling habit of val.
Criteria written before the run: aligned minus query >= 2 AP means implement; < 1 AP means do not implement.
E000 overfits train, so a weak result is not a complete rejection.

Usage: edge_transfer.py RUN {extract SPLIT SHARD NSHARDS | fit}
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

RUN = Path(sys.argv[1])
STAGE = sys.argv[2]
OUT = RUN / 'independent_verify_20261106/edge_probe'
ROOT = Path('/root/zxy/TCRA-SAR/datasets/SARDet_100K')
TANGENT = (-.3, 0., .3)
TRAIN_IMAGES = 6000


def split_data(split):
    data = json.load(open(ROOT / f'Annotations/{split}.json'))
    aircraft = next(c['id'] for c in data['categories'] if c['name'] == 'aircraft')
    gts = {}
    for a in data['annotations']:
        if a['category_id'] == aircraft:
            x, y, w, h = a['bbox']
            gts.setdefault(a['image_id'], []).append([x, y, x + w, y + h])
    return data, aircraft, {k: np.array(v, np.float32) for k, v in gts.items()}


def sample(feature, points):
    import torch.nn.functional as F
    shape = points.shape[:-1]
    grid = 2 * points.reshape(1, -1, 1, 2) - 1
    return F.grid_sample(feature, grid, mode='bilinear', padding_mode='zeros', align_corners=False)[0, :, :, 0].T.reshape(*shape, feature.shape[1])


def extract(split, shard, total):
    import torch
    import torchvision.transforms.functional as TF
    from engine.core import YAMLConfig
    from engine.deim.dfine_utils import weighting_function
    torch.set_num_threads(4)
    data, aircraft, gts = split_data(split)
    rows = data['images']
    if split == 'train':
        rows = [i for i in rows if i['id'] in gts]
        rows = [rows[k] for k in sorted(np.random.default_rng(0).permutation(len(rows))[:TRAIN_IMAGES])]
    rows = rows[shard::total]
    model = YAMLConfig(str(RUN / 'resolved_config.yml')).model
    model.load_state_dict(torch.load(RUN / 'best_stg2.pth', map_location='cpu')['ema']['module'], strict=False)
    model.eval()
    cap = {}
    model.backbone.stages[0].register_forward_hook(lambda m, i, o: cap.update(s2=o))
    model.encoder.register_forward_hook(lambda m, i, o: cap.update(p3=o[0] if torch.is_tensor(o[0]) else o[0][0]))
    model.decoder.dec_bbox_head[-1].register_forward_pre_hook(lambda m, i: cap.update(query=i[0]))
    model.decoder.decoder.register_forward_hook(lambda m, i, o: cap.update(logit=o[1][-1], corners=o[2][-1], ref=o[3][-1]))
    bins = weighting_function(model.decoder.reg_max, model.decoder.up, model.decoder.reg_scale).detach()
    scale = float(abs(model.decoder.reg_scale))
    pca = torch.load(OUT / 'pca.pth')
    index = aircraft - 1 if min(c['id'] for c in data['categories']) == 1 else aircraft

    def project(feature, key):
        p = pca[key]
        return (((feature[0].permute(1, 2, 0) - p['mean']) / p['std']) @ p['vectors']).permute(2, 0, 1)[None]

    keep = {k: [] for k in ('image_id', 'score', 'ref', 'corners', 'query', 'profile')}
    with torch.no_grad():
        for count, row in enumerate(rows):
            image = Image.open(ROOT / f'JPEGImages/{split}' / row['file_name']).convert('RGB')
            model(TF.to_tensor(TF.resize(image, [640, 640])).unsqueeze(0))
            prob = cap['logit'][0].sigmoid()
            score, label = prob.max(-1)
            pick = (score >= .01) & (label == index)
            if not pick.any():
                continue
            ref = cap['ref'][0][pick]
            s2, p3 = project(cap['s2'], 's2'), project(cap['p3'], 'p3')
            cx, cy, w, h = ref.unbind(-1)
            reach = (scale / 2 + bins)[None] / scale
            t = torch.tensor(TANGENT)
            along = torch.stack([cx[:, None] - reach * w[:, None], cy[:, None] - reach * h[:, None], cx[:, None] + reach * w[:, None], cy[:, None] + reach * h[:, None]], 1)
            points = torch.zeros(len(ref), 4, 33, 3, 2)
            for e in range(4):
                normal, tangent = (0, 1) if e in (0, 2) else (1, 0)
                center, extent = (cy, h) if e in (0, 2) else (cx, w)
                points[:, e, :, :, normal] = along[:, e, :, None]
                points[:, e, :, :, tangent] = (center[:, None] + t[None] * extent[:, None])[:, None]
            profile = torch.cat([sample(s2, points), sample(p3, points)], -1).mean(3)
            keep['image_id'].append(np.full(len(ref), row['id']))
            keep['score'].append(score[pick].numpy())
            keep['ref'].append(ref.numpy())
            keep['corners'].append(cap['corners'][0][pick].numpy())
            keep['query'].append(cap['query'][0][pick].numpy().astype(np.float16))
            keep['profile'].append(profile.numpy().astype(np.float16))
            if count % 200 == 0:
                print(split, shard, count, len(rows), flush=True)
    np.savez(OUT / f'transfer_{split}_{shard}.npz', bins=bins.numpy(), scale=scale, width=np.array([images_size(r)[0] for r in rows]), **{k: np.concatenate(v) for k, v in keep.items()})


def images_size(row):
    return row['width'], row['height']


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
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    where = np.searchsorted(recall, REC, side='left')
    out = np.zeros(101)
    ok = where < len(precision)
    out[ok] = precision[where[ok]]
    return out.mean()


def aircraft_ap(pi, ps, boxes, gts):
    n = sum(len(v) for v in gts.values())
    order = np.argsort(pi, kind='stable')
    values, first = np.unique(pi[order], return_index=True)
    where = dict(zip(values.tolist(), np.split(order, first[1:])))
    curve = []
    for th in THR:
        scores, hits = [], []
        for image_id, pos in where.items():
            pos = pos[np.argsort(-ps[pos], kind='mergesort')][:100]
            hit = np.zeros(len(pos), bool)
            cur = gts.get(image_id)
            if cur is not None:
                m = iou(boxes[pos], cur)
                used = set()
                for d in range(len(pos)):
                    chosen, value = -1, th
                    for g in range(len(cur)):
                        if g not in used and m[d, g] >= value:
                            value, chosen = m[d, g], g
                    if chosen >= 0:
                        used.add(chosen)
                        hit[d] = True
            scores.append(ps[pos])
            hits.append(hit)
        curve.append(ap_curve(np.concatenate(scores), np.concatenate(hits), n))
    return dict(AP=round(float(np.mean(curve)) * 100, 2), AP50=round(curve[0] * 100, 2), AP75=round(curve[5] * 100, 2), AP90=round(curve[8] * 100, 2))


def fit():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.set_num_threads(20)

    def load(split):
        shards = [np.load(p) for p in sorted(OUT.glob(f'transfer_{split}_*.npz'))]
        data, _, gts = split_data(split)
        sizes = {i['id']: (i['width'], i['height']) for i in data['images']}
        out = {k: np.concatenate([s[k] for s in shards]) for k in ('image_id', 'score', 'ref', 'corners', 'query', 'profile')}
        out['size'] = np.array([sizes[int(i)] for i in out['image_id']], np.float32)
        return out, gts, torch.as_tensor(shards[0]['bins']), float(shards[0]['scale'])

    train, train_gts, bins, scale = load('train')
    test, test_gts, _, _ = load('val')

    def tensors(d):
        return (torch.as_tensor(d['ref']), torch.as_tensor(d['corners']).reshape(-1, 4, 33), torch.as_tensor(d['query'].astype(np.float32)), torch.as_tensor(d['profile'].astype(np.float32)).nan_to_num())

    def decode(ref, logits):
        d = (logits.softmax(-1) * bins).sum(-1)
        cx, cy, w, h = ref.unbind(-1)
        return torch.stack([cx - (scale / 2 + d[:, 0]) * w / scale, cy - (scale / 2 + d[:, 1]) * h / scale, cx + (scale / 2 + d[:, 2]) * w / scale, cy + (scale / 2 + d[:, 3]) * h / scale], -1)

    def pixels(d, normalized):
        return normalized.numpy() * np.concatenate([d['size'], d['size']], 1)

    r_tr, c_tr, q_tr, p_tr = tensors(train)
    r_te, c_te, q_te, p_te = tensors(test)
    q_mean, q_std = q_tr.mean(0), q_tr.std(0).clamp(min=1e-6)
    p_mean, p_std = p_tr.mean((0, 1, 2)), p_tr.std((0, 1, 2)).clamp(min=1e-6)
    q_tr, q_te = (q_tr - q_mean) / q_std, (q_te - q_mean) / q_std
    p_tr, p_te = (p_tr - p_mean) / p_std, (p_te - p_mean) / p_std
    base_train = pixels(train, decode(r_tr, c_tr))
    target = np.zeros_like(base_train)
    best = np.zeros(len(base_train), np.float32)
    order = np.argsort(train['image_id'], kind='stable')
    values, first = np.unique(train['image_id'][order], return_index=True)
    for image_id, pos in zip(values.tolist(), np.split(order, first[1:])):
        cur = train_gts.get(image_id)
        if cur is not None:
            m = iou(base_train[pos], cur)
            target[pos] = cur[m.argmax(1)]
            best[pos] = m.max(1)
    use = torch.as_tensor(np.flatnonzero(best >= .4))
    normalized = torch.as_tensor(target / np.concatenate([train['size'], train['size']], 1))
    cx, cy, w, h = r_tr.unbind(-1)
    distance = torch.stack([(cx - normalized[:, 0]) / (w / scale), (cy - normalized[:, 1]) / (h / scale), (normalized[:, 2] - cx) / (w / scale), (normalized[:, 3] - cy) / (h / scale)], -1) - scale / 2
    distance = distance.clamp(float(bins[0]), float(bins[-1]))
    left = (torch.bucketize(distance, bins, right=True) - 1).clamp(0, 31)
    weight_right = ((distance - bins[left]) / (bins[left + 1] - bins[left])).clamp(0, 1)

    class QueryHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.edge = nn.Embedding(4, 32)
            self.net = nn.Sequential(nn.Linear(256 + 32 + 33, 384), nn.SiLU(), nn.Linear(384, 384), nn.SiLU(), nn.Linear(384, 33))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        def forward(self, q, p, c):
            n = len(q)
            return self.net(torch.cat([q[:, None].expand(n, 4, -1), self.edge.weight[None].expand(n, -1, -1), c.softmax(-1)], -1))

    class AlignedHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.edge = nn.Embedding(4, 32)
            self.context = nn.Sequential(nn.Linear(256 + 32, 128), nn.SiLU(), nn.Linear(128, 128))
            self.enter = nn.Linear(96 + 3, 128)
            self.convs = nn.ModuleList(nn.Sequential(nn.Conv1d(128, 128, 5, padding=2 * d, dilation=d), nn.GroupNorm(8, 128), nn.SiLU(), nn.Conv1d(128, 128, 1)) for d in (1, 2, 4))
            self.out = nn.Linear(128, 1)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

        def forward(self, q, p, c):
            n = len(q)
            context = self.context(torch.cat([q[:, None].expand(n, 4, -1), self.edge.weight[None].expand(n, -1, -1)], -1))
            position = (bins / 4)[None, None, :, None].expand(n, 4, 33, 1)
            token = self.enter(torch.cat([p, c.softmax(-1)[..., None], (c / 10)[..., None], position], -1)) + context[:, :, None]
            x = token.reshape(n * 4, 33, 128).transpose(1, 2)
            for conv in self.convs:
                x = x + conv(x)
            return self.out(x.transpose(1, 2)).reshape(n, 4, 33)

    base_test = pixels(test, decode(r_te, c_te))
    report = dict(train_detections=len(best), train_matched=len(use), train_base_iou=round(float(best[best >= .5].mean()), 4))
    report['base'] = aircraft_ap(test['image_id'], test['score'], base_test, test_gts)
    print('base', report['base'], report['train_base_iou'], flush=True)
    for name, make in (('query', QueryHead), ('aligned', AlignedHead)):
        torch.manual_seed(0)
        model = make()
        epochs, batch = 24, 256
        optimizer = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
        schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-3, total_steps=epochs * (len(use) // batch))
        for epoch in range(epochs):
            perm = use[torch.randperm(len(use))]
            total = 0.
            for s in range(0, len(perm) - batch + 1, batch):
                b = perm[s:s + batch]
                logits = c_tr[b] + model(q_tr[b], p_tr[b], c_tr[b])
                log_p = F.log_softmax(logits, -1)
                l, wr = left[b], weight_right[b]
                ce = -(log_p.gather(-1, l[..., None])[..., 0] * (1 - wr) + log_p.gather(-1, l[..., None] + 1)[..., 0] * wr).mean()
                reg = F.smooth_l1_loss((logits.softmax(-1) * bins).sum(-1) / scale, distance[b] / scale, beta=.05)
                loss = ce + 5 * reg
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                schedule.step()
                total += float(loss)
            print(name, epoch, round(total / (len(perm) // batch), 4), flush=True)
        model.eval()
        pieces = []
        with torch.no_grad():
            for s in range(0, len(r_te), 2048):
                pieces.append(model(q_te[s:s + 2048], p_te[s:s + 2048], c_te[s:s + 2048]))
        boxes = pixels(test, decode(r_te, c_te + torch.cat(pieces)))
        report[name] = aircraft_ap(test['image_id'], test['score'], boxes, test_gts)
        print(name, report[name], flush=True)
        (OUT / 'transfer.json').write_text(json.dumps(report, indent=1))
    print('DONE', flush=True)


if __name__ == '__main__':
    if STAGE == 'extract':
        extract(sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
    else:
        fit()
