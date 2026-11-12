"""D13: what does the surrounding of a box add as detection evidence, at which distance, at which level, and for
which targets? Follow-up of D12.3 ("add the surrounding, do not subtract it"). Frozen-feature probes; no training of
the detector, no change to it; train and val only.

Boxes: every GT (positive); one random box of the same size per GT without overlap (random negative, as in D12.2);
and the detector's own background false positives (hard negative): detections of this run with score >= .1 whose IoU
with every GT is < .1, at most five per image.
Features of a box, pooled from the encoder outputs (ENC: stride 8 / 16 / 32, 256 channels each, concatenated):
  inside   mean of 3x3 samples in the box
  r15, r20, r30, r40   mean of 8 samples (corners, edge midpoints) on the box scaled by 1.5 / 2 / 3 / 4
  glob     mean over the whole map
The backbone (BB) keeps inside, r20 and glob for reference. PIX is the pixel CFAR contrast of D12.3.
Probes are fitted on train split A (same split as D12.2) with random and hard negatives together: a linear probe for
every variant, and a small MLP (2 x 256) for the key variants. The MLP answers whether the surrounding carries
information that the box feature does not already contain, or whether the linear gain was only a readout effect.
Thresholds: 1 % and 0.1 % of val random negatives; 10 % and 1 % of val hard negatives.

Reading fixed before the run. A feature-level context module is supported only if, with the MLP probe on ENC,
`inside+rings` (or `inside+rings+glob`) beats `inside` on val targets with sim < .9, at the threshold passing 10 % of
the val hard negatives, by >= 2 points pooled and >= 5 points on the 512-px aircraft, with the same sign on both
runs, and the gain is present in the two lowest contrast quartiles of those aircraft. If the MLP on `inside` matches
`inside+rings`, the information is already in the box feature and D12.3's linear gain was a readout effect: no module.
The ring scale and the level ablation say where a module would have to look.

Usage (server; OUT directly under E000's independent_verify_20261106/):
  extract RUN OUT --d12 D12 --val-pred NPZ --train-pred NPZ [--device cuda:0]     OUT/context_{train,val}.npz
  analyse OUT --d12 D12 [--device cuda:0]                                         OUT/context_probe.json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clutter_relative_probe import pixel_cfar, ring_points  # noqa: E402
from e1_report import iou_xyxy, r3  # noqa: E402
from generalisation_probe import EvalImages, cell_name, fit_probe, negative_boxes, roi_mean  # noqa: E402

RINGS = {'r15': 1.5, 'r20': 2., 'r30': 3., 'r40': 4.}
ENC_PARTS = ['inside', 'r15', 'r20', 'r30', 'r40', 'glob']
BB_PARTS = ['inside', 'r20', 'glob']
LINEAR = {
    'inside': ['inside'], 'inside+r15': ['inside', 'r15'], 'inside+r20': ['inside', 'r20'], 'inside+r30': ['inside', 'r30'],
    'inside+r40': ['inside', 'r40'], 'inside+rings': ['inside', 'r15', 'r20', 'r30', 'r40'], 'inside+glob': ['inside', 'glob'],
    'inside+rings+glob': ['inside', 'r15', 'r20', 'r30', 'r40', 'glob'],
    'r20 only': ['r20'], 'rings only': ['r15', 'r20', 'r30', 'r40'], 'glob only': ['glob'],
}
MLP = ['inside', 'inside+rings', 'inside+glob', 'inside+rings+glob']


def ring_mean(fmap, boxes, scale):
    import torch.nn.functional as F
    grid = ring_points(boxes, (scale,))[None] * 2 - 1
    return F.grid_sample(fmap[None].float(), grid, mode='bilinear', padding_mode='border', align_corners=False)[0].mean(-1).T


def hard_negatives(pred, gt_px, width, height, limit=5):
    """pred: (scores, boxes xyxy px) of one image. Background false positives, normalised xyxy."""
    scores, boxes = pred
    if not len(scores):
        return np.zeros((0, 4))
    keep = scores >= .1
    if len(gt_px):
        keep &= iou_xyxy(boxes.astype(np.float64), gt_px).max(1) < .1
    idx = np.flatnonzero(keep)
    idx = idx[np.argsort(-scores[idx], kind='stable')][:limit]
    out = boxes[idx].astype(np.float64) / np.array([width, height, width, height])
    out = np.clip(out, 0, 1)
    return out[(out[:, 2] - out[:, 0] > 1e-4) & (out[:, 3] - out[:, 1] > 1e-4)]


def extract(args):
    import torch
    from engine.core import YAMLConfig
    from error_gallery import ROOT
    chosen = set(np.load(args.d12 / 'similarity.npz')['train_images'].tolist())
    cfg = YAMLConfig(str(args.run / 'resolved_config.yml'))
    model = cfg.model
    model.load_state_dict(torch.load(args.run / 'best_stg2.pth', map_location='cpu', weights_only=False)['ema']['module'])
    model = model.to(args.device).eval()
    grab = {}
    model.backbone.register_forward_hook(lambda m, i, o: grab.update(BB=[t.detach() for t in o]))
    model.encoder.register_forward_hook(lambda m, i, o: grab.update(ENC=[t.detach() for t in (o[0] if isinstance(o, tuple) else o)]))
    for split, pred_path in (('train', args.train_pred), ('val', args.val_pred)):
        with np.load(pred_path) as z:
            order = np.argsort(z['image_id'], kind='stable')
            p_img, p_score, p_box = z['image_id'][order], z['score'][order], z['box'][order]
        data = json.load(open(ROOT / f'Annotations/{split}.json'))
        by_image = defaultdict(list)
        for a in data['annotations']:
            by_image[a['image_id']].append(a)
        images = [i for i in data['images'] if split == 'val' or i['id'] in chosen]
        loader = torch.utils.data.DataLoader(EvalImages(ROOT / f'JPEGImages/{split}', images), batch_size=32, num_workers=8)
        rows = defaultdict(list)
        with torch.no_grad():
            for batch, indices in loader:
                batch = batch.to(args.device)
                model(batch)
                for k, index in enumerate(indices.tolist()):
                    image = images[index]
                    anns = by_image[image['id']]
                    if not anns:
                        continue
                    w, h = image['width'], image['height']
                    gt_px = np.array([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]] for a in anns], np.float64)
                    gt = np.clip(gt_px / np.array([w, h, w, h]), 0, 1)
                    rand = negative_boxes(gt, np.random.default_rng(image['id']))
                    s, e = np.searchsorted(p_img, image['id'], 'left'), np.searchsorted(p_img, image['id'], 'right')
                    hard = hard_negatives((p_score[s:e], p_box[s:e]), gt_px, w, h)
                    boxes = torch.as_tensor(np.concatenate([gt, rand, hard]), dtype=torch.float32, device=args.device)
                    for stage, parts in (('ENC', ENC_PARTS), ('BB', BB_PARTS)):
                        maps = [f[k].float() for f in grab[stage]]
                        for part in parts:
                            if part == 'inside':
                                v = torch.cat([roi_mean(f, boxes) for f in maps], 1)
                            elif part == 'glob':
                                v = torch.cat([f.mean((1, 2)) for f in maps])[None].expand(len(boxes), -1)
                            else:
                                v = torch.cat([ring_mean(f, boxes, RINGS[part]) for f in maps], 1)
                            rows[f'{stage}_{part}'].append(v.half().cpu().numpy())
                    rows['PIX'].append(pixel_cfar(batch[k], boxes).cpu().numpy())
                    rows['kind'].append(np.r_[np.zeros(len(gt), np.int8), np.ones(len(rand), np.int8), np.full(len(hard), 2, np.int8)])
                    rows['ann'].append(np.r_[[a['id'] for a in anns], np.full(len(rand) + len(hard), -1)])
                    rows['label'].append(np.r_[[a['category_id'] for a in anns], np.full(len(rand) + len(hard), -1)])
                    rows['image'].append(np.full(len(boxes), image['id']))
                    rows['size'].append(np.full(len(boxes), image['width']))
                    rows['long_side'].append(np.r_[[max(a['bbox'][2], a['bbox'][3]) for a in anns], np.full(len(rand) + len(hard), -1.)])
        out = {k: np.concatenate(v) for k, v in rows.items()}
        np.savez(args.out / f'context_{split}.npz', **out)
        print('written', args.out / f'context_{split}.npz', {n: int((out['kind'] == c).sum()) for c, n in enumerate(('pos', 'random', 'hard'))}, flush=True)


def fit_mlp(x, y, device, epochs=30, seed=0):
    import torch
    torch.manual_seed(seed)
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    y = torch.as_tensor(y, dtype=torch.float32, device=device)
    mean, std = x.mean(0, keepdim=True), x.std(0, keepdim=True) + 1e-6
    x = (x - mean) / std
    net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 256), torch.nn.ReLU(), torch.nn.Dropout(.1),
                              torch.nn.Linear(256, 256), torch.nn.ReLU(), torch.nn.Linear(256, 1)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * (len(x) // 1024 + 1))
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(len(x), device=device)
        for k in range(0, len(x), 1024):
            idx = perm[k:k + 1024]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(x[idx])[:, 0], y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    net.eval()

    def score(z):
        with torch.no_grad():
            z = torch.as_tensor(z, dtype=torch.float32, device=device)
            return torch.cat([net((z[k:k + 8192] - mean) / std)[:, 0] for k in range(0, len(z), 8192)]).cpu().numpy()
    return score


def feature(d, stage, parts, levels=None):
    """Concatenate parts; `levels` restricts the non-inside parts to some pyramid levels (ENC only: 256 channels each)."""
    out = []
    for part in parts:
        v = d[f'{stage}_{part}'].astype(np.float32)
        if levels is not None and part != 'inside':
            v = np.concatenate([v[:, 256 * l:256 * (l + 1)] for l in levels], 1)
        out.append(v)
    return np.concatenate(out, 1)


def evaluate(score, pops, va, edges):
    """Recall tables at the four thresholds."""
    rand, hard = score['val'][va['kind'] == 1], score['val'][va['kind'] == 2]
    thresholds = {'random 1%': np.quantile(rand, .99), 'random 0.1%': np.quantile(rand, .999),
                  'hard 10%': np.quantile(hard, .90), 'hard 1%': np.quantile(hard, .99)}
    out = {}
    for name, thr in thresholds.items():
        table = {}
        for k, p in pops.items():
            pos = p['m'] & (p['d']['kind'] == 0)
            hit = score[k] > thr
            row = dict(seen=r3(hit[pos & (p['sim'] >= .9)].mean() * 100), novel=r3(hit[pos & (p['sim'] < .9)].mean() * 100), cells={})
            for c in np.unique(p['cell'][pos]):
                mc = pos & (p['cell'] == c)
                if mc.sum() < 150:
                    continue
                row['cells'][c] = {lab: (r3(hit[mc & sel].mean() * 100) if (mc & sel).sum() >= 20 else None)
                                   for lab, sel in (('seen', p['sim'] >= .9), ('novel', p['sim'] < .9))}
            if k == 'val':
                air = pos & (p['cell'] == '512/aircraft') & (p['sim'] < .9)
                contrast = p['d']['PIX'][:, 3]
                row['aircraft_novel_by_contrast'] = [r3(hit[air & (contrast >= edges[b]) & (contrast < edges[b + 1])].mean() * 100) for b in range(4)]
            table[k] = row
        out[name] = table
    return out


def analyse(args):
    from error_gallery import ROOT
    names = {c['id']: c['name'] for c in json.load(open(ROOT / 'Annotations/val.json'))['categories']}
    sim = np.load(args.d12 / 'similarity.npz')
    sim_val = dict(zip(sim['val_ann'].tolist(), sim['val_sim'].tolist()))
    sim_train = dict(zip(sim['train_ann'].tolist(), sim['train_sim'].tolist()))
    tr, va = (dict(np.load(args.out / f'context_{s}.npz')) for s in ('train', 'val'))
    perm = np.random.default_rng(0).permutation(np.unique(tr['image']))
    split_a = set(perm[:len(perm) * 2 // 3].tolist())
    in_a = np.array([i in split_a for i in tr['image']])
    cells = lambda d: np.array([cell_name(s, names[int(c)], ls) if c >= 0 else '' for s, c, ls in zip(d['size'], d['label'], d['long_side'])])  # noqa: E731
    pops = {'train B': dict(d=tr, m=~in_a, sim=np.array([sim_train.get(int(a), np.nan) for a in tr['ann']]), cell=cells(tr)),
            'val': dict(d=va, m=np.ones(len(va['ann']), bool), sim=np.array([sim_val.get(int(a), np.nan) for a in va['ann']]), cell=cells(va))}
    p = pops['val']
    air = (va['kind'] == 0) & (p['cell'] == '512/aircraft') & (p['sim'] < .9)
    edges = [-1e9, *np.quantile(va['PIX'][air, 3], [.25, .5, .75]).tolist(), 1e9]
    y = tr['kind'][in_a] == 0
    report = dict(counts=dict(train_a={n: int((tr['kind'][in_a] == c).sum()) for c, n in enumerate(('pos', 'random', 'hard'))},
                              val={n: int((va['kind'] == c).sum()) for c, n in enumerate(('pos', 'random', 'hard'))}),
                  aircraft_contrast_quartile_edges=[r3(e) for e in edges[1:4]], linear={}, mlp={}, levels={}, backbone={})

    def run(fit, stage, parts, levels=None):
        probe = fit(feature(tr, stage, parts, levels)[in_a], y, args.device)
        return evaluate({k: probe(feature(q['d'], stage, parts, levels)) for k, q in pops.items()}, pops, va, edges)

    def show(tag, name, res):
        t = res['hard 10%']['val']
        print(tag, name, 'hard10% seen/novel', t['seen'], t['novel'], 'air', t['cells'].get('512/aircraft'), 'air by contrast', t['aircraft_novel_by_contrast'],
              '| random0.1% novel', res['random 0.1%']['val']['novel'], flush=True)

    for name, parts in LINEAR.items():
        report['linear'][name] = run(fit_probe, 'ENC', parts)
        show('ENC linear', name, report['linear'][name])
    for name in MLP:
        report['mlp'][name] = run(fit_mlp, 'ENC', LINEAR[name])
        show('ENC mlp', name, report['mlp'][name])
    for level, tag in enumerate(('P3', 'P4', 'P5')):
        report['levels'][f'inside+rings[{tag}]'] = run(fit_probe, 'ENC', LINEAR['inside+rings'], levels=[level])
        show('ENC linear', f'inside+rings[{tag}]', report['levels'][f'inside+rings[{tag}]'])
    for name, parts in (('inside', ['inside']), ('inside+r20', ['inside', 'r20']), ('inside+glob', ['inside', 'glob'])):
        report['backbone'][name] = run(fit_probe, 'BB', parts)
        show('BB linear', name, report['backbone'][name])
    (args.out / 'context_probe.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print('written', args.out / 'context_probe.json', flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    e = sub.add_parser('extract')
    e.add_argument('run', type=Path)
    e.add_argument('out', type=Path)
    e.add_argument('--d12', type=Path, required=True)
    e.add_argument('--val-pred', type=Path, required=True)
    e.add_argument('--train-pred', type=Path, required=True)
    e.add_argument('--device', default='cuda:0')
    a = sub.add_parser('analyse')
    a.add_argument('out', type=Path)
    a.add_argument('--d12', type=Path, required=True)
    a.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    {'extract': extract, 'analyse': analyse}[args.command](args)


if __name__ == '__main__':
    main()
