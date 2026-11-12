"""D12.3: do clutter-relative features make targets without a near-duplicate more separable? Zero-training check for
a feature-level clutter normalisation (the CFAR idea: a SAR target is evidence relative to its surrounding clutter).
No training of the detector, no change to it; train and val only.

Same boxes, splits, probe and thresholds as D12.2 (generalisation_probe.py). For the backbone (BB) and encoder (ENC)
outputs at stride 8 / 16 / 32, five variants of the pooled feature are probed:
  inside        mean of 3x3 samples in the box                                  (D12.2 reference)
  diff          inside - ring mean                                              local clutter subtracted
  cfar          (inside - ring mean) / (ring std + eps), per channel            local clutter normalised
  scene         (inside - image mean) / (image std + eps), per channel          scene normalised
  inside+ring   [inside, ring mean] concatenated                                control: the probe may form any difference
The ring is 16 samples on the box scaled by 1.5 and by 2 (corners and edge midpoints of each).
Pixel level, without any learning (PIX): the classical CFAR contrast (mean inside - ring mean) / ring std of the
640 input, with 7x7 samples inside and the samples of a 12x12 grid over the 2x box that lie outside the 1.25x box.
It answers a control question D12.2 left open: are targets without a near-duplicate physically weaker?

Reading fixed before the run. A feature-level clutter normalisation is supported only if `diff`, `cfar` or `scene`
raises the recall of val targets with sim < .9 over `inside` at the same false-positive rate, with the same sign on
all three runs, by at least 3 points at 1 % FPR (or 5 at 0.1 %) on the 512-px aircraft, and without lowering the
recall of targets with sim >= .9. Features were trained without such a normalisation, so a negative result here does
not exclude a trained module; it only removes the zero-training support for it.

Usage (server; OUT directly under E000's independent_verify_20261106/):
  extract RUN OUT --d12 D12 [--device cuda:0]       OUT/ring_{train,val}.npz
  analyse OUT --d12 D12 [--device cuda:0]           OUT/clutter_relative_probe.json
  matched OUT --d12 D12 [--device cuda:0]           OUT/contrast_matched.json (recall by pixel-contrast bin)
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_report import r3  # noqa: E402
from generalisation_probe import EvalImages, auc, cell_name, fit_probe, negative_boxes, roi_mean  # noqa: E402

VARIANTS = ['inside', 'diff', 'cfar', 'scene', 'inside+ring']
STAGES = ['BB', 'ENC']


def ring_points(boxes, scales=(1.5, 2.)):
    """boxes (N, 4) normalised xyxy tensor -> (N, 8 * len(scales), 2): corners and edge midpoints of the scaled boxes."""
    import torch
    cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    offsets = torch.tensor([[-.5, -.5], [0, -.5], [.5, -.5], [-.5, 0], [.5, 0], [-.5, .5], [0, .5], [.5, .5]], device=boxes.device)
    pts = [torch.stack([cx[:, None] + s * offsets[None, :, 0] * w[:, None], cy[:, None] + s * offsets[None, :, 1] * h[:, None]], -1) for s in scales]
    return torch.cat(pts, 1)


def ring_stats(fmap, boxes):
    """fmap (C, h, w); boxes (N, 4) -> ring mean (N, C), ring std (N, C)."""
    import torch.nn.functional as F
    grid = ring_points(boxes)[None] * 2 - 1
    out = F.grid_sample(fmap[None].float(), grid, mode='bilinear', padding_mode='border', align_corners=False)[0]     # C, N, P
    return out.mean(-1).T, out.std(-1).T


def pixel_cfar(image, boxes):
    """image (3, H, W) in [0, 1]; boxes (N, 4). Returns (N, 4): mean inside, ring mean, ring std, CFAR contrast."""
    import torch
    import torch.nn.functional as F
    grey = image.mean(0, keepdim=True)[None]
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]

    def grid_of(n, lo, hi):
        u = lo + (torch.arange(n, device=boxes.device, dtype=torch.float32) + .5) / n * (hi - lo)   # box coordinates
        gu, gv = torch.meshgrid(u, u, indexing='xy')
        return gu.reshape(-1), gv.reshape(-1)

    def take(gu, gv):
        pts = torch.stack([boxes[:, None, 0] + gu[None] * w[:, None], boxes[:, None, 1] + gv[None] * h[:, None]], -1)[None] * 2 - 1
        return F.grid_sample(grey, pts, mode='bilinear', padding_mode='border', align_corners=False)[0, 0]   # N, P

    inside = take(*grid_of(7, 0., 1.))
    gu, gv = grid_of(12, -.5, 1.5)
    outer = ((gu - .5).abs() > .625) | ((gv - .5).abs() > .625)
    ring = take(gu[outer], gv[outer])
    mu_in, mu_ring, sd_ring = inside.mean(1), ring.mean(1), ring.std(1)
    return torch.stack([mu_in, mu_ring, sd_ring, (mu_in - mu_ring) / (sd_ring + 1e-3)], 1)


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
    for split in ('train', 'val'):
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
                    gt = np.array([[a['bbox'][0] / w, a['bbox'][1] / h, (a['bbox'][0] + a['bbox'][2]) / w, (a['bbox'][1] + a['bbox'][3]) / h] for a in anns], np.float64)
                    gt = np.clip(gt, 0, 1)
                    neg = negative_boxes(gt, np.random.default_rng(image['id']))      # same seed as D12.2: same negatives
                    boxes = torch.as_tensor(np.concatenate([gt, neg]), dtype=torch.float32, device=args.device)
                    for stage in STAGES:
                        maps = [f[k].float() for f in grab[stage]]
                        ring = [ring_stats(f, boxes) for f in maps]
                        rows[f'{stage}_inside'].append(torch.cat([roi_mean(f, boxes) for f in maps], 1).half().cpu().numpy())
                        rows[f'{stage}_ring_mean'].append(torch.cat([r[0] for r in ring], 1).half().cpu().numpy())
                        rows[f'{stage}_ring_std'].append(torch.cat([r[1] for r in ring], 1).half().cpu().numpy())
                        scene = torch.cat([torch.cat([f.mean((1, 2)), f.std((1, 2))]) for f in maps])   # per image
                        rows[f'{stage}_scene'].append(scene.half().cpu().numpy()[None].repeat(len(boxes), 0))
                    rows['PIX'].append(pixel_cfar(batch[k], boxes).cpu().numpy())
                    rows['positive'].append(np.r_[np.ones(len(gt), bool), np.zeros(len(neg), bool)])
                    rows['ann'].append(np.r_[[a['id'] for a in anns], np.full(len(neg), -1)])
                    rows['label'].append(np.r_[[a['category_id'] for a in anns], np.full(len(neg), -1)])
                    rows['image'].append(np.full(len(boxes), image['id']))
                    rows['size'].append(np.r_[[image['width']] * len(boxes)])
                    rows['long_side'].append(np.r_[[max(a['bbox'][2], a['bbox'][3]) for a in anns], np.full(len(neg), -1.)])
        np.savez(args.out / f'ring_{split}.npz', **{k: np.concatenate(v) for k, v in rows.items()})
        print('written', args.out / f'ring_{split}.npz', sum(len(v) for v in rows['ann']), flush=True)


def variant(d, stage, name):
    inside = d[f'{stage}_inside'].astype(np.float32)
    if name == 'inside':
        return inside
    mean, std = d[f'{stage}_ring_mean'].astype(np.float32), d[f'{stage}_ring_std'].astype(np.float32)
    if name == 'diff':
        return inside - mean
    if name == 'cfar':
        return (inside - mean) / (std + 1e-2)
    if name == 'inside+ring':
        return np.concatenate([inside, mean], 1)
    scene = d[f'{stage}_scene'].astype(np.float32)
    c = inside.shape[1]
    # scene holds, per level, [mean (C_l), std (C_l)]; split it back into the concatenated level layout of `inside`
    levels = [c // 7 * k for k in (1, 2, 4)] if stage == 'BB' else [c // 3] * 3
    assert sum(levels) == c and scene.shape[1] == 2 * c
    means, stds, at = [], [], 0
    for n in levels:
        means.append(scene[:, at:at + n])
        stds.append(scene[:, at + n:at + 2 * n])
        at += 2 * n
    return (inside - np.concatenate(means, 1)) / (np.concatenate(stds, 1) + 1e-2)


def recall_tables(score, pops, va, names, fprs=(.01, .001)):
    out = {}
    val_neg = score['val'][~va['positive']]
    for fpr in fprs:
        thr = float(np.quantile(val_neg, 1 - fpr))
        table = {}
        for k, p in pops.items():
            pos, neg = p['m'] & p['d']['positive'], p['m'] & ~p['d']['positive']
            cell = p['cell']
            row = dict(negatives_fpr=r3((score[k][neg] > thr).mean() * 100),
                       seen=r3((score[k][pos & (p['sim'] >= .9)] > thr).mean() * 100),
                       novel=r3((score[k][pos & (p['sim'] < .9)] > thr).mean() * 100),
                       auc_novel=r3(auc(score[k][pos & (p['sim'] < .9)], score[k][neg])), cells={})
            for c in np.unique(cell[pos]):
                mc = pos & (cell == c)
                if mc.sum() < 150:
                    continue
                row['cells'][c] = {lab: (r3((score[k][mc & sel] > thr).mean() * 100) if (mc & sel).sum() >= 20 else None)
                                   for lab, sel in (('seen', p['sim'] >= .9), ('novel', p['sim'] < .9))}
            table[k] = row
        out[f'{fpr:g}'] = table
    return out


def analyse(args):
    from error_gallery import ROOT
    names = {c['id']: c['name'] for c in json.load(open(ROOT / 'Annotations/val.json'))['categories']}
    sim = np.load(args.d12 / 'similarity.npz')
    sim_val = dict(zip(sim['val_ann'].tolist(), sim['val_sim'].tolist()))
    sim_train = dict(zip(sim['train_ann'].tolist(), sim['train_sim'].tolist()))
    tr, va = (dict(np.load(args.out / f'ring_{s}.npz')) for s in ('train', 'val'))
    perm = np.random.default_rng(0).permutation(np.unique(tr['image']))          # same A / B split as D12.2
    split_a = set(perm[:len(perm) * 2 // 3].tolist())
    in_a = np.array([i in split_a for i in tr['image']])
    cells = lambda d: np.array([cell_name(s, names[int(c)], ls) if c >= 0 else '' for s, c, ls in zip(d['size'], d['label'], d['long_side'])])  # noqa: E731
    pops = {'train B': dict(d=tr, m=~in_a, sim=np.array([sim_train.get(int(a), np.nan) for a in tr['ann']]), cell=cells(tr)),
            'val': dict(d=va, m=np.ones(len(va['ann']), bool), sim=np.array([sim_val.get(int(a), np.nan) for a in va['ann']]), cell=cells(va))}
    report = dict(probes={}, pixel={})
    for stage in STAGES:
        for name in VARIANTS:
            probe = fit_probe(variant(tr, stage, name)[in_a], tr['positive'][in_a], args.device)
            score = {k: probe(variant(p['d'], stage, name)) for k, p in pops.items()}
            report['probes'][f'{stage}/{name}'] = recall_tables(score, pops, va, names)
            t = report['probes'][f'{stage}/{name}']['0.01']
            print(stage, name, 'val seen/novel', t['val']['seen'], t['val']['novel'], '512/aircraft', t['val']['cells'].get('512/aircraft'),
                  'train B novel', t['train B']['novel'], flush=True)
    # Pixel level: the CFAR contrast itself, no learning.
    score = {k: p['d']['PIX'][:, 3].astype(np.float64) for k, p in pops.items()}
    report['pixel']['cfar_contrast_recall'] = recall_tables(score, pops, va, names)
    med = {}
    for k, p in pops.items():
        pos = p['m'] & p['d']['positive']
        med[k] = dict(negatives=r3(np.median(score[k][p['m'] & ~p['d']['positive']])), cells={})
        for c in np.unique(p['cell'][pos]):
            mc = pos & (p['cell'] == c)
            if mc.sum() < 150:
                continue
            med[k]['cells'][c] = {lab: dict(n=int((mc & sel).sum()), contrast_median=r3(np.median(score[k][mc & sel])) if (mc & sel).sum() >= 20 else None,
                                            inside_mean_median=r3(np.median(p['d']['PIX'][mc & sel, 0])) if (mc & sel).sum() >= 20 else None,
                                            ring_mean_median=r3(np.median(p['d']['PIX'][mc & sel, 1])) if (mc & sel).sum() >= 20 else None)
                                  for lab, sel in (('seen', p['sim'] >= .9), ('novel', p['sim'] < .9))}
    report['pixel']['contrast_medians'] = med
    (args.out / 'clutter_relative_probe.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print('written', args.out / 'clutter_relative_probe.json', flush=True)


def matched(args):
    """Control for physical weakness: recall of the ENC `inside` probe at 1 % FPR, by pixel CFAR contrast bin, for
    train B, val targets with a near-duplicate and val targets without one. If the gap of the novel targets were only
    their lower contrast, it would vanish inside a contrast bin."""
    from error_gallery import ROOT
    names = {c['id']: c['name'] for c in json.load(open(ROOT / 'Annotations/val.json'))['categories']}
    sim = np.load(args.d12 / 'similarity.npz')
    sim_val = dict(zip(sim['val_ann'].tolist(), sim['val_sim'].tolist()))
    sim_train = dict(zip(sim['train_ann'].tolist(), sim['train_sim'].tolist()))
    tr, va = (dict(np.load(args.out / f'ring_{s}.npz')) for s in ('train', 'val'))
    perm = np.random.default_rng(0).permutation(np.unique(tr['image']))
    split_a = set(perm[:len(perm) * 2 // 3].tolist())
    in_a = np.array([i in split_a for i in tr['image']])
    probe = fit_probe(variant(tr, 'ENC', 'inside')[in_a], tr['positive'][in_a], args.device)
    s_tr, s_va = probe(variant(tr, 'ENC', 'inside')), probe(variant(va, 'ENC', 'inside'))
    thr = float(np.quantile(s_va[~va['positive']], .99))
    cells = lambda d: np.array([cell_name(z, names[int(c)], ls) if c >= 0 else '' for z, c, ls in zip(d['size'], d['label'], d['long_side'])])  # noqa: E731
    c_tr, c_va = cells(tr), cells(va)
    sim_tr = np.array([sim_train.get(int(a), np.nan) for a in tr['ann']])
    sim_va = np.array([sim_val.get(int(a), np.nan) for a in va['ann']])
    groups = {'train B (all)': (tr, s_tr, ~in_a & tr['positive'], c_tr),
              'val seen': (va, s_va, va['positive'] & (sim_va >= .9), c_va),
              'val novel': (va, s_va, va['positive'] & (sim_va < .9), c_va)}
    report = {}
    for cell in ('512/aircraft', '256/tank<16', '256/bridge', '256/ship', '800/ship'):
        # contrast quartiles of the val novel targets of this cell define the bins
        ref = va['PIX'][groups['val novel'][2] & (c_va == cell), 3]
        edges = [-1e9, *np.quantile(ref, [.25, .5, .75]).tolist(), 1e9]
        rows = {}
        for b in range(4):
            row = {}
            for g, (d, sc, m, cc) in groups.items():
                mm = m & (cc == cell) & (d['PIX'][:, 3] >= edges[b]) & (d['PIX'][:, 3] < edges[b + 1])
                row[g] = dict(n=int(mm.sum()), recall=r3((sc[mm] > thr).mean() * 100) if mm.sum() >= 20 else None)
            rows[f'contrast {edges[b]:.2f}..{edges[b + 1]:.2f}'.replace('-1000000000.00', '-inf').replace('1000000000.00', 'inf')] = row
        report[cell] = rows
        print(cell, json.dumps(rows), flush=True)
    (args.out / 'contrast_matched.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('extract', 'analyse', 'matched'):
        p = sub.add_parser(name)
        if name == 'extract':
            p.add_argument('run', type=Path)
        p.add_argument('out', type=Path)
        p.add_argument('--d12', type=Path, required=True)
        p.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    {'extract': extract, 'analyse': analyse, 'matched': matched}[args.command](args)


if __name__ == '__main__':
    main()
