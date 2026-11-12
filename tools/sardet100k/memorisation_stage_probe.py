"""D14: at which stage does the train-versus-val gap of targets without a near-duplicate open?

D13 (context_probe.py) left one unexplained number. A probe fitted on training images A separates the targets of training
images B from the detector's own false positives much better than it separates val targets, although neither B nor val was
used to fit the probe. For targets without a near-duplicate (sim < .9) the difference is 6-7 points on backbone features
and 13-17 points on encoder outputs. The detector was trained on B, so this difference is what the detector memorised
about individual training targets. This tool measures it stage by stage.

Stages (box-interior feature, 3x3 samples averaged, three pyramid levels concatenated):
  BB    backbone outputs
  PROJ  after the 1x1 input projections (256 channels each)
  AIFI  PROJ with the stride-32 level replaced by the transformer layer's output
  FPN   after the top-down path
  ENC   encoder outputs (after the bottom-up path)

Positives, random negatives and hard negatives (the run's own background false positives) are those of D13. Probes are
fitted on training images A with a linear model and with a small MLP, and thresholds come from val negatives.

Reported per stage: recall of train-B and val targets without a near-duplicate, the same with train B reweighted to the
cell composition of val ("matched"), and the gap in points and in logit units (logit, because recall levels differ a lot
between stages).

Reading, fixed before running (MLP probe, hard negatives passing 10%, matched gap in logit units):
  - a stage "opens the gap" if the gap grows by at least 0.3 across it in both runs and that growth is at least half of
    the growth from BB to ENC;
  - if the BB gap under the MLP is already at least two thirds of the ENC gap, the memorisation is in the backbone and
    D13's small backbone gap was a weak-readout effect of the linear probe;
  - otherwise the gap accumulates over stages and no single stage is the place to act.

  python tools/sardet100k/memorisation_stage_probe.py extract RUN OUT --d12 D12 --val-pred V.npz --train-pred T.npz
  python tools/sardet100k/memorisation_stage_probe.py analyse OUT --d12 D12
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_probe import fit_mlp, hard_negatives  # noqa: E402
from e1_report import r3  # noqa: E402
from generalisation_probe import EvalImages, cell_name, fit_probe, negative_boxes, roi_mean  # noqa: E402

STAGES = ['BB', 'PROJ', 'AIFI', 'FPN', 'ENC']


def extract(args):
    import torch
    from engine.core import YAMLConfig
    from error_gallery import ROOT
    chosen = set(np.load(args.d12 / 'similarity.npz')['train_images'].tolist())
    cfg = YAMLConfig(str(args.run / 'resolved_config.yml'))
    model = cfg.model
    model.load_state_dict(torch.load(args.run / 'best_stg2.pth', map_location='cpu', weights_only=False)['ema']['module'])
    model = model.to(args.device).eval()
    enc = model.encoder
    assert len(enc.input_proj) == 3 and len(enc.encoder) == 1 and list(enc.use_encoder_idx) == [2], 'unexpected encoder layout'
    grab = {}

    def keep(name):
        return lambda m, i, o: grab.__setitem__(name, o.detach())
    model.backbone.register_forward_hook(lambda m, i, o: grab.__setitem__('BB', [t.detach() for t in o]))
    for i, m in enumerate(enc.input_proj):
        m.register_forward_hook(keep(f'PROJ{i}'))
    enc.encoder[0].register_forward_hook(keep('AIFI'))         # (B, h*w, C) of the stride-32 level
    enc.lateral_convs[0].register_forward_hook(keep('LAT5'))   # stride 32 after the lateral conv
    enc.fpn_blocks[0].register_forward_hook(keep('FPN4'))      # stride 16 after the top-down fusion
    enc.fpn_blocks[1].register_forward_hook(keep('FPN3'))      # stride 8 after the top-down fusion
    enc.register_forward_hook(lambda m, i, o: grab.__setitem__('ENC', [t.detach() for t in (o[0] if isinstance(o, tuple) else o)]))

    def maps_of(k):
        p5 = grab['PROJ2']
        aifi = grab['AIFI'].permute(0, 2, 1).reshape(p5.shape)
        return {'BB': [f[k] for f in grab['BB']], 'PROJ': [grab['PROJ0'][k], grab['PROJ1'][k], p5[k]], 'AIFI5': [aifi[k]],
                'FPN': [grab['FPN3'][k], grab['FPN4'][k], grab['LAT5'][k]], 'ENC': [f[k] for f in grab['ENC']]}

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
                model(batch.to(args.device))
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
                    for stage, maps in maps_of(k).items():
                        rows[stage].append(torch.cat([roi_mean(f.float(), boxes) for f in maps], 1).half().cpu().numpy())
                    rows['kind'].append(np.r_[np.zeros(len(gt), np.int8), np.ones(len(rand), np.int8), np.full(len(hard), 2, np.int8)])
                    rows['ann'].append(np.r_[[a['id'] for a in anns], np.full(len(rand) + len(hard), -1)])
                    rows['label'].append(np.r_[[a['category_id'] for a in anns], np.full(len(rand) + len(hard), -1)])
                    rows['image'].append(np.full(len(boxes), image['id']))
                    rows['size'].append(np.full(len(boxes), image['width']))
                    rows['long_side'].append(np.r_[[max(a['bbox'][2], a['bbox'][3]) for a in anns], np.full(len(rand) + len(hard), -1.)])
        out = {k: np.concatenate(v) for k, v in rows.items()}
        np.savez(args.out / f'stages_{split}.npz', **out)
        print('written', args.out / f'stages_{split}.npz', {n: int((out['kind'] == c).sum()) for c, n in enumerate(('pos', 'random', 'hard'))},
              {s: out[s].shape[1] for s in ('BB', 'PROJ', 'AIFI5', 'FPN', 'ENC')}, flush=True)


def feature(d, stage, level=None):
    if stage == 'AIFI':
        x = np.concatenate([d['PROJ'][:, :512], d['AIFI5']], 1)
    else:
        x = d[stage]
    if level is not None:
        x = x[:, 256 * level:256 * (level + 1)]
    return x.astype(np.float32)


def logit(p):
    p = np.clip(p / 100, .005, .995)
    return float(np.log(p / (1 - p)))


def gap_table(score, pops):
    """Recall of the positives of train B and val at thresholds taken from val negatives; novel = sim < .9."""
    va = pops['val']['d']
    thresholds = {'hard 10%': np.quantile(score['val'][va['kind'] == 2], .90), 'random 0.1%': np.quantile(score['val'][va['kind'] == 1], .999)}
    out = {}
    for name, thr in thresholds.items():
        rec, cells = {}, {}
        for k, p in pops.items():
            pos = p['m'] & (p['d']['kind'] == 0)
            hit = score[k] > thr
            novel = pos & (p['sim'] < .9)
            rec[k] = dict(seen=hit[pos & (p['sim'] >= .9)].mean() * 100, novel=hit[novel].mean() * 100)
            cells[k] = {c: (hit[novel & (p['cell'] == c)].mean() * 100, int((novel & (p['cell'] == c)).sum())) for c in np.unique(p['cell'][novel])}
        common = [c for c in cells['val'] if c in cells['train B'] and min(cells['val'][c][1], cells['train B'][c][1]) >= 20]
        weight = np.array([cells['val'][c][1] for c in common], np.float64)
        matched = {k: float((weight * np.array([cells[k][c][0] for c in common])).sum() / weight.sum()) for k in pops}
        out[name] = dict(
            train_seen=r3(rec['train B']['seen']), val_seen=r3(rec['val']['seen']), train_novel=r3(rec['train B']['novel']), val_novel=r3(rec['val']['novel']),
            train_novel_matched=r3(matched['train B']), val_novel_matched=r3(matched['val']),
            gap_points=r3(matched['train B'] - matched['val']), gap_logit=r3(logit(matched['train B']) - logit(matched['val'])),
            seen_gap_logit=r3(logit(rec['train B']['seen']) - logit(rec['val']['seen'])),
            cells={c: dict(train=r3(cells['train B'][c][0]), val=r3(cells['val'][c][0]), n_train=cells['train B'][c][1], n_val=cells['val'][c][1]) for c in common})
    return out


def analyse(args):
    from error_gallery import ROOT
    names = {c['id']: c['name'] for c in json.load(open(ROOT / 'Annotations/val.json'))['categories']}
    sim = np.load(args.d12 / 'similarity.npz')
    sim_val = dict(zip(sim['val_ann'].tolist(), sim['val_sim'].tolist()))
    sim_train = dict(zip(sim['train_ann'].tolist(), sim['train_sim'].tolist()))
    tr, va = (dict(np.load(args.out / f'stages_{s}.npz')) for s in ('train', 'val'))
    perm = np.random.default_rng(0).permutation(np.unique(tr['image']))
    split_a = set(perm[:len(perm) * 2 // 3].tolist())
    in_a = np.array([i in split_a for i in tr['image']])
    cells = lambda d: np.array([cell_name(s, names[int(c)], ls) if c >= 0 else '' for s, c, ls in zip(d['size'], d['label'], d['long_side'])])  # noqa: E731
    pops = {'train B': dict(d=tr, m=~in_a, sim=np.array([sim_train.get(int(a), np.nan) for a in tr['ann']]), cell=cells(tr)),
            'val': dict(d=va, m=np.ones(len(va['ann']), bool), sim=np.array([sim_val.get(int(a), np.nan) for a in va['ann']]), cell=cells(va))}
    y = tr['kind'][in_a] == 0
    report = dict(counts=dict(train_a={n: int((tr['kind'][in_a] == c).sum()) for c, n in enumerate(('pos', 'random', 'hard'))},
                              train_b_novel=int(((~in_a) & (tr['kind'] == 0) & (pops['train B']['sim'] < .9)).sum()),
                              val_novel=int(((va['kind'] == 0) & (pops['val']['sim'] < .9)).sum())), linear={}, mlp={}, mlp_levels={})

    def run(fit, stage, level=None):
        probe = fit(feature(tr, stage, level)[in_a], y, args.device)
        return gap_table({k: probe(feature(q['d'], stage, level)) for k, q in pops.items()}, pops)

    def show(tag, name, res):
        for thr in ('hard 10%', 'random 0.1%'):
            t = res[thr]
            print(f"{tag:6s} {name:8s} {thr:11s} seen {t['train_seen']:5.1f}/{t['val_seen']:5.1f}  novel {t['train_novel']:5.1f}/{t['val_novel']:5.1f}  "
                  f"matched {t['train_novel_matched']:5.1f}/{t['val_novel_matched']:5.1f}  gap {t['gap_points']:+5.1f} pts {t['gap_logit']:+.2f} logit  "
                  f"(seen {t['seen_gap_logit']:+.2f})  512/aircraft {t['cells'].get('512/aircraft', {}).get('train')}/{t['cells'].get('512/aircraft', {}).get('val')}", flush=True)

    for stage in STAGES:
        report['linear'][stage] = run(fit_probe, stage)
        show('linear', stage, report['linear'][stage])
    for stage in STAGES:
        report['mlp'][stage] = run(fit_mlp, stage)
        show('mlp', stage, report['mlp'][stage])
    for stage in STAGES[1:]:
        for level, tag in enumerate(('P3', 'P4', 'P5')):
            if stage == 'AIFI' and level < 2:
                continue        # identical to PROJ on these levels
            report['mlp_levels'][f'{stage}[{tag}]'] = run(fit_mlp, stage, level)
            show('mlp', f'{stage}[{tag}]', report['mlp_levels'][f'{stage}[{tag}]'])
    (args.out / 'stage_probe.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print('written', args.out / 'stage_probe.json', flush=True)


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
