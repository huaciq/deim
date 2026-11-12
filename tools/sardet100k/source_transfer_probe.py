"""D16: is a class one thing across the merged sub-datasets, or one thing per source?

SARDet-100K merges several datasets; the same class name covers very different appearances (MSAR tank: 30-px ring on a dark
background; OGSOD tank: 6-px blob on a bright background). Three questions on the frozen box-interior features of D14
(memorisation_stage_probe.py, stages BB and ENC), probes fitted on training images A and evaluated on val:

  T1 transfer   A probe "class c of source s against the background of source s" is applied to class c of source s'.
                Within-source (s -> s) against cross-source (s -> s') tells whether the class evidence is shared.
  T2 readout    A probe fitted on one (source, class) cell against a probe fitted on all targets of all sources.
                If the cell probe is clearly better, a shared readout interferes and source-specific experts are supported.
  T3 direction  Cosine between the target directions (standardised mean of targets minus mean of background of the same
                source) of all cells. Same class / other source against other class / same source.

Source groups come from image width and file type: ogsod (256 png), 256jpg (MSAR and other 256 jpg), 224, 512, 800.
Reported: AUC against the random background boxes of the target's own source, and recall with 1% of them passing.

  python tools/sardet100k/source_transfer_probe.py STAGES_DIR OUT_JSON
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_probe import fit_mlp  # noqa: E402
from e1_report import r3  # noqa: E402


def auc(pos, neg):
    order = np.argsort(np.r_[pos, neg], kind='stable')
    rank = np.empty(len(order))
    rank[order] = np.arange(1, len(order) + 1)
    return float((rank[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stages', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--conditioning', action='store_true', help='run T4/T5 instead of T1-T3')
    args = parser.parse_args()
    from error_gallery import ROOT
    data = {s: json.load(open(ROOT / f'Annotations/{s}.json')) for s in ('train', 'val')}
    names = {c['id']: c['name'].lower() for c in data['train']['categories']}

    def group_of(image):
        w = image['width']
        if w == 256:
            return 'ogsod' if image['file_name'].endswith('.png') else '256jpg'
        return str(w) if w in (224, 512, 800) else 'misc'
    d = {}
    for split in ('train', 'val'):
        z = dict(np.load(args.stages / f'stages_{split}.npz'))
        g = {i['id']: group_of(i) for i in data[split]['images']}
        z['group'] = np.array([g[int(i)] for i in z['image']])
        z['cell'] = np.array([f"{a}/{names[int(c)]}" if c >= 0 else '' for a, c in zip(z['group'], z['label'])])
        d[split] = z
    tr, va = d['train'], d['val']
    perm = np.random.default_rng(0).permutation(np.unique(tr['image']))
    in_a = np.isin(tr['image'], perm[:len(perm) * 2 // 3])
    cells = [c for c in np.unique(tr['cell'][in_a & (tr['kind'] == 0)]) if ((tr['cell'] == c) & in_a).sum() >= 150 and (va['cell'] == c).sum() >= 100]
    print('cells', {c: (int(((tr['cell'] == c) & in_a).sum()), int((va['cell'] == c).sum())) for c in cells}, flush=True)
    report = dict(cells={c: dict(train_a=int(((tr['cell'] == c) & in_a).sum()), val=int((va['cell'] == c).sum())) for c in cells})

    def metrics(score, cell):
        g = cell.split('/')[0]
        pos = score[va['cell'] == cell]
        rand = score[(va['kind'] == 1) & (va['group'] == g)]
        return dict(auc=r3(auc(pos, rand) * 100), recall=r3((pos > np.quantile(rand, .99)).mean() * 100))

    groups = sorted(set(c.split('/')[0] for c in cells))
    onehot = lambda z: np.stack([(z['group'] == g) for g in groups], 1).astype(np.float32) * 4  # noqa: E731

    if args.conditioning:
        # T4: is the gain of the cell probes a matter of source or of class, and does a shared readout that is told the
        # source recover it? T5: can the source be told from the features of plain background?
        for stage in ('ENC', 'BB'):
            x_tr, x_va = tr[stage].astype(np.float32), va[stage].astype(np.float32)
            y = tr['kind'][in_a] == 0
            score = {'pooled': fit_mlp(x_tr[in_a], y, args.device)(x_va),
                     'pooled + source': fit_mlp(np.concatenate([x_tr, onehot(tr)], 1)[in_a], y, args.device)(np.concatenate([x_va, onehot(va)], 1))}
            expert = np.full(len(x_va), -1e9, np.float32)
            for g in groups:
                fit = in_a & (tr['group'] == g)
                sel = va['group'] == g
                expert[sel] = fit_mlp(x_tr[fit], tr['kind'][fit] == 0, args.device)(x_va[sel])
            score['one expert per source'] = expert
            table = {cell: {k: metrics(v, cell) for k, v in score.items()} for cell in cells}
            # One probe per class over all sources, and one per (source, class) cell, both against all backgrounds of
            # the sources involved: does a class shared across sources lose against a class split by source?
            for cls in sorted(set(c.split('/')[1] for c in cells)):
                members = [c for c in cells if c.split('/')[1] == cls]
                srcs = [c.split('/')[0] for c in members]
                neg = in_a & (tr['kind'] > 0) & np.isin(tr['group'], srcs)
                fit = (in_a & np.isin(tr['cell'], members)) | neg
                shared = fit_mlp(x_tr[fit], tr['kind'][fit] == 0, args.device)(x_va)
                told = fit_mlp(np.concatenate([x_tr, onehot(tr)], 1)[fit], tr['kind'][fit] == 0, args.device)(np.concatenate([x_va, onehot(va)], 1))
                for cell in members:
                    g = cell.split('/')[0]
                    own = (in_a & (tr['cell'] == cell)) | (in_a & (tr['kind'] > 0) & (tr['group'] == g))
                    table[cell]['class, all sources'] = metrics(shared, cell)
                    table[cell]['class, all sources + source'] = metrics(told, cell)
                    table[cell]['class, own source only'] = metrics(fit_mlp(x_tr[own], tr['kind'][own] == 0, args.device)(x_va), cell)
            for cell in cells:
                print(stage, 'conditioning', cell, {k: v['recall'] for k, v in table[cell].items()}, flush=True)
            # source from background features
            bg_a, bg_v = in_a & (tr['kind'] == 1) & np.isin(tr['group'], groups), (va['kind'] == 1) & np.isin(va['group'], groups)
            mean, std = x_tr[bg_a].mean(0), x_tr[bg_a].std(0) + 1e-6
            centre = np.stack([((x_tr[bg_a & (tr['group'] == g)] - mean) / std).mean(0) for g in groups])
            pred = (((x_va[bg_v] - mean) / std) @ centre.T - .5 * (centre ** 2).sum(1)).argmax(1)
            truth = np.array([groups.index(g) for g in va['group'][bg_v]])
            acc = {g: r3((pred[truth == k] == k).mean() * 100) for k, g in enumerate(groups)}
            print(stage, 'source from one background box (nearest class mean)', acc, flush=True)
            report[stage] = dict(conditioning=table, source_from_background=acc)
        args.out.write_text(json.dumps(report, indent=1))
        return

    for stage in ('ENC', 'BB'):
        x_tr, x_va = tr[stage].astype(np.float32), va[stage].astype(np.float32)
        out = dict(transfer={}, readout={}, direction={})
        pooled_fit = in_a
        pooled = fit_mlp(x_tr[pooled_fit], (tr['kind'][pooled_fit] == 0), args.device)(x_va)
        probes = {}
        for cell in cells:
            g = cell.split('/')[0]
            fit = in_a & ((tr['cell'] == cell) | ((tr['kind'] > 0) & (tr['group'] == g)))
            probes[cell] = fit_mlp(x_tr[fit], (tr['kind'][fit] == 0), args.device)(x_va)
            out['readout'][cell] = dict(pooled=metrics(pooled, cell), cell_probe=metrics(probes[cell], cell))
            print(stage, 'readout', cell, out['readout'][cell], flush=True)
        for target in cells:
            cls = target.split('/')[1]
            row = {source: metrics(probes[source], target) for source in cells if source.split('/')[1] == cls}
            if len(row) > 1:
                out['transfer'][target] = row
                print(stage, 'transfer to', target, {k: (v['auc'], v['recall']) for k, v in row.items()}, flush=True)
        mean, std = x_tr[in_a].mean(0), x_tr[in_a].std(0) + 1e-6
        direction = {}
        for cell in cells:
            g = cell.split('/')[0]
            v = ((x_tr[in_a & (tr['cell'] == cell)] - mean) / std).mean(0) - ((x_tr[in_a & (tr['kind'] == 1) & (tr['group'] == g)] - mean) / std).mean(0)
            direction[cell] = v / np.linalg.norm(v)
        out['direction'] = {a: {b: r3(float(direction[a] @ direction[b])) for b in cells} for a in cells}
        same_class = [out['direction'][a][b] for a in cells for b in cells if a < b and a.split('/')[1] == b.split('/')[1]]
        same_source = [out['direction'][a][b] for a in cells for b in cells if a < b and a.split('/')[0] == b.split('/')[0]]
        neither = [out['direction'][a][b] for a in cells for b in cells if a < b and a.split('/')[0] != b.split('/')[0] and a.split('/')[1] != b.split('/')[1]]
        out['direction_summary'] = dict(same_class_other_source=r3(np.mean(same_class)), same_source_other_class=r3(np.mean(same_source)), neither=r3(np.mean(neither)))
        print(stage, 'direction', out['direction_summary'], flush=True)
        report[stage] = out
    args.out.write_text(json.dumps(report, indent=1))
    print('written', args.out, flush=True)


if __name__ == '__main__':
    main()
