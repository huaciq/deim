"""D7.3: what is the localisation-quality information of a probe signal worth in AP? GT-aware swap oracle. Diagnostic
upper bound for "let the score read information the network already has" - it uses GT to form the candidate sets and
to train the rankers, so it is not a deployable method. No training of the detector, no change to it.

Inputs, both written by one run of `quality_signal_probe.py extract --all-cells` (one CPU forward pass per val image):
probe_features.pt (candidates of every GT: queries whose highest-IoU same-class GT is this GT, IoU >= .5, at least two
per GT) and queries.npz (last decoder layer, 300 queries x 6 classes, every val image). A candidate is addressed by its
query index; its box and logit are checked against queries.npz, and a GT that fails the check is left untouched and
counted. The queries.npz of `e1_followup.py collect` cannot be used here: it comes from a GPU forward pass whose
encoder top-300 differs from the CPU pass at the margin, so query indices do not correspond (about half match).

Swap: within one GT's candidates, the class-c logits are permuted so that their order follows the signal (the candidate
with the highest signal gets the highest of the existing logits, and so on). The multiset of scores, every box, every
other query and every other class logit stay as they are; detections are then rebuilt with the postprocessor rule
(sigmoid, top 300 over query x class) and evaluated with the COCO rule of e1_followup.

Signals: S1_final (the logit itself: reproduces the baseline), vector signals of the probe as out-of-fold linear
pairwise rankers (same pairs, folds, weights and L2 search as D7.1/D7.2, trained per cell), and `iou` (the true IoU:
the ceiling of any reordering inside these candidate sets under this rule).

Usage (server, CPU; PROBE_DIR and OUT directly under E000's independent_verify_20261106/):
  python quality_swap_oracle.py PROBE_DIR/probe_features.pt PROBE_DIR --out OUT [--cells ogsod/tank ...] [--boot 200]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_followup import Val, ap_arrays, ap_from, detections, evaluate, group_ap, paired_bootstrap  # noqa: E402
from e1_report import r3  # noqa: E402
from quality_signal_probe import add_joint_features, oof_vector, probe_setup  # noqa: E402

SIGNALS = ['S1_final', 'S0+S1', 'S4_h1', 'S0+S1+S4_h1', 'iou']
TRAINED = ['S0+S1', 'S4_h1', 'S0+S1+S4_h1']


def locate(records, data, cls):
    """Check that every probe candidate is the query of the same index in queries.npz. Sets r['q'] and r['ok']."""
    row = {int(i): n for n, i in enumerate(data['image_id'])}
    for r in records:
        n = row[int(r['image'])]
        c = r['S0'][:, :4].astype(np.float64)
        box = np.concatenate([c[:, :2] - c[:, 2:] / 2, c[:, :2] + c[:, 2:] / 2], 1)
        r['q'] = np.asarray(r['query'])
        r['ok'] = bool(np.abs(box - data['boxes'][n][r['q']]).max() < 1e-5 and np.allclose(data['logits'][n, r['q'], cls], r['S1_final'], atol=1e-4))


def swap(logits, row_of, records, signal, cls):
    """logits: (images, 300, classes) of the images in row_of, changed in place."""
    for k, r in enumerate(records):
        if not r['ok']:
            continue
        n = row_of[int(r['image'])]
        order = np.argsort(-np.asarray(signal[k], np.float64), kind='stable')
        logits[n, r['q'][order], cls] = np.sort(logits[n, r['q'], cls])[::-1]


def ranking_stats(records, signal):
    top = [int(np.argmax(signal[k])) for k, r in enumerate(records) if r['ok']]
    iou = [r['iou'] for r in records if r['ok']]
    top_iou = np.array([v[t] for v, t in zip(iou, top)])
    return dict(top_is_best=r3(np.mean([t == int(np.argmax(v)) for v, t in zip(iou, top)])), top_iou_median=r3(np.median(top_iou)),
                top_iou_ge075=r3(np.mean(top_iou >= .75)), best_iou_median=r3(np.median([v.max() for v in iou])),
                best_iou_ge075=r3(np.mean([v.max() >= .75 for v in iou])))


def metrics(ap):
    return dict(AP=r3(np.mean(ap) * 100), AP50=r3(ap[0] * 100), AP75=r3(ap[5] * 100))


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('features', type=Path)
    parser.add_argument('followup', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cells', nargs='*')
    parser.add_argument('--boot', type=int, default=200)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--signals', nargs='*', help='trained signals ("+"-joined record keys); S1_final and iou are always added')
    parser.add_argument('--name', default='swap_oracle.json')
    args = parser.parse_args()
    trained = args.signals or TRAINED
    signal_names = ['S1_final'] + trained + ['iou']
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    val = Val(args.followup)
    id_of = {v: k for k, v in val.names.items()}
    with np.load(args.followup / 'queries.npz') as z:
        data = {k: z[k] for k in z.files}
    saved = torch.load(args.features, weights_only=False)
    cells = {}
    for cell in sorted({r['cell'] for r in saved['records']}):
        if args.cells and cell not in args.cells:
            continue
        records = [r for r in saved['records'] if r['cell'] == cell]
        pairs, per_gt, weights, image_of_pair, folds = probe_setup(records)
        if len(pairs) < 50 or len(folds) < 10:
            print('skipped', cell, 'pairs', len(pairs), 'images', len(folds), flush=True)
            continue
        cells[cell] = dict(records=records, setup=(pairs, weights, folds, image_of_pair))
    class_of = {cell: id_of[cell.split('/')[1]] for cell in cells}
    report = dict(meta=dict(features=str(args.features), followup=str(args.followup), probe=saved['meta']), cells={}, all_val={})
    signals = {}
    for cell, c in cells.items():
        locate(c['records'], data, class_of[cell])
        ok = [r['ok'] for r in c['records']]
        add_joint_features(c['records'], trained)
        pairs, weights, folds, image_of_pair = c['setup']
        sig = {'S1_final': {k: r['S1_final'] for k, r in enumerate(c['records'])}, 'iou': {k: r['iou'] for k, r in enumerate(c['records'])}}
        for name in trained:
            sig[name] = oof_vector(c['records'], pairs, weights, name, folds, image_of_pair)
        signals[cell] = sig
        report['cells'][cell] = dict(gts_with_candidates=len(ok), gts_located=int(np.sum(ok)),
                                     pairs=len(pairs), ranking={s: ranking_stats(c['records'], sig[s]) for s in signal_names}, ap={})
        print(cell, 'gts', len(ok), 'located', int(np.sum(ok)), flush=True)

    # Baseline on all val; every variant re-evaluates only the images that have swapped GTs.
    all_images = sorted(val.info)
    base_dets = detections(val, data)
    base_det, npos = evaluate(val, base_dets)
    changed = sorted({int(r['image']) for c in cells.values() for r in c['records'] if r['ok']})
    row_all = {int(i): n for n, i in enumerate(data['image_id'])}
    rows = np.array([row_all[i] for i in changed])
    row_of = {i: n for n, i in enumerate(changed)}
    keep = ~np.isin(base_det['image'], changed)
    group_images = {g: [i for i in all_images if val.group[i] == g] for g in {cell.split('/')[0] for cell in cells}}

    def tables(det):
        by_group = {g: ap_arrays(det, npos, val, ids) for g, ids in group_images.items()}
        return ap_arrays(det, npos, val, all_images), {cell: {class_of[cell]: by_group[cell.split('/')[0]][class_of[cell]]} for cell in cells}

    base_all, base_cell = tables(base_det)
    for name in signal_names:
        logits = data['logits'][rows].copy()
        for cell, c in cells.items():
            swap(logits, row_of, c['records'], signals[cell][name], class_of[cell])
        sub = dict(image_id=np.array(changed), logits=logits, boxes=data['boxes'][rows])
        det_changed, _ = evaluate(val, detections(val, sub))
        det = {k: np.concatenate([base_det[k][keep], det_changed[k]]) for k in base_det}
        arr_all, arr_cell = tables(det)
        mean, per_class = group_ap(arr_all)
        boot = 0 if name == 'S1_final' else args.boot  # S1_final only checks that the swap reproduces the baseline
        report['all_val'][name] = dict(metrics(mean), per_class={val.names[c]: metrics(v) for c, v in per_class.items()})
        if boot:
            report['all_val'][name]['paired_bootstrap_vs_baseline'] = paired_bootstrap(base_all, arr_all, len(all_images), boot)
        for cell in cells:
            c = class_of[cell]
            n_group = len(group_images[cell.split('/')[0]])
            report['cells'][cell]['ap'][name] = metrics(ap_from(arr_cell[cell][c])[0])
            if boot:
                report['cells'][cell]['ap'][name]['paired_bootstrap_vs_baseline'] = paired_bootstrap(base_cell[cell], arr_cell[cell], n_group, boot)
        print(name, 'all val', {k: report['all_val'][name][k] for k in ('AP', 'AP50', 'AP75')},
              {cell: [report['cells'][cell]['ap'][name][k] for k in ('AP', 'AP50', 'AP75')] for cell in cells}, flush=True)
        (args.out / args.name).write_text(json.dumps(report, indent=1, ensure_ascii=False))
    mean, per_class = group_ap(base_all)
    report['all_val']['baseline'] = dict(metrics(mean), per_class={val.names[c]: metrics(v) for c, v in per_class.items()})
    for cell in cells:
        report['cells'][cell]['ap']['baseline'] = metrics(ap_from(base_cell[cell][class_of[cell]])[0])
    (args.out / args.name).write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print('baseline', report['all_val']['baseline'], {cell: report['cells'][cell]['ap']['baseline'] for cell in cells}, flush=True)
    print('written', args.out / args.name, flush=True)


if __name__ == '__main__':
    main()
