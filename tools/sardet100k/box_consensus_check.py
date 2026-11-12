"""D8.3: do the near-duplicate queries of one object carry complementary localisation information? Zero-training check,
no GT in the method, no training, no model change.

One-to-one training keeps one query per object and pushes the others to score 0; their boxes are discarded even
though the union set keeps supervising them. If the box errors of these queries are partly independent, a consensus
of their boxes is more accurate than the box of the winner, and the choice of the winner matters less.

Method: every query's box is replaced by a weighted mean of the boxes of all queries of the same image whose IoU with
it is >= THR (itself included). Scores, classes and the top-300 rule are untouched, so duplicate suppression is not
affected. Weights: `uniform`, or `kernel` = exp(-(1 - IoU)^2 / 0.025).
Evaluation: the COCO rule of e1_followup on all val, paired image bootstrap against the unchanged boxes.

Usage (server, CPU; DIR holds a queries.npz and lies directly under E000's independent_verify_20261106/):
  python box_consensus_check.py DIR --out OUT --name NAME.json --mode uniform --thr 0.5 [--boot 200]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_followup import Val, ap_arrays, ap_from, detections, evaluate, group_ap, paired_bootstrap  # noqa: E402
from e1_report import iou_xyxy, r3  # noqa: E402

CELLS = ['ogsod/tank', 'ogsod/bridge', 'ogsod/harbor', 'other_512/aircraft', 'other_512/car', 'other_256/ship',
         'other_800/ship', 'msar/ship', 'msar/tank']


def consensus(boxes, mode, thr):
    out = np.empty_like(boxes)
    neighbours = []
    for n in range(len(boxes)):
        b = boxes[n].astype(np.float64)
        iou = iou_xyxy(b, b)
        np.fill_diagonal(iou, 1.)
        w = (iou >= thr).astype(np.float64)
        if mode == 'kernel':
            w = w * np.exp(-(1 - iou) ** 2 / .025)
        out[n] = (w @ b) / w.sum(1, keepdims=True)
        neighbours.append((iou >= thr).sum(1) - 1)
    return out, np.concatenate(neighbours)


def metrics(ap):
    return dict(AP=r3(np.mean(ap) * 100), AP50=r3(ap[0] * 100), AP75=r3(ap[5] * 100), AP90=r3(ap[8] * 100))


def tables(val, det, npos, all_images, group_images, id_of):
    arr_all = ap_arrays(det, npos, val, all_images)
    by_group = {g: ap_arrays(det, npos, val, ids) for g, ids in group_images.items()}
    arr_cell = {cell: {id_of[cell.split('/')[1]]: by_group[cell.split('/')[0]][id_of[cell.split('/')[1]]]} for cell in CELLS}
    return arr_all, arr_cell


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--mode', choices=['uniform', 'kernel'], required=True)
    parser.add_argument('--thr', type=float, required=True)
    parser.add_argument('--boot', type=int, default=200)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    val = Val(args.dir)
    id_of = {v: k for k, v in val.names.items()}
    with np.load(args.dir / 'queries.npz') as z:
        data = {k: z[k] for k in z.files}
    all_images = sorted(val.info)
    group_images = {g: [i for i in all_images if val.group[i] == g] for g in {c.split('/')[0] for c in CELLS}}
    report = dict(source=str(args.dir), mode=args.mode, thr=args.thr)
    arrs = {}
    for label in ('baseline', 'consensus'):
        d = dict(data)
        if label == 'consensus':
            d['boxes'], nb = consensus(data['boxes'], args.mode, args.thr)
            report['neighbours_per_query'] = dict(mean=r3(nb.mean()), share_with_any=r3((nb > 0).mean()))
        det, npos = evaluate(val, detections(val, d))
        arr_all, arr_cell = tables(val, det, npos, all_images, group_images, id_of)
        arrs[label] = (arr_all, arr_cell)
        mean, per_class = group_ap(arr_all)
        report[label] = dict(all_val=dict(metrics(mean), per_class={val.names[c]: metrics(v) for c, v in per_class.items()}),
                             cells={cell: metrics(ap_from(list(arr_cell[cell].values())[0])[0]) for cell in CELLS})
        print(label, report[label]['all_val'], flush=True)
    report['paired_bootstrap'] = dict(all_val=paired_bootstrap(arrs['baseline'][0], arrs['consensus'][0], len(all_images), args.boot),
                                      cells={cell: paired_bootstrap(arrs['baseline'][1][cell], arrs['consensus'][1][cell],
                                                                    len(group_images[cell.split('/')[0]]), args.boot) for cell in CELLS})
    (args.out / args.name).write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print('written', args.out / args.name, report['paired_bootstrap']['all_val'], flush=True)


if __name__ == '__main__':
    main()
