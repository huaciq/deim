"""BCLQ design review: what is a per-query, per-class localisation-quality prediction worth, as a function of how it is
fused with the classification score and of how accurate it is? Zero-training check with a simulated quality head.
No training, no model change. Diagnostic only: the simulated head reads the GT.

Simulated head: Q[i, c] = clip(t[i, c] + noise, .02, .98), with t[i, c] = max IoU of query i's box with the class-c GTs
of the image (the BCLQ training target) and noise ~ N(0, sigma^2), independent per query and class. sigma = 0 is a
perfect head. Real heads make correlated errors on near-duplicate queries; the within-GT pairwise accuracy reached by
each sigma is therefore reported, and results should be read against that accuracy, not against sigma.

Fusions:
  add   z_final = z_cls + beta * logit(Q)                      (the BCLQ proposal)
  swap  per class, queries are grouped without GT: sorted by z_cls, the first ungrouped query is a seed and takes every
        ungrouped query whose box IoU with it is >= .5; inside a group the existing z_cls values are re-assigned in
        the order of Q. The multiset of scores is unchanged, so one-to-one duplicate suppression is kept by
        construction; only which box of a group carries which score changes.

Read-outs (all val, COCO rule of e1_followup): AP, AP50, AP75; duplicate / localisation / background false positives
that outrank the median true positive of their class; pairwise accuracy of Q and of z_cls on the competing pairs of
D7 (same GT, both IoU >= .5, at least one score >= .1, GT-weighted).

Reading fixed before the run:
  add   viable only if, at a sigma whose pairwise accuracy a trained head can plausibly reach, some beta gives
        AP >= +0.5 without AP50 falling by more than 0.5.
  swap  the pairwise accuracy at which AP first reaches +0.5 is the target a quality head must beat; compare it with
        the accuracy of z_cls (D7.3) and of every probe so far (none above z_cls outside OGSOD tank).

Usage (server, CPU; DIR holds queries.npz and lies directly under E000's independent_verify_20261106/):
  python quality_fusion_oracle.py DIR --out OUT --mode add --sigma 0.1 --beta 2
  python quality_fusion_oracle.py DIR --out OUT --mode swap --sigma 0.1
  python quality_fusion_oracle.py DIR --out OUT --mode none
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_followup import Val, ap_arrays, ap_from, detections, evaluate, group_ap  # noqa: E402
from e1_report import iou_xyxy, r3, sigmoid  # noqa: E402

CELLS = ['ogsod/tank', 'ogsod/bridge', 'other_512/aircraft', 'other_256/ship', 'other_800/ship', 'msar/ship', 'msar/tank']


def quality_targets(val, data):
    """t (N, 300, C): max IoU of each query box with the class-c GTs of its image."""
    n_class = data['logits'].shape[2]
    col = {c: k for k, c in enumerate(val.classes)}
    t = np.zeros(data['logits'].shape, np.float32)
    for n, image_id in enumerate(data['image_id'].tolist()):
        gt = val.gt[image_id]
        if not len(gt['label']):
            continue
        iou = iou_xyxy(data['boxes'][n].astype(np.float64), gt['norm'].astype(np.float64))
        for c in np.unique(gt['label']):
            t[n, :, col[int(c)]] = iou[:, gt['label'] == c].max(1)
    assert n_class == len(val.classes)
    return t


def swap(logits, boxes, q):
    out = logits.copy()
    for n in range(len(logits)):
        b = boxes[n].astype(np.float64)
        near = iou_xyxy(b, b) >= .5
        for c in range(logits.shape[2]):
            z = logits[n, :, c]
            free = np.ones(len(z), bool)
            for seed in np.argsort(-z, kind='stable'):
                if not free[seed]:
                    continue
                group = np.flatnonzero(near[seed] & free)
                free[group] = False
                if len(group) > 1:
                    out[n, group[np.argsort(-q[n, group, c], kind='stable')], c] = np.sort(z[group])[::-1]
    return out


def pair_accuracy(val, data, signal):
    """GT-weighted share of competing pairs (D7 rule) ordered correctly by signal (N, 300, C); per cell and overall."""
    col = {c: k for k, c in enumerate(val.classes)}
    acc = {}
    for n, image_id in enumerate(data['image_id'].tolist()):
        gt = val.gt[image_id]
        if not len(gt['label']):
            continue
        iou_all = iou_xyxy(data['boxes'][n].astype(np.float64), gt['norm'].astype(np.float64))
        for c in np.unique(gt['label']):
            idx = np.flatnonzero(gt['label'] == c)
            iou = iou_all[:, idx]
            owner, best = iou.argmax(1), iou.max(1)
            k = col[int(c)]
            score = sigmoid(data['logits'][n, :, k].astype(np.float64))
            cell = f'{val.group[image_id]}/{val.names[int(c)]}'
            for j in range(len(idx)):
                cand = np.flatnonzero((owner == j) & (best >= .5))
                if len(cand) < 2:
                    continue
                i, s, v = iou[cand, j], score[cand], signal[n, cand, k]
                a, b = np.triu_indices(len(cand), 1)
                keep = (np.maximum(s[a], s[b]) >= .1) & (i[a] != i[b])
                if not keep.any():
                    continue
                good = np.where(v[a] == v[b], .5, (v[a] > v[b]) == (i[a] > i[b]))[keep].mean()
                for key in (cell, 'all'):
                    acc.setdefault(key, []).append(good)
    return {k: r3(np.mean(v)) for k, v in acc.items() if k == 'all' or k in CELLS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=['none', 'add', 'swap'], required=True)
    parser.add_argument('--sigma', type=float, default=0.)
    parser.add_argument('--beta', type=float, default=0.)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    val = Val(args.dir)
    id_of = {v: k for k, v in val.names.items()}
    with np.load(args.dir / 'queries.npz') as z:
        data = {k: z[k] for k in z.files}
    t = quality_targets(val, data)
    rng = np.random.default_rng(int(round(args.sigma * 1000)))
    q = np.clip(t + rng.normal(0, args.sigma, t.shape).astype(np.float32) if args.sigma > 0 else t, .02, .98)
    report = dict(mode=args.mode, sigma=args.sigma, beta=args.beta)
    d = dict(data)
    if args.mode == 'add':
        d['logits'] = data['logits'] + args.beta * np.log(q / (1 - q))
    elif args.mode == 'swap':
        d['logits'] = swap(data['logits'], data['boxes'], q)
    if args.mode == 'none':
        report['pair_accuracy_cls'] = pair_accuracy(val, data, data['logits'])
    else:
        report['pair_accuracy_q'] = pair_accuracy(val, data, q)
    det, npos = evaluate(val, detections(val, d))
    all_images = sorted(val.info)
    arr = ap_arrays(det, npos, val, all_images)
    mean, per_class = group_ap(arr)
    m = lambda ap: dict(AP=r3(np.mean(ap) * 100), AP50=r3(ap[0] * 100), AP75=r3(ap[5] * 100))  # noqa: E731
    report['all_val'] = dict(m(mean), per_class={val.names[c]: m(v) for c, v in per_class.items()})
    report['cells'] = {}
    for cell in CELLS:
        g, c = cell.split('/')[0], id_of[cell.split('/')[1]]
        a = ap_arrays(det, npos, val, [i for i in all_images if val.group[i] == g])
        report['cells'][cell] = m(ap_from(a[c])[0])
    # false positives ranked above the median true positive of their class (score scale differs between fusions)
    above = {k: 0 for k in ('duplicate', 'localisation', 'background')}
    for c in val.classes:
        sel = det['label'] == c
        tp = sel & (det['kind'] == 0)
        if not tp.any():
            continue
        thr = np.median(det['score'][tp])
        for kind, name in ((1, 'duplicate'), (3, 'localisation'), (4, 'background')):
            above[name] += int((sel & (det['kind'] == kind) & (det['score'] > thr)).sum())
    report['fp_above_median_tp'] = above
    name = f'{args.mode}_s{args.sigma:g}_b{args.beta:g}.json'
    (args.out / name).write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(name, report['all_val']['AP'], report['all_val']['AP50'], report['all_val']['AP75'], above,
          report.get('pair_accuracy_q', report.get('pair_accuracy_cls', {})).get('all'), flush=True)


if __name__ == '__main__':
    main()
