"""D4: class-wise NMS threshold sweep on the final detections of two runs, same thresholds for both. No training.

Input: the queries.npz written by `e1_followup.py collect` (last decoder layer, 300 queries x 6 classes, own EMA
weights) for each run. Detections are rebuilt with the configured postprocessor rule (sigmoid, top 300 over
query x class). NMS is applied per image and class to that 300-detection list (greedy by score, a detection is
removed when its IoU with a kept higher-scored detection of the same class exceeds the threshold). Boxes and scores
are untouched. Evaluation is the COCO rule of e1_followup (maxDets 100 per image and class, 101 recall points).

Usage (repository root; DIRs as written by e1_followup collect; OUT directly under independent_verify_20261106/):
  python nms_sweep.py BASE_DIR RUN_DIR --out OUT [--labels E000 E1] [--boot 200]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_followup import TYPES, Val, ap_arrays, ap_from, detections, evaluate, group_ap, load_run, paired_bootstrap  # noqa: E402
from e1_report import iou_xyxy, r3  # noqa: E402

THRESHOLDS = [None, .5, .55, .6, .65, .7, .75, .8]
CELLS = [('ogsod', 'tank'), ('ogsod', 'bridge')]


def nms(dets, thr):
    if thr is None:
        return dets
    out = {}
    for image_id, d in dets.items():
        keep = np.ones(len(d['score']), bool)
        for c in np.unique(d['label']):
            idx = np.flatnonzero(d['label'] == c)  # already in descending score order
            if len(idx) < 2:
                continue
            ious = iou_xyxy(d['box'][idx], d['box'][idx])
            removed = np.zeros(len(idx), bool)
            for x in range(len(idx)):
                if not removed[x]:
                    removed[x + 1:] |= ious[x, x + 1:] > thr
            keep[idx[removed]] = False
        out[image_id] = {k: v[keep] for k, v in d.items()}
    return out


def summarise(val, det, npos, names_to_id):
    images = sorted(val.info)
    arr = ap_arrays(det, npos, val, images)
    ap = group_ap(arr)[0]
    row = dict(AP=r3(np.mean(ap) * 100), AP50=r3(ap[0] * 100), AP75=r3(ap[5] * 100), detections_evaluated=int(len(det['score'])))
    for k, name in enumerate(TYPES):
        mk = det['kind'] == k
        row[f'{name}_ge005'] = int((mk & (det['score'] >= .05)).sum())
        row[f'{name}_ge03'] = int((mk & (det['score'] >= .3)).sum())
        row[f'{name}_ge05'] = int((mk & (det['score'] >= .5)).sum())
    tp = det['kind'] == 0
    row['tp_score_q10_q50_q90'] = [r3(np.quantile(det['score'][tp], q)) for q in (.1, .5, .9)]
    # TP@.75 count (detections matched at IoU .75) and TP@.5 that are not TP@.75.
    row['tp75_all'] = int(det['tp'][:, 5].sum())
    for g, cls in CELLS:
        ids = [i for i in images if val.group[i] == g]
        a = ap_arrays(det, npos, val, ids)
        c = names_to_id[cls]
        v = ap_from(a[c])[0]
        sel = (det['group'] == g) & (det['label'] == c)
        row[f'{g}/{cls}'] = dict(AP=r3(np.mean(v) * 100), AP50=r3(v[0] * 100), AP75=r3(v[5] * 100),
                                 tp_ge03=int((sel & tp & (det['score'] >= .3)).sum()),
                                 duplicate_ge03=int((sel & (det['kind'] == 1) & (det['score'] >= .3)).sum()),
                                 duplicate_ge005=int((sel & (det['kind'] == 1) & (det['score'] >= .05)).sum()),
                                 tp75=int((sel & det['tp'][:, 5]).sum()))
    return row, arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('base', type=Path)
    parser.add_argument('run', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--labels', nargs=2, default=['E000', 'E1'])
    parser.add_argument('--boot', type=int, default=200)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    val = Val(args.base)
    names_to_id = {v: k for k, v in val.names.items()}
    dets = {lab: detections(val, load_run(d)[0]) for lab, d in zip(args.labels, (args.base, args.run))}
    result = {}
    for thr in THRESHOLDS:
        key = 'none' if thr is None else f'{thr:.2f}'
        result[key] = {}
        arrs = []
        for lab in args.labels:
            det, npos = evaluate(val, nms(dets[lab], thr))
            row, arr = summarise(val, det, npos, names_to_id)
            result[key][lab] = row
            arrs.append(arr)
            print(key, lab, {k: row[k] for k in ('AP', 'AP50', 'AP75', 'duplicate_ge03', 'tp_ge03', 'tp_ge05')}, flush=True)
        result[key]['paired_bootstrap'] = paired_bootstrap(arrs[0], arrs[1], len(val.info), args.boot)
        print(key, 'bootstrap', result[key]['paired_bootstrap'], flush=True)
        (args.out / 'nms_sweep.json').write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print('written', args.out / 'nms_sweep.json', flush=True)


if __name__ == '__main__':
    main()
