#!/usr/bin/env python3
"""
Decompose the AP gap of a COCO-format detector into its upper bounds.

Given val annotations and a predictions json (from tools/generate_predictions.py),
re-evaluates the same detections under controlled corruptions. All variants keep
the detection set fixed, so any AP difference is attributable to exactly one
factor:

  AP         reference, should reproduce the number in log.txt
  AP_GTBOX   every detection matched at IoU>=thr takes the GT box instead of its
             own. Upper bound reachable by improving box precision alone.
  AP_TPFP    matched detections keep box and score, every other detection gets
             score 0. Upper bound reachable by improving foreground/background
             separation alone.
  AP_IQS     boxes untouched, score multiplied by the IoU against the best GT.
             Upper bound reachable by ranking detections by true box quality.
  AP_BOTH    GT box for matched detections and score 0 for the rest.

Reading the result:
  AP_GTBOX >> AP   -> localization precision is the bottleneck
  AP_TPFP  >> AP   -> false positives are the bottleneck
  AP_IQS   >> AP   -> the score does not track localization quality

Usage:
python tools/localization_ceiling.py \
    --gt /data2/zxy/sar/datasets/OGSOD-1.0/sar/RTDETR_COCO/val.json \
    --dt predictions_baseline.json \
    --iou-thr 0.5 \
    --output localization_ceiling.txt
"""

import argparse
import io
import json
import contextlib
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def iou_1toN(box, boxes):
    """IoU of one xywh box against N xywh boxes."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.float64)
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    bx1, by1 = box[0], box[1]
    bx2, by2 = box[0] + box[2], box[1] + box[3]
    gx1, gy1 = boxes[:, 0], boxes[:, 1]
    gx2, gy2 = boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]
    iw = np.clip(np.minimum(bx2, gx2) - np.maximum(bx1, gx1), 0, None)
    ih = np.clip(np.minimum(by2, gy2) - np.maximum(by1, gy1), 0, None)
    inter = iw * ih
    union = box[2] * box[3] + boxes[:, 2] * boxes[:, 3] - inter
    out = np.zeros(len(boxes), dtype=np.float64)
    ok = union > 0
    out[ok] = inter[ok] / union[ok]
    return out


def evaluate(coco_gt, dets, name):
    if not dets:
        print(f"{name:12s} (no detections)")
        return None
    coco_dt = coco_gt.loadRes(dets)
    ev = COCOeval(coco_gt, coco_dt, 'bbox')
    ev.evaluate()
    ev.accumulate()
    with contextlib.redirect_stdout(io.StringIO()):
        ev.summarize()
    s = ev.stats
    print(f"{name:12s} AP={s[0]*100:6.2f}  AP50={s[1]*100:6.2f}  "
          f"AP75={s[2]*100:6.2f}  APs={s[3]*100:6.2f}  "
          f"APm={s[4]*100:6.2f}  APl={s[5]*100:6.2f}  "
          f"AR50={s[6]*100:6.2f}  AR75={s[7]*100:6.2f}  AR100={s[8]*100:6.2f}")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', required=True)
    ap.add_argument('--dt', required=True)
    ap.add_argument('--iou-thr', type=float, default=0.5,
                    help='threshold at which a detection counts as matched')
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    coco_gt = COCO(args.gt)
    dets = json.load(open(args.dt))
    print(f"loaded {len(dets)} detections from {args.dt}")

    gts_by_img = defaultdict(list)
    for ann in coco_gt.dataset['annotations']:
        gts_by_img[ann['image_id']].append(ann)
    dts_by_img = defaultdict(list)
    for d in dets:
        dts_by_img[d['image_id']].append(d)

    gtbox, tpfp, iqs, both = [], [], [], []

    for img_id, dlist in dts_by_img.items():
        dlist = sorted(dlist, key=lambda d: -d['score'])
        gts = gts_by_img.get(img_id, [])
        used = [False] * len(gts)

        for d in dlist:
            cand = [i for i, g in enumerate(gts)
                    if g['category_id'] == d['category_id']]
            best_i, best_iou = -1, 0.0
            if cand:
                ious = iou_1toN(d['bbox'], [gts[i]['bbox'] for i in cand])
                k = int(np.argmax(ious))
                if ious[k] > best_iou:
                    best_i, best_iou = cand[k], float(ious[k])

            hit = best_i >= 0 and best_iou >= args.iou_thr and not used[best_i]
            if hit:
                used[best_i] = True

            base = {'image_id': img_id, 'category_id': d['category_id'],
                    'score': float(d['score'])}
            gt_bbox = list(gts[best_i]['bbox']) if hit else list(d['bbox'])

            gtbox.append({**base, 'bbox': gt_bbox})
            tpfp.append({**base, 'bbox': list(d['bbox']),
                         'score': float(d['score']) if hit else 0.0})
            iqs.append({**base, 'bbox': list(d['bbox']),
                        'score': float(d['score']) * max(best_iou, 0.0)})
            both.append({**base, 'bbox': gt_bbox,
                         'score': float(d['score']) if hit else 0.0})

    variants = [('AP', dets), ('AP_GTBOX', gtbox), ('AP_TPFP', tpfp),
                ('AP_IQS', iqs), ('AP_BOTH', both)]
    print()
    rows, ref = [], None
    for name, dd in variants:
        s = evaluate(coco_gt, dd, name)
        if s is None:
            continue
        if name == 'AP':
            ref = s[0] * 100
        rows.append((name, s))

    print()
    hdr = (f"{'variant':12s} {'AP':>7s} {'AP50':>7s} {'AP75':>7s} {'APs':>7s} "
           f"{'APm':>7s} {'APl':>7s} {'AR100':>7s} {'dAP':>7s}")
    print(hdr)
    out_lines = [hdr]
    for name, s in rows:
        line = (f"{name:12s} {s[0]*100:7.2f} {s[1]*100:7.2f} {s[2]*100:7.2f} "
                f"{s[3]*100:7.2f} {s[4]*100:7.2f} {s[5]*100:7.2f} "
                f"{s[8]*100:7.2f} {(s[0]*100 - ref):+7.2f}")
        print(line)
        out_lines.append(line)

    if args.output:
        with open(args.output, 'w') as f:
            f.write('\n'.join(out_lines) + '\n')
        print(f"\nwritten {args.output}")


if __name__ == '__main__':
    main()
