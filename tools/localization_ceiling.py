#!/usr/bin/env python3
"""
Decompose the AP gap of a COCO-format detector into box precision,
TP/FP separation and localization-quality ranking.

Given val annotations and a predictions json produced by
tools/generate_predictions.py, reports:

  AP            baseline, must reproduce the number in log.txt
  AP_GTBOX      every detection that matches a GT at IoU>=thr keeps its score
                but takes the GT box. Ceiling reachable by pure box-precision
                improvement, with nothing else changed.
  AP_TPFP       detections that match a GT keep their box and score, all other
                detections get score 0. Ceiling reachable by perfect
                background/foreground separation.
  AP_IQS        boxes untouched, score multiplied by the IoU with the best GT.
                Ceiling reachable by ranking detections by their real
                localization quality.
  AP_BOTH       GT boxes for matched detections and score 0 for the rest.

Usage:
python tools/localization_ceiling.py \
    --gt /data2/zxy/sar/datasets/OGSOD-1.0/sar/RTDETR_COCO/val.json \
    --dt predictions_baseline.json \
    --iou-thr 0.5 \
    --output localization_ceiling.txt
"""

import argparse
import json
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def xywh_to_xyxy(b):
    x, y, w, h = b
    return np.array([x, y, x + w, y + h], dtype=np.float64)


def iou_matrix(a, b):
    """IoU between two sets of xywh boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    A, B = xywh_to_xyxy(a[0])[None], np.stack([xywh_to_xyxy(x) for x in b])
    # vectorised
    ax1, ay1, ax2, ay2 = A[:, 0], A[:, 1], A[:, 2], A[:, 3]
    bx1, by1, bx2, by2 = B[:, 0], B[:, 1], B[:, 2], B[:, 3]
    aw, ah = ax2 - ax1, ay2 - ay1
    bw, bh = bx2 - bx1, by2 - by1
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    union = (aw * ah) + (bw * bh) - inter
    out = np.zeros((len(a), len(b)))
    ok = union > 0
    out[ok] = inter[ok] / union[ok]
    if len(a) == 1:
        return out
    return out


def evaluate(coco_gt, dets, label):
    if len(dets) == 0:
        print(f"{label:12s} no detections")
        return None
    coco_dt = coco_gt.loadRes(dets)
    ev = COCOeval(coco_gt, coco_dt, 'bbox')
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    s = ev.stats
    print(f"{label:12s} AP={s[0]*100:6.2f}  AP50={s[1]*100:6.2f}  "
          f"AP75={s[2]*100:6.2f}  APs={s[3]*100:6.2f}  "
          f"APm={s[4]*100:6.2f}  APl={s[5]*100:6.2f}  "
          f"AR100={s[8]*100:6.2f}")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', required=True)
    ap.add_argument('--dt', required=True)
    ap.add_argument('--iou-thr', type=float, default=0.5)
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    coco_gt = COCO(args.gt)
    dets = json.load(open(args.dt))
    print(f"loaded {len(dets)} detections")

    gts_by_img = defaultdict(list)
    for ann in coco_gt.datasets[0]['annotations']:
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
            same_cls = [i for i, g in enumerate(gts) if g['category_id'] == d['category_id']]
            best_i, best_iou = -1, 0.0
            for i in same_cls:
                iou = iou_matrix([d['bbox']], [gts[i]['bbox']])[0, 0]
                if iou > best_iou:
                    best_i, best_iou = i, iou

            hit = best_iou >= args.iou_thr and not used[best_i] if best_i >= 0 else False
            if hit:
                used[best_i] = True

            base = {'image_id': img_id, 'category_id': d['category_id'], 'score': d['score']}

            gtbox.append({**base, 'bbox': list(gts[best_i]['bbox']) if hit else d['bbox']})
            tpfp.append({**base, 'bbox': d['bbox'], 'score': d['score'] if hit else 0.0})
            iqs.append({**base, 'bbox': d['bbox'],
                        'score': d['score'] * max(best_iou, 0.0)})
            both.append({**base, 'bbox': list(gts[best_i]['bbox']) if hit else d['bbox'],
                         'score': d['score'] if hit else 0.0})

    lines = []
    for name, dd in [('AP', dets), ('AP_GTBOX', gtbox), ('AP_TPFP', tpfp),
                     ('AP_IQS', iqs), ('AP_BOTH', both)]:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            s = evaluate(coco_gt, dd, name)
        print(buf.getvalue().strip())
        if s is not None:
            lines.append(f"{name:12s} {s[0]*100:.2f} {s[1]*100:.2f} {s[2]*100:.2f} "
                         f"{s[3]*100:.2f} {s[4]*100:.2f} {s[5]*100:.2f} {s[8]*100:.2f}")

    header = f"{'variant':12s} {'AP':>6s} {'AP50':>6s} {'AP75':>6s} {'APs':>6s} {'APm':>6s} {'APl':>6s} {'AR100':>6s}"
    print('\n' + header)
    for l in lines:
        print(l)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(header + '\n')
            f.write('\n'.join(lines) + '\n')
        print(f"\nwritten {args.output}")


if __name__ == '__main__':
    main()
