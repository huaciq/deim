#!/usr/bin/env python3
"""
Decompose the box error of matched detections into center error and size error.

The ceiling analysis says localization precision is the whole gap on OGSOD
(AP 59.4 -> 92.4 when matched boxes are replaced by GT boxes), while TP/FP
separation is saturated (+1.05). A ceiling does not say *which* property of the
box is wrong, and the fix is different for each:

  center error dominates -> sub-pixel localization of the object
  size error dominates   -> scale estimation from insufficient evidence
  systematic bias        -> a calibration, not a new module
  random scatter         -> only more resolution helps

For every matched (detection, GT) pair this reports, per GT size bin:

  dcx, dcy      center offset normalised by the GT width/height
  s, t          w_pred/w_gt and h_pred/h_gt (1.0 = perfect)
  IoU           actual overlap
  IoU_c         GT size with the predicted center  -> "if only the center were fixed"
  IoU_s         GT center with the predicted size  -> "if only the size were fixed"

IoU_c >> IoU and IoU_s ~ IoU  =>  size is wrong.
IoU_s >> IoU and IoU_c ~ IoU  =>  center is wrong.
Both >> IoU                   =>  both, independently.

Usage:
python tools/box_error_decomposition.py \
    --gt /data2/zxy/sar/datasets/OGSOD-1.0/sar/RTDETR_COCO/val.json \
    --dt predictions_b24.json \
    --iou-thr 0.5
"""

import argparse
import json
from collections import defaultdict

import numpy as np

BINS = [('tiny', 0, 32 ** 2), ('small', 32 ** 2, 96 ** 2),
        ('medium', 96 ** 2, 256 ** 2), ('large', 256 ** 2, float('inf'))]


def iou_1toN(box, boxes):
    if len(boxes) == 0:
        return np.zeros(0)
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    bx1, by1 = box[0], box[1]
    bx2, by2 = box[0] + box[2], box[1] + box[3]
    gx1, gy1, gx2, gy2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(bx2, gx2) - np.maximum(bx1, gx1), 0, None)
    ih = np.clip(np.minimum(by2, gy2) - np.maximum(by1, gy1), 0, None)
    inter = iw * ih
    union = box[2] * box[3] + b[:, 2] * b[:, 3] - inter
    out = np.zeros(len(b), dtype=np.float64)
    ok = union > 0
    out[ok] = inter[ok] / union[ok]
    return out


def to_cxcywh(b):
    return np.array([b[0] + b[2] / 2.0, b[1] + b[3] / 2.0, b[2], b[3]],
                    dtype=np.float64)


def iou_cxcywh(a, b):
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', required=True)
    ap.add_argument('--dt', required=True)
    ap.add_argument('--iou-thr', type=float, default=0.5)
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    gt_data = json.load(open(args.gt))
    dets = json.load(open(args.dt))
    print(f"GT images={len(gt_data.get('images', []))} "
          f"anns={len(gt_data.get('annotations', []))} dets={len(dets)}")

    gts_by_img = defaultdict(list)
    for a in gt_data['annotations']:
        gts_by_img[a['image_id']].append(a)
    dts_by_img = defaultdict(list)
    for d in dets:
        dts_by_img[d['image_id']].append(d)

    rows = []
    for img_id, dlist in dts_by_img.items():
        dlist = sorted(dlist, key=lambda d: -d['score'])
        gts = gts_by_img.get(img_id, [])
        used = [False] * len(gts)

        for d in dlist:
            cand = [i for i, g in enumerate(gts) if g['category_id'] == d['category_id']]
            if not cand:
                continue
            ious = iou_1toN(d['bbox'], [gts[i]['bbox'] for i in cand])
            k = int(np.argmax(ious))
            gi = cand[k]
            if ious[k] < args.iou_thr or used[gi]:
                continue
            used[gi] = True

            g = to_cxcywh(gts[gi]['bbox'])
            p = to_cxcywh(d['bbox'])
            if g[2] <= 0 or g[3] <= 0:
                continue
            iou = iou_cxcywh(p, g)
            iou_c = iou_cxcywh(np.array([p[0], p[1], g[2], g[3]]), g)
            iou_s = iou_cxcywh(np.array([g[0], g[1], p[2], p[3]]), g)

            rows.append({
                'area': g[2] * g[3],
                'dcx': (p[0] - g[0]) / g[2],
                'dcy': (p[1] - g[1]) / g[3],
                's': p[2] / g[2],
                't': p[3] / g[3],
                'iou': iou, 'iou_c': iou_c, 'iou_s': iou_s,
            })

    if not rows:
        print("no matched detections")
        return

    hdr = (f"{'bin':8s} {'n':>6s} {'IoU':>7s} {'IoU_c':>7s} {'IoU_s':>7s} "
           f"{'dcx':>7s} {'sdcx':>7s} {'dcy':>7s} {'sdcy':>7s} "
           f"{'s':>7s} {'sds':>7s} {'t':>7s} {'sdt':>7s}")
    lines = [hdr]
    print('\n' + hdr)

    def emit(name, sel):
        if not sel:
            return
        a = np.array([[r['iou'], r['iou_c'], r['iou_s'], r['dcx'], r['dcy'],
                       r['s'], r['t']] for r in sel])
        line = (f"{name:8s} {len(sel):6d} {a[:,0].mean():7.3f} {a[:,1].mean():7.3f} "
                f"{a[:,2].mean():7.3f} {a[:,3].mean():+7.3f} {a[:,3].std():7.3f} "
                f"{a[:,4].mean():+7.3f} {a[:,4].std():7.3f} "
                f"{a[:,5].mean():7.3f} {a[:,5].std():7.3f} "
                f"{a[:,6].mean():7.3f} {a[:,6].std():7.3f}")
        print(line)
        lines.append(line)

    emit('all', rows)
    for name, lo, hi in BINS:
        emit(name, [r for r in rows if lo <= r['area'] < hi])

    print()
    for name, lo, hi in [('all', 0, float('inf'))] + BINS:
        sel = [r for r in rows if lo <= r['area'] < hi]
        if not sel:
            continue
        a = np.array([[r['iou'], r['iou_c'], r['iou_s'], r['s'], r['t']] for r in sel])
        # iou_c = GT size + predicted centre  -> what fixing the SIZE buys.
        # iou_s = GT centre + predicted size   -> what fixing the CENTRE buys.
        gain_size = a[:, 1].mean() - a[:, 0].mean()
        gain_centre = a[:, 2].mean() - a[:, 0].mean()
        scale_bias = np.sqrt(a[:, 3] * a[:, 4]).mean() - 1.0
        aspect = (a[:, 3] / a[:, 4]).mean() - 1.0
        verdict = ('CENTRE' if gain_centre > 2 * gain_size
                   else 'SIZE' if gain_size > 2 * gain_centre
                   else 'both')
        line = (f"{name:8s} n={len(sel):5d}  gain_if_size_fixed={gain_size:+.3f}  "
                f"gain_if_centre_fixed={gain_centre:+.3f}  -> {verdict}"
                f"   scale_bias={scale_bias:+.3f}  aspect_bias={aspect:+.3f}")
        print(line)
        lines.append(line)

    if args.output:
        with open(args.output, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f"\nwritten {args.output}")


if __name__ == '__main__':
    main()
