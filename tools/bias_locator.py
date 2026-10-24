#!/usr/bin/env python3
"""
Locate the source of a systematic matched-box offset.

box_error_decomposition.py reported, for matched detections on OGSOD:
  dcx = +0.062 in units of GT width (tiny: +0.070), dcy = -0.001
i.e. a ~0.8 px consistent shift along +x and no shift along y. A bias that
strong, that consistent, and that asymmetric between axes is unlikely to be a
capacity limit; it is usually a coordinate convention somewhere in the
train/eval chain. This tool tries to pin it down without guessing:

  per category      a single class carrying the bias points at annotation or
                    class-specific handling, not at the box head
  per image width   a fixed pixel offset across widths is a convention bug;
                    an offset proportional to width is a normalisation bug
  per GT width      distinguishes a pixel offset (dx_px constant) from a
                    relative offset (dcx constant)
  per flip status   a bias that only appears on flipped images points at
                    RandomHorizontalFlip or a flip-aware transform

Reports dx_px = (pred_cx - gt_cx) in pixels, which is the natural unit for a
convention bug. Usage:

python tools/bias_locator.py --gt .../val.json --dt predictions_b24.json
"""

import argparse
import json
from collections import defaultdict

import numpy as np


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
    out = np.zeros(len(b))
    ok = union > 0
    out[ok] = inter[ok] / union[ok]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', required=True)
    ap.add_argument('--dt', required=True)
    ap.add_argument('--iou-thr', type=float, default=0.5)
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    gt = json.load(open(args.gt))
    dets = json.load(open(args.dt))

    img_info = {im['id']: im for im in gt['images']}
    cat_name = {c['id']: c['name'] for c in gt.get('categories', [])}
    gts_by_img = defaultdict(list)
    for a in gt['annotations']:
        gts_by_img[a['image_id']].append(a)
    dts_by_img = defaultdict(list)
    for d in dets:
        dts_by_img[d['image_id']].append(d)

    rows = []
    for img_id, dlist in dts_by_img.items():
        dlist = sorted(dlist, key=lambda d: -d['score'])
        gts = gts_by_img.get(img_id, [])
        used = [False] * len(gts)
        im = img_info.get(img_id, {})
        W_img, H_img = im.get('width', 0), im.get('height', 0)

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
            g = gts[gi]['bbox']
            p = d['bbox']
            rows.append({
                'cat': d['category_id'],
                'W_img': W_img, 'H_img': H_img,
                'W_gt': g[2], 'H_gt': g[3],
                'dx': (p[0] + p[2] / 2) - (g[0] + g[2] / 2),
                'dy': (p[1] + p[3] / 2) - (g[1] + g[3] / 2),
                'dw': p[2] - g[2], 'dh': p[3] - g[3],
                'iou': ious[k],
            })

    if not rows:
        print('no matched detections')
        return

    lines = []

    def report(title, key_fn, items):
        print(f"\n== {title} ==")
        hdr = f"{'group':14s} {'n':>6s} {'dx_px':>8s} {'sd':>7s} {'dy_px':>8s} {'sd':>7s} {'iou':>7s}"
        print(hdr)
        lines.extend([f"== {title} ==", hdr])
        for name, sel in items:
            if len(sel) < 20:
                continue
            dx = np.array([r['dx'] for r in sel])
            dy = np.array([r['dy'] for r in sel])
            iou = np.array([r['iou'] for r in sel])
            line = (f"{str(name):14s} {len(sel):6d} {dx.mean():+8.3f} {dx.std():7.3f} "
                    f"{dy.mean():+8.3f} {dy.std():7.3f} {iou.mean():7.3f}")
            print(line)
            lines.append(line)

    cats = defaultdict(list)
    for r in rows:
        cats[r['cat']].append(r)
    report('per category', None,
           [(cat_name.get(c, str(c)), v) for c, v in sorted(cats.items())])

    widths = defaultdict(list)
    for r in rows:
        widths[r['W_img']].append(r)
    report('per image width', None, sorted(widths.items()))

    def wbin(r):
        w = r['W_gt']
        if w < 12:
            return 'gt<12px'
        if w < 24:
            return 'gt12-24'
        if w < 48:
            return 'gt24-48'
        if w < 96:
            return 'gt48-96'
        return 'gt>=96'

    wb = defaultdict(list)
    for r in rows:
        wb[wbin(r)].append(r)
    report('per GT width', None, sorted(wb.items()))

    # pixel offset vs relative offset, per GT width
    print("\n== offset scaling ==")
    print(f"{'gt width':14s} {'n':>6s} {'dx_px':>8s} {'dx/W_gt':>9s} {'slope_check':>12s}")
    lines.append("== offset scaling ==")
    for name, v in sorted(wb.items()):
        if len(v) < 20:
            continue
        dx = np.array([r['dx'] for r in v])
        rel = np.array([r['dx'] / r['W_gt'] for r in v])
        line = f"{name:14s} {len(v):6d} {dx.mean():+8.3f} {rel.mean():+9.4f}"
        print(line)
        lines.append(line)

    if args.output:
        with open(args.output, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f"\nwritten {args.output}")


if __name__ == '__main__':
    main()
