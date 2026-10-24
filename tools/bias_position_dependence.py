#!/usr/bin/env python3
"""
Decide whether the matched-box x offset is a translation bug or a scale bug.

bias_shift_search found that shifting every detection by dx = -0.75 source
pixels is worth +5.12 AP, and dy = 0. A constant offset has several possible
origins and only one of them is a bug we can fix once and keep:

  constant in x, independent of position   -> translation convention
                                              (e.g. a flip that maps
                                              x -> w - x instead of w - 1 - x)
  proportional to x                        -> scale convention
                                              (e.g. dividing by 255 vs 256)
  constant but only on one side of the
  image                                    -> flip-specific, because a flipped
                                              sample moves to the mirrored side

This reports dx_px in source pixels binned by the GT centre x, and binned by
its distance to the nearer edge, which separate those three cases.

Usage:
python tools/bias_position_dependence.py --gt .../val.json --dt predictions_b24.json
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
    ap.add_argument('--bins', type=int, default=8)
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    gt = json.load(open(args.gt))
    dets = json.load(open(args.dt))
    img_info = {im['id']: im for im in gt['images']}
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
        W = img_info.get(img_id, {}).get('width', 256)
        H = img_info.get(img_id, {}).get('height', 256)
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
            gx = g[0] + g[2] / 2
            gy = g[1] + g[3] / 2
            rows.append({
                'gx': gx, 'gy': gy, 'W': W, 'H': H,
                'dx': (p[0] + p[2] / 2) - gx,
                'dy': (p[1] + p[3] / 2) - gy,
            })

    if not rows:
        print('no matches')
        return

    lines = []
    def p(s):
        print(s)
        lines.append(s)

    dx_all = np.array([r['dx'] for r in rows])
    dy_all = np.array([r['dy'] for r in rows])
    p(f"n={len(rows)}  dx={dx_all.mean():+.3f} (sd {dx_all.std():.3f})  "
      f"dy={dy_all.mean():+.3f} (sd {dy_all.std():.3f})")

    # 1. dx vs x position
    gx = np.array([r['gx'] for r in rows])
    edges = np.quantile(gx, np.linspace(0, 1, args.bins + 1))
    edges[-1] += 1e-6
    p("\n== dx vs GT centre x ==")
    p(f"{'x range':16s} {'n':>6s} {'dx':>8s} {'sd':>7s} {'dx/gx':>8s}")
    for i in range(args.bins):
        sel = (gx >= edges[i]) & (gx < edges[i + 1])
        if sel.sum() < 20:
            continue
        dxs = dx_all[sel]
        p(f"{edges[i]:6.1f}-{edges[i+1]:6.1f} {sel.sum():6d} {dxs.mean():+8.3f} "
          f"{dxs.std():7.3f} {dxs.mean()/gx[sel].mean():+8.4f}")

    # 2. dx vs distance to the nearer horizontal edge
    dist = np.array([min(r['gx'], r['W'] - r['gx']) for r in rows])
    de = np.quantile(dist, np.linspace(0, 1, args.bins + 1))
    de[-1] += 1e-6
    p("\n== dx vs distance to nearer edge (flip test) ==")
    p(f"{'dist range':16s} {'n':>6s} {'dx':>8s} {'sd':>7s}")
    for i in range(args.bins):
        sel = (dist >= de[i]) & (dist < de[i + 1])
        if sel.sum() < 20:
            continue
        dxs = dx_all[sel]
        p(f"{de[i]:6.1f}-{de[i+1]:6.1f} {sel.sum():6d} {dxs.mean():+8.3f} {dxs.std():7.3f}")

    # 3. is a pure scale bug enough? fit dx = a*x + b
    A = np.stack([gx, np.ones_like(gx)], axis=1)
    coef, *_ = np.linalg.lstsq(A, dx_all, rcond=None)
    pred = A @ coef
    resid = dx_all - pred
    p(f"\nfitted dx = {coef[0]:+.5f}*x + {coef[1]:+.3f}   "
      f"residual sd {resid.std():.3f}")
    p(f"implied scale error {coef[0]*100:+.2f}%  "
      f"(1/255 vs 1/256 would be +{100*(1/255-1/256)/(1/256):.2f}%)")

    if args.output:
        with open(args.output, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f"\nwritten {args.output}")


if __name__ == '__main__':
    main()
