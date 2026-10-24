#!/usr/bin/env python3
"""
Measure the annotation noise floor of a COCO dataset.

box_error_decomposition found sd(w_pred/w_gt) = 0.104 on objects below 32^2 px.
Before attributing that to the model, check what the labels themselves can
support:

  integer_or_half   are box edges on integer pixels? then the label itself has
                    +-0.5 px quantisation, which for an 11 px object is 4.5%
                    of its width, i.e. a large share of the 10.4% observed
  edge_convention   which pixel-centre convention do the boxes follow
                    (x1 inclusive / x2 exclusive, or both inclusive)
  size_histogram    median object size and the share below one stride-8 cell
                    at the network input resolution
  duplicate_pairs   near-identical boxes on the same image: their disagreement
                    is a direct estimate of labelling variance

The last one is the only measurement of true annotation noise: if the same
object was annotated twice, the spread between the two boxes is the floor no
model can beat.

Usage:
python tools/annotation_precision.py --gt .../train.json --input-size 640
"""

import argparse
import json
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', required=True)
    ap.add_argument('--input-size', type=int, default=640,
                    help='network input resolution for the stride footprint')
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    data = json.load(open(args.gt))
    imgs = {im['id']: im for im in data['images']}
    anns = data['annotations']
    print(f"images={len(imgs)} annotations={len(anns)}")

    widths = np.array([im['width'] for im in imgs.values()], dtype=float)
    heights = np.array([im['height'] for im in imgs.values()], dtype=float)
    print(f"image size: width {widths.min():.0f}-{widths.max():.0f} "
          f"(mean {widths.mean():.1f}), height {heights.min():.0f}-{heights.max():.0f}")

    b = np.array([a['bbox'] for a in anns], dtype=np.float64)
    x, y, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    area = w * h

    lines = []

    def p(s):
        print(s)
        lines.append(s)

    p("\n== coordinate quantisation ==")
    for name, v in (('x', x), ('y', y), ('w', w), ('h', h)):
        frac = np.abs(v - np.round(v))
        p(f"{name:2s} all-integer={np.mean(frac < 1e-6)*100:6.2f}%  "
          f"half-integer={np.mean(np.abs(frac-0.5) < 1e-6)*100:6.2f}%  "
          f"max deviation from integer={frac.max():.4f}")

    p("\n== derived edge positions ==")
    x2, y2 = x + w, y + h
    for name, v in (('x1', x), ('x2', x2), ('y1', y), ('y2', y2)):
        p(f"{name} integer share={np.mean(np.abs(v-np.round(v)) < 1e-6)*100:6.2f}%")

    p("\n== object size ==")
    scale = args.input_size / np.median(widths)
    p(f"median image width {np.median(widths):.0f}; upsampled to input "
      f"{args.input_size} (x{scale:.3f})")
    for name, v in (('width', w), ('height', h)):
        p(f"{name:6s} median={np.median(v):6.2f} mean={v.mean():6.2f} "
          f"sd={v.std():6.2f}")
    p(f"area   median={np.median(area):8.1f} px^2  "
      f"below 32^2: {np.mean(area < 32**2)*100:5.1f}%  "
      f"below 12^2: {np.mean(area < 12**2)*100:5.1f}%")

    w_in = w * scale
    p(f"\nat input {args.input_size}, native widths become x{scale:.2f}:")
    p(f"  stride-8 cells per object: median {np.median(w_in)/8:.2f}, "
      f"5th pct {np.percentile(w_in,5)/8:.2f}")
    p(f"  stride-4 cells per object: median {np.median(w_in)/4:.2f}, "
      f"5th pct {np.percentile(w_in,5)/4:.2f}")
    p(f"  objects below one stride-8 cell: {np.mean(w_in < 8)*100:.1f}%")

    p("\n== label quantisation vs observed model error ==")
    p(f"integer-pixel labels imply +-0.5 px edge uncertainty;")
    med_w = np.median(w)
    p(f"  as a fraction of width, sd contribution = 0.5*sqrt(2)/median_w "
      f"= {0.5*np.sqrt(2)/med_w:.4f} (median width {med_w:.1f} px)")
    for sub, lo, hi in (('<12^2', 0, 12**2), ('12-32^2', 12**2, 32**2),
                        ('32-96^2', 32**2, 96**2)):
        sel = (area >= lo) & (area < hi)
        if sel.sum() < 10:
            continue
        mw = np.median(w[sel])
        floor = 0.5 * np.sqrt(2) / mw
        p(f"  {sub:9s} n={sel.sum():6d} median_w={mw:6.2f}px  "
          f"label floor sd_s ~ {floor:.4f}  "
          f"(observed tiny sd_s = 0.104)")

    d = defaultdict(list)
    for a in anns:
        d[a['image_id']].append(a)
    p("\n== near-duplicate boxes (labelling variance) ==")
    pairs = 0
    ratios = []
    for img_id, lst in d.items():
        if len(lst) < 2:
            continue
        for i in range(len(lst)):
            bi = lst[i]['bbox']
            for j in range(i + 1, len(lst)):
                bj = lst[j]['bbox']
                ci = (bi[0] + bi[2] / 2, bi[1] + bi[3] / 2)
                cj = (bj[0] + bj[2] / 2, bj[1] + bj[3] / 2)
                if abs(ci[0] - cj[0]) > 3 or abs(ci[1] - cj[1]) > 3:
                    continue
                if bi[2] <= 0 or bj[2] <= 0:
                    continue
                pairs += 1
                ratios.append(bi[2] / bj[2])
    p(f"pairs within 3 px of each other: {pairs}")
    if pairs:
        r = np.array([v for v in ratios if 0.5 < v < 2.0])
        p(f"width ratio among them: mean {r.mean():.3f} sd {r.std():.4f} "
          f"(n={len(r)})")
        p("a large sd here is a lower bound on annotation noise for near-identical objects")

    if args.output:
        with open(args.output, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f"\nwritten {args.output}")


if __name__ == '__main__':
    main()
