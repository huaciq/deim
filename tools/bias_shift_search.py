#!/usr/bin/env python3
"""
Measure how much AP a systematic box offset costs, by brute force.

bias_locator reports dx ~ +0.79 px across every size bin and every category,
with dy ~ 0 on all of them. Whether that is worth chasing depends on how many
AP points it actually costs, so this applies the offset to every detection and
re-evaluates, rather than reasoning about it.

For each (dx, dy) in a grid, shift every box in COCO output space, evaluate,
and report AP/AP50/AP75/APs. Also reports the per-detection optimal shift
distribution, so a spread-out optimum means "this is noise, do not fit it"
while a sharp optimum means "there is a real constant offset", which is a
convention bug and not something to learn.

Usage:
python tools/bias_shift_search.py \
    --gt .../val.json --dt predictions_b24.json \
    --range 2.0 --step 0.25
"""

import argparse
import io
import json
import contextlib

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate(coco_gt, dets):
    if not dets:
        return None
    coco_dt = coco_gt.loadRes(dets)
    ev = COCOeval(coco_gt, coco_dt, 'bbox')
    ev.evaluate()
    ev.accumulate()
    with contextlib.redirect_stdout(io.StringIO()):
        ev.summarize()
    return ev.stats


def shift(dets, dx, dy):
    out = []
    for d in dets:
        x, y, w, h = d['bbox']
        out.append({**d, 'bbox': [x + dx, y + dy, w, h]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', required=True)
    ap.add_argument('--dt', required=True)
    ap.add_argument('--range', type=float, default=2.0, help='search +-range pixels')
    ap.add_argument('--step', type=float, default=0.25)
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    coco_gt = COCO(args.gt)
    dets = json.load(open(args.dt))
    print(f"{len(dets)} detections")

    n = int(round(args.range / args.step))
    grid = [i * args.step for i in range(-n, n + 1)]

    base = evaluate(coco_gt, dets)
    print(f"\nbase       AP={base[0]*100:6.2f} AP50={base[1]*100:6.2f} "
          f"AP75={base[2]*100:6.2f} APs={base[3]*100:6.2f}")

    lines = [f"base {base[0]*100:.2f} {base[1]*100:.2f} {base[2]*100:.2f} {base[3]*100:.2f}"]

    print("\n== dx sweep (dy=0) ==")
    print(f"{'dx':>6s} {'AP':>7s} {'AP50':>7s} {'AP75':>7s} {'APs':>7s} {'dAP':>7s}")
    best = (-1, None)
    for dx in grid:
        s = evaluate(coco_gt, shift(dets, dx, 0.0))
        if s is None:
            continue
        line = (f"{dx:+6.2f} {s[0]*100:7.2f} {s[1]*100:7.2f} {s[2]*100:7.2f} "
                f"{s[3]*100:7.2f} {(s[0]-base[0])*100:+7.2f}")
        print(line)
        lines.append(line)
        if s[0] > best[0]:
            best = (s[0], dx, s)

    print(f"\nbest dx={best[1]:+.2f}  AP={best[2][0]*100:.2f} "
          f"(+{(best[2][0]-base[0])*100:.2f})  AP75={best[2][2]*100:.2f} "
          f"(+{(best[2][2]-base[2])*100:.2f})  APs={best[2][3]*100:.2f} "
          f"(+{(best[2][3]-base[3])*100:.2f})")
    lines.append(f"best_dx {best[1]:+.2f} AP {best[2][0]*100:.2f} AP75 {best[2][2]*100:.2f} APs {best[2][3]*100:.2f}")

    print("\n== dy sweep (dx=best) ==")
    print(f"{'dy':>6s} {'AP':>7s} {'AP75':>7s} {'dAP':>7s}")
    besty = (-1, None)
    for dy in grid:
        s = evaluate(coco_gt, shift(dets, best[1], dy))
        if s is None:
            continue
        line = f"{dy:+6.2f} {s[0]*100:7.2f} {s[2]*100:7.2f} {(s[0]-base[0])*100:+7.2f}"
        print(line)
        lines.append(line)
        if s[0] > besty[0]:
            besty = (s[0], dy, s)
    print(f"\nbest dy={besty[1]:+.2f}  AP={besty[2][0]*100:.2f} "
          f"(+{(besty[2][0]-base[0])*100:.2f})  "
          f"AP75={besty[2][2]*100:.2f}  APs={besty[2][3]*100:.2f}")

    if args.output:
        with open(args.output, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f"\nwritten {args.output}")


if __name__ == '__main__':
    main()
