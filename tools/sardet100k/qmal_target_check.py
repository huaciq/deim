"""Zero-training check for Q-MAL: on E000 val true positives, does the edge-tolerant target stop depending on target size?

Usage: python tools/sardet100k/qmal_target_check.py RUN_DIR [--tolerance 1.0]

For each GT, the best-scoring same-class prediction with IoU >= .5 is taken. Boxes are mapped to the 640x640 network input,
where the training tolerance is defined. Reported per GT-size bin: median score, MAL target IoU^1.5 and Q-MAL target.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from engine.deim.deim_criterion import edge_tolerant_iou  # noqa: E402

ROOT = Path('/root/zxy/TCRA-SAR/datasets/SARDet_100K')
BINS = [(0, 8), (8, 16), (16, 32), (32, 64), (64, 1 << 20)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--tolerance', type=float, default=1.)
    parser.add_argument('--gamma', type=float, default=1.5)
    args = parser.parse_args()
    saved = np.load(args.run / 'independent_verify_20261106/val_predictions.npz')
    PI, PC, PS, PB = saved['image_id'], saved['label'].astype(int), saved['score'], saved['box']
    val = json.load(open(ROOT / 'Annotations/val.json'))
    sizes = {i['id']: (i['width'], i['height']) for i in val['images']}
    gts = defaultdict(list)
    for a in val['annotations']:
        x, y, w, h = a['bbox']
        gts[a['image_id']].append((a['category_id'], np.array([x, y, x + w, y + h], np.float64)))
    order = np.argsort(PI, kind='stable')
    vals, first = np.unique(PI[order], return_index=True)
    slices = dict(zip(vals.tolist(), np.split(order, first[1:])))
    rows = []
    for image_id, items in gts.items():
        pos = slices.get(image_id)
        if pos is None:
            continue
        W, H = sizes[image_id]
        scale = np.array([640 / W, 640 / H] * 2)
        for c, g in items:
            q = pos[PC[pos] == c]
            if not len(q):
                continue
            B = PB[q].astype(np.float64)
            iw = np.clip(np.minimum(B[:, 2], g[2]) - np.maximum(B[:, 0], g[0]), 0, None)
            ih = np.clip(np.minimum(B[:, 3], g[3]) - np.maximum(B[:, 1], g[1]), 0, None)
            inter = iw * ih
            iou = inter / np.maximum((B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) + (g[2] - g[0]) * (g[3] - g[1]) - inter, 1e-9)
            j = int(np.argmax(PS[q] * (iou >= .5)))
            if iou[j] < .5:
                continue
            rows.append((max(g[2] - g[0], g[3] - g[1]), np.abs(B[j] - g).mean(), PS[q][j], iou[j], B[j] * scale, g * scale))
    long_side, error, score, iou = (np.array([r[i] for r in rows]) for i in range(4))
    to_cxcywh = lambda b: torch.tensor(np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1) / 640)
    pred, gt = np.stack([r[4] for r in rows]), np.stack([r[5] for r in rows])
    q_iou = edge_tolerant_iou(to_cxcywh(pred), to_cxcywh(gt), args.tolerance, (640, 640)).numpy()
    report = {}
    for name, mask in [('edge error 0-1 px', error < 1), ('edge error 1-2 px', (error >= 1) & (error < 2)), ('all TPs', error >= 0)]:
        report[name] = {}
        for lo, hi in BINS:
            k = mask & (long_side >= lo) & (long_side < hi)
            if k.sum() < 20:
                continue
            report[name][f'{lo}-{hi if hi < 1 << 20 else "inf"}'] = dict(
                n=int(k.sum()), score=round(float(np.median(score[k])), 3),
                mal_target=round(float(np.median(iou[k] ** args.gamma)), 3),
                qmal_target=round(float(np.median(q_iou[k] ** args.gamma)), 3))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
