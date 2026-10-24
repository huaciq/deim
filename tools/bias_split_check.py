#!/usr/bin/env python3
"""
Test whether train and val annotations share a coordinate convention.

Reasoning: the inference chain (resize, flip, normalisation, postprocessing)
has been verified exact to 1e-4, yet the trained model's boxes sit +0.79 px to
the right of the val annotations, with dy = -0.02 (i.e. no y offset). A model
cannot invent a constant one-axis shift, so either it learned one from the
training targets, or the two annotation files disagree.

This measures the same statistic on TRAIN images with the trained model:

  dx ~ 0    on train, dx ~ 0.79 on val  ->  the two annotation files use
                                            different conventions. The fix is
                                            to re-export annotations, not code.
  dx ~ 0.79 on both                     ->  the shift is in the model or in a
                                            shared part of the pipeline.

The run uses the val transform on train images, so geometry is identical and
only the labels differ. Only N images are used, which is enough: the effect is
0.68 sd with thousands of samples.

Usage:
python tools/bias_split_check.py \
    -c configs/deim_dfine/deim_hgnetv2_s_ogsod_120e_1gpu_noaug_b24.yml \
    -r "$RUN/best_stg1.pth" \
    --split-file .../RTDETR_COCO/train.json \
    --img-folder /data2/zxy/sar/datasets/OGSOD-1.0/sar \
    --limit 600 --output bias_train.json
"""

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import yaml
import numpy as np
from tqdm import tqdm

from engine.core import YAMLConfig


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


def build_cfg(config_path, ann_file, img_folder):
    """Write an override config next to the original.

    __include__ paths are resolved relative to the file that contains them
    (engine/core/yaml_utils.py:42), so a temp file in /tmp cannot find the
    original's siblings. Keeping the override in the same directory leaves the
    relative includes intact.
    """
    with open(config_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # __include__ is resolved relative to the file that holds it
    # (engine/core/yaml_utils.py:42), and this override lives elsewhere.
    # Rewrite the includes to absolute paths so they still resolve.
    if '__include__' in cfg:
        base_dir = os.path.dirname(os.path.abspath(config_path))
        includes = cfg['__include__']
        if isinstance(includes, str):
            includes = [includes]
        cfg['__include__'] = [
            inc if os.path.isabs(inc) else os.path.join(base_dir, inc)
            for inc in includes
        ]

    cfg.setdefault('val_dataloader', {}).setdefault('dataset', {})
    cfg['val_dataloader']['dataset']['ann_file'] = ann_file
    cfg['val_dataloader']['dataset']['img_folder'] = img_folder
    cfg['val_dataloader']['total_batch_size'] = 8
    cfg['val_dataloader']['num_workers'] = 4
    # Drop paths that would let this temporary config write anywhere.
    cfg.pop('output_dir', None)
    cfg.pop('summary_dir', None)
    tmp = os.path.join(tempfile.gettempdir(), 'bias_split_cfg.yml')
    with open(tmp, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    return tmp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-r', '--resume', required=True)
    ap.add_argument('--split-file', required=True)
    ap.add_argument('--img-folder', required=True)
    ap.add_argument('--limit', type=int, default=600)
    ap.add_argument('--iou-thr', type=float, default=0.5)
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    tmp_cfg = build_cfg(args.config, args.split_file, args.img_folder)
    cfg = YAMLConfig(tmp_cfg, resume=args.resume)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = cfg.model.to(device)
    ckpt = torch.load(args.resume, map_location='cpu')
    state = ckpt['ema']['module'] if ckpt.get('ema') else ckpt['model']
    model.load_state_dict(state)
    model.eval()
    print(f"loaded {args.resume}")

    gt_data = json.load(open(args.split_file))
    gts_by_img = defaultdict(list)
    for a in gt_data['annotations']:
        gts_by_img[a['image_id']].append(a)

    loader = cfg.val_dataloader
    preds = defaultdict(list)
    seen = 0
    with torch.no_grad():
        for samples, targets in tqdm(loader, total=min(args.limit, len(loader))):
            samples = samples.to(device)
            outputs = model(samples)
            pb = outputs['pred_boxes']
            pl = outputs['pred_logits']
            for i in range(pb.shape[0]):
                img_id = targets[i]['image_id'].item()
                orig = targets[i]['orig_size']
                scores = pl[i].sigmoid()
                sc, lb = scores.max(dim=-1)
                keep = sc > 0.3
                seen += 1
                if keep.sum() == 0:
                    continue
                boxes = pb[i][keep]
                cx, cy, w, h = boxes.unbind(-1)
                xyxy = torch.stack([
                    (cx - 0.5 * w) * orig[1], (cy - 0.5 * h) * orig[0],
                    (cx + 0.5 * w) * orig[1], (cy + 0.5 * h) * orig[0]], dim=-1)
                bb = torch.cat([xyxy[:, :2], (xyxy[:, 2:] - xyxy[:, :2])], dim=-1).cpu().numpy()
                for b, s, l in zip(bb, sc[keep].cpu().numpy(), lb[keep].cpu().numpy()):
                    preds[img_id].append((float(s), int(l), b))
            if seen >= args.limit:
                break

    dxs, dys = [], []
    for img_id, lst in preds.items():
        lst.sort(key=lambda r: -r[0])
        gts = gts_by_img.get(img_id, [])
        used = [False] * len(gts)
        for s, l, b in lst:
            cand = [k for k, g in enumerate(gts) if g['category_id'] == l]
            if not cand:
                continue
            ious = iou_1toN(b, [gts[k]['bbox'] for k in cand])
            j = int(np.argmax(ious))
            gk = cand[j]
            if ious[j] < args.iou_thr or used[gk]:
                continue
            used[gk] = True
            g = gts[gk]['bbox']
            dxs.append((b[0] + b[2] / 2) - (g[0] + g[2] / 2))
            dys.append((b[1] + b[3] / 2) - (g[1] + g[3] / 2))

    dxs = np.array(dxs)
    dys = np.array(dys)
    if len(dxs) == 0:
        print("no matches")
        return
    print(f"\nimages={seen}  matches={len(dxs)}")
    print(f"dx = {dxs.mean():+.3f} px (sd {dxs.std():.3f})")
    print(f"dy = {dys.mean():+.3f} px (sd {dys.std():.3f})")
    print("\ncompare with val: dx=+0.792  dy=-0.020")
    if abs(dxs.mean()) < 0.3:
        print("=> train dx ~ 0, val dx ~ 0.79: the two annotation files disagree.")
    else:
        print("=> train dx ~ val dx: the shift is in the model or the shared pipeline.")

    if args.output:
        out = {'images': seen, 'matches': len(dxs),
               'dx_mean': float(dxs.mean()), 'dx_sd': float(dxs.std()),
               'dy_mean': float(dys.mean()), 'dy_sd': float(dys.std()),
               'split_file': args.split_file}
        json.dump(out, open(args.output, 'w'), indent=2)
        print(f"written {args.output}")


if __name__ == '__main__':
    main()
