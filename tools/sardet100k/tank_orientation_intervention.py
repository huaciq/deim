"""Two checks after tank_orientation_recall.py. No training; weights and annotations unchanged.

Usage: python tools/sardet100k/tank_orientation_intervention.py RUN OUT [--device cuda:0]

1. Where does image content sit relative to the box, per image frame? Tells a constant offset in every frame from an error
   in flipped copies only: under a constant offset, two mirrored copies show the same content offset in their own frames.
2. Infer on vertically flipped val tiles and split them by whether train holds that orientation. Horizontal flips are part of
   training, so a vertical flip and a 180 degree rotation are the same condition. If only tiles absent from train lose recall,
   the difference is familiarity with these pixels; if all lose it, the flip itself hurts. MSAR is the control.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from error_gallery import ROOT, iou_matrix, match_image  # noqa: E402
from tank_dihedral_layout import orient_array, thumb_array, unit  # noqa: E402
from tank_feature_diagnosis import SCORE, IOU, TANK, Hooks, load_model, load_val, xyxy  # noqa: E402

REACH = 12
SIZE = 256
SAME = .98


def content_offset(path, boxes):
    """Brightness centroid of the stacked targets of one image, relative to the box centre, in that image's own frame."""
    values = np.asarray(Image.open(path).convert('L'), np.float32)
    stack, count = np.zeros((2 * REACH + 1, 2 * REACH + 1)), 0
    for x1, y1, x2, y2 in boxes:
        cx, cy = int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))
        if REACH <= cx < SIZE - REACH and REACH <= cy < SIZE - REACH:
            crop = values[cy - REACH:cy + REACH + 1, cx - REACH:cx + REACH + 1]
            stack += (crop - np.median(crop)) / (crop.std() + 1e-6)
            count += 1
    if count < 3:
        return None
    stack /= count
    grid = np.arange(-REACH, REACH + 1)
    keep = np.abs(grid) <= 6
    core = stack[np.ix_(keep, keep)]
    weights = np.clip(core - np.percentile(core, 70), 0, None)
    if weights.sum() <= 0:
        return None
    return float((weights.sum(0) * grid[keep]).sum() / weights.sum()), float((weights.sum(1) * grid[keep]).sum() / weights.sum())


def main():
    import torch
    import torchvision.transforms.functional as F
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    report = {}
    rows = json.load(open(args.out / 'gt_rows.json'))
    orientation = json.load(open(args.out / 'orientation_recall.json'))
    by = defaultdict(list)
    for r in rows:
        if r['native_gw'] > 0 and r['native_gh'] > 0:
            by[r['file_name']].append(r)
    folder = ROOT / 'JPEGImages/val'

    offsets = {f: content_offset(folder / f, [(r['gt_x1'], r['gt_y1'], r['gt_x2'], r['gt_y2']) for r in p]) for f, p in by.items()}
    for source in ('ogsod', 'msar'):
        values = np.array([offsets[f] for f, p in by.items() if p[0]['source'] == source and offsets[f]])
        report[f'content_offset_{source}'] = dict(images=len(values), median_dx=round(float(np.median(values[:, 0])), 3), median_dy=round(float(np.median(values[:, 1])), 3), mad_dx=round(float(np.median(np.abs(values[:, 0] - np.median(values[:, 0])))), 3), mad_dy=round(float(np.median(np.abs(values[:, 1] - np.median(values[:, 1])))), 3))
    gaps = defaultdict(list)
    for p in orientation['pairs']:
        if p['matched'] >= .8 * p['of'] and offsets.get(p['a']) and offsets.get(p['b']):
            a, b = offsets[p['a']], offsets[p['b']]
            gaps[p['orientation']].append((abs(a[0] - b[0]), abs(a[1] - b[1])))
    report['content_offset_pair_gap'] = {k: dict(pairs=len(v), median_abs_dx_gap=round(float(np.median([x[0] for x in v])), 3), median_abs_dy_gap=round(float(np.median([x[1] for x in v])), 3)) for k, v in gaps.items()}
    print(json.dumps(report), flush=True)

    # Which val tiles does train hold upside down?
    annotations, source_of, grouped = load_val(args.out)
    train = json.load(open(ROOT / 'Annotations/train.json'))
    names = {i['id']: i['file_name'] for i in train['images'] if i['width'] == 256 and i['height'] == 256}
    train_files = sorted({names[a['image_id']] for a in train['annotations'] if a['category_id'] == TANK and a['image_id'] in names})
    matrix = np.stack([unit(thumb_array(ROOT / 'JPEGImages/train' / f)) for f in train_files]).astype(np.float32)
    held = {}
    for f in by:
        thumb = thumb_array(folder / f)
        same_way = max(float((matrix @ unit(orient_array(thumb, n)).astype(np.float32)).max()) for n in ('identity', 'flip_x'))
        upside = max(float((matrix @ unit(orient_array(thumb, n)).astype(np.float32)).max()) for n in ('flip_y', 'rot180'))
        held[f] = dict(same_way=same_way >= SAME, upside_down=upside >= SAME, same_way_corr=round(same_way, 3), upside_corr=round(upside, 3))
    print('held', Counter((by[f][0]['source'], v['same_way'], v['upside_down']) for f, v in held.items()), flush=True)

    model, post = load_model(args.run, args.device)
    hooks = Hooks(model)
    results = {}
    with torch.no_grad():
        for count, f in enumerate(by):
            image = Image.open(folder / f).convert('RGB')
            w, h = image.size
            s = 640 / w
            targets = grouped[by[f][0]['image_id']]
            g_cls = np.array([a['category_id'] for a in targets], int)
            g_box = np.array([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]] for a in targets], np.float32)
            valid = (g_box[:, 2] > g_box[:, 0]) & (g_box[:, 3] > g_box[:, 1])
            g_cls, g_box = g_cls[valid], g_box[valid]
            source = by[f][0]['source']
            for condition in ('original', 'flip_y'):
                view = image if condition == 'original' else image.transpose(Image.FLIP_TOP_BOTTOM)
                boxes = g_box.copy()
                if condition == 'flip_y':
                    boxes[:, 1], boxes[:, 3] = h - g_box[:, 3], h - g_box[:, 1]
                    if source == 'ogsod':
                        boxes[:, [1, 3]] -= 2  # The shift real OGSOD vertical copies carry relative to their pixels.
                x = F.to_tensor(F.resize(view, [640, 640]))[None].to(args.device)
                stages, enc, layers, last = hooks.run(x)
                res = post(last, torch.tensor([[w, h]], device=args.device, dtype=torch.float32))[0]
                labels, scores, pred = res['labels'].cpu().numpy().astype(int), res['scores'].cpu().numpy(), res['boxes'].cpu().numpy()
                keep, kinds, _, fn_kind, _, taken = match_image(labels, scores, pred, g_cls, boxes, SCORE, IOU)
                centers = []
                for stride, (H, W) in zip((8, 16, 32), enc['shapes']):
                    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
                    centers.append(np.stack([(xx.ravel() + .5) * stride, (yy.ravel() + .5) * stride], 1))
                centers = np.concatenate(centers)
                for g in np.flatnonzero(g_cls == TANK):
                    b = boxes[g] * s
                    inside = (centers[:, 0] >= b[0]) & (centers[:, 0] < b[2]) & (centers[:, 1] >= b[1]) & (centers[:, 1] < b[3])
                    probability = float(enc['prob'][inside, TANK].max()) if inside.any() else float('nan')
                    tp_iou = None
                    if taken[g] >= 0:
                        tp_iou = float(iou_matrix(pred[keep[taken[g]]][None], boxes[g][None])[0, 0])
                    results.setdefault(f, {}).setdefault(condition, []).append(dict(status=fn_kind[g] or 'TP', enc=probability, tp_iou=tp_iou, side=float(max(boxes[g][2] - boxes[g][0], boxes[g][3] - boxes[g][1]) * s), index=int(g)))
                bkg = sum(k == 'Bkg' and labels[d] == TANK for d, k in zip(keep, kinds))
                results[f].setdefault('fp', {})[condition] = int(bkg)
            print('flip', count + 1, len(by), flush=True)

    def summary(files, condition):
        part = [x for f in files for x in results[f][condition]]
        if not part:
            return None
        tps = [x['tp_iou'] for x in part if x['tp_iou'] is not None]
        return dict(images=len(files), gt=len(part), recall=round(float(np.mean([x['status'] == 'TP' for x in part])), 3), enc=round(float(np.nanmedian([x['enc'] for x in part])), 3), tp_iou=round(float(np.median(tps)), 3) if tps else None, fp_bkg=sum(results[f]['fp'][condition] for f in files),
                    status=dict(Counter(x['status'] for x in part)))

    groups = {
        'ogsod|upside_down_not_in_train': [f for f in by if by[f][0]['source'] == 'ogsod' and held[f]['same_way'] and not held[f]['upside_down']],
        'ogsod|upside_down_also_in_train': [f for f in by if by[f][0]['source'] == 'ogsod' and held[f]['same_way'] and held[f]['upside_down']],
        'ogsod|only_upside_down_in_train': [f for f in by if by[f][0]['source'] == 'ogsod' and not held[f]['same_way'] and held[f]['upside_down']],
        'msar|upside_down_not_in_train': [f for f in by if by[f][0]['source'] == 'msar' and not held[f]['upside_down']],
        'msar|upside_down_in_train': [f for f in by if by[f][0]['source'] == 'msar' and held[f]['upside_down']],
    }
    report['flip_test'] = {name: dict(original=summary(files, 'original'), flip_y=summary(files, 'flip_y')) for name, files in groups.items() if files}
    paired = {}
    for name, files in groups.items():
        counts = Counter()
        for f in files:
            for a, b in zip(results[f]['original'], results[f]['flip_y']):
                counts[(a['status'] == 'TP', b['status'] == 'TP')] += 1
        paired[name] = dict(both_hit=counts[(True, True)], original_only=counts[(True, False)], flipped_only=counts[(False, True)], both_miss=counts[(False, False)])
    report['flip_paired'] = paired
    # Image-level so that one busy tile cannot carry the answer.
    per_image = {}
    for name, files in groups.items():
        changes = [float(np.mean([x['status'] == 'TP' for x in results[f]['flip_y']]) - np.mean([x['status'] == 'TP' for x in results[f]['original']])) for f in files]
        if changes:
            per_image[name] = dict(images=len(changes), median_change=round(float(np.median(changes)), 3), share_lower=round(float(np.mean(np.array(changes) < 0)), 3), share_higher=round(float(np.mean(np.array(changes) > 0)), 3), quantiles=np.round(np.percentile(changes, [10, 25, 75, 90]), 3).tolist())
    report['flip_per_image'] = per_image
    trace = json.load(open(args.out / 'trace/trace.json'))
    report['held_sample_image'] = held.get(trace['image'])
    report['held'] = held
    (args.out / 'orientation_intervention.json').write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != 'held'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
