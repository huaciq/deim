"""CPU analysis of saved predictions: metrics, FP/FN taxonomy, confusion matrix, and GT-vs-prediction error images.

Usage: python tools/sardet100k/error_gallery.py RUN_DIR [--split val] [--score 0.3] [--iou 0.5] [--per-bucket 10] [--no-images]

An error is defined at one operating point (score >= --score, IoU >= --iou), with greedy matching in score order.
FP kinds:  Bkg (IoU < .1 with every GT), Loc (same-class IoU in [.1, iou)), Both (other-class IoU in [.1, iou)),
           Cls (other-class IoU >= iou), Dupe (same-class IoU >= iou, GT already taken).
FN kinds:  LowScore (a same-class box has IoU >= iou and a score in [.05, threshold)), Loc (a same-class box above the
           threshold reaches only IoU in [.1, iou)), Cls (taken by an other-class box), Crowd (the box above the
           threshold went to a neighbouring GT), NoBox (no same-class box scoring >= .05 at IoU >= iou).
"""
import argparse
import csv
import glob
import json
import multiprocessing as mp
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = Path('/root/zxy/TCRA-SAR/datasets')
ROOT = BASE / 'SARDet_100K'
SIZE_BINS = [(0, 16), (16, 32), (32, 64), (64, 128), (128, 1 << 20)]
THRESHOLDS = np.linspace(.5, .95, 10)
RECALLS = np.linspace(0, 1, 101)
PANEL = 640
LOW_SCORE_FLOOR = .05  # With 300 queries some box nearly always overlaps a GT by chance; below this it does not count as a detection.
GT_COLOR, MISS_COLOR = (60, 220, 90), (255, 200, 0)
TP_COLOR, FP_COLOR, LOW_COLOR = (70, 190, 255), (255, 60, 60), (230, 110, 255)
FONT = ImageFont.load_default(size=14)
BIG_FONT = ImageFont.load_default(size=17)


def iou_matrix(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    x1, y1 = np.maximum(a[:, None, 0], b[:, 0]), np.maximum(a[:, None, 1], b[:, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[:, 2]), np.minimum(a[:, None, 3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None] + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return inter / np.maximum(union, 1e-9)


def match_image(p_cls, p_score, p_box, g_cls, g_box, score_thr, iou_thr):
    """Return (kind per kept prediction, GT index per prediction, FN kind per GT or None, best low-score box per GT)."""
    keep = np.flatnonzero(p_score >= score_thr)
    keep = keep[np.argsort(-p_score[keep], kind='mergesort')]
    m_all = iou_matrix(p_box, g_box)
    m = m_all[keep]
    same = p_cls[keep][:, None] == g_cls[None, :] if len(g_cls) else np.zeros((len(keep), 0), bool)
    taken = np.full(len(g_cls), -1)
    kinds, targets = ['']*len(keep), [-1]*len(keep)
    for d in range(len(keep)):
        options = np.where(same[d] & (taken < 0), m[d], -1) if len(g_cls) else np.zeros(0)
        if len(options) and options.max() >= iou_thr:
            g = int(options.argmax())
            taken[g] = d
            kinds[d], targets[d] = 'TP', g
    cls_taken = np.zeros(len(g_cls), bool)
    for d in range(len(keep)):
        if kinds[d] or not len(g_cls):
            continue
        same_row, other_row = np.where(same[d], m[d], 0), np.where(same[d], 0, m[d])
        if same_row.max() >= iou_thr:
            kinds[d], targets[d] = 'Dupe', int(same_row.argmax())
        elif other_row.max() >= iou_thr:
            g = int(other_row.argmax())
            kinds[d], targets[d] = 'Cls', g
            if taken[g] < 0:
                cls_taken[g] = True
        elif same_row.max() >= .1:
            kinds[d], targets[d] = 'Loc', int(same_row.argmax())
        elif other_row.max() >= .1:
            kinds[d], targets[d] = 'Both', int(other_row.argmax())
        else:
            kinds[d] = 'Bkg'
    kinds = [k or 'Bkg' for k in kinds]
    fn_kind, low_box = [None] * len(g_cls), [None] * len(g_cls)
    for g in np.flatnonzero(taken < 0):
        same_any = (p_cls == g_cls[g]) & (p_score >= LOW_SCORE_FLOOR)
        row = np.where(same_any, m_all[:, g], 0) if len(p_cls) else np.zeros(0)
        strong = p_score >= score_thr
        if len(row) and (row * strong).max() >= iou_thr:
            fn_kind[g] = 'Crowd'
        elif cls_taken[g]:
            fn_kind[g] = 'Cls'
        elif len(row) and row.max() >= iou_thr:
            candidates = np.flatnonzero(row >= iou_thr)
            best = candidates[np.argmax(p_score[candidates])]
            fn_kind[g], low_box[g] = 'LowScore', int(best)
        elif len(row) and (row * strong).max() >= .1:
            fn_kind[g] = 'Loc'
        else:
            fn_kind[g] = 'NoBox'
    return keep, kinds, targets, fn_kind, low_box, taken


def ap_curve(scores, hits, positives):
    if not positives or not len(scores):
        return 0.
    rank = np.argsort(-scores, kind='mergesort')
    hit = hits[rank]
    tps, fps = np.cumsum(hit), np.cumsum(~hit)
    recall, precision = tps / positives, tps / np.maximum(tps + fps, 1e-12)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    where = np.searchsorted(recall, RECALLS, side='left')
    sampled = np.zeros(101)
    inside = where < len(precision)
    sampled[inside] = precision[where[inside]]
    return float(sampled.mean())


def coco_ap(image_ids, data):
    """COCO-style per-class AP over the given images (100 detections per class per image)."""
    out = {}
    for c in data['classes']:
        lines = [([], []) for _ in THRESHOLDS]
        positives = 0
        for image_id in image_ids:
            gt = data['gt'].get(image_id, ([], np.zeros((0, 4), np.float32)))
            select = np.flatnonzero(np.asarray(gt[0], int) == c) if len(gt[0]) else np.zeros(0, int)
            boxes = gt[1][select]
            positives += len(select)
            pos = data['slices'].get(image_id)
            if pos is None:
                continue
            pos = pos[data['label'][pos] == c]
            pos = pos[np.argsort(-data['score'][pos], kind='mergesort')][:100]
            if not len(pos):
                continue
            m = iou_matrix(data['box'][pos], boxes)
            for t, threshold in enumerate(THRESHOLDS):
                used = np.zeros(len(boxes), bool)
                hit = np.zeros(len(pos), bool)
                for d in range(len(pos)):
                    if len(boxes):
                        row = np.where(used, -1, m[d])
                        if row.max() >= threshold:
                            used[int(row.argmax())] = True
                            hit[d] = True
                lines[t][0].append(data['score'][pos])
                lines[t][1].append(hit)
        if positives:
            values = [ap_curve(np.concatenate(s) if s else np.zeros(0), np.concatenate(h) if h else np.zeros(0, bool), positives) for s, h in lines]
            out[c] = dict(gt=positives, AP=round(np.mean(values) * 100, 2), AP50=round(values[0] * 100, 2), AP75=round(values[5] * 100, 2))
    return out


def thumb(path):
    image = Image.open(path).convert('L')
    t = np.asarray(image.resize((32, 32), Image.BILINEAR), np.float32).ravel()
    t = t - t.mean()
    n = np.linalg.norm(t)
    return (t / n if n > 0 else t), image.size


def same_pixels(task):
    a, b = task
    A = np.asarray(Image.open(a).convert('L'), np.float32)
    B = np.asarray(Image.open(b).convert('L'), np.float32)
    return A.shape == B.shape and np.abs(A - B).mean() < 2


def identify_sources(paths):
    """Same pixel match as source_val.py (thumbnail correlation >= .98, mean absolute difference < 2), on CPU."""
    references = {'ogsod': sorted(glob.glob(str(BASE / 'OGSOD-1.0/sar/images/*/*'))),
                  'msar': sorted(glob.glob(str(BASE / 'MSAR-1.0 dataset/JPEGImages/*')))}
    labels = np.array(['other'] * len(paths), dtype=object)
    with mp.Pool(32) as pool:
        rows = pool.map(thumb, paths, chunksize=64)
        query, sizes = np.stack([r[0] for r in rows]), [r[1] for r in rows]
        for name, refs in references.items():
            rows = pool.map(thumb, refs, chunksize=64)
            R, ref_sizes = np.stack([r[0] for r in rows]), [r[1] for r in rows]
            candidates = []
            for start in range(0, len(query), 512):
                sim = query[start:start + 512] @ R.T
                best = sim.argmax(1)
                for offset, b in enumerate(best):
                    q = start + offset
                    if sim[offset, b] >= .98 and sizes[q] == ref_sizes[b]:
                        candidates.append((q, int(b)))
            ok = pool.map(same_pixels, [(paths[q], refs[r]) for q, r in candidates], chunksize=64)
            labels[[q for (q, _), good in zip(candidates, ok) if good]] = name
    return labels


def prf(tp, fp, fn):
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(TP=tp, FP=fp, FN=fn, precision=round(precision, 4), recall=round(recall, 4),
                F1=round(2 * precision * recall / max(precision + recall, 1e-9), 4))


def analyse(data, score_thr, iou_thr):
    """Full-dataset tally at one operating point. Returns tables plus per-image error rows when requested."""
    classes, names = data['classes'], data['names']
    n = len(classes)
    confusion = np.zeros((n + 1, n + 1), int)  # row: GT class or background; column: predicted class or missed
    fp, fn, tp = defaultdict(Counter), defaultdict(Counter), Counter()
    errors = []
    size = defaultdict(lambda: np.zeros(3, int))
    fp_size = defaultdict(Counter)
    per_source = defaultdict(lambda: defaultdict(Counter))
    images_with_error = 0
    for image_id, image in data['images'].items():
        g_cls, g_box = data['gt'].get(image_id, ([], np.zeros((0, 4), np.float32)))
        g_cls = np.asarray(g_cls, int)
        pos = data['slices'].get(image_id, np.zeros(0, int))
        keep, kinds, targets, fn_kind, low_box, taken = match_image(data['label'][pos], data['score'][pos], data['box'][pos], g_cls, g_box, score_thr, iou_thr)
        source = data['source'].get(image_id, 'unknown')
        had_error = False
        for d, kind, g in zip(keep, kinds, targets):
            c = int(data['label'][pos[d]])
            s = float(data['score'][pos[d]])
            box = data['box'][pos[d]]
            if kind == 'TP':
                tp[c] += 1
                confusion[classes.index(c), classes.index(c)] += 1
                per_source[source][c]['TP'] += 1
                continue
            had_error = True
            fp[c][kind] += 1
            per_source[source][c]['FP'] += 1
            side = float(max(box[2] - box[0], box[3] - box[1]))
            fp_size[names[c]][next(f'{a}-{b}' for a, b in SIZE_BINS if a <= side < b)] += 1
            if kind == 'Cls' and fn_kind[g] == 'Cls':
                confusion[classes.index(int(g_cls[g])), classes.index(c)] += 1
            else:
                confusion[n, classes.index(c)] += 1
            errors.append(dict(image_id=image_id, group='FP', kind=kind, cls=names[c], score=round(s, 4), box=[round(float(v), 1) for v in box], other=names[int(g_cls[g])] if kind in ('Cls', 'Both') else '', side=round(side, 1), source=source, target=g, pred=int(pos[d])))
        for g, kind in enumerate(fn_kind):
            c = int(g_cls[g])
            box = g_box[g]
            side = float(max(box[2] - box[0], box[3] - box[1]))
            bin_label = next(f'{a}-{b}' for a, b in SIZE_BINS if a <= side < b)
            size[f'{names[c]}|{bin_label}'][0] += 1
            if kind is None:
                continue
            had_error = True
            fn[c][kind] += 1
            size[f'{names[c]}|{bin_label}'][1] += 1
            per_source[source][c]['FN'] += 1
            if kind == 'Cls':
                pass
            else:
                confusion[classes.index(c), n] += 1
            best = float(data['score'][pos[low_box[g]]]) if low_box[g] is not None else 0.
            errors.append(dict(image_id=image_id, group='FN', kind=kind, cls=names[c], score=round(best, 4), box=[round(float(v), 1) for v in box], other='', side=round(side, 1), source=source, target=g, pred=int(pos[low_box[g]]) if low_box[g] is not None else -1))
        images_with_error += had_error
    per_class = {}
    for c in classes:
        per_class[names[c]] = dict(prf(tp[c], sum(fp[c].values()), sum(fn[c].values())), FP_kinds=dict(fp[c]), FN_kinds=dict(fn[c]))
    t, f, m = sum(tp.values()), sum(sum(v.values()) for v in fp.values()), sum(sum(v.values()) for v in fn.values())
    return dict(score=score_thr, iou=iou_thr, overall=prf(t, f, m), per_class=per_class,
                confusion=dict(labels=[names[c] for c in classes] + ['background'], matrix=confusion.tolist()),
                FN_by_long_side={k: dict(gt=int(v[0]), missed=int(v[1]), miss_rate=round(float(v[1] / max(v[0], 1)), 4)) for k, v in sorted(size.items())},
                FP_by_long_side={k: dict(v) for k, v in fp_size.items()},
                by_source={s: {names[c]: prf(v['TP'], v['FP'], v['FN']) for c, v in rows.items()} for s, rows in per_source.items()},
                images_with_error=images_with_error, images=len(data['images'])), errors


def best_f1(data):
    """Per-class score threshold that maximises F1 at IoU .5, using each detection's global same-class match."""
    out = {}
    scan = np.arange(.05, .96, .05)
    for score_thr in scan:
        report, _ = analyse(data, float(score_thr), .5)
        for name, row in report['per_class'].items():
            if name not in out or row['F1'] > out[name]['F1']:
                out[name] = dict(score=round(float(score_thr), 2), **{k: row[k] for k in ('precision', 'recall', 'F1', 'TP', 'FP', 'FN')})
    return out


# ---- drawing ----
def draw_box(draw, box, color, width=2, dashed=False):
    x1, y1, x2, y2 = box
    if dashed:
        for a, b in [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]:
            length = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
            for t in np.arange(0, max(length, 1), 8):
                u, v = t / max(length, 1), min(t + 4, length) / max(length, 1)
                draw.line([a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[0] + (b[0] - a[0]) * v, a[1] + (b[1] - a[1]) * v], fill=color, width=width)
    else:
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)


def draw_label(draw, box, text, color, panel_size):
    left, top, right, bottom = draw.textbbox((0, 0), text, font=FONT)
    w, h = right - left, bottom - top
    x = float(np.clip(box[0], 0, max(panel_size[0] - w - 4, 0)))
    y = box[1] - h - 5 if box[1] - h - 5 >= 0 else min(box[3] + 2, panel_size[1] - h - 4)
    draw.rectangle([x, y, x + w + 4, y + h + 4], fill=(0, 0, 0))
    draw.text((x + 2, y + 2 - top), text, fill=color, font=FONT)


def overlay(base, offset, scale, items, labelled=True):
    """items: (xyxy in image pixels, color, label text, thick, dashed). base is modified in place."""
    draw = ImageDraw.Draw(base)
    for box, color, text, thick, dashed in items:
        scaled = [(box[0] - offset[0]) * scale, (box[1] - offset[1]) * scale, (box[2] - offset[0]) * scale, (box[3] - offset[1]) * scale]
        if scaled[2] < 0 or scaled[3] < 0 or scaled[0] > base.width or scaled[1] > base.height:
            continue
        draw_box(draw, scaled, color, 3 if thick else 2, dashed)
        if text and labelled:
            draw_label(draw, scaled, text, color, base.size)


def render(task):
    """Four panels: raw image | GT | predictions | zoom on the error, with a header line."""
    image_path, header, gt_items, pred_items, focus, out_path = task
    image = Image.open(image_path).convert('RGB')
    w, h = image.size
    scale = PANEL / max(w, h)
    size = (max(round(w * scale), 1), max(round(h * scale), 1))
    resample = Image.BICUBIC if scale >= 1 else Image.LANCZOS
    raw = image.resize(size, resample)
    gt_panel, pred_panel = raw.copy(), raw.copy()
    many_gt, many_pred = len(gt_items) > 15, len(pred_items) > 15
    overlay(gt_panel, (0, 0), scale, [(b, c, (t if (not many_gt or thick) else ''), thick, d) for b, c, t, thick, d in gt_items])
    overlay(pred_panel, (0, 0), scale, [(b, c, (t if (not many_pred or thick) else ''), thick, d) for b, c, t, thick, d in pred_items])

    span = max(max(focus[2] - focus[0], focus[3] - focus[1]) * 4, 64)
    cx, cy = (focus[0] + focus[2]) / 2, (focus[1] + focus[3]) / 2
    x0, y0 = int(max(min(cx - span / 2, w - span), 0)), int(max(min(cy - span / 2, h - span), 0))
    x1, y1 = int(min(x0 + span, w)), int(min(y0 + span, h))
    zoom_scale = PANEL / max(x1 - x0, y1 - y0, 1)
    zoom = image.crop((x0, y0, x1, y1)).resize((max(round((x1 - x0) * zoom_scale), 1), max(round((y1 - y0) * zoom_scale), 1)), Image.BICUBIC)
    # In the zoom, GT is drawn in green/yellow and predictions in their colours, so a pair can be compared directly.
    zoom_items = [(b, c, '', thick, d) for b, c, _, thick, d in gt_items] + pred_items
    overlay(zoom, (x0, y0), zoom_scale, zoom_items)

    titles = ['raw', 'GT (green=found, yellow=missed)', 'pred (blue=TP, red=FP, purple dashed=low score)', f'zoom x{zoom_scale:.1f}: GT + pred']
    panels = [raw, gt_panel, pred_panel, zoom]
    bar, title_bar, gap = 28, 22, 6
    canvas = Image.new('RGB', (PANEL * 4 + gap * 3, bar + title_bar + PANEL), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 4), header, fill=(240, 240, 240), font=BIG_FONT)
    for i, (panel, title) in enumerate(zip(panels, titles)):
        x = i * (PANEL + gap)
        draw.text((x + 4, bar + 3), title, fill=(190, 190, 190), font=FONT)
        canvas.paste(panel, (x, bar + title_bar))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=90)
    return str(out_path)


def build_task(error, data, score_thr, iou_thr, out_path):
    image_id = error['image_id']
    image = data['images'][image_id]
    g_cls, g_box = data['gt'].get(image_id, ([], np.zeros((0, 4), np.float32)))
    g_cls = np.asarray(g_cls, int)
    pos = data['slices'].get(image_id, np.zeros(0, int))
    keep, kinds, targets, fn_kind, low_box, _ = match_image(data['label'][pos], data['score'][pos], data['box'][pos], g_cls, g_box, score_thr, iou_thr)
    names = data['names']
    is_fn = error['group'] == 'FN'
    gt_items = []
    for g in range(len(g_cls)):
        missed = fn_kind[g] is not None
        focus_here = is_fn and g == error['target']
        gt_items.append((g_box[g].tolist(), MISS_COLOR if missed else GT_COLOR, f"{names[int(g_cls[g])]}{' ' + fn_kind[g] if missed else ''}", focus_here, False))
    pred_items = []
    for d, kind in zip(keep, kinds):
        p = pos[d]
        focus_here = not is_fn and int(p) == error['pred']
        label = f"{names[int(data['label'][p])]} {data['score'][p]:.2f}" + ('' if kind == 'TP' else f' {kind}')
        pred_items.append((data['box'][p].tolist(), TP_COLOR if kind == 'TP' else FP_COLOR, label, focus_here, False))
    for g, low in enumerate(low_box):
        if low is not None:
            p = pos[low]
            pred_items.append((data['box'][p].tolist(), LOW_COLOR, f"{names[int(data['label'][p])]} {data['score'][p]:.2f}", is_fn and g == error['target'], True))
    tally = Counter(kinds)
    fns = sum(k is not None for k in fn_kind)
    header = (f"{image['file_name']}  {image['width']}x{image['height']}  source={error['source']}  |  {error['group']}_{error['kind']} {error['cls']}"
              f"  score={error['score']:.2f}  long_side={error['side']:.0f}px  |  image: GT={len(g_cls)} TP={tally['TP']} FP={len(kinds) - tally['TP']} FN={fns}")
    return data['folder'] / image['file_name'], header, gt_items, pred_items, error['box'], out_path


def choose(errors, per_bucket, seed):
    """For each (group, kind, class): half the worst cases and half a seeded random sample, at most one per image."""
    buckets = defaultdict(list)
    for e in errors:
        buckets[(e['group'], e['kind'], e['cls'])].append(e)
    chosen = []
    rng = random.Random(seed)
    for key in sorted(buckets):
        rows = buckets[key]
        # Worst: FP with the highest score, FN with the largest target (a large miss is the most surprising).
        rows_sorted = sorted(rows, key=lambda e: -(e['score'] if e['group'] == 'FP' else e['side']))
        seen = set()
        picks = []
        for e in rows_sorted:
            if len(picks) >= per_bucket // 2:
                break
            if e['image_id'] not in seen:
                seen.add(e['image_id'])
                picks.append(('worst', e))
        rest = [e for e in rows if e['image_id'] not in seen]
        rng.shuffle(rest)
        for e in rest:
            if len(picks) >= per_bucket:
                break
            if e['image_id'] not in seen:
                seen.add(e['image_id'])
                picks.append(('random', e))
        chosen.extend((key, tag, e) for tag, e in picks)
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--split', default='val')
    parser.add_argument('--score', type=float, default=.3)
    parser.add_argument('--iou', type=float, default=.5)
    parser.add_argument('--per-bucket', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-images', action='store_true')
    args = parser.parse_args()
    verify = args.run / 'independent_verify_20261106'
    out = verify / f'error_gallery_{args.split}'
    out.mkdir(parents=True, exist_ok=True)
    annotations = json.load(open(ROOT / f'Annotations/{args.split}.json'))
    saved = np.load(verify / f'{args.split}_predictions.npz')
    label, score, box, pred_image = saved['label'].astype(int), saved['score'], saved['box'], saved['image_id']
    order = np.argsort(pred_image, kind='stable')
    values, first = np.unique(pred_image[order], return_index=True)
    grouped = defaultdict(lambda: ([], []))
    for a in annotations['annotations']:
        x, y, w, h = a['bbox']
        grouped[a['image_id']][0].append(a['category_id'])
        grouped[a['image_id']][1].append([x, y, x + w, y + h])
    folder = ROOT / 'JPEGImages' / args.split
    images = {i['id']: i for i in annotations['images']}
    source_file = out / 'source.json'
    if source_file.exists():
        source = {int(k): v for k, v in json.load(open(source_file)).items()}
    else:
        labels = identify_sources([str(folder / i['file_name']) for i in annotations['images']])
        source = {i['id']: str(s) for i, s in zip(annotations['images'], labels)}
        source_file.write_text(json.dumps(source))
    data = dict(classes=sorted(c['id'] for c in annotations['categories']), names={c['id']: c['name'] for c in annotations['categories']},
                images=images, gt={k: (v[0], np.array(v[1], np.float32)) for k, v in grouped.items()}, slices=dict(zip(values.tolist(), np.split(order, first[1:]))),
                label=label, score=score, box=box, source=source, folder=folder)

    report = dict(run=str(args.run), split=args.split, predictions=f'{args.split}_predictions.npz', source_counts=dict(Counter(source.values())),
                  images_without_gt=sum(i not in grouped for i in images))
    ids = list(images)
    report['AP'] = {data['names'][c]: v for c, v in coco_ap(ids, data).items()}
    report['AP_mean'] = {k: round(float(np.mean([v[k] for v in report['AP'].values()])), 2) for k in ('AP', 'AP50', 'AP75')}
    report['AP_by_source'] = {}
    for s in sorted(set(source.values())):
        rows = coco_ap([i for i in ids if source[i] == s], data)
        report['AP_by_source'][s] = {data['names'][c]: v for c, v in rows.items()}
    report['operating_points'] = {}
    errors = None
    for s in (.1, .3, .5):
        result, found = analyse(data, s, args.iou)
        report['operating_points'][f'score_{s}'] = result
        if abs(s - args.score) < 1e-9:
            errors = found
    if errors is None:
        result, errors = analyse(data, args.score, args.iou)
        report['operating_points'][f'score_{args.score}'] = result
    report['best_f1_per_class'] = best_f1(data)
    (out / 'report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    with open(out / 'errors.csv', 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['file_name', 'source', 'group', 'kind', 'class', 'score', 'x1', 'y1', 'x2', 'y2', 'long_side', 'other_class'])
        for e in errors:
            writer.writerow([images[e['image_id']]['file_name'], e['source'], e['group'], e['kind'], e['cls'], e['score'], *e['box'], e['side'], e['other']])
    print(json.dumps({k: report[k] for k in ('AP_mean', 'source_counts', 'images_without_gt')}), len(errors), flush=True)

    if not args.no_images:
        tasks, index = [], []
        for (group, kind, cls), tag, e in choose(errors, args.per_bucket, args.seed):
            name = f"{tag}_{e['score']:.2f}_{e['side']:.0f}px_{e['source']}_{Path(images[e['image_id']]['file_name']).stem}.jpg"
            path = out / 'images' / f'{group}_{kind}' / cls / name
            tasks.append(build_task(e, data, args.score, args.iou, path))
            index.append([str(path.relative_to(out)), tag, group, kind, cls, e['score'], e['side'], e['source'], e['other']])
        with mp.Pool(16) as pool:
            pool.map(render, tasks, chunksize=8)
        with open(out / 'gallery_index.csv', 'w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['path', 'selection', 'group', 'kind', 'class', 'score', 'long_side', 'source', 'other_class'])
            writer.writerows(index)
        print('rendered', len(tasks), flush=True)


if __name__ == '__main__':
    main()
