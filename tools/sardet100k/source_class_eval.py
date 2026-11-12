"""Evaluate saved predictions per source group x class with COCOeval, plus error counts at one operating point.

Usage: python tools/sardet100k/source_class_eval.py VERIFY_DIR OUT_DIR [--split val] [--score 0.3] [--iou 0.5]

VERIFY_DIR holds {split}_predictions.npz (written for the E000 independent verification).

Each group is evaluated on its own images only, so a cell's precision sees only that group's predictions and its AP
is not a share of the global AP. A class with predictions but no GT in a group has AP undefined; its FP counts remain.
Sizes use the box long side, both in original pixels and after resizing the long side to --input-size.
Recall figures: AR = COCO AR@100 over IoU .50:.95 and R50 = recall@IoU .5 with up to 100 boxes per class and no score
threshold; `recall` = recall at the operating point, whose complement is the miss count.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from faster_coco_eval import COCO, COCOeval_faster

from dataset_profile import group_of
from error_gallery import ROOT, identify_sources, match_image

GROUP_ORDER = ['ogsod', 'msar', 'other_256', 'other_512', 'other_800', 'other_224', 'other_1000', 'other_misc']
SIZE_BINS = [('all', 0, 1e5), ('<16', 0, 16), ('16-32', 16, 32), ('32-64', 32, 64), ('64-128', 64, 128), ('>=128', 128, 1e5)]
MIN_POOL_GT = 30
FP_KINDS, FN_KINDS =['Bkg', 'Loc', 'Both', 'Cls', 'Dupe'], ['NoBox', 'LowScore', 'Loc', 'Cls', 'Crowd']


def load_sources(split, annotations, out):
    path = out / f'{split}_source_groups.json'
    if path.exists():
        return {int(k): v for k, v in json.load(open(path)).items()}
    folder = ROOT / 'JPEGImages' / split
    labels = identify_sources([str(folder / i['file_name']) for i in annotations['images']])
    groups = {i['id']: group_of(str(s), i['width'], i['height']) for i, s in zip(annotations['images'], labels)}
    path.write_text(json.dumps(groups))
    return groups


def long_side_gt(coco, scales, use_input):
    for a in coco.anns.values():
        side = max(a['bbox'][2], a['bbox'][3]) * (scales[a['image_id']] if use_input else 1)
        a['area'] = float(side ** 2)


def run_eval(gt, dt, image_ids, cat_ids):
    """Returns {(class id, bin name): metrics} for one image set. `area` already holds long side squared."""
    e = COCOeval_faster(gt, dt, 'bbox')
    e.params.imgIds = sorted(image_ids)
    e.params.catIds = sorted(cat_ids)
    # COCOeval ranges are closed; the small margin keeps a box of exactly 16 px out of '<16', matching size_label.
    e.params.areaRng = [[float(a ** 2), float(b ** 2) - 1e-6] for _, a, b in SIZE_BINS]
    e.params.areaRngLbl = [name for name, _, _ in SIZE_BINS]
    e.evaluate()
    e.accumulate()
    out = {}
    precision, recall = np.asarray(e.eval['precision']), np.asarray(e.eval['recall'])  # precision: T,R,K,A,M; recall: T,K,A,M
    t50, t75, last = 0, 5, len(e.params.maxDets) - 1
    for k, c in enumerate(e.params.catIds):
        for a, (name, _, _) in enumerate(SIZE_BINS):
            p, r = precision[:, :, k, a, last], recall[:, k, a, last]
            if (r < 0).all():  # no GT of this class in this bin
                continue
            out[c, name] = dict(AP=float(p[p > -1].mean()) * 100, AP50=float(p[t50][p[t50] > -1].mean()) * 100,
                                AP75=float(p[t75][p[t75] > -1].mean()) * 100, AR=float(r[r > -1].mean()) * 100, R50=float(r[t50]) * 100)
    return out


def size_label(side):
    return next(name for name, a, b in SIZE_BINS[1:] if a <= side < b)


def tally(annotations, saved, groups, scales, score, iou):
    """Per group x class x size: GT, TP, FN and FP kinds at the operating point, using the same matching as error_gallery."""
    grouped = defaultdict(lambda: ([], []))
    for a in annotations['annotations']:
        x, y, w, h = a['bbox']
        grouped[a['image_id']][0].append(a['category_id'])
        grouped[a['image_id']][1].append([x, y, x + w, y + h])
    label, scores, boxes, pred_image = saved['label'].astype(int), saved['score'], saved['box'], saved['image_id']
    order = np.argsort(pred_image, kind='stable')
    values, first = np.unique(pred_image[order], return_index=True)
    slices = dict(zip(values.tolist(), np.split(order, first[1:])))
    rows = defaultdict(Counter)  # key: (group, class id, size system, bin); fields GT, TP, FN, FN_<kind>, FP, FP_<kind>
    for image in annotations['images']:
        image_id = image['id']
        group = groups[image_id]
        g_cls = np.asarray(grouped[image_id][0], int)
        g_box = np.asarray(grouped[image_id][1], np.float32).reshape(-1, 4)
        pos = slices.get(image_id, np.zeros(0, int))
        keep, kinds, _, fn_kind, _, _ = match_image(label[pos], scores[pos], boxes[pos], g_cls, g_box, score, iou)

        def bins(box):
            side = float(max(box[2] - box[0], box[3] - box[1]))
            return [('orig', 'all'), ('orig', size_label(side)), ('input', size_label(side * scales[image_id]))]

        for g, kind in enumerate(fn_kind):
            for system, name in bins(g_box[g]):
                row = rows[group, int(g_cls[g]), system, name]
                row['GT'] += 1
                row['FN' if kind else 'TP'] += 1
                if kind:
                    row[f'FN_{kind}'] += 1
        for d, kind in zip(keep, kinds):
            if kind == 'TP':
                continue
            for system, name in bins(boxes[pos[d]]):
                row = rows[group, int(label[pos[d]]), system, name]
                row['FP'] += 1
                row[f'FP_{kind}'] += 1
    return rows


def cell(metrics, counts):
    out = {k: round(v, 2) for k, v in metrics.items()} if metrics else dict(AP=None, AP50=None, AP75=None, AR=None, R50=None)
    gt, fn = counts['GT'], counts['FN']
    out.update(GT=gt, TP=counts['TP'], missed=fn, recall=round(counts['TP'] / gt * 100, 2) if gt else None,
               FP=counts['FP'], FP_bkg=counts['FP_Bkg'], FP_kinds={k: counts[f'FP_{k}'] for k in FP_KINDS}, FN_kinds={k: counts[f'FN_{k}'] for k in FN_KINDS})
    return out


def merged(rows, keys):
    total = Counter()
    for key in keys:
        total.update(rows.get(key, {}))
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('verify', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--split', default='val')
    parser.add_argument('--score', type=float, default=.3)
    parser.add_argument('--iou', type=float, default=.5)
    parser.add_argument('--input-size', type=int, default=640)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    path = ROOT / f'Annotations/{args.split}.json'
    annotations = json.load(open(path))
    names = {c['id']: c['name'] for c in annotations['categories']}
    cat_ids = sorted(names)
    groups = load_sources(args.split, annotations, args.out)
    scales = {i['id']: args.input_size / max(i['width'], i['height']) for i in annotations['images']}
    members = defaultdict(list)
    for image_id, group in groups.items():
        members[group].append(image_id)
    saved = np.load(args.verify / f'{args.split}_predictions.npz')
    results = [dict(image_id=int(i), category_id=int(c), score=float(s), bbox=[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])])
               for i, c, s, b in zip(saved['image_id'], saved['label'], saved['score'], saved['box'])]

    # Official-protocol check: the untouched annotation areas with COCO's default ranges reproduce the run's headline AP.
    official_gt = COCO(str(path))
    official = COCOeval_faster(official_gt, official_gt.loadRes(results), 'bbox')
    official.evaluate()
    official.accumulate()
    official.summarize()
    report = dict(split=args.split, score=args.score, iou=args.iou, input_size=args.input_size,
                  official_coco_stats=[round(float(v) * 100, 2) for v in official.stats[:6]],
                  images={g: len(v) for g, v in members.items()})

    metrics = {}
    for system in ('orig', 'input'):
        gt = COCO(str(path))
        long_side_gt(gt, scales, system == 'input')
        dt = gt.loadRes(results)
        long_side_gt(dt, scales, system == 'input')
        for group in [*[g for g in GROUP_ORDER if g in members], 'all']:
            ids = members[group] if group != 'all' else list(groups)
            for (c, name), row in run_eval(gt, dt, ids, cat_ids).items():
                metrics[group, c, system, name] = row
            print('evaluated', system, group, flush=True)

    rows = tally(annotations, saved, groups, scales, args.score, args.iou)
    report['cells'], report['group_summary'], report['size'] = {}, {}, {}
    for group in [*[g for g in GROUP_ORDER if g in members], 'all']:
        group_list = list(members) if group == 'all' else [group]
        report['cells'][group] = {}
        report['size'][group] = {}
        for c in cat_ids:
            total = merged(rows, [(g, c, 'orig', 'all') for g in group_list])
            if total:
                report['cells'][group][names[c]] = cell(metrics.get((group, c, 'orig', 'all')), total)
            for system in ('orig', 'input'):
                for name, _, _ in SIZE_BINS[1:]:
                    counts = merged(rows, [(g, c, system, name) for g in group_list])
                    if counts:
                        report['size'][group].setdefault(system, {}).setdefault(names[c], {})[name] = cell(metrics.get((group, c, system, name)), counts)
        for system in ('orig', 'input'):  # Counts cover every class; AP averages classes with enough GT, since one box can swing a cell's AP from 0 to 100.
            pooled = {}
            for name, _, _ in SIZE_BINS[1:]:
                counts = merged(rows, [(g, c, system, name) for g in group_list for c in cat_ids])
                if counts:
                    present = [metrics[group, c, system, name] for c in cat_ids if (group, c, system, name) in metrics and merged(rows, [(g, c, system, name) for g in group_list])['GT'] >= MIN_POOL_GT]
                    pooled[name] = cell({k: float(np.mean([p[k] for p in present])) for k in present[0]} if present else None, counts)
            report['size'][group].setdefault(system, {})['ALL'] = pooled
        have_gt = [metrics[group, c, 'orig', 'all'] for c in cat_ids if (group, c, 'orig', 'all') in metrics]
        total = merged(rows, [(g, c, 'orig', 'all') for g in group_list for c in cat_ids])
        report['group_summary'][group] = dict(cell({k: float(np.mean([p[k] for p in have_gt])) for k in have_gt[0]}, total), images=len(members[group]) if group != 'all' else len(groups), classes_with_gt=len(have_gt))
    (args.out / f'{args.split}_report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(dict(official=report['official_coco_stats'], summary={g: {k: v[k] for k in ('images', 'GT', 'AP', 'AP50', 'AP75', 'recall', 'missed', 'FP_bkg')} for g, v in report['group_summary'].items()})), flush=True)


if __name__ == '__main__':
    main()
