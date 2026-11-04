#!/usr/bin/env python3
"""CPU diagnosis of one exported run on test: where does the AP50 go?

Reads predictions from tools/export_analysis_predictions.py and asks three questions:

1. Is the AP50 lost to type confusion, background false positives or the rest
   (misses and poor localization)? Answered with oracle AP50 values.
2. How do targets seen in the training set compare with unseen ones? Seen means a
   near-identical crop exists in --reference (same matcher as check_split_overlap.py).
3. Is an unseen aircraft not found at all, or found with a split type score?
   Answered with class-agnostic AP50 and the located rate at two score thresholds.

Diagnostic only: nothing here selects a checkpoint or changes the official metric.
Writes OUTPUT/{report.md,summary.json,targets.csv}; no inference or training.
"""

import argparse
from collections import defaultdict
import contextlib
import copy
import io
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.check_split_overlap import best_match, targets
from tools.compare_rem_102e import class_agnostic, confusion, write_csv
from tools.sar_analysis_core import iou_matrix, load_json, save_json

SEEN_THRESHOLD = .9
LOW_SCORE = .05
SCORE = .3
BACKGROUND_IOU = .1
SUBSETS = ('all', 'seen', 'unseen')


def evaluate(dataset, predictions, work):
    """AP, AP50, AP75 and per-class AP50; GT flagged iscrowd is ignored, as in COCO."""
    from faster_coco_eval import COCO, COCOeval_faster

    path = Path(work) / '_gt.json'
    save_json(path, dataset)
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(path))
        if predictions:
            dt = gt.loadRes(copy.deepcopy(predictions))
        else:
            dt = COCO()
            dt.dataset = dict(images=gt.dataset['images'], categories=gt.dataset['categories'], annotations=[])
            dt.createIndex()
        evaluator = COCOeval_faster(gt, dt, 'bbox')
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    path.unlink()
    names = {c['id']: c['name'] for c in dataset['categories']}
    precision = evaluator.eval['precision'][0, :, :, 0, -1]  # IoU 0.50, all areas, maxDets 100
    classes = {}
    for index, category in enumerate(evaluator.params.catIds):
        values = precision[:, index]
        values = values[values > -1]
        classes[names[category]] = float(values.mean() * 100) if values.size else None

    def stat(value):
        return float(value * 100) if value >= 0 else None

    return dict(AP=stat(evaluator.stats[0]), AP50=stat(evaluator.stats[1]), AP75=stat(evaluator.stats[2]), classes=classes)


def best_copies(predictions):
    """Query x class flattening repeats a box once per class; keep its highest-scoring copy."""
    best = {}
    for d in predictions:
        key = (d['image_id'], tuple(round(float(v), 3) for v in d['bbox']))
        if key not in best or d['score'] > best[key]['score']:
            best[key] = d
    return list(best.values())


def with_overlap(dataset, predictions):
    """Yield (prediction, highest IoU with any GT, class of that GT) for the best copy of each box."""
    gts = defaultdict(list)
    for a in dataset['annotations']:
        gts[a['image_id']].append(a)
    for d in best_copies(predictions):
        annotations = gts[d['image_id']]
        if not annotations:
            yield d, 0., None
            continue
        overlap = iou_matrix([d['bbox']], [a['bbox'] for a in annotations])[0]
        index = int(overlap.argmax())
        yield d, float(overlap[index]), annotations[index]['category_id']


def type_oracle(dataset, predictions, iou=.5):
    """Every box with IoU>=iou to a GT takes that GT's class: no type confusion is left."""
    return [dict(d, category_id=category) if overlap >= iou else d
            for d, overlap, category in with_overlap(dataset, predictions)]


def without_background(dataset, predictions, iou=BACKGROUND_IOU):
    """Remove every box that touches no GT, keeping the original class copies of the rest."""
    kept = {(d['image_id'], tuple(round(float(v), 3) for v in d['bbox']))
            for d, overlap, _ in with_overlap(dataset, predictions) if overlap >= iou}
    return [d for d in predictions
            if (d['image_id'], tuple(round(float(v), 3) for v in d['bbox'])) in kept]


def ignore_others(dataset, keep):
    """Keep every image; GT outside the subset becomes ignored, so boxes on it are neither TP nor FP."""
    return dict(dataset, annotations=[dict(a, iscrowd=0 if a['id'] in keep else 1) for a in dataset['annotations']])


def only(dataset, keep):
    return dict(dataset, annotations=[a for a in dataset['annotations'] if a['id'] in keep])


def annotation_ids(dataset):
    """Annotation ids in the order of check_split_overlap.targets."""
    by_image = defaultdict(list)
    for a in dataset['annotations']:
        by_image[a['image_id']].append(a['id'])
    return [i for im in dataset['images'] for i in by_image[im['id']]]


def seen_scores(test, reference, data_root):
    """Best match correlation of each test GT against the reference split, by annotation id."""
    query = targets(test, data_root)
    best = best_match(query, targets(reference, data_root))
    return dict(zip(annotation_ids(test), best.tolist()))


def totals(rows):
    gt = sum(r['gt'] for r in rows)
    located = sum(r['located'] for r in rows)
    correct = sum(r['correct'] for r in rows)
    return dict(gt=gt, located_rate=located / gt if gt else None, correct_given_located=correct / located if located else None)


def top_confusions(rows, limit=10):
    pairs = []
    for row in rows:
        for key, value in row.items():
            if key.startswith('pred_') and key[5:] != row['gt_category'] and value:
                pairs.append(dict(gt=row['gt_category'], predicted=key[5:], count=value, fraction_of_gt=value / row['gt']))
    return sorted(pairs, key=lambda p: -p['count'])[:limit]


def fmt(value, digits=2):
    return '—' if value is None else f'{value:.{digits}f}'


def analyze(predictions_path, data_root, output, reference='trainval', split='test', threshold=SEEN_THRESHOLD):
    data_root, output = Path(data_root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    dataset = load_json(data_root / f'COCO/{split}.json')
    predictions = load_json(predictions_path)
    names = [c['name'] for c in dataset['categories']]
    scores = seen_scores(dataset, load_json(data_root / f'COCO/{reference}.json'), data_root)
    groups = dict(all=set(scores), seen={i for i, v in scores.items() if v >= threshold},
                  unseen={i for i, v in scores.items() if v < threshold})
    relabelled = type_oracle(dataset, predictions)
    clean = without_background(dataset, predictions)
    both = type_oracle(dataset, clean)
    agnostic_data, agnostic_predictions = class_agnostic(dataset, predictions)
    subsets, confusions = {}, {}
    for name in SUBSETS:
        keep = groups[name]
        view = ignore_others(dataset, keep)
        rows = confusion(only(dataset, keep), predictions, threshold=SCORE)
        confusions[name] = top_confusions(rows)
        subsets[name] = dict(official=evaluate(view, predictions, output),
                             type_oracle=evaluate(view, relabelled, output),
                             no_background=evaluate(view, clean, output),
                             both_oracles=evaluate(view, both, output),
                             class_agnostic=evaluate(ignore_others(agnostic_data, keep), agnostic_predictions, output),
                             rates=totals(rows),
                             low_score_located_rate=totals(confusion(only(dataset, keep), predictions, threshold=LOW_SCORE))['located_rate'],
                             per_class={r['gt_category']: dict(gt=r['gt'], located_rate=r['located_rate'],
                                                               correct_given_located=r['correct_given_located'],
                                                               missed=r['missed']) for r in rows})
    summary = dict(split=split, reference=reference, seen_threshold=threshold,
                   gt=len(scores), seen=len(groups['seen']), unseen=len(groups['unseen']),
                   subsets=subsets, top_confusions=confusions, definitions=dict(
                       seen=f'normalized correlation of 24x24 box crops >= {threshold} against {reference}, box sizes within 10%',
                       subset_AP='GT outside the subset is flagged iscrowd (ignored); images are unchanged',
                       type_oracle='best copy of each box; IoU>=0.5 to a GT takes that GT class',
                       no_background=f'boxes with IoU<{BACKGROUND_IOU} to every GT removed',
                       located=f'any-class prediction with IoU>=0.5; score>={SCORE}, or score>={LOW_SCORE} for the low-score rate',
                       remaining='100 minus both oracles: misses, poor localization and duplicates'))
    save_json(output / 'summary.json', summary)
    category = {c['id']: c['name'] for c in dataset['categories']}
    write_csv(output / 'targets.csv', [dict(annotation_id=a['id'], image_id=a['image_id'], category=category[a['category_id']],
                                            match_score=scores[a['id']], seen=scores[a['id']] >= threshold)
                                       for a in dataset['annotations']])
    (output / 'report.md').write_text(render(summary, names), encoding='utf-8')
    print(output / 'report.md', flush=True)
    return summary


def render(s, names):
    lines = [f"# {s['split']} 错误拆解", '',
             f"GT {s['gt']}：见过 {s['seen']}，没见过 {s['unseen']}（相对 {s['reference']}，相关 ≥ {s['seen_threshold']}）。", '',
             '## 1. AP50 分解', '',
             '| 子集 | GT | AP | AP50 | AP75 | 去掉型号混淆 | 去掉背景误检 | 两者都去掉 | 剩余缺口 | 类别无关 AP50 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in SUBSETS:
        r = s['subsets'][name]
        both = r['both_oracles']['AP50']
        lines.append(f"| {name} | {r['rates']['gt']} | {fmt(r['official']['AP'])} | {fmt(r['official']['AP50'])} | "
                     f"{fmt(r['official']['AP75'])} | {fmt(r['type_oracle']['AP50'])} | {fmt(r['no_background']['AP50'])} | "
                     f"{fmt(both)} | {fmt(None if both is None else 100 - both)} | {fmt(r['class_agnostic']['AP50'])} |")
    lines += ['', '三个 oracle 列是假设修好对应错误后的 AP50，用来比较各类错误的上限，不是可达到的指标。', '',
              '## 2. 定位与型号', '',
              f'| 子集 | 定位率（score≥{SCORE}） | 定位率（score≥{LOW_SCORE}） | 定位后型号正确率 |', '|---|---:|---:|---:|']
    for name in SUBSETS:
        r = s['subsets'][name]
        lines.append(f"| {name} | {fmt(r['rates']['located_rate'], 3)} | {fmt(r['low_score_located_rate'], 3)} | "
                     f"{fmt(r['rates']['correct_given_located'], 3)} |")
    lines += ['', 'Q4 判读：没见过子集的类别无关 AP50 仍低，说明飞机本身没被检出，属于目标性问题，Q4 放弃；'
              '类别无关 AP50 高而分类别 AP50 低，才值得试共享的目标性分数。', '',
              '## 3. 各类', '',
              '| 类别 | GT | 见过 | AP50 | 去掉型号混淆 | 见过 AP50 | 没见过 AP50 | 定位率 | 型号正确率 | 没见过定位率 | 没见过型号正确率 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c in names:
        a, seen, unseen = (s['subsets'][k] for k in SUBSETS)
        lines.append(f"| {c} | {a['per_class'][c]['gt']} | {seen['per_class'][c]['gt']} | {fmt(a['official']['classes'][c])} | "
                     f"{fmt(a['type_oracle']['classes'][c])} | {fmt(seen['official']['classes'][c])} | "
                     f"{fmt(unseen['official']['classes'][c])} | {fmt(a['per_class'][c]['located_rate'], 3)} | "
                     f"{fmt(a['per_class'][c]['correct_given_located'], 3)} | {fmt(unseen['per_class'][c]['located_rate'], 3)} | "
                     f"{fmt(unseen['per_class'][c]['correct_given_located'], 3)} |")
    lines += ['', '## 4. 最大的型号混淆（score≥0.3，IoU≥0.5）', '', '| 子集 | GT 类别 | 预测类别 | 数量 | 占该子集该类 GT |', '|---|---|---|---:|---:|']
    for name in SUBSETS:
        lines += [f"| {name} | {p['gt']} | {p['predicted']} | {p['count']} | {fmt(p['fraction_of_gt'], 3)} |"
                  for p in s['top_confusions'][name]]
    lines += ['', '## 定义', ''] + [f'- {k}：{v}' for k, v in s['definitions'].items()]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--predictions', type=Path, required=True, help='predictions.json.gz of the run on --split')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new output directory (must not exist)')
    parser.add_argument('--reference', default='trainval', help='split the run was trained on')
    parser.add_argument('--split', default='test')
    args = parser.parse_args()
    analyze(args.predictions, args.data_root, args.output, args.reference, args.split)


if __name__ == '__main__':
    main()
