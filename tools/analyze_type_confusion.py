#!/usr/bin/env python3
"""CPU follow-up on a finished paired diagnosis: is type confusion tied to image size?

Every image is resized to 640 for the network, so an aircraft's apparent size
depends on its source image size (800/1000/1200/1500) while its size in source
pixels does not. This reads the exported predictions already in a
compare_rem_102e.py output directory and reports, per source image size,
class-aware AP, class-agnostic AP50 and the localized-type accuracy, plus the
per-class box size in source and 640-input pixels and the largest confusions.
Writes DIAG/type_confusion/{report.md,summary.json}; no inference or training.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.compare_rem_102e import ROLES, class_agnostic, confusion, evaluate_coco, split_annotation
from tools.sar_analysis_core import load_json, save_json


def subset(dataset, predictions, image_ids):
    images = [im for im in dataset['images'] if im['id'] in image_ids]
    data = dict(dataset, images=images, annotations=[a for a in dataset['annotations'] if a['image_id'] in image_ids])
    return data, [d for d in predictions if d['image_id'] in image_ids]


def located_accuracy(rows):
    located = sum(r['located'] for r in rows)
    total = sum(r['gt'] for r in rows)
    correct = sum(r['correct'] for r in rows)
    return dict(gt=total, located_rate=located / total if total else None,
                correct_given_located=correct / located if located else None)


def size_table(dataset):
    images = {im['id']: im for im in dataset['images']}
    names = {c['id']: c['name'] for c in dataset['categories']}
    values = defaultdict(lambda: dict(source=[], input=[]))
    for a in dataset['annotations']:
        w, h = a['bbox'][2:]
        side = images[a['image_id']]['width']
        values[a['category_id']]['source'].append(max(w, h))
        values[a['category_id']]['input'].append(max(w, h) * 640 / side)
    rows = []
    for category in sorted(values):
        row = dict(category=names[category], gt=len(values[category]['source']))
        for kind in ('source', 'input'):
            q = np.quantile(values[category][kind], [.1, .5, .9])
            row.update({f'{kind}_p10': float(q[0]), f'{kind}_p50': float(q[1]), f'{kind}_p90': float(q[2])})
        # Relative spread: a size cue is usable only if within-class spread is small.
        row['source_iqr_over_median'] = float(np.subtract(*np.quantile(values[category]['source'], [.75, .25]))
                                              / np.median(values[category]['source']))
        row['input_iqr_over_median'] = float(np.subtract(*np.quantile(values[category]['input'], [.75, .25]))
                                             / np.median(values[category]['input']))
        rows.append(row)
    return rows


def top_confusions(rows, limit=8):
    pairs = []
    for row in rows:
        for key, value in row.items():
            if key.startswith('pred_') and key[5:] != row['gt_category'] and value:
                pairs.append(dict(gt=row['gt_category'], predicted=key[5:], count=value,
                                  fraction_of_gt=value / row['gt']))
    return sorted(pairs, key=lambda p: -p['count'])[:limit]


def fmt(value, digits=2):
    return '—' if value is None else f'{value:.{digits}f}'


def analyze(diag):
    diag = Path(diag).resolve()
    annotation, split = split_annotation(diag)
    dataset = load_json(annotation)
    predictions = {role: load_json(diag / role / 'predictions.json.gz') for role in ROLES}
    groups = defaultdict(set)
    for im in dataset['images']:
        groups[f"{im['width']}x{im['height']}"].add(im['id'])
    gt_per_group = Counter()
    images = {im['id']: im for im in dataset['images']}
    class_counts = defaultdict(Counter)
    names = {c['id']: c['name'] for c in dataset['categories']}
    for a in dataset['annotations']:
        im = images[a['image_id']]
        key = f"{im['width']}x{im['height']}"
        gt_per_group[key] += 1
        class_counts[key][names[a['category_id']]] += 1
    out = diag / 'type_confusion'
    out.mkdir(exist_ok=True)
    by_size = []
    for key in sorted(groups, key=lambda k: int(k.split('x')[0])):
        row = dict(image_size=key, images=len(groups[key]), gt=gt_per_group[key], classes=dict(class_counts[key]))
        for role in ROLES:
            data, dets = subset(dataset, predictions[role], groups[key])
            path = out / 'subset_gt.json'
            save_json(path, data)
            metrics = evaluate_coco(path, dets)['metrics']
            row[f'{role}_AP'], row[f'{role}_AP50'] = metrics['AP'], metrics['AP50']
            agnostic_data, agnostic_dets = class_agnostic(data, dets)
            save_json(path, agnostic_data)
            row[f'{role}_agnostic_AP50'] = evaluate_coco(path, agnostic_dets)['metrics']['AP50']
            row.update({f'{role}_{k}': v for k, v in located_accuracy(confusion(data, dets)).items() if k != 'gt'})
        by_size.append(row)
    (out / 'subset_gt.json').unlink()
    sizes = size_table(dataset)
    confusions = {role: top_confusions(confusion(dataset, predictions[role])) for role in ROLES}
    summary = dict(split=split, by_image_size=by_size, class_box_size=sizes, top_confusions=confusions,
                   definitions=dict(size='max(w, h) of the GT box; source = original pixels, input = after resize to 640',
                                    located='any-class prediction, score>=0.3, IoU>=0.5; not one-to-one'))
    save_json(out / 'summary.json', summary)

    lines = [f'# 型号混淆与源图尺寸（{split}）', '',
             '每张图都缩放到 640 输入，同一型号在不同尺寸源图里的输入尺寸不同；源图像素尺寸不受缩放影响。', '',
             '## 按源图尺寸', '',
             '| 源图 | 图 | GT | baseline AP | REM AP | baseline AP50 | baseline 类别无关 AP50 | '
             'baseline 定位率 | baseline 定位后型号正确率 | REM 定位后型号正确率 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in by_size:
        lines.append(f"| {r['image_size']} | {r['images']} | {r['gt']} | {fmt(r['baseline_AP'])} | {fmt(r['rem_AP'])} | "
                     f"{fmt(r['baseline_AP50'])} | {fmt(r['baseline_agnostic_AP50'])} | {fmt(r['baseline_located_rate'], 3)} | "
                     f"{fmt(r['baseline_correct_given_located'], 3)} | {fmt(r['rem_correct_given_located'], 3)} |")
    lines += ['', '各尺寸的类别 GT 数：', '']
    lines += [f"- {r['image_size']}：" + '，'.join(f'{k} {v}' for k, v in sorted(r['classes'].items())) for r in by_size]
    lines += ['', '## 各类框尺寸（max(w,h)，像素）', '',
              '| 类别 | GT | 源图 p10 | 源图 p50 | 源图 p90 | 源图 IQR/中位数 | 640 输入 p50 | 640 输入 IQR/中位数 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in sizes:
        lines.append(f"| {r['category']} | {r['gt']} | {fmt(r['source_p10'], 1)} | {fmt(r['source_p50'], 1)} | "
                     f"{fmt(r['source_p90'], 1)} | {fmt(r['source_iqr_over_median'], 3)} | {fmt(r['input_p50'], 1)} | "
                     f"{fmt(r['input_iqr_over_median'], 3)} |")
    lines += ['', 'IQR/中位数越小，尺寸越能区分型号；源图明显小于 640 输入，说明缩放把尺寸线索打散了。', '',
              '## 最大的型号混淆（score>=0.3，IoU>=0.5）', '',
              '| 模型 | GT 类别 | 预测类别 | 数量 | 占该类 GT |', '|---|---|---|---:|---:|']
    for role in ROLES:
        lines += [f"| {role} | {p['gt']} | {p['predicted']} | {p['count']} | {fmt(p['fraction_of_gt'], 3)} |"
                  for p in confusions[role]]
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(out / 'report.md', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('diag', nargs='+', type=Path, help='finished compare_rem_102e.py output directory')
    for diag in parser.parse_args().diag:
        analyze(diag)


if __name__ == '__main__':
    main()
