#!/usr/bin/env python3
"""CPU check: how many targets of one split reappear (overlapping tiles) in another.

Uses the same crop matcher as analyze_sar_characteristics.py (normalized
correlation of 24x24 box crops, box sizes within 10%). The key number for the
training-set choice is how many test targets are unseen in train but present in
val, i.e. what training on train+val would add. Writes OUTPUT/{report.md,summary.json}.
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.analyze_sar_characteristics import DUP_SIDE, DUP_THRESHOLDS, crop_vector, grey, side, similar
from tools.sar_analysis_core import load_json, save_json


def targets(dataset, data_root):
    names = {c['id']: c['name'] for c in dataset['categories']}
    by_image = defaultdict(list)
    for a in dataset['annotations']:
        by_image[a['image_id']].append(a)
    rows, vectors = [], []
    for im in dataset['images']:
        values, _ = grey(data_root / im['file_name'])
        for a in by_image[im['id']]:
            rows.append(dict(category=names[a['category_id']], size=side(a['bbox'])))
            vectors.append(crop_vector(values, a['bbox']))
    return rows, np.array(vectors, np.float32).reshape(-1, DUP_SIDE * DUP_SIDE)


def best_match(query, reference):
    """Best correlation of each query target against the reference split (0 if none passes size/lowest threshold)."""
    sizes_q = np.array([r['size'] for r in query[0]], float)
    sizes_r = np.array([r['size'] for r in reference[0]], float)
    best = np.zeros(len(sizes_q))
    for i, _, score in similar(query[1], reference[1], sizes_q, sizes_r):
        best[i] = max(best[i], score)
    return best


def analyze(data_root, output, splits=('train', 'val', 'test')):
    data_root, output = Path(data_root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    data = {}
    for split in splits:
        data[split] = targets(load_json(data_root / f'COCO/{split}.json'), data_root)
        print(f'loaded {split}: {len(data[split][0])} targets', flush=True)
    pairs = [('train', 'val'), ('train', 'test'), ('val', 'test')]
    best = {(ref, query): best_match(data[query], data[ref]) for ref, query in pairs}
    summary = dict(splits=list(splits), pairs=[], test_by_class=[], definitions=dict(
        matched=f'normalized correlation of {DUP_SIDE}x{DUP_SIDE} box crops >= threshold, box sizes within 10%',
        val_only='test target with no train match but a val match: seen only when training on train+val'))
    for ref, query in pairs:
        for t in DUP_THRESHOLDS:
            summary['pairs'].append(dict(reference=ref, query=query, threshold=t, gt=len(best[(ref, query)]),
                                         matched=int((best[(ref, query)] >= t).sum())))
    train_test, val_test = best[('train', 'test')], best[('val', 'test')]
    for t in DUP_THRESHOLDS:
        in_train, in_val = train_test >= t, val_test >= t
        summary['pairs'].append(dict(reference='train+val', query='test', threshold=t, gt=len(train_test),
                                     matched=int((in_train | in_val).sum()), val_only=int((~in_train & in_val).sum())))
    classes = sorted({r['category'] for r in data['test'][0]})
    for c in classes:
        mask = np.array([r['category'] == c for r in data['test'][0]])
        in_train, in_val = (train_test >= .9) & mask, (val_test >= .9) & mask
        summary['test_by_class'].append(dict(category=c, gt=int(mask.sum()), in_train=int(in_train.sum()),
                                             in_train_or_val=int((in_train | in_val).sum()),
                                             val_only=int((~in_train & in_val).sum())))
    save_json(output / 'summary.json', summary)
    lines = ['# 划分间重复目标', '', '| 参照 | 查询 | 阈值 | 查询 GT | 有匹配 | 比例 | 仅 val 有匹配 |', '|---|---|---:|---:|---:|---:|---:|']
    lines += [f"| {r['reference']} | {r['query']} | {r['threshold']} | {r['gt']} | {r['matched']} | "
              f"{r['matched'] / r['gt']:.3f} | {r.get('val_only', '—')} |" for r in summary['pairs']]
    lines += ['', '## test 各类（阈值 0.9）', '', '| 类别 | GT | train 中有匹配 | train+val 中有匹配 | 仅 val 有匹配 |',
              '|---|---:|---:|---:|---:|']
    lines += [f"| {r['category']} | {r['gt']} | {r['in_train']} | {r['in_train_or_val']} | {r['val_only']} |"
              for r in summary['test_by_class']]
    lines += ['', '## 定义', ''] + [f'- {k}：{v}' for k, v in summary['definitions'].items()]
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(output / 'report.md', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new output directory (must not exist)')
    args = parser.parse_args()
    analyze(args.data_root, args.output)


if __name__ == '__main__':
    main()
