#!/usr/bin/env python3
"""Analyze a frozen SAR baseline: curves, COCO metrics, sizes and error counts.

All AP/AR calculations retain full-score official predictions. Operational
error counts use predeclared score thresholds, not an AP confidence cutoff.
"""
import argparse
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from tools.sar_analysis_core import (ERROR_NAMES, SCALE_EDGES, SCALE_LABELS, error_analysis,
                                    load_json, object_size, save_json, summarize_values,
                                    training_summary)

METRICS = ('AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
           'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large')
CN = dict(background='背景误检', localization='定位错误', duplicate='重复检测',
          classification='分类错误', classification_localization='分类与定位同时错误')
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 180, 'figure.dpi': 110})


def csv_write(path, rows):
    rows = list(rows)
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def coco_evaluate(dataset, predictions, custom_scales=False):
    data = copy.deepcopy(dataset)
    detections = copy.deepcopy(predictions)
    if custom_scales:
        image_map = {im['id']: im for im in data['images']}
        for annotation in data['annotations']:
            im = image_map[annotation['image_id']]
            sx, sy = 640 / im['width'], 640 / im['height']
            x, y, w, h = annotation['bbox']
            annotation['bbox'] = [x * sx, y * sy, w * sx, h * sy]
            annotation['area'] = w * h * sx * sy
        for detection in detections:
            im = image_map[detection['image_id']]
            sx, sy = 640 / im['width'], 640 / im['height']
            x, y, w, h = detection['bbox']
            detection['bbox'] = [x * sx, y * sy, w * sx, h * sy]
        for im in data['images']:
            im['width'] = im['height'] = 640
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = data
        coco.createIndex()
        if detections:
            dt = coco.loadRes(detections)
        else:
            dt = COCO()
            dt.dataset = dict(images=data['images'], categories=data['categories'], annotations=[])
            dt.createIndex()
        evaluation = COCOeval(coco, dt, 'bbox')
        if custom_scales:
            evaluation.params.areaRng = [[0, 1e10]] + [[low ** 2, np.nextafter(high ** 2, -np.inf)
                                                      if np.isfinite(high) else 1e10]
                                                     for low, high in zip(SCALE_EDGES[:-1], SCALE_EDGES[1:])]
            evaluation.params.areaRngLbl = ['all'] + list(SCALE_LABELS)
        evaluation.evaluate()
        evaluation.accumulate()
        if not custom_scales:
            evaluation.summarize()
    return evaluation


def mean_valid(values):
    valid = np.asarray(values)[np.asarray(values) >= 0]
    return float(valid.mean()) if valid.size else None


def area_metrics(evaluation, area=0, category=None):
    precision = evaluation.eval['precision']
    recall = evaluation.eval['recall']
    k = slice(None) if category is None else slice(category, category + 1)
    return dict(AP=mean_valid(precision[:, :, k, area, -1]),
                AP50=mean_valid(precision[0, :, k, area, -1]),
                AP75=mean_valid(precision[5, :, k, area, -1]),
                AR100=mean_valid(recall[:, k, area, -1]),
                recall50=mean_valid(recall[0, k, area, -1]),
                recall75=mean_valid(recall[5, k, area, -1]))


def figure_save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def training_plots(log, output, summary):
    epoch = np.array([r['epoch'] for r in log])
    values = np.array([r['test_coco_eval_bbox'] for r in log]) * 100
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for i, label in [(0, 'mAP'), (1, 'AP50'), (2, 'AP75')]:
        ax[0, 0].plot(epoch, values[:, i], label=label)
    ax[0, 0].scatter([summary['best_epoch']], [100 * summary['best_metrics'][0]], color='black', zorder=5)
    for i, label in [(6, 'AR@1'), (7, 'AR@10'), (8, 'AR@100')]:
        ax[0, 1].plot(epoch, values[:, i], label=label)
    for i, label in [(3, 'AP small (original)'), (4, 'AP medium (original)'), (5, 'AP large (original)')]:
        ax[1, 0].plot(epoch, values[:, i], label=label)
    ax[1, 1].plot(epoch, [r['train_loss'] for r in log], label='Total training loss', color='#bc3d42')
    for a in ax.flat:
        for boundary in [4, 49, 90]:
            a.axvline(boundary, color='gray', linestyle=':', alpha=.55)
        a.set_xlabel('Epoch (zero-based)')
        a.grid(alpha=.2)
        a.legend(fontsize=8)
    ax[0, 0].set_title('Validation AP (%)')
    ax[0, 1].set_title('COCO average recall (%) across IoU .50:.95')
    ax[1, 0].set_title('COCO area = original annotation area')
    ax[1, 1].set_title('Training loss; no validation loss was recorded')
    figure_save(fig, output / 'training_curves.png')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for family in ['bbox', 'giou', 'mal', 'fgl', 'ddf']:
        keys = [k for k in log[0] if k.startswith('train_loss_' + family)]
        axes[0].plot(epoch, [sum(r.get(k, 0) for k in keys) for r in log], label=family.upper())
    axes[0].legend()
    axes[0].set_title('Sum of logged weighted loss terms by family')
    backbone_lr = np.array([r['train_lr'] for r in log])
    axes[1].plot(epoch, backbone_lr, label='Backbone non-BN (logged)')
    axes[1].plot(epoch, backbone_lr * 10, label='Main LR (same schedule, x10)')
    axes[1].set_yscale('log')
    axes[1].set_title('Epoch-average LR; group 0 is backbone non-BN')
    axes[1].legend(fontsize=8)
    for a in axes:
        a.set_xlabel('Epoch')
        a.grid(alpha=.2)
        for boundary in [4, 49, 90]:
            a.axvline(boundary, color='gray', linestyle=':', alpha=.55)
    figure_save(fig, output / 'loss_and_lr.png')
    keys = [k for k in log[0] if k.startswith('train_loss_')]
    fig, axes = plt.subplots(int(np.ceil(len(keys) / 4)), 4, figsize=(15, np.ceil(len(keys) / 4) * 2.0))
    for a, key in zip(axes.flat, keys):
        a.plot(epoch, [r[key] for r in log], linewidth=1)
        a.set_title(key.removeprefix('train_'), fontsize=8)
        a.tick_params(labelsize=7)
        a.grid(alpha=.2)
        a.axvline(90, color='gray', linestyle=':', alpha=.5)
    for a in list(axes.flat)[len(keys):]:
        a.set_visible(False)
    figure_save(fig, output / 'all_loss_terms.png')


def scale_analysis(splits, objects, evaluation, output):
    rows, summaries = [], {}
    categories = {c['id']: c['name'] for c in splits['val']['categories']}
    for split, dataset in splits.items():
        images = {im['id']: im for im in dataset['images']}
        split_rows = []
        for g in dataset['annotations']:
            size = object_size(g, images[g['image_id']])
            row = dict(split=split, gt_id=g['id'], image_id=g['image_id'],
                       category=categories[g['category_id']], **size,
                       input_side_480=size['input_side'] * .75,
                       input_side_800=size['input_side'] * 1.25,
                       scale_bin=__import__('tools.sar_analysis_core', fromlist=['scale_bin']).scale_bin(size['input_side']))
            split_rows.append(row)
        rows.extend(split_rows)
        summaries[split] = dict(images=len(images), objects=len(split_rows),
                               original_side=summarize_values([r['side'] for r in split_rows]),
                               input_side_640=summarize_values([r['input_side'] for r in split_rows]),
                               input_min_side=summarize_values([min(r['input_width'], r['input_height']) for r in split_rows]),
                               input_scale_bins={label:sum(r['scale_bin'] == label for r in split_rows) for label in SCALE_LABELS})
    csv_write(output / 'object_sizes_all_splits.csv', rows)
    bin_results = []
    for i, label in enumerate(SCALE_LABELS, 1):
        members = [o for o in objects if o['scale_bin'] == label]
        bin_results.append(dict(scale=label, gt=len(members), **area_metrics(evaluation, i),
                                operational_tp=sum(o['detected'] for o in members),
                                operational_recall=sum(o['detected'] for o in members) / len(members) if members else None))
    csv_write(output / 'input_scale_performance.csv', bin_results)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = {'train':'#34699a', 'val':'#df8c29', 'test':'#54966a'}
    edges = np.geomspace(4, 600, 36)
    for split in splits:
        selected = [r for r in rows if r['split'] == split]
        axes[0, 0].hist([r['side'] for r in selected], bins=edges, density=True, histtype='step',
                        label=split, color=colors[split], linewidth=1.6)
        axes[0, 1].hist([r['input_side'] for r in selected], bins=edges, density=True, histtype='step',
                        label=split, color=colors[split], linewidth=1.6)
    for a in axes[0]:
        a.set_xscale('log')
        a.set_xlabel('sqrt(box area), pixels')
        a.set_ylabel('Density')
        a.legend()
    axes[0, 0].set_title('Original annotation sizes')
    axes[0, 1].set_title('Sizes after direct resize to 640x640')
    x = np.arange(len(bin_results))
    axes[1, 0].bar(x, [r['gt'] for r in bin_results], color='#34699a')
    axes[1, 0].set_xticks(x, SCALE_LABELS)
    axes[1, 0].set_title('Validation GT support at 640 input')
    for key, label in [('AP', 'COCO AP'), ('AP50', 'COCO AP50'), ('operational_recall', 'Recall score>=.3, IoU>=.5')]:
        axes[1, 1].plot(x, [r[key] * 100 if r[key] is not None else np.nan for r in bin_results], marker='o', label=label)
    axes[1, 1].set_xticks(x, SCALE_LABELS)
    axes[1, 1].set_ylim(0, 102)
    axes[1, 1].set_ylabel('%')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_title('Performance by input sqrt(area)')
    for a in axes.flat:
        a.grid(alpha=.2)
    figure_save(fig, output / 'target_scales.png')
    return summaries, bin_results


def format_percent(value):
    return '—' if value is None or value < 0 else f'{value * 100:.2f}'


def make_report(args, training, official, reproduced, class_results, scales, bins, errors, sensitivities, output, metadata):
    best = training['best_metrics']
    final = training['last_metrics']
    original_counts = dict(small=sum(a['area'] < 32 ** 2 for a in load_json(args.annotations / 'val.json')['annotations']),
                           medium=sum(32 ** 2 <= a['area'] < 96 ** 2 for a in load_json(args.annotations / 'val.json')['annotations']),
                           large=sum(a['area'] >= 96 ** 2 for a in load_json(args.annotations / 'val.json')['annotations']))
    lines = [
        '# SAR-AIRcraft DEIM-M baseline 结果分析', '',
        f"训练已完成 {training['completed_epochs']} 轮（日志 epoch {training['first_epoch']}–{training['last_epoch']}）。"
        f"验证集最佳为 **epoch {training['best_epoch']}（第 {training['best_epoch'] + 1} 轮）**："
        f"**mAP {format_percent(best[0])}、AP50 {format_percent(best[1])}、AP75 {format_percent(best[2])}**。", '',
        '本报告的性能、错误和案例均来自 **val 的 442 张图**；日志中的 `test_coco_eval_bbox` 实际也是 val。'
        'test 仅统计标注尺寸，不运行预测，不用于选权重或诊断方法收益。', '',
        '## 1. 协议与最佳权重', '',
        f'- 原运行：`{args.run}`；训练 revision：`{args.revision}`。',
        '- DEIM-D-FINE-M，7 类；COCO M 权重初始化；seed 0；GPU 0/1；全局 batch 28；102 epochs。',
        '- 主 LR 3.5e-4，backbone 非 BN 3.5e-5；AdamW；2000 步 warmup，flat 49，最后 12 轮固定最小 LR。',
        '- 训练基础 640，前 90 轮保留 480–800 多尺度；关闭 RandomZoomOut/RandomIoUCrop；其余固定协议保留。',
        '- EMA 权重；验证输入固定 640×640；使用原后处理 top-300、不加 NMS、不预先按置信度过滤；FP32 推理与训练时验证一致。',
        f"- 选用 `{Path(metadata['checkpoint']).name}`，检查点内部 last_epoch={metadata['checkpoint_last_epoch']}。"
        '官方 solver 在收尾阶段会回载 best_stg1 并回退内部轮次，因此应以验证日志的逻辑 epoch 选模，不能把内部 last_epoch 当作最佳逻辑 epoch。', '',
        '| 指标（%） | 日志最佳 | 最佳权重复算 | 末轮 |', '|---|---:|---:|---:|']
    for i, key in [(0, 'AP'), (1, 'AP50'), (2, 'AP75'), (8, 'AR100')]:
        lines.append(f'| {key} | {format_percent(best[i])} | {format_percent(reproduced[key])} | {format_percent(final[i])} |')
    lines += ['', '**AR100 是 IoU=.50:.95 的平均召回，不是固定 IoU=.50 的 Recall。**',
              f"同一最佳权重的 COCO Recall@IoU.50={format_percent(reproduced['recall50'])}%，"
              f"Recall@IoU.75={format_percent(reproduced['recall75'])}%（maxDets=100，每图每类）。", '',
              '### 各类结果', '', '| 类别 | GT | AP | AP50 | AP75 | AR100 | Recall（score≥0.3, IoU≥0.5） |', '|---|---:|---:|---:|---:|---:|---:|']
    for row in class_results:
        lines.append('| ' + ' | '.join([row['category'], str(row['gt'])] + [format_percent(row[k]) for k in ['AP', 'AP50', 'AP75', 'AR100', 'operational_recall']]) + ' |')
    lines += ['', '## 2. 训练曲线与收敛', '',
              f"首次达到距离全程最佳不超过 0.5 AP 点：epoch {training['first_epoch_within_0_5_ap_point']}。"
              f"最后 12 轮 AP 均值 {100 * training['last12_ap_mean']:.4f}，标准差 {100 * training['last12_ap_std']:.4f} 点，"
              f"极差 {100 * training['last12_ap_range']:.4f} 点；最佳到末轮差 {(best[0] - final[0]) * 100:.4f} AP 点。", '',
              '| 阶段（零起始 epoch） | 阶段最佳 epoch | 最佳 AP | 阶段末 AP |', '|---|---:|---:|---:|']
    for row in training['stages']:
        lines.append(f"| {row['start']}–{row['end']} | {row['best_epoch']} | {100 * row['best_ap']:.4f} | {100 * row['last_ap']:.4f} |")
    lines += ['', f"训练总损失从 {training['initial_loss']:.4f} 下降至 {training['final_loss']:.4f}；"
              f"有 FP32 异常重算的 epoch 数为 {training['retry_epochs']}。"
              '增强和 EMA 回载会改变不同阶段的损失与 AP，不能将阶段边界的跃变单独解释成优化不稳定。'
              '日志未记录验证损失，因此不能凭训练损失证明不存在过拟合。', '',
              '![训练与验证曲线](training_curves.png)', '', '![损失族与学习率](loss_and_lr.png)', '',
              '全部逐项损失见 [全损失曲线](all_loss_terms.png) 与 [每轮指标 CSV](epoch_metrics.csv)。'
              '损失族图对日志中已经加权的主分支、辅助、encoder、去噪项求和；它不是未加权原始损失。', '',
              '## 3. 目标尺度', '',
              '原始尺寸以 bbox 宽/高和 √(w×h) 统计；网络尺寸按 x×640/W、y×640/H 换算。'
              '480/800 仅展示相同原始框直接 resize 的理论上下尺度，未声称重建 Mosaic/MixUp 后的实际训练框分布。', '',
              '| split | 图像 | 目标 | 原图 √area 中位数 | 640 输入 √area 中位数 | 640 输入 P10 / P90 |', '|---|---:|---:|---:|---:|---:|']
    for split, stat in scales.items():
        s = stat['input_side_640']
        lines.append(f"| {split} | {stat['images']} | {stat['objects']} | {stat['original_side']['p50']:.2f} | {s['p50']:.2f} | {s['p10']:.2f} / {s['p90']:.2f} |")
    lines += ['', '### 按 640 输入中的 √area 分组', '',
              '| 输入尺度 px | GT | AP | AP50 | AP75 | AR100 | Recall（score≥0.3, IoU≥0.5） |', '|---|---:|---:|---:|---:|---:|---:|']
    for row in bins:
        lines.append('| ' + ' | '.join([row['scale'], str(row['gt'])] + [format_percent(row[k]) for k in ['AP', 'AP50', 'AP75', 'AR100', 'operational_recall']]) + ' |')
    lines += ['', '输入尺度 AP 在把 GT 与预测框一同换算到 640 后重新执行 COCO area-range 评估，'
              '保留所有图像和预测，并按 COCO 规则忽略分组之外的 GT/检测；不通过简单删 GT 制造额外背景误检。', '',
              f"原图 COCO small/medium/large 的 GT 数分别为 {original_counts['small']}/{original_counts['medium']}/{original_counts['large']}。"
              '原图 APs 不能直接代表网络中的小目标性能；样本极少的分组不作稳定结论。', '',
              '![目标尺度分布和分组性能](target_scales.png)', '',
              '完整宽高、√area、480/640/800 换算及分组在 [目标尺寸 CSV](object_sizes_all_splits.csv)。', '',
              '## 4. 错误类型与阈值敏感性', '',
              f"主诊断阈值预先固定 score≥0.3、IoU≥0.5：GT {errors['gt']}，保留预测 {errors['predictions']}，"
              f"TP {errors['tp']}，FP {errors['fp']}，FN {errors['fn']}；"
              f"Precision {100 * errors['precision']:.2f}%，Recall {100 * errors['recall']:.2f}%，F1 {100 * errors['f1']:.2f}%。", '',
              f"**漏检 FN={errors['fn']}，占 GT 的 {100 * errors['miss_fraction_of_gt']:.2f}%。**"
              '漏检按 GT 计数，以下误检按预测计数；二者可能描述同一失败目标，不相加作为互斥错误比例。', '',
              '| 互斥 FP 类型 | 数量 | 占全部 FP | 占保留预测 |', '|---|---:|---:|---:|']
    for name in ERROR_NAMES:
        stat = errors['fp_types'][name]
        lines.append(f"| {CN[name]} | {stat['count']} | {100 * stat['fraction_of_fp']:.2f}% | {100 * stat['fraction_of_predictions']:.2f}% |")
    lines += ['', '先按分数降序与尚未匹配的同类 GT 一对一匹配。未成为 TP 的预测按以下顺序判定：', '',
              '1. 同类 IoU≥0.5 且 GT 已匹配：重复检测。',
              '2. 与异类 GT IoU≥0.5：分类错误。',
              '3. 与同类 GT 0.1≤IoU<0.5：定位错误。',
              '4. 只与异类 GT 有 0.1≤IoU<0.5：分类与定位同时错误。',
              '5. 与所有 GT 的最大 IoU<0.1：背景误检。', '',
              '额外保留“分类与定位同时错误”以完整覆盖预测，避免把它混入背景或纯定位。'
              '这是本项目的数量分解，不是 TIDE 的 dAP 贡献分析；IoU 阈值、置信度和类别占比都会影响数量。', '',
              '| score 阈值 | TP | FP | FN | Precision | Recall | F1 |', '|---|---:|---:|---:|---:|---:|---:|']
    for row in sensitivities:
        lines.append(f"| {row['score_threshold']} | {row['tp']} | {row['fp']} | {row['fn']} | {100 * row['precision']:.2f} | {100 * row['recall']:.2f} | {100 * row['f1']:.2f} |")
    lines += ['', '![错误统计](errors_and_confusion.png)', '',
              '明细见 [预测分类 CSV](prediction_errors.csv)、[漏检 CSV](missed_gt.csv)、[各类指标 CSV](class_performance.csv)。', '',
              '## 5. SAR 困难样本案例库', '',
              '图例、完整候选案例与筛选依据见 [案例库](cases/index.html)、[案例清单](cases/case_manifest.csv)。'
              '案例按背景纹理/杂波候选、低对比度、密集、外观稀疏/细长，以及具体错误类型组织；同一图可属于多种现象。'
              '这些是图像与框区域的可测代理，不能单凭框内亮度自动认定 SAR 物理成因；已抽查记录与剩余候选状态分开保存。', '',
              '## 6. 复现与限制', '',
              '- 所有 AP/AR 使用未阈值过滤的官方后处理结果；0.3 只用于错误计数和可视化。',
              '- 一次 seed=0 的验证结果不能证明模块在其他 seed、全集或其他数据集上的收益。',
              '- 不修改训练配置、标签、检查点或原结果；分析输出单独保存。',
              '- 本报告不以观察到的失败样本直接断言标注错误，也不将外观代理当成因果解释。',
              '- 精确运行与恢复命令见 `HANDOFF.md`；结构化结果见 `summary.json`。', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--revision', required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    log = [json.loads(line) for line in (args.source / 'log.txt').read_text().splitlines() if line.strip()]
    metadata = load_json(str(args.predictions) + '.metadata.json')
    splits = {name:load_json(args.annotations / f'{name}.json') for name in ['train', 'val', 'test']}
    predictions = load_json(args.predictions)
    summary = training_summary(log)
    csv_write(output / 'epoch_metrics.csv', [dict(epoch=r['epoch'],
                                                **{name:r['test_coco_eval_bbox'][i] * 100 for i, name in enumerate(METRICS)},
                                                **{k:v for k, v in r.items() if k.startswith('train_')}) for r in log])
    training_plots(log, output, summary)
    evaluation = coco_evaluate(splits['val'], predictions)
    print('Best checkpoint COCO metrics:', evaluation.stats.tolist(), flush=True)
    metrics = area_metrics(evaluation)
    sensitivity = [error_analysis(splits['val'], predictions, t)[0] for t in [.1, .3, .5]]
    errors, records, misses, objects = error_analysis(splits['val'], predictions, .3)
    csv_write(output / 'prediction_errors.csv', records)
    csv_write(output / 'missed_gt.csv', misses)
    csv_write(output / 'validation_objects.csv', objects)
    csv_write(output / 'threshold_sensitivity.csv', sensitivity)
    class_results = []
    for k, category in enumerate(evaluation.params.catIds):
        name = next(c['name'] for c in splits['val']['categories'] if c['id'] == category)
        members = [o for o in objects if o['category_id'] == category]
        class_results.append(dict(category=name, category_id=category, gt=len(members),
                                  **area_metrics(evaluation, category=k),
                                  operational_recall=sum(o['detected'] for o in members) / len(members) if members else None))
    csv_write(output / 'class_performance.csv', class_results)
    scaled_evaluation = coco_evaluate(splits['val'], predictions, custom_scales=True)
    scales, bins = scale_analysis(splits, objects, scaled_evaluation, output)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].bar(range(5), [errors['fp_types'][name]['count'] for name in ERROR_NAMES], color='#b65d54')
    axes[0].set_xticks(range(5), ['Background', 'Localization', 'Duplicate', 'Class', 'Class+Loc'], rotation=15)
    axes[0].set_title(f"Exclusive FP types; separate FN={errors['fn']}")
    categories = [c['category'] for c in class_results]
    matrix = np.zeros((len(categories), len(categories)), dtype=int)
    ids = list(evaluation.params.catIds)
    for row in records:
        if row['kind'] in ['classification', 'classification_localization']:
            matrix[ids.index(row['gt_category_id']), ids.index(row['category_id'])] += 1
    im = axes[1].imshow(matrix, cmap='Blues')
    axes[1].set_xticks(range(len(categories)), categories, rotation=45, ha='right')
    axes[1].set_yticks(range(len(categories)), categories)
    axes[1].set_xlabel('Predicted class')
    axes[1].set_ylabel('Associated GT class')
    axes[1].set_title('Class/class+loc FP associations (counts, not unique GT)')
    for i in range(len(categories)):
        for j in range(len(categories)):
            if matrix[i, j]:
                axes[1].text(j, i, str(matrix[i, j]), ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=axes[1], fraction=.045)
    figure_save(fig, output / 'errors_and_confusion.png')
    csv_write(output / 'classification_associations.csv', [dict(gt=categories[i], predicted=categories[j], count=int(matrix[i, j]))
                                                         for i in range(len(categories)) for j in range(len(categories))])
    result = dict(training=summary, reproduced_metrics=metrics, reproduced_coco_stats=evaluation.stats.tolist(),
                  metric_reproduction_delta_ap=metrics['AP'] - summary['best_metrics'][0],
                  classes=class_results, scales=scales, input_scale_performance=bins,
                  errors=errors, threshold_sensitivity=sensitivity, prediction_metadata=metadata)
    save_json(output / 'summary.json', result)
    make_report(args, summary, evaluation, metrics, class_results, scales, bins, errors, sensitivity, output, metadata)
    print(json.dumps(dict(best_epoch=summary['best_epoch'], metrics=metrics, errors=errors,
                         scale_bins=bins, reproduction_delta_ap=result['metric_reproduction_delta_ap']), indent=2), flush=True)


if __name__ == '__main__':
    main()
