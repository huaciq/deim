#!/usr/bin/env python3
"""Export two saved runs on one server and compare their detections on val or test."""

import argparse
from collections import Counter, defaultdict
import contextlib
import copy
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.sar_analysis_core import classify_image, error_analysis, iou_matrix, load_json, save_json, summarize_values


ROLES = ('baseline', 'rem')
METRICS = ('AP', 'AP50', 'AP75', 'APs', 'APm', 'APl')
IOU_THRESHOLDS = (.50, .75, .85, .90)
SCORE_THRESHOLD = .3


def delta(value, reference):
    return value - reference if value is not None and reference is not None else None


def display(value):
    return f'{value:.4f}' if value is not None else 'n/a'


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def epoch_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    return sorted((row for row in rows if 'test_coco_eval_bbox' in row), key=lambda row: row['epoch'])


def select_checkpoint(run, config, rows):
    best = max(rows, key=lambda row: row['test_coco_eval_bbox'][0])
    stop = config['train_dataloader']['collate_fn']['stop_epoch']
    path = run / ('best_stg2.pth' if best['epoch'] >= stop else 'best_stg1.pth')
    if not path.is_file():
        raise FileNotFoundError(f'Best logged validation AP is at epoch {best["epoch"]}; transfer {path}.')
    return path, best


def prepare_run(run, role_dir, data_root, annotation):
    run = Path(run).resolve()
    role_dir.mkdir(parents=True, exist_ok=True)
    original = yaml.safe_load((run / 'resolved_config.yml').read_text(encoding='utf-8'))
    rows = epoch_rows(run / 'log.txt')
    checkpoint, best = select_checkpoint(run, original, rows)
    for filename in ('resolved_config.yml', 'log.txt', 'git_revision.txt', 'run_metadata.json'):
        source = run / filename
        if source.is_file():
            (role_dir / filename).write_bytes(source.read_bytes())
    runtime = copy.deepcopy(original)
    runtime.pop('__include__', None)
    runtime['HGNetv2']['pretrained'] = False
    runtime['val_dataloader']['dataset'].update(img_folder=str(data_root), ann_file=str(annotation))
    runtime['val_dataloader']['total_batch_size'] = 16  # export: 8 per GPU, independent of training batch
    runtime.update(output_dir=str(role_dir), summary_dir=str(role_dir / 'summary'))
    config_path = role_dir / 'inference_config.yml'
    config_path.write_text(yaml.safe_dump(runtime, sort_keys=False, allow_unicode=True), encoding='utf-8')
    selection = dict(source_run=str(run), checkpoint=str(checkpoint), logical_epoch=best['epoch'],
                     logged_best_metrics=best['test_coco_eval_bbox'][:6],
                     inference_config=str(config_path), predictions=str(role_dir / 'predictions.json.gz'))
    save_json(role_dir / 'selection.json', selection)
    return selection


def split_annotation(output):
    """The archived split annotation; outputs made before --split are val."""
    manifest = output / 'manifest.json'
    split = load_json(manifest).get('split', 'val') if manifest.is_file() else 'val'
    return output / f'{split}.json', split


def prepare(baseline_run, rem_run, data_root, output, split='val'):
    output = Path(output).resolve()
    data_root = Path(data_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    annotation = data_root / f'COCO/{split}.json'
    # One common, user-confirmed dataset; no transferred annotations are required.
    (output / f'{split}.json').write_bytes(annotation.read_bytes())
    sources = dict(baseline=baseline_run, rem=rem_run)
    selections = {role: prepare_run(sources[role], output / role, data_root, annotation) for role in ROLES}
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    manifest = dict(output=str(output), data_root=str(data_root), annotation_file=str(annotation), split=split,
                    analysis_git_revision=revision, runs=selections,
                    export=dict(gpus=[0, 1], precision='float32', batch_per_gpu=8,
                                confidence_filter=None, extra_nms=False),
                    diagnostic_score_threshold=SCORE_THRESHOLD,
                    diagnostic_iou_thresholds=IOU_THRESHOLDS)
    save_json(output / 'manifest.json', manifest)
    return manifest


def evaluate_coco(annotation, predictions):
    from faster_coco_eval import COCO, COCOeval_faster

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(annotation))
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
    precision = evaluator.eval['precision']  # T,R,K,A,M; -1 means absent GT

    def ap(values):
        values = values[values >= 0]
        return float(values.mean() * 100) if values.size else None

    per_iou = {f'{threshold:.2f}': ap(precision[index, :, :, 0, -1])
               for index, threshold in enumerate(evaluator.params.iouThrs)}
    class_rows = []
    at75 = np.isclose(evaluator.params.iouThrs, .75)
    for index, category in enumerate(evaluator.params.catIds):
        class_rows.append(dict(category_id=int(category), AP=ap(precision[:, :, index, 0, -1]),
                               AP75=ap(precision[at75, :, index, 0, -1]),
                               APm=ap(precision[:, :, index, 2, -1]),
                               APl=ap(precision[:, :, index, 3, -1])))
    return dict(metrics={key: float(value * 100) if value >= 0 else None
                         for key, value in zip(METRICS, evaluator.stats[:6])},
                per_iou_ap=per_iou, classes=class_rows)


def class_agnostic(dataset, predictions):
    """Single-class copy of GT and predictions.

    Query x class flattening repeats a query's box once per class; keep only the
    highest-scoring copy of each identical box so duplicates are not counted as FP.
    """
    data = dict(dataset, categories=[dict(id=0, name='aircraft')],
                annotations=[dict(a, category_id=0) for a in dataset['annotations']])
    best = {}
    for d in predictions:
        key = (d['image_id'], tuple(round(float(v), 3) for v in d['bbox']))
        if key not in best or d['score'] > best[key]['score']:
            best[key] = dict(d, category_id=0)
    return data, list(best.values())


def confusion(dataset, predictions, threshold=SCORE_THRESHOLD, iou=.5):
    """GT class versus the class of the highest-scoring overlapping prediction.

    Any-class predictions with score>=threshold and IoU>=iou count; this is not a
    one-to-one assignment, it separates "not localized" from "localized, wrong type".
    """
    by_image = defaultdict(list)
    for d in predictions:
        if d['score'] >= threshold:
            by_image[d['image_id']].append(d)
    ids = [c['id'] for c in dataset['categories']]
    names = {c['id']: c['name'] for c in dataset['categories']}
    counts = {category: Counter() for category in ids}
    for g in dataset['annotations']:
        detections = by_image[g['image_id']]
        label = 'missed'
        if detections:
            overlap = iou_matrix([g['bbox']], [d['bbox'] for d in detections])[0]
            hits = [d for d, value in zip(detections, overlap) if value >= iou]
            if hits:
                label = max(hits, key=lambda d: d['score'])['category_id']
        counts[g['category_id']][label] += 1
    rows = []
    for category in ids:
        total = sum(counts[category].values())
        located = total - counts[category]['missed']
        rows.append(dict(gt_category_id=category, gt_category=names[category], gt=total, located=located,
                         correct=counts[category][category], missed=counts[category]['missed'],
                         located_rate=located / total if total else None,
                         correct_given_located=counts[category][category] / located if located else None,
                         **{f'pred_{names[c]}': counts[category][c] for c in ids}))
    return rows


def size_group(annotation):
    area = annotation.get('area', annotation['bbox'][2] * annotation['bbox'][3])
    return 'small' if area < 32 ** 2 else 'medium' if area < 96 ** 2 else 'large'


def geometry(prediction, truth, image):
    x, y, w, h = prediction
    gx, gy, gw, gh = truth
    sx, sy = 640 / image['width'], 640 / image['height']
    return dict(dx640=(x + w / 2 - gx - gw / 2) * sx,
                dy640=(y + h / 2 - gy - gh / 2) * sy,
                width_error_pct=(w / gw - 1) * 100, height_error_pct=(h / gh - 1) * 100,
                left640=(x - gx) * sx, top640=(y - gy) * sy,
                right640=(x + w - gx - gw) * sx, bottom640=(y + h - gy - gh) * sy)


def pair_detections(dataset, predictions):
    images = {image['id']: image for image in dataset['images']}
    categories = {category['id']: category['name'] for category in dataset['categories']}
    gts = defaultdict(list)
    dts = {role: defaultdict(list) for role in ROLES}
    for annotation in dataset['annotations']:
        gts[annotation['image_id']].append(annotation)
    for role in ROLES:
        for detection in predictions[role]:
            dts[role][detection['image_id']].append(detection)
    pairs, common, candidates = [], [], []
    for image_id, annotations in gts.items():
        for threshold in IOU_THRESHOLDS:
            matches = {}
            for role in ROLES:
                records, _ = classify_image(annotations, dts[role][image_id], SCORE_THRESHOLD, threshold)
                matches[role] = {record['gt_id']: record for record in records if record['kind'] == 'tp'}
            for gt in annotations:
                b, r = matches['baseline'].get(gt['id']), matches['rem'].get(gt['id'])
                status = 'both' if b and r else 'gained' if r else 'lost' if b else 'neither'
                row = dict(image_id=image_id, gt_id=gt['id'], category_id=gt['category_id'],
                           category=categories[gt['category_id']], size=size_group(gt),
                           iou_threshold=threshold, status=status,
                           baseline_score=b['score'] if b else None, rem_score=r['score'] if r else None,
                           baseline_iou=b['iou'] if b else None, rem_iou=r['iou'] if r else None)
                pairs.append(row)
                if threshold == .5 and status == 'both':
                    item = dict(row, delta_iou=r['iou'] - b['iou'], delta_score=r['score'] - b['score'])
                    for role, record in (('baseline', b), ('rem', r)):
                        item.update({f'{role}_{key}': value for key, value in
                                     geometry(record['bbox'], gt['bbox'], images[image_id]).items()})
                    common.append(item)
        for gt in annotations:
            item = dict(image_id=image_id, gt_id=gt['id'], category=categories[gt['category_id']], size=size_group(gt))
            for role in ROLES:
                detections = sorted(dts[role][image_id], key=lambda d: -d['score'])
                overlap = iou_matrix([gt['bbox']], [d['bbox'] for d in detections])[0]
                same = [i for i, d in enumerate(detections) if d['category_id'] == gt['category_id']]
                same_iou = max((float(overlap[i]) for i in same), default=0.)
                high_quality_scores = [detections[i]['score'] for i in same if overlap[i] >= .75]
                wrong_scores = [d['score'] for i, d in enumerate(detections)
                                if d['category_id'] != gt['category_id'] and overlap[i] >= .75]
                item.update({f'{role}_best_same_class_iou': same_iou,
                             f'{role}_best_any_class_iou': float(overlap.max()) if overlap.size else 0.,
                             f'{role}_max_correct_score_iou75': max(high_quality_scores, default=None),
                             f'{role}_max_wrong_score_iou75': max(wrong_scores, default=None)})
            candidates.append(item)
    grouped = defaultdict(lambda: dict(gt=0, both=0, gained=0, lost=0, neither=0))
    for row in pairs:
        for category, size in (('all', 'all'), ('all', row['size']), (row['category'], 'all'),
                               (row['category'], row['size'])):
            group = grouped[(row['iou_threshold'], category, size)]
            group['gt'] += 1
            group[row['status']] += 1
    groups = [dict(iou_threshold=threshold, category=category, size=size, **counts,
                   net_gained=counts['gained'] - counts['lost'])
              for (threshold, category, size), counts in sorted(grouped.items())]
    return dict(pairs=pairs, groups=groups, common_geometry=common, candidates=candidates)


def summarize_training(rows, config):
    best = max(rows, key=lambda row: row['test_coco_eval_bbox'][0])
    stop = config['train_dataloader']['collate_fn']['stop_epoch']

    def mean_metrics(selected):
        return {key: float(np.mean([row['test_coco_eval_bbox'][index] for row in selected]) * 100)
                if selected else None for index, key in enumerate(METRICS)}

    return dict(best_epoch=best['epoch'], last_logged_epoch=rows[-1]['epoch'],
                first_within_05_ap=next(row['epoch'] for row in rows
                                       if row['test_coco_eval_bbox'][0] >= best['test_coco_eval_bbox'][0] - .005),
                last12_means=mean_metrics(rows[-12:]),
                before_stop_last12_means=mean_metrics([r for r in rows if stop - 12 <= r['epoch'] < stop]),
                after_stop_means=mean_metrics([r for r in rows if r['epoch'] >= stop]))


def analyze(output):
    output = Path(output).resolve()
    annotation, split = split_annotation(output)
    dataset = load_json(annotation)
    predictions = {role: load_json(output / role / 'predictions.json.gz') for role in ROLES}
    report = output / 'report'
    report.mkdir(exist_ok=True)
    evaluations = {role: evaluate_coco(annotation, predictions[role]) for role in ROLES}
    agnostic, confusions = {}, {}
    for role in ROLES:
        data, detections = class_agnostic(dataset, predictions[role])
        save_json(report / 'class_agnostic_gt.json', data)
        agnostic[role] = evaluate_coco(report / 'class_agnostic_gt.json', detections)['metrics']
        confusions[role] = confusion(dataset, predictions[role])
        write_csv(report / f'confusion_{role}.csv', confusions[role])
    paired = pair_detections(dataset, predictions)
    for key, filename in (('pairs', 'gt_pairs.csv'), ('groups', 'paired_groups.csv'),
                          ('common_geometry', 'common_gt_geometry.csv'), ('candidates', 'candidate_quality.csv')):
        write_csv(report / filename, paired[key])
    classes = []
    names = {c['id']: c['name'] for c in dataset['categories']}
    baseline_classes = {c['category_id']: c for c in evaluations['baseline']['classes']}
    for rem in evaluations['rem']['classes']:
        base = baseline_classes[rem['category_id']]
        item = dict(category_id=rem['category_id'], category=names[rem['category_id']])
        for key in ('AP', 'AP75', 'APm', 'APl'):
            item.update({f'baseline_{key}': base[key], f'rem_{key}': rem[key],
                         f'delta_{key}': delta(rem[key], base[key])})
        classes.append(item)
    write_csv(report / 'class_metrics.csv', classes)
    epochs = []
    selections = {}
    training = {}
    for role in ROLES:
        selections[role] = load_json(output / role / 'selection.json')
        rows = epoch_rows(output / role / 'log.txt')
        training[role] = summarize_training(rows, yaml.safe_load((output / role / 'resolved_config.yml').read_text(encoding='utf-8')))
        for row in rows:
            epochs.append(dict(role=role, epoch=row['epoch'],
                               **{key: value * 100 for key, value in zip(METRICS, row['test_coco_eval_bbox'][:6])},
                               rem_alpha=row.get('train_rem_alpha')))
    write_csv(report / 'epoch_metrics.csv', epochs)
    geometry_summary = {}
    for role in ROLES:
        geometry_summary[role] = {}
        for key in ('dx640', 'dy640', 'width_error_pct', 'height_error_pct',
                    'left640', 'right640', 'top640', 'bottom640'):
            values = [row[f'{role}_{key}'] for row in paired['common_geometry']]
            geometry_summary[role][key] = dict(summarize_values(values),
                                               mean_absolute=float(np.mean(np.abs(values))) if values else None)
    errors = {role: error_analysis(dataset, predictions[role], SCORE_THRESHOLD)[0] for role in ROLES}
    classification_gap = {role: {key: delta(agnostic[role][key], evaluations[role]['metrics'][key])
                                 for key in METRICS} for role in ROLES}
    summary = dict(split=split, evaluations=evaluations, class_agnostic=agnostic,
                   classification_gap=classification_gap, confusion_at_score03=confusions, selections=selections, training=training, paired_groups=paired['groups'],
                   errors_at_score03=errors, common_gt_geometry=geometry_summary,
                   common_gt_delta_iou=summarize_values([r['delta_iou'] for r in paired['common_geometry']]),
                   common_gt_delta_score=summarize_values([r['delta_score'] for r in paired['common_geometry']]),
                   delta_metrics={key: delta(evaluations['rem']['metrics'][key], evaluations['baseline']['metrics'][key])
                                  for key in METRICS},
                   logged_to_export_ap_delta={role: evaluations[role]['metrics']['AP'] -
                                              selections[role]['logged_best_metrics'][0] * 100 for role in ROLES},
                   definitions=dict(score_threshold=SCORE_THRESHOLD,
                       gt_pairs='same-class score-greedy one-to-one matches; counts are not AP attribution',
                       common_geometry='only GT matched in both at score>=0.3 and IoU>=0.5',
                       candidate_quality='full exported candidates; per-GT best IoU is not one-to-one recall'))
    save_json(report / 'summary.json', summary)
    lines = [f'# REM 102 轮配对诊断（{split}）', '',
             f'两组均使用各自最佳验证 AP 权重，在 {split} 上统一 FP32、640 和原后处理。', '',
             '| 指标 | baseline | REM | 差值（AP 点） |', '|---|---:|---:|---:|']
    for key in METRICS:
        lines.append(f'| {key} | {display(evaluations["baseline"]["metrics"][key])} | '
                     f'{display(evaluations["rem"]["metrics"][key])} | {display(summary["delta_metrics"][key])} |')
    lines += ['', '完整复算指标与训练日志的 AP 差值（AP 点）：' +
              json.dumps(summary['logged_to_export_ap_delta'], ensure_ascii=False), '']
    lines += ['', '## 类别无关 AP 与分类缺口', '',
              '所有类别合并为一类，同一框只保留最高类别分数；缺口 = 类别无关 − 原 AP，'
              '近似表示"框对了但型号错了或型号分数排序差"造成的损失。', '',
              '| 指标 | baseline 原 | baseline 类别无关 | baseline 缺口 | REM 原 | REM 类别无关 | REM 缺口 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for key in ('AP', 'AP50', 'AP75'):
        lines.append(f'| {key} | ' + ' | '.join(
            f'{display(evaluations[role]["metrics"][key])} | {display(agnostic[role][key])} | '
            f'{display(classification_gap[role][key])}' for role in ROLES) + ' |')
    lines += ['', '## 各类 AP', '', '| 类别 | baseline AP | REM AP | 差值 | baseline AP75 | REM AP75 |',
              '|---|---:|---:|---:|---:|---:|']
    for row in classes:
        lines.append(f'| {row["category"]} | {display(row["baseline_AP"])} | {display(row["rem_AP"])} | '
                     f'{display(row["delta_AP"])} | {display(row["baseline_AP75"])} | {display(row["rem_AP75"])} |')
    lines += ['', '## 定位与型号（score>=0.3，IoU>=0.5，任意类别预测）', '',
              '| 类别 | GT | baseline 定位率 | baseline 定位后型号正确率 | REM 定位率 | REM 定位后型号正确率 |',
              '|---|---:|---:|---:|---:|---:|']
    for base, rem in zip(confusions['baseline'], confusions['rem']):
        lines.append(f'| {base["gt_category"]} | {base["gt"]} | ' + ' | '.join(
            display(row[key]) for row in (base, rem) for key in ('located_rate', 'correct_given_located')) + ' |')
    lines += ['', '完整混淆计数见 confusion_baseline.csv / confusion_rem.csv（行 GT，列预测类别）。']
    lines += ['', '## 同一 GT 的检测得失', '',
              '固定 score>=0.3；计数不等于 AP 损失。完整 AP 使用全部导出预测，不作该阈值过滤。', '',
              '| IoU | 分组 | GT | 两者检出 | REM 新增 | REM 损失 | 都未检出 | 净新增 |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in paired['groups']:
        if row['category'] == 'all':
            lines.append(f'| {row["iou_threshold"]:.2f} | {row["size"]} | {row["gt"]} | {row["both"]} | '
                         f'{row["gained"]} | {row["lost"]} | {row["neither"]} | {row["net_gained"]:+d} |')
    lines += ['', '## 后续定位', '',
              '- class_metrics.csv：先查中目标 AP 损失集中在哪些类别。',
              '- gt_pairs.csv / paired_groups.csv：按 image_id、gt_id 配对具体新增与损失目标。',
              '- common_gt_geometry.csv：共同检出目标的 IoU、分数、宽高和四边误差变化。',
              '- candidate_quality.csv：区分候选框不足与高 IoU 正确类候选分数偏低；候选覆盖不是召回率。',
              '- epoch_metrics.csv：对齐完整曲线，检查第 90 轮收尾阶段及 APm 变化。',
              '- summary.json：保留逐 IoU AP、官方指标、错误计数和完整定义。', '',
              '这些结果用于筛选 REM 的失效机制，不把 AP75 直接等同于纯定位能力，也不据此断言某个新结构有效。', '']
    (report / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(report=str(report), delta_metrics=summary['delta_metrics']), ensure_ascii=False), flush=True)
    return summary


def package(output):
    destination = output / 'diagnosis_bundle.tar.gz'
    with tarfile.open(destination, 'w:gz') as archive:
        for filename in ('manifest.json', split_annotation(output)[0].name):
            archive.add(output / filename, arcname=filename)
        for role in ROLES:
            for filename in ('resolved_config.yml', 'inference_config.yml', 'log.txt', 'selection.json',
                             'git_revision.txt', 'run_metadata.json', 'predictions.json.gz',
                             'predictions.json.gz.metadata.json', 'export.console.log'):
                path = output / role / filename
                if path.is_file():
                    archive.add(path, arcname=f'{role}/{filename}')
        archive.add(output / 'report', arcname='report')
    print(f'PACKAGE={destination}', flush=True)


def run_exports(output, manifest):
    for role in ROLES:
        selection = manifest['runs'][role]
        marker = output / role / 'export.done'
        if marker.is_file():
            print(f'SKIP_COMPLETED_EXPORT role={role}', flush=True)
            continue
        arguments = ['torchrun', '--standalone', '--nproc_per_node=2',
                     str(ROOT / 'tools/export_analysis_predictions.py'),
                     '-c', selection['inference_config'], '-r', selection['checkpoint'],
                     '--ann-file', manifest['annotation_file'], '--batch-per-gpu', '8',
                     '--output', selection['predictions']]
        save_json(output / role / 'export_command.json', arguments)
        print(f'EXPORT_START role={role} log={output / role / "export.console.log"}', flush=True)
        with (output / role / 'export.console.log').open('w', encoding='utf-8') as stream:
            subprocess.run(arguments, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES='0,1', PYTHONUNBUFFERED='1'),
                           stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, check=True)
        marker.write_text('completed\n', encoding='utf-8')
        print(f'EXPORT_DONE role={role}', flush=True)
    analyze(output)
    package(output)
    (output / 'DONE').write_text('exports and paired analysis completed\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    new = sub.add_parser('run')
    new.add_argument('--baseline-run', type=Path, required=True)
    new.add_argument('--rem-run', type=Path, required=True)
    new.add_argument('--data-root', type=Path, required=True)
    new.add_argument('--output', type=Path, required=True)
    new.add_argument('--split', choices=('val', 'test'), default='val')
    for name in ('resume', 'analyze'):
        command = sub.add_parser(name)
        command.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == 'analyze':
        analyze(output)
        package(output)
    else:
        manifest = (prepare(args.baseline_run, args.rem_run, args.data_root, output, args.split)
                    if args.action == 'run' else load_json(output / 'manifest.json'))
        run_exports(output, manifest)


if __name__ == '__main__':
    main()
