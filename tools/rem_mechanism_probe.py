#!/usr/bin/env python3
"""Frozen-checkpoint raw-logit/LQE readout and REM band attenuation experiments."""

import argparse
from collections import Counter
import copy
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_rem_102e import evaluate_coco, write_csv
from tools.rem_probe_utils import (ARRAY_KEYS, BANDS, RESPONSE_NAMES, SCALES, VARIANTS, EvidenceReadout,
                                   anchor_evidence, make_anchors, merge_query_shards,
                                   paired_decomposition, pool_responses, postprocess_with_queries,
                                   raw_only_predictions)
from tools.sar_analysis_core import load_json, save_json


def prepare(comparison, output):
    comparison, output = Path(comparison).resolve(), Path(output).resolve()
    parent = load_json(comparison / 'manifest.json')
    dataset = load_json(comparison / 'val.json')
    reference = {role: load_json(comparison / role / 'predictions.json.gz') for role in ('baseline', 'rem')}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'val.json').write_bytes((comparison / 'val.json').read_bytes())
    (output / 'reference_summary.json').write_bytes((comparison / 'report/summary.json').read_bytes())
    anchors = make_anchors(dataset, reference)
    save_json(output / 'anchors.json', anchors)
    runs = {}
    for variant in VARIANTS:
        role = 'baseline' if variant == 'baseline' else 'rem'
        destination = output / variant
        destination.mkdir()
        for filename in ('resolved_config.yml', 'selection.json'):
            source = comparison / role / filename
            (destination / filename).write_bytes(source.read_bytes())
        cfg = yaml.safe_load((comparison / role / 'inference_config.yml').read_text(encoding='utf-8'))
        cfg['HGNetv2']['pretrained'] = False
        cfg['val_dataloader']['dataset'].update(img_folder=parent['data_root'], ann_file=str(output / 'val.json'))
        cfg['val_dataloader']['total_batch_size'] = 16
        cfg.update(output_dir=str(destination), summary_dir=str(destination / 'summary'))
        config_path = destination / 'inference_config.yml'
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding='utf-8')
        runs[variant] = dict(config=str(config_path), checkpoint=parent['runs'][role]['checkpoint'],
                             branch_scales=SCALES[variant], source_role=role)
    manifest = dict(comparison=str(comparison), output=str(output), runs=runs,
                    data_root=parent['data_root'], anchor_counts=dict(Counter(a['group'] for a in anchors)),
                    source_selections=parent['runs'], export=dict(gpus=[0, 1], batch_per_gpu=8,
                    precision='float32', postprocessor='original, no confidence filter or extra NMS'),
                    response_names=RESPONSE_NAMES,
                    analysis_git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    save_json(output / 'manifest.json', manifest)
    return manifest


def worker(args):
    import torch.distributed as distributed
    from torch.utils.data import DataLoader, Subset
    from engine.core import YAMLConfig

    world, rank = int(os.environ.get('WORLD_SIZE', '1')), int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    if world > 1:
        distributed.init_process_group('nccl')
    device = torch.device('cuda', local_rank)
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    cfg = YAMLConfig(str(args.config))
    cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    model = cfg.model.to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    use_ema = bool(checkpoint.get('ema'))
    model.load_state_dict(checkpoint['ema']['module'] if use_ema else checkpoint['model'], strict=True)
    checkpoint_epoch = checkpoint.get('last_epoch')
    del checkpoint
    postprocessor = cfg.postprocessor.to(device).eval()
    if (model.encoder.rem is None) != (args.variant == 'baseline'):
        raise ValueError('Variant and the saved baseline/REM architecture do not agree.')
    readout = EvidenceReadout(model, args.variant)
    source_loader = cfg.val_dataloader
    indices = list(range(rank, len(source_loader.dataset), world))
    loader = DataLoader(Subset(source_loader.dataset, indices), batch_size=8, shuffle=False,
                        num_workers=2, pin_memory=True, drop_last=False, collate_fn=source_loader.collate_fn)
    arrays, predictions = {key: [] for key in ARRAY_KEYS}, []
    maximum_error = 0.
    try:
        with torch.inference_mode():
            for step, (samples, targets) in enumerate(loader):
                readout.clear()
                outputs = model(samples.to(device))
                raw, quality, final = readout.raw, readout.quality, outputs['pred_logits']
                error = (final - (raw + quality)).abs().max().item()
                maximum_error = max(maximum_error, error)
                if error > 1e-5:
                    raise ValueError(f'Final logits contain more than raw class logits plus LQE: max error={error}.')
                sizes = torch.stack([t['orig_size'] for t in targets]).to(device)
                image_ids = [int(t['image_id'].item()) for t in targets]
                predictions.extend(postprocess_with_queries(postprocessor, outputs, sizes, image_ids))
                core, ring, valid = pool_responses(readout.bands.maps(), outputs['pred_boxes'])
                batch = dict(image_ids=np.asarray(image_ids, dtype=np.int64), orig_sizes=sizes.cpu().numpy(),
                             boxes=outputs['pred_boxes'].cpu().numpy(), raw_logits=raw.cpu().numpy(),
                             quality=quality.cpu().numpy(), final_logits=final.cpu().numpy(),
                             core_abs=core.cpu().numpy(), ring_abs=ring.cpu().numpy(), valid_fractions=valid.cpu().numpy())
                for key, value in batch.items():
                    arrays[key].append(value)
                if step % 10 == 0:
                    print(f'variant={args.variant} rank={rank} batch={step+1}/{len(loader)}', flush=True)
    finally:
        readout.close()
    np.savez_compressed(destination / f'queries.rank{rank}.npz', **{k: np.concatenate(v) for k, v in arrays.items()})
    with gzip.open(destination / f'predictions.rank{rank}.json.gz', 'wt', encoding='utf-8') as stream:
        json.dump(predictions, stream)
    save_json(destination / f'metadata.rank{rank}.json', dict(max_decomposition_error=maximum_error))
    if world > 1:
        distributed.barrier()
    if rank == 0:
        shards, predictions = [], []
        for index in range(world):
            with np.load(destination / f'queries.rank{index}.npz') as data:
                shards.append({key: data[key] for key in ARRAY_KEYS})
            predictions.extend(load_json(destination / f'predictions.rank{index}.json.gz'))
        merged = merge_query_shards(shards)
        ids = merged['image_ids'].tolist()
        if len(ids) != len(set(ids)) or set(ids) != set(source_loader.dataset.ids):
            raise ValueError('Each validation image must occur exactly once in the merged readout.')
        np.savez_compressed(destination / 'queries.npz', **merged)
        predictions.sort(key=lambda d: (d['image_id'], -d['score']))
        with gzip.open(destination / 'predictions.json.gz', 'wt', encoding='utf-8') as stream:
            json.dump(predictions, stream)
        alpha = model.encoder.rem.alpha.detach().item() if model.encoder.rem is not None else None
        save_json(destination / 'export_metadata.json', dict(variant=args.variant, checkpoint=str(args.checkpoint),
            ema=use_ema, checkpoint_internal_epoch=checkpoint_epoch, alpha=alpha, branch_scales=SCALES[args.variant],
            images=len(ids), num_queries=int(merged['boxes'].shape[1]), num_classes=int(merged['raw_logits'].shape[-1]),
            topk=postprocessor.num_top_queries, precision='float32', response_names=RESPONSE_NAMES,
            max_decomposition_error=max(load_json(destination / f'metadata.rank{i}.json')['max_decomposition_error'] for i in range(world)),
            response_definition='channel mean absolute learned feature response; not physical SAR scattering power',
            pool_definition='3x3 core at .75 box; 12 outer-side points at 1.05 box; invalid samples excluded'))
        print(f'EXPORTED variant={args.variant} images={len(ids)}', flush=True)
    if world > 1:
        distributed.destroy_process_group()


def summarize_evidence(records, differences):
    groups, pairs = [], []
    names = sorted({r['group'] for r in records})
    for variant in VARIANTS:
        for group in names:
            all_rows = [r for r in records if r['variant'] == variant and r['group'] == group]
            rows = [r for r in all_rows if r['present']]
            item = dict(variant=variant, group=group, anchors=len(all_rows), present=len(rows))
            labeled = [r for r in rows if r['gt_category'] is not None]
            item.update(target_raw_top1=sum(r['raw_top_class'] == r['target_category'] for r in rows),
                        gt_labeled=len(labeled),
                        gt_raw_top1=sum(r['raw_top_class'] == r['gt_category'] for r in labeled) if labeled else None)
            for key in ('raw_logit', 'quality_logit', 'final_logit', 'score', 'raw_margin', 'iou',
                        *[f'{location}_{band}_abs' for location in ('core', 'ring') for band in RESPONSE_NAMES]):
                item[f'mean_{key}'] = float(np.mean([r[key] for r in rows])) if rows else None
            groups.append(item)
            delta = [r for r in differences if r['variant'] == variant and r['group'] == group]
            if variant != 'baseline':
                pair = dict(variant=variant, reference='baseline' if variant == 'rem' else 'rem', group=group, paired=len(delta))
                for key in ('raw_logit', 'quality_logit', 'final_logit', 'score', 'raw_margin', 'iou'):
                    pair[f'mean_delta_{key}'] = float(np.mean([r[f'delta_{key}'] for r in delta])) if delta else None
                pairs.append(pair)
    return groups, pairs


def collect_examples(output, manifest, dataset, anchors, records, differences):
    examples = output / 'examples'
    examples.mkdir(exist_ok=True)
    by_anchor = {a['anchor_id']: a for a in anchors}
    by_image = {im['id']: im for im in dataset['images']}
    selected = sorted((r for r in differences if r['variant'] == 'rem' and r['group'] == 'a220_wrongclass'),
                      key=lambda r: -r['delta_final_logit'])[:5]
    reference = {(r['variant'], r['anchor_id']): r for r in records}
    selected_ids = [r['anchor_id'] for r in selected]
    selected_ids += [a['anchor_id'] for a in anchors if a['group'] == 'a330_true' and
                     reference[('baseline', a['anchor_id'])]['present'] and
                     reference[('baseline', a['anchor_id'])]['score'] < .1]
    rows = []
    data_root = Path(manifest['data_root']).resolve()
    for anchor_id in selected_ids:
        anchor = by_anchor[anchor_id]
        image = by_image[anchor['image_id']]
        source = (data_root / image['file_name']).resolve()
        # Copy selected original images unchanged, only from the confirmed dataset.
        if source.is_relative_to(data_root) and source.is_file():
            shutil.copyfile(source, examples / f'{image["id"]}_{source.name}')
        rows.append(dict(anchor=anchor, image=image,
                         evidence={v: reference[(v, anchor_id)] for v in VARIANTS}))
    save_json(examples / 'cases.json', rows)


def analyze(output, copy_examples=False):
    output = Path(output).resolve()
    manifest, dataset, anchors = [load_json(output / name) for name in ('manifest.json', 'val.json', 'anchors.json')]
    report = output / 'report'
    report.mkdir(exist_ok=True)
    evaluations, raw_only, evidence = {}, {}, []
    for variant in VARIANTS:
        predictions = load_json(output / variant / 'predictions.json.gz')
        evaluations[variant] = evaluate_coco(output / 'val.json', predictions)
        with np.load(output / variant / 'queries.npz') as source:
            arrays = {key: source[key] for key in ARRAY_KEYS}
        evidence.extend(anchor_evidence(anchors, arrays, variant))
        if variant in ('baseline', 'rem'):
            topk = load_json(output / variant / 'export_metadata.json')['topk']
            raw_only[variant] = evaluate_coco(output / 'val.json', raw_only_predictions(arrays, topk))
    differences = paired_decomposition(evidence)
    groups, pairs = summarize_evidence(evidence, differences)
    write_csv(report / 'anchor_evidence.csv', evidence)
    write_csv(report / 'paired_decomposition.csv', differences)
    write_csv(report / 'group_summary.csv', groups)
    write_csv(report / 'paired_summary.csv', pairs)
    save_json(report / 'anchor_evidence.json', evidence)
    metric_rows = []
    for variant, evaluation in evaluations.items():
        classes = {c['category_id']: c for c in evaluation['classes']}
        metric_rows.append(dict(variant=variant, **evaluation['metrics'], A220_AP=classes[0]['AP'],
                                A220_AP75=classes[0]['AP75'], A220_APm=classes[0]['APm'], A330_AP=classes[2]['AP']))
    write_csv(report / 'intervention_metrics.csv', metric_rows)
    parent_summary = load_json(output / 'reference_summary.json')
    summary = dict(evaluations=evaluations, raw_only_evaluations=raw_only, anchor_groups=groups,
                   paired_groups=pairs, source_manifest=manifest,
                   reference_reproduction_delta={v: evaluations[v]['metrics']['AP'] - parent_summary['evaluations'][v]['metrics']['AP']
                                                 for v in ('baseline', 'rem')},
                   interpretation=dict(pairing='same GT or fixed background region, not Query index across networks',
                       attribution='delta final logit = delta raw class logit + delta LQE on jointly available matched candidates',
                       raw_only='same decoded queries and boxes; LQE removed; top-K reselected from all query/class pairs',
                       interventions='test-time frozen-weight branch attenuation, not retrained ablation results',
                       response='learned feature mean absolute magnitude; component magnitudes do not add to total magnitude'))
    save_json(report / 'summary.json', summary)
    lines = ['# REM 分类分数 / LQE / 子带残差机制诊断', '',
             '同一最佳权重上的推理干预；所有版本统一FP32、640、原后处理。', '',
             '| 版本 | AP | AP75 | APm | A220 AP | A330 AP |', '|---|---:|---:|---:|---:|---:|']
    for row in metric_rows:
        lines.append('| ' + row['variant'] + ' | ' + ' | '.join(f'{row[k]:.4f}' for k in ('AP', 'AP75', 'APm', 'A220_AP', 'A330_AP')) + ' |')
    lines += ['', '## 相同目标的分数分解', '',
              '原始项与LQE项均从同一次前向读取。只对两版本均存在满足几何条件的候选求配对差；',
              '缺失数量另见group_summary.csv，不能将缺失候选当作logit为0。', '',
              '| 版本 | 参照 | 分组 | 配对数量 | Δraw | ΔLQE | Δfinal | Δscore |',
              '|---|---|---|---:|---:|---:|---:|---:|']
    for row in pairs:
        values = [row[f'mean_delta_{k}'] for k in ('raw_logit', 'quality_logit', 'final_logit', 'score')]
        lines.append(f'| {row["variant"]} | {row["reference"]} | {row["group"]} | {row["paired"]} | ' +
                     ' | '.join('n/a' if x is None else f'{x:+.5f}' for x in values) + ' |')
    lines += ['', '## 移除LQE的固定框排序对照', '',
              '| 版本 | 原AP | 无LQE AP | 原A220 AP | 无LQE A220 AP |', '|---|---:|---:|---:|---:|']
    for variant in ('baseline', 'rem'):
        full, raw = evaluations[variant], raw_only[variant]
        a = next(c['AP'] for c in full['classes'] if c['category_id'] == 0)
        b = next(c['AP'] for c in raw['classes'] if c['category_id'] == 0)
        lines.append(f'| {variant} | {full["metrics"]["AP"]:.4f} | {raw["metrics"]["AP"]:.4f} | {a:.4f} | {b:.4f} |')
    lines += ['', '移除LQE不重新计算框和Decoder，只重新选择top-K；这是机制反事实，不是新模型性能。', '',
              '## 结果用途', '',
              '- 先看A220错类/背景组的Δraw与ΔLQE，再看A220真阳性和A330真阳性是否同样变化。',
              '- 半强度、LL半强度、HF半强度使用同一个REM checkpoint，均不训练、不改权重。',
              '- 子带响应是特征幅度代理，不等于物理散射能量；不得据此直接判定前景/杂波。',
              '- examples中至多保存5个A220错类案例及低分A330目标的原始图片；图像未修改，坐标/类别见cases.json。',
              '- 所有版本均需报告，不能挑某个干预的最高AP冒充正式训练结果。', '']
    (report / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    if copy_examples:
        collect_examples(output, manifest, dataset, anchors, evidence, differences)
    print(f'ANALYSIS_DONE report={report / "report.md"}', flush=True)
    return summary


def package(output):
    destination = output / 'mechanism_bundle.tar.gz'
    with tarfile.open(destination, 'w:gz') as archive:
        for name in ('manifest.json', 'anchors.json', 'val.json', 'reference_summary.json', 'report'):
            archive.add(output / name, arcname=name)
        if (output / 'examples').is_dir():
            archive.add(output / 'examples', arcname='examples')
        for variant in VARIANTS:
            for name in ('resolved_config.yml', 'inference_config.yml', 'selection.json', 'export_metadata.json',
                         'queries.npz', 'predictions.json.gz', 'export.console.log'):
                path = output / variant / name
                if path.is_file():
                    archive.add(path, arcname=f'{variant}/{name}')
    print(f'PACKAGE={destination}', flush=True)


def run_jobs(output, manifest):
    for variant in VARIANTS:
        directory = output / variant
        if (directory / 'export.done').is_file():
            print(f'SKIP_COMPLETED variant={variant}', flush=True)
            continue
        job = manifest['runs'][variant]
        arguments = ['torchrun', '--standalone', '--nproc_per_node=2', str(Path(__file__).resolve()), 'worker',
                     '--config', job['config'], '--checkpoint', job['checkpoint'], '--variant', variant, '--output', str(directory)]
        save_json(directory / 'command.json', arguments)
        print(f'EXPORT_START variant={variant} log={directory / "export.console.log"}', flush=True)
        with (directory / 'export.console.log').open('w', encoding='utf-8') as stream:
            subprocess.run(arguments, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES='0,1', PYTHONUNBUFFERED='1'), check=True)
        (directory / 'export.done').write_text('completed\n', encoding='utf-8')
        print(f'EXPORT_DONE variant={variant}', flush=True)
    analyze(output, copy_examples=True)
    package(output)
    (output / 'DONE').write_text('all readouts, interventions and analysis completed\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    new = sub.add_parser('run')
    new.add_argument('--comparison', type=Path, required=True)
    new.add_argument('--output', type=Path, required=True)
    for name in ('resume', 'analyze'):
        command = sub.add_parser(name)
        command.add_argument('--output', type=Path, required=True)
    export = sub.add_parser('worker')
    export.add_argument('--config', type=Path, required=True)
    export.add_argument('--checkpoint', type=Path, required=True)
    export.add_argument('--variant', choices=VARIANTS, required=True)
    export.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'worker':
        worker(args)
    elif args.action == 'analyze':
        analyze(args.output)
        package(args.output)
    else:
        output = args.output.resolve()
        manifest = prepare(args.comparison, output) if args.action == 'run' else load_json(output / 'manifest.json')
        run_jobs(output, manifest)


if __name__ == '__main__':
    main()
