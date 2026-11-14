"""Prepare exact unequal-crop descriptors and immutable v3 plans on CPU."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.core.yaml_utils import load_config
from engine.data.region_dense_adaptive import (
    AdaptiveSourceQueuePlanner, adaptive_epoch_summary, extract_strip_features, protected_geometry)
from tools.build_region_dense_sar import build_geometry


def write_adaptive_plans(bank, strips, valid, settings, global_batch, world_size, output, metadata):
    n = len(bank['image_ids'])
    if global_batch % world_size:
        raise ValueError('Global batch must divide across the selected GPUs')
    count = (math.ceil(n / world_size) // (global_batch // world_size)) * global_batch
    planner = AdaptiveSourceQueuePlanner(bank, strips, valid, size=settings['size'],
        adaptive=settings['adaptive_partition'], **{k: settings[k] for k in (
            'probability', 'queue_capacity', 'nearest_candidates', 'max_targets', 'seed')})
    report = dict(epochs={}, ordinary_gt_per_canvas=float(bank['counts'].sum() / n))
    for epoch in range(*settings['epochs']):
        rows = np.array([planner.next() for _ in range(count)], dtype=np.int32)
        np.save(output / f'epoch_{epoch:03d}.npy', rows, allow_pickle=False)
        report['epochs'][str(epoch)] = adaptive_epoch_summary(rows, bank)
        print(f'Planned epoch {epoch}: {json.dumps(report["epochs"][str(epoch)])}', flush=True)
    report.update(completed_source_passes=planner.completed_passes,
                  remaining_pass_sources_used=int(planner.pass_usage.sum()),
                  total_source_exposures=int(planner.usage.sum()),
                  min_instance_exposure=int(planner.usage.min()), max_instance_exposure=int(planner.usage.max()),
                  total_class_counts=(planner.usage @ bank['counts']).tolist(),
                  total_native_size_counts=(planner.usage @ bank['native_size_counts']).tolist(),
                  total_input_size_counts=(planner.usage @ bank['input_size_counts']).tolist(),
                  feature_selection_fallbacks=planner.feature_fallbacks,
                  adaptive_feature_fallbacks=planner.adaptive_feature_fallbacks)
    metadata.update(version=3, settings=settings, global_batch_size=global_batch, build_world_size=world_size,
                    output_canvases_per_epoch=count, optimizer_steps_per_epoch=count // global_batch,
                    source_images=n, plan_columns=['layout', 'source0', 'source1', 'source2', 'source3', 'cut'],
                    layout_names={'-1': 'whole', '0': 'four', '1': 'equal_vertical', '2': 'equal_horizontal',
                                  '3': 'adaptive_vertical', '4': 'adaptive_horizontal'},
                    epoch_semantics='Fixed E010 output-canvas/step budget; continuous complete source passes',
                    resume='Immutable epoch plans; epoch-boundary restore reads the same next plan')
    (output / 'plan_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (output / 'plan_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--base-plan-dir', required=True, help='Completed E010 version 2 plan directory')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--world-size', type=int, default=2)
    args = parser.parse_args()
    started = time.monotonic()
    cfg = load_config(args.config, {})
    ds = cfg['train_dataloader']['dataset']
    settings = ds['plan_settings']
    base = Path(args.base_plan_dir)
    prior = json.loads((base / 'plan_metadata.json').read_text(encoding='utf-8'))
    inherited = {k: v for k, v in settings.items() if k != 'adaptive_partition'}
    if prior['version'] != 2 or prior['settings'] != inherited:
        raise ValueError('The E010 base geometry and descriptor settings must be preserved')
    if Path(prior['annotation_path']).resolve() != Path(ds['ann_file']).resolve():
        raise ValueError('The E010 bank must use this training annotation')
    annotation = json.loads(Path(ds['ann_file']).read_text(encoding='utf-8'))
    geometry, boxes, images, categories = build_geometry(annotation, settings)
    with np.load(base / 'bank.npz', allow_pickle=False) as archive:
        bank = {key: archive[key].copy() for key in archive.files}
    for key in ('image_ids', 'counts', 'crops'):
        if not np.array_equal(geometry[key], bank[key]):
            raise ValueError(f'Original E010 geometry differs: {key}')
    bank['regions'], bank['required_wh'] = protected_geometry(boxes, settings['size'], settings['context_margin'])
    output = Path(ds['plan_dir'])
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'bank.npz', **bank)
    shape = (len(images), 2, settings['size'] + 1)
    strips = np.lib.format.open_memmap(output / 'strip_features.npy', mode='w+', dtype=np.float64, shape=(*shape, 5))
    valid = np.lib.format.open_memmap(output / 'strip_valid.npy', mode='w+', dtype=bool, shape=shape)
    tasks = ((i, str(Path(ds['img_folder']) / im['file_name']), settings['size'], boxes[i], bank['regions'][i],
              settings['background_mask_margin'], settings['min_background_pixels'],
              bank['feature_center'], bank['feature_scale']) for i, im in enumerate(images))
    print(f'Exact strip descriptors: {len(images)} train images, {args.workers} CPU workers, '
          f'{strips.nbytes / 2**30:.2f} GiB feature table', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, (index, features, available) in enumerate(pool.map(extract_strip_features, tasks, chunksize=16), 1):
            strips[index], valid[index] = features, available
            if done % 2000 == 0:
                print(f'Strip features {done}/{len(images)}, elapsed {time.monotonic()-started:.1f}s', flush=True)
    strips.flush(); valid.flush()
    feature_elapsed = time.monotonic() - started
    metadata = dict(git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    base_plan_dir=str(base.resolve()), base_plan_git_revision=prior.get('git_revision'),
                    annotation_path=str(Path(ds['ann_file']).resolve()), image_root=str(Path(ds['img_folder']).resolve()),
                    categories=categories, feature_preparation_seconds=feature_elapsed,
                    descriptor='Same E010 five background quantiles, measured on the actual unequal crop',
                    calibration='Unchanged train-only E010 pooled IQR, floor 1 gray level',
                    computation='Exact uint8 prefix histograms, linear quantile interpolation; cached only for preparation',
                    geometry='All GT and original protected neighborhoods; integer midpoint of legal origins and split',
                    unchanged='E010 fixed routes first, FIFO source passes, probability, query limit, model and training protocol')
    (output / 'feature_preparation.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Descriptors ready after {feature_elapsed:.1f}s; planning epochs', flush=True)
    write_adaptive_plans(bank, strips, valid, settings, cfg['train_dataloader']['total_batch_size'],
                         args.world_size, output, metadata)
    print(f'Prepared {output}; total elapsed {time.monotonic()-started:.1f}s', flush=True)


if __name__ == '__main__':
    main()
