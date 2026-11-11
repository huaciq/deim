"""Prepare RegionDenseSAR background features and fixed epoch plans, using train only."""

import argparse
from collections import defaultdict
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
from engine.data.region_bank import context_boxes, origin_range, tile_sizes
from engine.data.region_dense_sar import extract_image_features, SourceQueuePlanner, epoch_plan_summary


def build_geometry(annotation, settings):
    images = sorted(annotation['images'], key=lambda x: x['id'])
    categories = sorted(annotation['categories'], key=lambda x: x['id'])
    if [c['id'] for c in categories] != list(range(len(categories))):
        raise ValueError('RegionDenseSAR requires zero-based contiguous categories')
    by_image = defaultdict(list)
    for ann in annotation['annotations']:
        if not ann.get('iscrowd', 0):
            by_image[ann['image_id']].append(ann)
    n, size = len(images), settings['size']
    bank = dict(image_ids=np.array([x['id'] for x in images], dtype=np.int64),
                counts=np.zeros((n, len(categories)), dtype=np.int64),
                crops=np.full((n, 3, 4), -1, dtype=np.int32),
                native_size_counts=np.zeros((n, 3), dtype=np.int64),
                input_size_counts=np.zeros((n, 3), dtype=np.int64))
    boxes = []
    for index, info in enumerate(images):
        anns = by_image[info['id']]
        b = np.array([a['bbox'] for a in anns], dtype=np.float32).reshape(-1, 4)
        b[:, 2:] += b[:, :2]
        b[:, 0::2] = np.clip(b[:, 0::2], 0, info['width'])
        b[:, 1::2] = np.clip(b[:, 1::2], 0, info['height'])
        keep = np.all(b[:, 2:] > b[:, :2], axis=1)
        labels = np.array([a['category_id'] for a in anns], dtype=np.int64)[keep]
        areas = np.array([a['area'] for a in anns])[keep]
        b = b[keep].astype(np.float64) * np.array([size / info['width'], size / info['height']] * 2)
        boxes.append(b)
        bank['counts'][index] = np.bincount(labels, minlength=len(categories))
        bank['native_size_counts'][index] = np.bincount(np.digitize(areas, [32**2, 96**2]), minlength=3)
        area640 = (b[:, 2:] - b[:, :2]).prod(1)
        bank['input_size_counts'][index] = np.bincount(np.digitize(area640, [32**2, 96**2]), minlength=3)
        if not len(b):
            continue
        contexts = context_boxes(b, size, settings['context_margin'])
        region = np.r_[contexts[:, :2].min(0), contexts[:, 2:].max(0)]
        for layout, tile in enumerate(tile_sizes(size)):
            lo, hi = origin_range(region, tile, size)
            if np.all(lo <= hi):
                xy = (lo + hi) // 2
                bank['crops'][index, layout] = np.r_[xy, xy + tile]
    return bank, boxes, images, categories


def write_plans(bank, settings, global_batch, world_size, output, metadata):
    n = len(bank['image_ids'])
    per_gpu = global_batch // world_size
    count = (math.ceil(n / world_size) // per_gpu) * global_batch
    planner = SourceQueuePlanner(bank, **{k: settings[k] for k in (
        'probability', 'queue_capacity', 'nearest_candidates', 'max_targets', 'seed')})
    report = dict(epochs={}, ordinary_gt_per_canvas=float(bank['counts'].sum() / n))
    for epoch in range(*settings['epochs']):
        rows = np.array([planner.next() for _ in range(count)], dtype=np.int32)
        np.save(output / f'epoch_{epoch:03d}.npy', rows, allow_pickle=False)
        report['epochs'][str(epoch)] = epoch_plan_summary(rows, bank)
        print(f'Planned epoch {epoch}: {json.dumps(report["epochs"][str(epoch)])}', flush=True)
    report.update(completed_source_passes=planner.completed_passes,
                  remaining_pass_sources_used=int(planner.pass_usage.sum()),
                  total_source_exposures=int(planner.usage.sum()),
                  min_instance_exposure=int(planner.usage.min()), max_instance_exposure=int(planner.usage.max()),
                  total_class_counts=(planner.usage @ bank['counts']).tolist(),
                  total_native_size_counts=(planner.usage @ bank['native_size_counts']).tolist(),
                  total_input_size_counts=(planner.usage @ bank['input_size_counts']).tolist(),
                  feature_selection_fallbacks=planner.feature_fallbacks)
    metadata.update(settings=settings, global_batch_size=global_batch, build_world_size=world_size,
                    output_canvases_per_epoch=count, optimizer_steps_per_epoch=count // global_batch,
                    source_images=n, plan_columns=['layout', 'source0', 'source1', 'source2', 'source3'],
                    epoch_semantics='Fixed baseline output-canvas/step budget; source passes continue across active epochs',
                    resume='Epoch-indexed immutable plans; epoch-boundary resume reads the same next plan')
    (output / 'plan_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (output / 'plan_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--world-size', type=int, default=2)
    args = parser.parse_args()
    started = time.monotonic()
    cfg = load_config(args.config, {})
    ds = cfg['train_dataloader']['dataset']
    settings = ds['plan_settings']
    output = Path(ds['plan_dir'])
    output.mkdir(parents=True, exist_ok=False)
    annotation = json.loads(Path(ds['ann_file']).read_text(encoding='utf-8'))
    bank, boxes, images, categories = build_geometry(annotation, settings)
    features = np.zeros((len(images), 3, 5), dtype=np.float64)
    valid = np.zeros((len(images), 3), dtype=bool)
    tasks = [(i, str(Path(ds['img_folder']) / im['file_name']), settings['size'], boxes[i], bank['crops'][i],
              settings['background_mask_margin'], settings['min_background_pixels'])
             for i, im in enumerate(images) if (bank['crops'][i, :, 0] >= 0).any()]
    print(f'Extracting features for {len(tasks)}/{len(images)} images with {args.workers} CPU workers', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, (index, values, available) in enumerate(pool.map(extract_image_features, tasks, chunksize=32), 1):
            features[index], valid[index] = values, available
            if done % 5000 == 0:
                print(f'Features {done}/{len(tasks)}, elapsed {time.monotonic()-started:.1f}s', flush=True)
    values = features[valid]
    center = np.median(values, axis=0)
    scale = np.maximum(np.quantile(values, .75, axis=0) - np.quantile(values, .25, axis=0), 1.)
    bank.update(features=(features - center) / scale, feature_valid=valid,
                raw_features=features, feature_center=center, feature_scale=scale)
    np.savez_compressed(output / 'bank.npz', **bank)
    metadata = dict(version=2, git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    annotation_path=str(Path(ds['ann_file']).resolve()), image_root=str(Path(ds['img_folder']).resolve()),
                    categories=categories, feature_valid_regions=int(valid.sum()),
                    descriptor='background gray q25/q50/q75 and absolute adjacent-difference q50/q90; GT+2px masked',
                    calibration='Train-only pooled feature IQR, floor 1 gray value; display proxies, not physical SAR parameters',
                    geometry='All GT and v1 protected neighborhoods; center of legal crop-origin interval')
    write_plans(bank, settings, cfg['train_dataloader']['total_batch_size'], args.world_size, output, metadata)
    print(f'Prepared {output}; elapsed {time.monotonic()-started:.1f}s', flush=True)


if __name__ == '__main__':
    main()
