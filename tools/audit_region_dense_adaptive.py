"""Verify v3 DDP coverage, actual unequal crops and the saved background descriptors."""

import argparse
import copy
import json
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.core.yaml_utils import load_config
from engine.data.dataset.region_dense_adaptive_dataset import RegionDenseAdaptiveCocoDetection
from engine.data.region_dense_sar import background_features
from engine.data.transforms import Compose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--pixel-checks', type=int, default=200)
    args = parser.parse_args()
    torch.set_num_threads(1)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    cfg = load_config(args.config, {})
    options = copy.deepcopy(cfg['train_dataloader']['dataset'])
    transforms = options.pop('transforms'); options.pop('type')
    ds = RegionDenseAdaptiveCocoDetection(transforms=None, **options)
    pipeline = Compose(**{k: v for k, v in transforms.items() if k != 'type'})
    with np.load(ds.plan_dir / 'bank.npz') as bank:
        required, center, scale = bank['required_wh'], bank['feature_center'], bank['feature_scale']
    strips = np.load(ds.plan_dir / 'strip_features.npy', mmap_mode='r')
    available = np.load(ds.plan_dir / 'strip_valid.npy', mmap_mode='r')
    coverage, selected = [], []
    for epoch in range(*ds.plan_settings['epochs']):
        ds.set_epoch(epoch)
        parts = [list(ds.make_sampler(cfg['train_dataloader']['total_batch_size']//2, 2, rank, True, True))
                 for rank in (0, 1)]
        if not np.array_equal(np.sort(np.concatenate(parts)), np.arange(ds.output_canvases)):
            raise AssertionError('DDP plan rows have repetitions or omissions')
        rows = np.load(ds.plan_dir / f'epoch_{epoch:03d}.npy', mmap_mode='r')
        sources = rows[:, 1:5]
        for a in range(4):
            for b in range(a+1, 4):
                if np.any((sources[:, a] >= 0) & (sources[:, a] == sources[:, b])):
                    raise AssertionError('Repeated source within a canvas')
        totals = (ds.source_counts[np.maximum(sources, 0)].sum(2) * (sources >= 0)).sum(1)
        if np.any(totals[rows[:, 0] >= 0] > ds.plan_settings['max_targets']):
            raise AssertionError('Packed canvas exceeds query count')
        for layout in (0, 1, 2):
            used = sources[rows[:, 0] == layout]
            if np.any(ds.source_crops[used[used >= 0], layout, 0] < 0):
                raise AssertionError('Invalid fixed crop')
        for layout in (3, 4):
            group = rows[rows[:, 0] == layout]
            axis = layout - 3
            if np.any(group[:, 3:5] != -1):
                raise AssertionError('Unequal layout must contain exactly two sources')
            if np.any(group[:, 5] < required[group[:, 1], axis]) or np.any(
                    ds.plan_settings['size'] - group[:, 5] < required[group[:, 2], axis]):
                raise AssertionError('Unequal split cuts a protected target region')
        coverage.append(dict(epoch=epoch, slots=len(rows), per_rank=len(parts[0]),
                             optimizer_steps=len(parts[0]) // (cfg['train_dataloader']['total_batch_size']//2)))
        for layout in (3, 4, 0, 1, 2):
            candidates = np.flatnonzero(rows[:, 0] == layout)
            picks = np.random.default_rng(epoch*10+layout).choice(candidates, min(3, len(candidates)), replace=False)
            selected.extend((epoch, int(i), layout) for i in picks)
    checked = dropped = descriptor_checks = 0
    max_side_error = max_descriptor_error = 0.
    routes = {}
    for epoch, index, layout in selected[:args.pixel_checks]:
        if ds.epoch != epoch:
            ds.set_epoch(epoch)
        packed, target, plan, details = ds.load_planned_item(index, trace=True)
        for sample, detail in zip(plan, details):
            source, original = ds.load_item(ds.dataset_indices[sample['image_id']])
            resized = source.resize((640, 640), Image.Resampling.BILINEAR)
            expected = resized.crop(sample['crop'])
            x, y = sample['offset']
            if not np.array_equal(np.asarray(expected), np.asarray(packed.crop((x, y, x+expected.width, y+expected.height)))):
                raise AssertionError('Changed source pixels')
            if len(detail['gt_indices']) != len(original['labels']):
                raise AssertionError('Lost source GT')
            before, after = np.array(detail['input_boxes']), np.array(detail['output_boxes'])
            max_side_error = max(max_side_error, float(np.abs((before[:, 2:]-before[:, :2])-(after[:, 2:]-after[:, :2])).max()))
            if layout >= 3 and descriptor_checks < 40:
                raw, valid = background_features(resized, before, [sample['crop']],
                    ds.plan_settings['background_mask_margin'], ds.plan_settings['min_background_pixels'])
                source_idx = ds.dataset_indices[sample['image_id']]
                axis = layout-3; length = expected.size[axis]
                if bool(valid[0]) != bool(available[source_idx, axis, length]):
                    raise AssertionError('Wrong descriptor validity')
                if valid[0]:
                    error = float(np.abs((raw[0]-center)/scale - strips[source_idx, axis, length]).max())
                    max_descriptor_error = max(max_descriptor_error, error)
                    if error > 1e-10:
                        raise AssertionError('Cached descriptor differs from actual crop')
                descriptor_checks += 1
        pixels, transformed, _ = pipeline(packed.copy(), copy.deepcopy(target), ds)
        if tuple(pixels.shape) != (3, 640, 640) or not torch.isfinite(transformed['boxes']).all():
            raise AssertionError('Invalid training transform output')
        dropped += len(target['labels']) - len(transformed['labels'])
        routes[str(layout)] = routes.get(str(layout), 0) + 1
        checked += 1
    report = dict(config=args.config, all_active_epoch_ddp_coverage=coverage, real_pixel_checks=checked,
                  checked_routes=routes, max_box_side_change_px=max_side_error,
                  targets_removed_by_sanitize=dropped, exact_actual_crop_descriptor_checks=descriptor_checks,
                  max_descriptor_error=max_descriptor_error, plan_report=ds.plan_report,
                  limitation='Data construction and reproducibility verification, no trained AP claim')
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('plan_report', 'all_active_epoch_ddp_coverage')}, indent=2))


if __name__ == '__main__':
    main()
