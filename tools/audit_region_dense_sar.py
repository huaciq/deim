"""Verify RegionDenseSAR source plans, DDP coverage and actual image/GT construction on CPU."""

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
from engine.data.dataset.region_dense_sar_dataset import RegionDenseSARCocoDetection
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
    transforms = options.pop('transforms')
    options.pop('type')
    ds = RegionDenseSARCocoDetection(transforms=None, **options)
    pipeline = Compose(**{k: v for k, v in transforms.items() if k != 'type'})
    epochs = range(*ds.plan_settings['epochs'])
    coverage = []
    packed_choices = []
    for epoch in epochs:
        ds.set_epoch(epoch)
        parts = [list(ds.make_sampler(cfg['train_dataloader']['total_batch_size']//2, 2, r, True, True)) for r in (0, 1)]
        merged = np.sort(np.concatenate(parts))
        if not np.array_equal(merged, np.arange(ds.output_canvases)):
            raise AssertionError('DDP plan slots have omissions or repetitions')
        rows = np.load(ds.plan_dir / f'epoch_{epoch:03d}.npy', mmap_mode='r')
        sources = rows[:, 1:]
        for a in range(4):
            for b in range(a+1, 4):
                if np.any((sources[:, a] >= 0) & (sources[:, a] == sources[:, b])):
                    raise AssertionError('A canvas reuses the same source')
        for layout in (0, 1, 2):
            selected = sources[rows[:, 0] == layout]
            used = selected[selected >= 0]
            if np.any(ds.source_crops[used, layout, 0] < 0):
                raise AssertionError('Plan selected an ineligible source crop')
            totals = ds.source_counts[np.maximum(selected, 0)].sum(2) * (selected >= 0)
            if np.any(totals.sum(1) > ds.plan_settings['max_targets']):
                raise AssertionError('Packed canvas exceeded query budget')
        coverage.append(dict(epoch=epoch, slots=len(merged), per_rank=len(parts[0]),
                             optimizer_steps=len(parts[0]) // (cfg['train_dataloader']['total_batch_size']//2)))
        indices = np.flatnonzero(rows[:, 0] >= 0)
        chosen = np.random.default_rng(epoch).choice(indices, min(20, len(indices)), replace=False)
        packed_choices.extend((epoch, int(i)) for i in chosen)
    checked = dropped = 0
    max_side_error = 0.
    for epoch, index in packed_choices[:args.pixel_checks]:
        if ds.epoch != epoch:
            ds.set_epoch(epoch)
        packed, target, plan, details = ds.load_planned_item(index, trace=True)
        for sample, detail in zip(plan, details):
            source, original = ds.load_item(ds.dataset_indices[sample['image_id']])
            expected = source.resize((640, 640), Image.Resampling.BILINEAR).crop(sample['crop'])
            x, y = sample['offset']
            actual = packed.crop((x, y, x+expected.width, y+expected.height))
            if not np.array_equal(np.asarray(actual), np.asarray(expected)):
                raise AssertionError('Region composition changed source pixels')
            if len(detail['gt_indices']) != len(original['labels']):
                raise AssertionError('A source lost a GT')
            before, after = np.array(detail['input_boxes']), np.array(detail['output_boxes'])
            max_side_error = max(max_side_error, float(np.abs((before[:, 2:]-before[:, :2])-(after[:, 2:]-after[:, :2])).max()))
        pixels, transformed, _ = pipeline(packed.copy(), copy.deepcopy(target), ds)
        if tuple(pixels.shape) != (3, 640, 640) or not torch.isfinite(transformed['boxes']).all():
            raise AssertionError('Invalid full training pipeline result')
        dropped += len(target['labels']) - len(transformed['labels'])
        checked += 1
    report = dict(config=args.config, all_active_epoch_ddp_coverage=coverage,
                  real_pixel_checks=checked, all_source_gt_kept_before_sanitize=True,
                  max_box_side_change_px=max_side_error, targets_removed_by_following_sanitize=dropped,
                  plan_report=ds.plan_report,
                  limitation='Data construction verification only; no model forward or training/AP claim')
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('plan_report', 'all_active_epoch_ddp_coverage')}, indent=2))


if __name__ == '__main__':
    main()
