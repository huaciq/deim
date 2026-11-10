"""CPU geometry replay and real-pixel verification of a RegionDense training configuration."""

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.core.yaml_utils import load_config
from engine.data.dataset.coco_dataset import CocoDetection
from engine.data.transforms import Compose, RegionDense


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--draws', type=int, default=10000)
    parser.add_argument('--pixel-checks', type=int, default=200)
    parser.add_argument('--seed', type=int, default=20261110)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = load_config(args.config, {})
    dataset_cfg = cfg['train_dataloader']['dataset']
    dataset = CocoDetection(dataset_cfg['img_folder'], dataset_cfg['ann_file'], transforms=None)
    dataset.set_epoch(8)
    transform_cfg = copy.deepcopy(dataset_cfg['transforms']['ops'][0])
    transform_cfg.pop('type')
    transform = RegionDense(**transform_cfg)
    transform.load_bank(dataset)
    bank = transform.bank
    remainder = Compose(copy.deepcopy(dataset_cfg['transforms']['ops'][1:]))
    rng = random.Random(args.seed)
    original = np.zeros(bank.classes, dtype=np.int64)
    added = np.zeros_like(original)
    routes = [0, 0, 0]
    pixel_checks = pipeline_dropped = 0
    saved_classes = set()
    max_size_error = 0.0
    start = time.monotonic()
    for draw in range(args.draws):
        index = rng.randrange(len(bank.image_ids))
        image_id = int(bank.image_ids[index])
        original += bank.counts[index]
        plan = bank.plan(image_id, rng, max_targets=transform.max_targets)
        if plan is None:
            continue
        layout = int(bank.primary[index])
        routes[layout] += 1
        added += sum((s['counts'] for s in plan if not s['primary']), start=np.zeros_like(added))
        if pixel_checks >= args.pixel_checks:
            continue
        image, target = dataset.load_item(transform.dataset_indices[image_id])
        packed, packed_target, trace = transform.apply_plan(image, target, dataset, plan, trace=True)
        for sample, detail in zip(plan, trace):
            source, _ = dataset.load_item(transform.dataset_indices[sample['image_id']])
            ordinary = source.resize((bank.size, bank.size), Image.Resampling.BILINEAR)
            expected = ordinary.crop(sample['crop'])
            x, y = sample['offset']
            actual = packed.crop((x, y, x + expected.width, y + expected.height))
            if not np.array_equal(np.asarray(actual), np.asarray(expected)):
                raise AssertionError('Packing changed pixels beyond the ordinary training resize')
            before, after = np.array(detail['input_boxes']), np.array(detail['output_boxes'])
            size_error = float(np.max(np.abs((before[:, 2:] - before[:, :2]) - (after[:, 2:] - after[:, :2]))))
            max_size_error = max(max_size_error, size_error)
            if sample['primary'] and len(before) != len(target['boxes']):
                raise AssertionError('Primary target was dropped')
        pixels, normalized, _ = remainder(packed.copy(), copy.deepcopy(packed_target), dataset)
        if tuple(pixels.shape) != (3, bank.size, bank.size) or not torch.isfinite(normalized['boxes']).all():
            raise AssertionError('Invalid transformed training sample')
        pipeline_dropped += len(packed_target['labels']) - len(normalized['labels'])
        primary_class = int(bank.counts[index].argmax())
        if primary_class not in saved_classes or pixel_checks < 4:
            normal = image.resize((bank.size, bank.size), Image.Resampling.BILINEAR)
            left_draw, right_draw = ImageDraw.Draw(normal), ImageDraw.Draw(packed)
            width, height = image.size
            for box in target['boxes'].tolist():
                left_draw.rectangle([box[0] * bank.size / width, box[1] * bank.size / height,
                                     box[2] * bank.size / width, box[3] * bank.size / height], outline='lime', width=1)
            for detail in trace:
                color = 'lime' if detail['primary'] else 'orange'
                for box in detail['output_boxes']:
                    right_draw.rectangle(box, outline=color, width=1)
                x, y = detail['offset']
                crop = detail['crop']
                right_draw.rectangle((x, y, x + crop[2] - crop[0] - 1, y + crop[3] - crop[1] - 1), outline='cyan')
            comparison = Image.new('RGB', (bank.size * 2, bank.size + 30), 'white')
            comparison.paste(normal, (0, 30))
            comparison.paste(packed, (bank.size, 30))
            painter = ImageDraw.Draw(comparison)
            painter.text((8, 8), f'Ordinary resize: {len(target["labels"])} targets', fill='black')
            painter.text((bank.size + 8, 8), f'RegionDense: {len(packed_target["labels"])}; green=primary, orange=donor', fill='black')
            comparison.save(output / f'example_{pixel_checks:03d}_{image_id}.jpg', quality=95)
            saved_classes.add(primary_class)
        pixel_checks += 1
    density = (original + added).sum() / args.draws
    base = original.sum() / args.draws
    report = dict(seed=args.seed, draws=args.draws, forced_attempts=True,
                  mean_ordinary_targets=float(base), mean_targets_when_attempted=float(density),
                  configured_probability=transform.probability,
                  expected_mean_during_active_epochs=float(base + transform.probability * (density - base)),
                  packed_route_counts=routes, whole_fallback=args.draws - sum(routes),
                  real_pixel_checks=pixel_checks, max_box_side_change_px=max_size_error,
                  targets_removed_by_following_sanitize=pipeline_dropped,
                  class_exposure={name: dict(original=int(original[k]), added=int(added[k]),
                                             forced_attempt_ratio=float((original[k] + added[k]) / max(original[k], 1)),
                                             expected_active_ratio=float(1 + transform.probability * added[k] / max(original[k], 1)))
                                  for k, name in enumerate(bank.metadata['category_names'])},
                  seconds=time.monotonic() - start,
                  limitation='Training annotations and pixels only; no model forward, AP, matching-quality or speedup claim.')
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
