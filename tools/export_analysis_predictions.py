#!/usr/bin/env python3
"""Export the official, full-score postprocessor output for offline analysis.

Run with torchrun on two GPUs. Each rank owns disjoint dataset indices; there
is no sampler padding, confidence filtering, extra NMS, or test-time scaling.
FP32 evaluation matches engine/solver/det_engine.py, even for an AMP run.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as distributed
from torch.utils.data import DataLoader, Subset

from engine.core import YAMLConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-r', '--checkpoint', required=True)
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-per-gpu', type=int, default=8)
    args = parser.parse_args()
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    if world > 1:
        distributed.init_process_group('nccl')
    device = torch.device('cuda', local_rank)
    cfg = YAMLConfig(args.config)
    cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    cfg.yaml_cfg['val_dataloader']['dataset']['ann_file'] = args.ann_file
    cfg.yaml_cfg['val_dataloader']['total_batch_size'] = args.batch_per_gpu * world
    model = cfg.model.to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    use_ema = bool(checkpoint.get('ema'))
    state = checkpoint['ema']['module'] if use_ema else checkpoint['model']
    model.load_state_dict(state, strict=True)
    saved_epoch = checkpoint.get('last_epoch')
    del state, checkpoint
    postprocessor = cfg.postprocessor.to(device).eval()
    base_loader = cfg.val_dataloader
    indices = list(range(rank, len(base_loader.dataset), world))
    loader = DataLoader(Subset(base_loader.dataset, indices),
                        batch_size=args.batch_per_gpu, shuffle=False,
                        num_workers=2, collate_fn=base_loader.collate_fn,
                        pin_memory=True, drop_last=False)
    predictions = []
    image_ids = []
    with torch.inference_mode():
        for step, (images, targets) in enumerate(loader):
            outputs = model(images.to(device))
            original_sizes = torch.stack([t['orig_size'] for t in targets]).to(device)
            results = postprocessor(outputs, original_sizes)
            for target, result in zip(targets, results):
                image_id = int(target['image_id'].item())
                image_ids.append(image_id)
                boxes = result['boxes'].detach().cpu().clone()
                boxes[:, 2:] -= boxes[:, :2]
                for box, score, category in zip(boxes.tolist(), result['scores'].tolist(),
                                                result['labels'].tolist()):
                    predictions.append(dict(image_id=image_id, category_id=int(category),
                                            bbox=box, score=float(score)))
            if step % 10 == 0:
                print(f'rank={rank} batch={step + 1}/{len(loader)} images={len(image_ids)}', flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    part = Path(str(output) + f'.rank{rank}.json.gz')
    with gzip.open(part, 'wt', encoding='utf-8') as stream:
        json.dump(dict(image_ids=image_ids, predictions=predictions), stream)
    if world > 1:
        distributed.barrier()
    if rank == 0:
        all_predictions, all_ids = [], []
        for shard in range(world):
            with gzip.open(str(output) + f'.rank{shard}.json.gz', 'rt', encoding='utf-8') as stream:
                content = json.load(stream)
            all_predictions.extend(content['predictions'])
            all_ids.extend(content['image_ids'])
        expected_ids = set(base_loader.dataset.ids)
        if len(all_ids) != len(set(all_ids)) or set(all_ids) != expected_ids:
            raise ValueError('Prediction coverage must contain every split image exactly once.')
        all_predictions.sort(key=lambda d: (d['image_id'], -d['score']))
        with gzip.open(output, 'wt', encoding='utf-8') as stream:
            json.dump(all_predictions, stream)
        metadata = dict(config=str(Path(args.config).resolve()),
                        checkpoint=str(Path(args.checkpoint).resolve()),
                        checkpoint_last_epoch=saved_epoch, ema=use_ema,
                        annotation_file=str(Path(args.ann_file).resolve()),
                        images=len(all_ids), predictions=len(all_predictions),
                        world_size=world, batch_per_gpu=args.batch_per_gpu,
                        precision='float32, matching training-time evaluation',
                        confidence_filter=None, extra_nms=False,
                        postprocessor=repr(postprocessor))
        Path(str(output) + '.metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print(json.dumps(metadata, indent=2), flush=True)
    if world > 1:
        distributed.destroy_process_group()


if __name__ == '__main__':
    main()
