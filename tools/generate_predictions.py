"""
Generate COCO-format predictions for analysis
Run on validation set to get predictions.json
"""

import torch
import argparse
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pycocotools.coco import COCO
from tqdm import tqdm

from engine.core import YAMLConfig
from engine.misc import dist_utils


def main(args):
    """Generate predictions on validation set"""

    # Load config
    cfg = YAMLConfig(args.config, resume=args.resume)

    # Setup
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model
    model = cfg.model.to(device)

    # Load checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint and checkpoint['ema'] is not None:
            state_dict = checkpoint['ema']['module']
        else:
            state_dict = checkpoint['model']
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {args.resume}")

    model.eval()

    # Build dataloader
    val_dataloader = cfg.val_dataloader

    # Generate predictions
    predictions = []

    with torch.no_grad():
        for samples, targets in tqdm(val_dataloader, desc='Generating predictions'):
            samples = samples.to(device)

            # Forward
            outputs = model(samples)

            # Post-process
            if hasattr(cfg, 'postprocessor'):
                orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
                results = cfg.postprocessor(outputs, orig_target_sizes)
            else:
                results = outputs

            # Convert to COCO format
            for i, result in enumerate(results):
                image_id = targets[i]['image_id'].item()
                boxes = result['boxes'].cpu()  # [N, 4] xyxy
                scores = result['scores'].cpu()  # [N]
                labels = result['labels'].cpu()  # [N]

                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box.tolist()
                    predictions.append({
                        'image_id': image_id,
                        'category_id': int(label),
                        'bbox': [x1, y1, x2 - x1, y2 - y1],  # COCO format: xywh
                        'score': float(score)
                    })

    # Save predictions
    output_file = args.output if args.output else 'predictions.json'
    with open(output_file, 'w') as f:
        json.dump(predictions, f)

    print(f"Saved {len(predictions)} predictions to {output_file}")

    # Also save GT for easier analysis
    gt_file = output_file.replace('.json', '_gt.json')
    coco_gt = COCO(cfg.val_dataloader.dataset.ann_file)
    gt_data = {
        'images': [coco_gt.imgs[img_id] for img_id in coco_gt.imgs],
        'annotations': [coco_gt.anns[ann_id] for ann_id in coco_gt.anns],
        'categories': [coco_gt.cats[cat_id] for cat_id in coco_gt.cats]
    }
    with open(gt_file, 'w') as f:
        json.dump(gt_data, f)
    print(f"Saved GT annotations to {gt_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Config file')
    parser.add_argument('-r', '--resume', type=str, required=True, help='Checkpoint path')
    parser.add_argument('-o', '--output', type=str, default='predictions.json', help='Output file')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    args = parser.parse_args()
    main(args)
