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

            # Post-process: convert outputs to detection results
            # outputs['pred_boxes']: [B, num_queries, 4] cxcywh normalized
            # outputs['pred_logits']: [B, num_queries, num_classes]

            pred_boxes = outputs['pred_boxes']  # [B, num_queries, 4]
            pred_logits = outputs['pred_logits']  # [B, num_queries, num_classes]

            batch_size = pred_boxes.shape[0]

            for i in range(batch_size):
                image_id = targets[i]['image_id'].item()
                orig_size = targets[i]['orig_size']  # [H, W]

                # Get scores and labels for this image
                scores = pred_logits[i].sigmoid()  # [num_queries, num_classes]
                scores_per_query, labels_per_query = scores.max(dim=-1)  # [num_queries]

                # Apply score threshold
                keep = scores_per_query > args.score_thr

                if keep.sum() == 0:
                    continue

                # Filter boxes, scores, labels
                boxes_keep = pred_boxes[i][keep]  # [N, 4] cxcywh normalized
                scores_keep = scores_per_query[keep]  # [N]
                labels_keep = labels_per_query[keep]  # [N]

                # Convert cxcywh normalized to xyxy pixel coordinates
                cx, cy, w, h = boxes_keep.unbind(-1)
                boxes_xyxy = torch.stack([
                    (cx - 0.5 * w) * orig_size[1],
                    (cy - 0.5 * h) * orig_size[0],
                    (cx + 0.5 * w) * orig_size[1],
                    (cy + 0.5 * h) * orig_size[0]
                ], dim=-1)

                # Apply NMS per class
                from torchvision.ops import nms

                if args.no_nms:
                    keep_nms = torch.arange(len(labels_keep), device=labels_keep.device)
                else:
                    keep_all = []
                    for class_id in labels_keep.unique():
                        class_mask = labels_keep == class_id
                        class_boxes = boxes_xyxy[class_mask]
                        class_scores = scores_keep[class_mask]

                        keep_class = nms(class_boxes, class_scores, args.nms_thr)
                        keep_all.append(torch.where(class_mask)[0][keep_class])

                    if len(keep_all) == 0:
                        continue

                    keep_nms = torch.cat(keep_all)

                # Final boxes in xyxy pixel coordinates
                boxes_final = boxes_xyxy[keep_nms]
                scores_final = scores_keep[keep_nms]
                labels_final = labels_keep[keep_nms]

                # Convert to COCO format (xywh)
                for box, score, label in zip(boxes_final, scores_final, labels_final):
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
    parser.add_argument('--score-thr', type=float, default=0.3,
                        help='Score threshold. Use a very small value (e.g. 0.001) '
                             'for ceiling analysis, which must see the full ranking.')
    parser.add_argument('--nms-thr', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--no-nms', action='store_true',
                        help='Skip NMS. COCOeval already applies its own greedy matching, '
                             'so NMS only removes detections before it sees them.')

    args = parser.parse_args()
    main(args)
