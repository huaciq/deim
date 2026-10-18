"""
Query Recall diagnosis for DETR-like detectors
Analyzes how well Query Selection captures ground-truth objects

Metrics:
- Query Recall @ IoU (QR@X): % of GTs matched by top-K queries
- Best-query rank: Average rank of best matching query for each GT
"""

import torch
import argparse
import json
import numpy as np
from collections import defaultdict
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pycocotools.coco import COCO
from tqdm import tqdm

from engine.core import YAMLConfig


def box_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes
    boxes1, boxes2: [N, 4] and [M, 4] in xyxy format
    Returns: [N, M] IoU matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou


def cxcywh_to_xyxy(boxes):
    """Convert boxes from cxcywh to xyxy format"""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def query_diagnosis(args):
    """Run query diagnosis on validation set"""

    # Load config
    cfg = YAMLConfig(args.config, resume=args.resume)

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

    # Storage for statistics
    stats = {
        'query_recall': defaultdict(list),  # QR at different IoU thresholds
        'best_query_rank': [],              # Rank of best-matching query
        'per_class': defaultdict(lambda: {
            'query_recall': defaultdict(list),
            'best_query_rank': []
        }),
        'per_size': defaultdict(lambda: {
            'query_recall': defaultdict(list),
            'best_query_rank': []
        })
    }

    iou_thresholds = [0.3, 0.5, 0.7, 0.75, 0.9]

    print(f"Running query diagnosis...")
    print(f"IoU thresholds: {iou_thresholds}")

    with torch.no_grad():
        for samples, targets in tqdm(val_dataloader, desc='Query diagnosis'):
            samples = samples.to(device)
            batch_size = samples.shape[0]

            # Forward (get decoder outputs before NMS)
            outputs = model(samples, targets)

            # Extract query predictions (last decoder layer)
            # outputs['pred_boxes']: [B, num_queries, 4] cxcywh normalized
            # outputs['pred_logits']: [B, num_queries, num_classes]
            pred_boxes = outputs['pred_boxes']  # [B, num_queries, 4]
            pred_logits = outputs['pred_logits']  # [B, num_queries, num_classes]

            for i in range(batch_size):
                # Get GT for this image
                gt_boxes = targets[i]['boxes'].to(device)  # [N, 4] cxcywh normalized - move to device
                gt_labels = targets[i]['labels'].to(device)  # [N] - move to device
                orig_size = targets[i]['orig_size'].to(device)  # [H, W] - move to device

                if len(gt_boxes) == 0:
                    continue

                # Convert query boxes: cxcywh normalized -> xyxy pixel
                query_boxes = cxcywh_to_xyxy(pred_boxes[i])  # [num_queries, 4] xyxy normalized
                query_boxes[:, [0, 2]] = query_boxes[:, [0, 2]] * orig_size[1]  # x coords
                query_boxes[:, [1, 3]] = query_boxes[:, [1, 3]] * orig_size[0]  # y coords

                # Convert GT boxes: cxcywh normalized -> xyxy pixel
                gt_boxes_xyxy = cxcywh_to_xyxy(gt_boxes)  # [N, 4] xyxy normalized
                gt_boxes_xyxy[:, [0, 2]] = gt_boxes_xyxy[:, [0, 2]] * orig_size[1]  # x coords
                gt_boxes_xyxy[:, [1, 3]] = gt_boxes_xyxy[:, [1, 3]] * orig_size[0]  # y coords

                # Compute IoU between all queries and all GTs
                iou_matrix = box_iou(query_boxes, gt_boxes_xyxy)  # [num_queries, num_gts]

                # For each GT, find best matching query
                best_ious, best_query_indices = iou_matrix.max(dim=0)  # [num_gts]

                # Compute query scores (max class score)
                query_scores = pred_logits[i].sigmoid().max(dim=-1)[0]  # [num_queries]

                # Sort queries by score (descending)
                sorted_indices = torch.argsort(query_scores, descending=True)
                query_ranks = torch.argsort(sorted_indices)  # Rank of each query

                # Analyze each GT
                for gt_idx in range(len(gt_boxes)):
                    best_query_idx = best_query_indices[gt_idx].item()
                    best_iou = best_ious[gt_idx].item()
                    rank = query_ranks[best_query_idx].item() + 1  # 1-indexed rank

                    gt_label = gt_labels[gt_idx].item()
                    gt_area = (gt_boxes[gt_idx][2] * gt_boxes[gt_idx][3] * orig_size[0] * orig_size[1]).item()

                    # Determine size category
                    if gt_area < 32 * 32:
                        size_cat = 'tiny'
                    elif gt_area < 96 * 96:
                        size_cat = 'small'
                    else:
                        size_cat = 'medium'

                    # Record rank
                    stats['best_query_rank'].append(rank)
                    stats['per_class'][gt_label]['best_query_rank'].append(rank)
                    stats['per_size'][size_cat]['best_query_rank'].append(rank)

                    # Check query recall at different IoU thresholds and top-K
                    for iou_thr in iou_thresholds:
                        matched = best_iou >= iou_thr

                        # Check if best query is in top-K
                        for top_k in [100, 300, 500, 900]:
                            in_topk = rank <= top_k
                            recall = 1 if (matched and in_topk) else 0

                            key = f'QR@{iou_thr}_top{top_k}'
                            stats['query_recall'][key].append(recall)
                            stats['per_class'][gt_label]['query_recall'][key].append(recall)
                            stats['per_size'][size_cat]['query_recall'][key].append(recall)

    # Compute summary statistics
    print("\n" + "=" * 80)
    print("QUERY DIAGNOSIS REPORT")
    print("=" * 80)

    # Overall query recall
    print("\nOVERALL QUERY RECALL:")
    print("-" * 80)
    print(f"{'Metric':<25} {'Top-100':<12} {'Top-300':<12} {'Top-500':<12} {'Top-900':<12}")
    print("-" * 80)

    for iou_thr in iou_thresholds:
        metrics = []
        for top_k in [100, 300, 500, 900]:
            key = f'QR@{iou_thr}_top{top_k}'
            qr = np.mean(stats['query_recall'][key]) * 100
            metrics.append(f"{qr:.1f}%")
        print(f"QR @ IoU={iou_thr:<7.2f} {metrics[0]:<12} {metrics[1]:<12} {metrics[2]:<12} {metrics[3]:<12}")

    # Best-query rank
    mean_rank = np.mean(stats['best_query_rank'])
    median_rank = np.median(stats['best_query_rank'])
    print(f"\nBest-query rank:")
    print(f"  Mean: {mean_rank:.1f}")
    print(f"  Median: {median_rank:.1f}")

    # Per-class analysis
    print("\n" + "-" * 80)
    print("PER-CLASS QUERY RECALL @ IoU=0.5 (Top-300):")
    print("-" * 80)

    coco_gt = COCO(cfg.val_dataloader.dataset.ann_file)
    cat_names = {cat['id']: cat['name'] for cat in coco_gt.dataset['categories']}

    for cat_id in sorted(stats['per_class'].keys()):
        cat_name = cat_names.get(cat_id, f'Class_{cat_id}')
        key = 'QR@0.5_top300'
        qr = np.mean(stats['per_class'][cat_id]['query_recall'][key]) * 100
        cat_rank = np.mean(stats['per_class'][cat_id]['best_query_rank'])
        print(f"{cat_name:<20} QR@0.5: {qr:>6.1f}%   Mean rank: {cat_rank:>6.1f}")

    # Per-size analysis
    print("\n" + "-" * 80)
    print("PER-SIZE QUERY RECALL @ IoU=0.5 (Top-300):")
    print("-" * 80)

    for size_cat in ['tiny', 'small', 'medium']:
        if size_cat in stats['per_size']:
            key = 'QR@0.5_top300'
            qr = np.mean(stats['per_size'][size_cat]['query_recall'][key]) * 100
            size_rank = np.mean(stats['per_size'][size_cat]['best_query_rank'])
            print(f"{size_cat:<20} QR@0.5: {qr:>6.1f}%   Mean rank: {size_rank:>6.1f}")

    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS:")
    print("=" * 80)

    qr_05_top300 = np.mean(stats['query_recall']['QR@0.5_top300']) * 100
    qr_075_top300 = np.mean(stats['query_recall']['QR@0.75_top300']) * 100

    print(f"1. Top-300 queries capture {qr_05_top300:.1f}% of GTs at IoU=0.5")
    print(f"2. Top-300 queries capture {qr_075_top300:.1f}% of GTs at IoU=0.75")
    print(f"3. Gap: {qr_05_top300 - qr_075_top300:.1f} percentage points")

    if qr_05_top300 > 90 and qr_075_top300 < 60:
        print("\n→ DIAGNOSIS: Objects are DETECTED but LOCALIZATION is poor")
        print("   Recommendation: Focus on improving box regression quality")
    elif qr_05_top300 < 80:
        print("\n→ DIAGNOSIS: Many objects are MISSED by Query Selection")
        print("   Recommendation: Improve feature extraction or query initialization")

    print("=" * 80)

    # Save detailed results
    output_file = args.output if args.output else 'query_diagnosis.txt'
    with open(output_file, 'w') as f:
        f.write("Query Diagnosis Results\n")
        f.write("=" * 80 + "\n\n")

        f.write("Overall Query Recall:\n")
        for iou_thr in iou_thresholds:
            for top_k in [100, 300, 500, 900]:
                key = f'QR@{iou_thr}_top{top_k}'
                qr = np.mean(stats['query_recall'][key]) * 100
                f.write(f"{key}: {qr:.2f}%\n")

        f.write(f"\nMean best-query rank: {mean_rank:.2f}\n")
        f.write(f"Median best-query rank: {median_rank:.2f}\n")

    print(f"\nDetailed results saved to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Config file')
    parser.add_argument('-r', '--resume', type=str, required=True, help='Checkpoint path')
    parser.add_argument('-o', '--output', type=str, default='query_diagnosis.txt', help='Output file')

    args = parser.parse_args()
    query_diagnosis(args)
