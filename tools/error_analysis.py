"""
Comprehensive error analysis for object detection
Analyzes false negatives (FN), false positives (FP), and localization errors

Usage:
python tools/error_analysis.py \
    --gt predictions_gt.json \
    --dt predictions.json \
    --output error_analysis.txt
"""

import json
import argparse
import numpy as np
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def compute_iou(box1, box2):
    """Compute IoU between two boxes in xywh format"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def analyze_errors(gt_file, dt_file, iou_threshold=0.5, output_file='error_analysis.txt'):
    """Analyze detection errors"""

    # Load GT and predictions
    coco_gt = COCO(gt_file)
    coco_dt = coco_gt.loadRes(dt_file)

    # Get category names
    cat_names = {cat['id']: cat['name'] for cat in coco_gt.dataset['categories']}

    print(f"Analyzing with IoU threshold = {iou_threshold}")
    print(f"Categories: {cat_names}")

    # Storage for analysis
    stats = {
        'total_gt': 0,
        'total_dt': 0,
        'tp': 0,
        'fp': 0,
        'fn': 0,
        'per_class': defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'gt_count': 0}),
        'size_bins': {
            'tiny': {'tp': 0, 'fn': 0, 'total': 0},      # area < 32^2
            'small': {'tp': 0, 'fn': 0, 'total': 0},     # 32^2 <= area < 96^2
            'medium': {'tp': 0, 'fn': 0, 'total': 0},    # 96^2 <= area < 256^2 (很少)
        },
        'localization_errors': [],  # IoU between 0.1 and iou_threshold
        'background_fps': [],       # FPs with max IoU < 0.1
    }

    # Process each image
    img_ids = coco_gt.getImgIds()

    for img_id in img_ids:
        # Get GT annotations
        gt_ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        gt_anns = coco_gt.loadAnns(gt_ann_ids)

        # Get predictions
        dt_ann_ids = coco_dt.getAnnIds(imgIds=img_id)
        dt_anns = coco_dt.loadAnns(dt_ann_ids)

        stats['total_gt'] += len(gt_anns)
        stats['total_dt'] += len(dt_anns)

        # Track matched GTs
        matched_gt = set()

        # For each prediction, find best matching GT
        for dt_ann in dt_anns:
            dt_box = dt_ann['bbox']
            dt_cat = dt_ann['category_id']
            dt_score = dt_ann['score']

            best_iou = 0
            best_gt_idx = -1

            # Find best matching GT of same category
            for gt_idx, gt_ann in enumerate(gt_anns):
                if gt_ann['category_id'] != dt_cat:
                    continue

                gt_box = gt_ann['bbox']
                iou = compute_iou(dt_box, gt_box)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            # Classify detection
            if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
                # True positive
                stats['tp'] += 1
                stats['per_class'][dt_cat]['tp'] += 1
                matched_gt.add(best_gt_idx)

                # Record GT area
                gt_area = gt_anns[best_gt_idx]['area']
                if gt_area < 32 * 32:
                    stats['size_bins']['tiny']['tp'] += 1
                elif gt_area < 96 * 96:
                    stats['size_bins']['small']['tp'] += 1
                else:
                    stats['size_bins']['medium']['tp'] += 1

            elif 0.1 <= best_iou < iou_threshold:
                # Localization error (detected but IoU too low)
                stats['localization_errors'].append({
                    'iou': best_iou,
                    'score': dt_score,
                    'category': dt_cat,
                    'dt_box': dt_box,
                    'gt_box': gt_anns[best_gt_idx]['bbox'] if best_gt_idx >= 0 else None
                })
                stats['fp'] += 1
                stats['per_class'][dt_cat]['fp'] += 1

            else:
                # Background false positive (no nearby GT)
                stats['background_fps'].append({
                    'score': dt_score,
                    'category': dt_cat,
                    'box': dt_box,
                    'max_iou': best_iou
                })
                stats['fp'] += 1
                stats['per_class'][dt_cat]['fp'] += 1

        # Count false negatives (unmatched GTs)
        for gt_idx, gt_ann in enumerate(gt_anns):
            gt_cat = gt_ann['category_id']
            gt_area = gt_ann['area']

            stats['per_class'][gt_cat]['gt_count'] += 1

            if gt_idx not in matched_gt:
                stats['fn'] += 1
                stats['per_class'][gt_cat]['fn'] += 1

                # Record size distribution of FNs
                if gt_area < 32 * 32:
                    stats['size_bins']['tiny']['fn'] += 1
                    stats['size_bins']['tiny']['total'] += 1
                elif gt_area < 96 * 96:
                    stats['size_bins']['small']['fn'] += 1
                    stats['size_bins']['small']['total'] += 1
                else:
                    stats['size_bins']['medium']['fn'] += 1
                    stats['size_bins']['medium']['total'] += 1
            else:
                # Count total GTs for recall calculation
                if gt_area < 32 * 32:
                    stats['size_bins']['tiny']['total'] += 1
                elif gt_area < 96 * 96:
                    stats['size_bins']['small']['total'] += 1
                else:
                    stats['size_bins']['medium']['total'] += 1

    # Compute summary statistics
    precision = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0
    recall = stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Generate report
    report = []
    report.append("=" * 80)
    report.append("DETECTION ERROR ANALYSIS")
    report.append("=" * 80)
    report.append(f"\nIoU Threshold: {iou_threshold}")
    report.append(f"\nTotal GT annotations: {stats['total_gt']}")
    report.append(f"Total predictions: {stats['total_dt']}")
    report.append(f"\nTrue Positives (TP): {stats['tp']}")
    report.append(f"False Positives (FP): {stats['fp']}")
    report.append(f"False Negatives (FN): {stats['fn']}")
    report.append(f"\nPrecision: {precision:.3f}")
    report.append(f"Recall: {recall:.3f}")
    report.append(f"F1 Score: {f1:.3f}")

    # Per-class analysis
    report.append("\n" + "-" * 80)
    report.append("PER-CLASS ANALYSIS")
    report.append("-" * 80)
    report.append(f"{'Category':<20} {'GT Count':<12} {'TP':<8} {'FP':<8} {'FN':<8} {'Precision':<12} {'Recall':<12}")
    report.append("-" * 80)

    for cat_id in sorted(stats['per_class'].keys()):
        cat_stats = stats['per_class'][cat_id]
        cat_name = cat_names.get(cat_id, f'Class_{cat_id}')
        cat_prec = cat_stats['tp'] / (cat_stats['tp'] + cat_stats['fp']) if (cat_stats['tp'] + cat_stats['fp']) > 0 else 0
        cat_rec = cat_stats['tp'] / (cat_stats['tp'] + cat_stats['fn']) if (cat_stats['tp'] + cat_stats['fn']) > 0 else 0

        report.append(f"{cat_name:<20} {cat_stats['gt_count']:<12} {cat_stats['tp']:<8} {cat_stats['fp']:<8} {cat_stats['fn']:<8} {cat_prec:<12.3f} {cat_rec:<12.3f}")

    # Size-based analysis
    report.append("\n" + "-" * 80)
    report.append("SIZE-BASED ANALYSIS")
    report.append("-" * 80)
    report.append(f"{'Size Bin':<15} {'Total GT':<12} {'TP':<8} {'FN':<8} {'Recall':<12} {'Miss Rate':<12}")
    report.append("-" * 80)

    for size_name, size_stats in stats['size_bins'].items():
        if size_stats['total'] > 0:
            size_recall = size_stats['tp'] / size_stats['total']
            size_miss = size_stats['fn'] / size_stats['total']
            report.append(f"{size_name:<15} {size_stats['total']:<12} {size_stats['tp']:<8} {size_stats['fn']:<8} {size_recall:<12.3f} {size_miss:<12.3f}")

    # Localization error analysis
    report.append("\n" + "-" * 80)
    report.append("LOCALIZATION ERROR ANALYSIS")
    report.append("-" * 80)
    report.append(f"Total localization errors: {len(stats['localization_errors'])}")
    report.append(f"  (Detections with 0.1 <= IoU < {iou_threshold})")

    if len(stats['localization_errors']) > 0:
        loc_ious = [err['iou'] for err in stats['localization_errors']]
        report.append(f"  Mean IoU: {np.mean(loc_ious):.3f}")
        report.append(f"  Median IoU: {np.median(loc_ious):.3f}")
        report.append(f"  IoU range: [{np.min(loc_ious):.3f}, {np.max(loc_ious):.3f}]")

        # Per-class localization errors
        loc_per_class = defaultdict(list)
        for err in stats['localization_errors']:
            loc_per_class[err['category']].append(err['iou'])

        report.append(f"\n  Per-class localization errors:")
        for cat_id in sorted(loc_per_class.keys()):
            cat_name = cat_names.get(cat_id, f'Class_{cat_id}')
            cat_ious = loc_per_class[cat_id]
            report.append(f"    {cat_name}: {len(cat_ious)} errors, mean IoU = {np.mean(cat_ious):.3f}")

    # Background FP analysis
    report.append("\n" + "-" * 80)
    report.append("BACKGROUND FALSE POSITIVE ANALYSIS")
    report.append("-" * 80)
    report.append(f"Total background FPs: {len(stats['background_fps'])}")
    report.append(f"  (Detections with max IoU < 0.1)")

    if len(stats['background_fps']) > 0:
        bg_scores = [fp['score'] for fp in stats['background_fps']]
        report.append(f"  Mean score: {np.mean(bg_scores):.3f}")
        report.append(f"  Score range: [{np.min(bg_scores):.3f}, {np.max(bg_scores):.3f}]")

        # Per-class background FPs
        bg_per_class = defaultdict(int)
        for fp in stats['background_fps']:
            bg_per_class[fp['category']] += 1

        report.append(f"\n  Per-class background FPs:")
        for cat_id in sorted(bg_per_class.keys()):
            cat_name = cat_names.get(cat_id, f'Class_{cat_id}')
            report.append(f"    {cat_name}: {bg_per_class[cat_id]}")

    # Key findings summary
    report.append("\n" + "=" * 80)
    report.append("KEY FINDINGS")
    report.append("=" * 80)

    # Find worst-performing category
    worst_cat = None
    worst_recall = 1.0
    for cat_id, cat_stats in stats['per_class'].items():
        if cat_stats['gt_count'] > 0:
            cat_rec = cat_stats['tp'] / (cat_stats['tp'] + cat_stats['fn'])
            if cat_rec < worst_recall:
                worst_recall = cat_rec
                worst_cat = cat_id

    if worst_cat:
        report.append(f"1. Worst-performing category: {cat_names[worst_cat]} (Recall: {worst_recall:.3f})")

    # Find most problematic size
    worst_size = None
    worst_size_miss = 0
    for size_name, size_stats in stats['size_bins'].items():
        if size_stats['total'] > 0:
            miss_rate = size_stats['fn'] / size_stats['total']
            if miss_rate > worst_size_miss:
                worst_size_miss = miss_rate
                worst_size = size_name

    if worst_size:
        report.append(f"2. Most problematic size: {worst_size} (Miss rate: {worst_size_miss:.3f})")

    # Localization vs detection issue
    loc_error_ratio = len(stats['localization_errors']) / stats['fp'] if stats['fp'] > 0 else 0
    bg_fp_ratio = len(stats['background_fps']) / stats['fp'] if stats['fp'] > 0 else 0

    report.append(f"3. False positive breakdown:")
    report.append(f"   - Localization errors: {loc_error_ratio:.1%} ({len(stats['localization_errors'])} / {stats['fp']})")
    report.append(f"   - Background FPs: {bg_fp_ratio:.1%} ({len(stats['background_fps'])} / {stats['fp']})")

    if loc_error_ratio > 0.5:
        report.append(f"   → Main issue: LOCALIZATION (boxes are detected but not accurate enough)")
    elif bg_fp_ratio > 0.5:
        report.append(f"   → Main issue: BACKGROUND CLUTTER (false detections in empty areas)")

    report.append("\n" + "=" * 80)

    # Print and save report
    report_text = '\n'.join(report)
    print(report_text)

    with open(output_file, 'w') as f:
        f.write(report_text)

    print(f"\nReport saved to {output_file}")

    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, required=True, help='GT annotations (predictions_gt.json)')
    parser.add_argument('--dt', type=str, required=True, help='Predictions (predictions.json)')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold')
    parser.add_argument('--output', type=str, default='error_analysis.txt', help='Output file')

    args = parser.parse_args()

    analyze_errors(args.gt, args.dt, args.iou, args.output)
