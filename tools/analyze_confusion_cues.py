#!/usr/bin/env python3
"""CPU follow-up on a finished paired diagnosis: what separates the confused types?

Reads the exported predictions of a compare_rem_102e.py output directory and asks
four questions about localized-but-wrong-type aircraft (any-class prediction,
score>=0.3, IoU>=0.5, as in compare_rem_102e.confusion):

1. score ranking: is the true type the runner-up with a small margin?
2. geometry: does the size relative to the other aircraft in the same image
   separate the confused pairs, although source-pixel size does not?
3. clustering: are the errors spread over images or concentrated in a few?
4. crowding: does the error rate depend on neighbour distance?

It also writes crop sheets (fixed source-pixel window, no rescaling) of the
largest baseline confusions next to correctly typed examples of both classes.
Writes DIAG/confusion_cues/{report.md,summary.json,crops/*.png}; no inference.
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.compare_rem_102e import ROLES, SCORE_THRESHOLD, split_annotation
from tools.sar_analysis_core import iou_matrix, load_json, save_json

FOCUS_PAIRS = 4
CROP_WINDOW = 176  # source pixels; p90 box side of every class is below 133
CROP_SAMPLES = 16
NN_BINS = ((0, 1.5), (1.5, 3), (3, float('inf')))
COUNT_BINS = ((1, 3), (3, 6), (6, 11), (11, float('inf')))


def side(box):
    return max(box[2], box[3])


def box_key(d):
    return d['image_id'], tuple(round(float(v), 3) for v in d['bbox'])


def records(dataset, predictions, threshold=SCORE_THRESHOLD, iou=.5):
    """One row per GT: localization, class-score ranking and scale-free geometry."""
    names = {c['id']: c['name'] for c in dataset['categories']}
    images = {im['id']: im for im in dataset['images']}
    scores = defaultdict(dict)  # all classes exported for an identical box, any score
    for d in predictions:
        key = box_key(d)
        scores[key][d['category_id']] = max(d['score'], scores[key].get(d['category_id'], 0.))
    confident = defaultdict(dict)  # class-agnostic: best copy of each box
    for d in predictions:
        key = box_key(d)
        if d['score'] >= threshold and (key not in confident[d['image_id']]
                                        or d['score'] > confident[d['image_id']][key]['score']):
            confident[d['image_id']][key] = d
    confident = {k: list(v.values()) for k, v in confident.items()}
    gts = defaultdict(list)
    for a in dataset['annotations']:
        gts[a['image_id']].append(a)
    rows = []
    for image_id, annotations in gts.items():
        dets = confident.get(image_id, [])
        sizes = [side(a['bbox']) for a in annotations]
        centers = np.array([[a['bbox'][0] + a['bbox'][2] / 2, a['bbox'][1] + a['bbox'][3] / 2] for a in annotations])
        overlap = iou_matrix([a['bbox'] for a in annotations], [d['bbox'] for d in dets]) if dets else None
        for i, a in enumerate(annotations):
            others = sizes[:i] + sizes[i + 1:]
            distance = np.delete(np.linalg.norm(centers - centers[i], axis=1), i)
            row = dict(image_id=image_id, gt=names[a['category_id']], bbox=a['bbox'], size=sizes[i],
                       input_size=sizes[i] * 640 / images[image_id]['width'],
                       aspect=max(a['bbox'][2:]) / max(min(a['bbox'][2:]), 1e-6),
                       relative_gt=sizes[i] / float(np.median(others)) if len(others) >= 2 else None,
                       neighbour_gap=float(distance.min()) / sizes[i] if len(distance) else None,
                       image_gt=len(annotations), located=False, predicted=None, relative_pred=None,
                       true_rank=None, top_score=None, true_score=None)
            hits = [j for j in range(len(dets)) if overlap[i, j] >= iou] if dets else []
            if hits:
                j = max(hits, key=lambda k: dets[k]['score'])
                best = dets[j]
                ranked = sorted(scores[box_key(best)].items(), key=lambda kv: -kv[1])
                order = [c for c, _ in ranked]
                rest = [side(d['bbox']) for k, d in enumerate(dets)
                        if k != j and iou_matrix([best['bbox']], [d['bbox']])[0, 0] < iou]
                row.update(located=True, predicted=names[best['category_id']], top_score=ranked[0][1],
                           true_score=scores[box_key(best)].get(a['category_id']),
                           true_rank=order.index(a['category_id']) + 1 if a['category_id'] in order else None,
                           relative_pred=side(best['bbox']) / float(np.median(rest)) if len(rest) >= 2 else None)
            rows.append(row)
    return rows


def auc(positive, negative):
    """P(positive > negative) with ties counted half (Mann-Whitney)."""
    positive, negative = np.asarray(positive, float), np.asarray(negative, float)
    if not len(positive) or not len(negative):
        return None
    values = np.concatenate([positive, negative])
    order = values.argsort(kind='mergesort')
    ranks = np.empty(len(values))
    ranks[order] = np.arange(1, len(values) + 1)
    for value in np.unique(values):  # average ranks of ties
        tied = values == value
        ranks[tied] = ranks[tied].mean()
    return float((ranks[:len(positive)].sum() - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative)))


def median(values):
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def top_pairs(rows, limit=FOCUS_PAIRS):
    counts = Counter((r['gt'], r['predicted']) for r in rows if r['located'] and r['predicted'] != r['gt'])
    return [pair for pair, _ in counts.most_common(limit)]


def ranking_table(rows, classes):
    table = []
    for name in classes:
        located = [r for r in rows if r['gt'] == name and r['located']]
        wrong = [r for r in located if r['predicted'] != name]
        table.append(dict(category=name, located=len(located),
                          top1=sum(r['true_rank'] == 1 for r in located) / len(located) if located else None,
                          top2=sum(r['true_rank'] in (1, 2) for r in located) / len(located) if located else None,
                          wrong=len(wrong),
                          wrong_true_second=sum(r['true_rank'] == 2 for r in wrong) / len(wrong) if wrong else None,
                          wrong_true_absent=sum(r['true_rank'] is None for r in wrong) / len(wrong) if wrong else None,
                          wrong_median_margin=median([r['top_score'] - r['true_score'] for r in wrong
                                                      if r['true_score'] is not None])))
    return table


FEATURES = ('size', 'input_size', 'relative_gt', 'relative_pred', 'aspect')


def geometry_table(rows, pairs):
    table = []
    for a, b in pairs:
        row = dict(gt=a, predicted=b)
        for feature in FEATURES:
            value = auc([r[feature] for r in rows if r['gt'] == a and r[feature] is not None],
                        [r[feature] for r in rows if r['gt'] == b and r[feature] is not None])
            row[f'{feature}_auc'] = value
            row[f'{feature}_separability'] = None if value is None else max(value, 1 - value)
        for label, subset in (('a_correct', [r for r in rows if r['gt'] == a and r['predicted'] == a]),
                              ('a_as_b', [r for r in rows if r['gt'] == a and r['predicted'] == b]),
                              ('b_correct', [r for r in rows if r['gt'] == b and r['predicted'] == b])):
            row[f'{label}_n'] = len(subset)
            row[f'{label}_relative_gt'] = median([r['relative_gt'] for r in subset])
            row[f'{label}_size'] = median([r['size'] for r in subset])
        table.append(row)
    return table


def clustering_table(rows, classes):
    """Over-dispersion of per-image wrong-type counts against a common rate."""
    table = []
    for name in classes:
        per_image = defaultdict(lambda: [0, 0])
        for r in rows:
            if r['gt'] == name and r['located']:
                per_image[r['image_id']][0] += 1
                per_image[r['image_id']][1] += r['predicted'] != name
        multi = {k: v for k, v in per_image.items() if v[0] >= 2}
        n = np.array([v[0] for v in multi.values()], float)
        k = np.array([v[1] for v in multi.values()], float)
        rate = k.sum() / n.sum() if n.sum() else None
        variance = (n * rate * (1 - rate)).sum() if rate is not None else 0
        wrong = sorted((v[1] for v in per_image.values()), reverse=True)
        total = sum(wrong)
        half = next((i + 1 for i, c in enumerate(np.cumsum(wrong)) if c >= total / 2), None) if total else None
        table.append(dict(category=name, images=len(per_image), images_with_error=sum(w > 0 for w in wrong),
                          errors=total, images_for_half_errors=half, multi_images=len(multi), rate=rate,
                          dispersion=float(((k - n * rate) ** 2).sum() / variance) if variance else None,
                          errors_in_mostly_wrong_images=(float(k[(k / n >= .8)].sum()) / k.sum()
                                                         if len(k) and k.sum() else None)))
    return table


def binned(rows, key, bins):
    table = []
    for low, high in bins:
        subset = [r for r in rows if r[key] is not None and low <= r[key] < high]
        located = [r for r in subset if r['located']]
        table.append(dict(bin=f'[{low:g}, {high:g})', gt=len(subset),
                          located_rate=len(located) / len(subset) if subset else None,
                          correct_given_located=(sum(r['predicted'] == r['gt'] for r in located) / len(located)
                                                 if located else None)))
    return table


def to_8bit(image):
    if image.mode in ('L', 'RGB'):
        return image.copy()
    values = np.asarray(image, dtype=float)
    high = np.percentile(values, 99.5) or 1.
    return Image.fromarray((np.clip(values / high, 0, 1) * 255).astype(np.uint8))


def crop(image, box, label, colour):
    cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
    half = CROP_WINDOW / 2
    left, top = int(round(cx - half)), int(round(cy - half))
    tile = image.crop((left, top, left + CROP_WINDOW, top + CROP_WINDOW)).convert('RGB')
    draw = ImageDraw.Draw(tile)
    draw.rectangle([box[0] - left, box[1] - top, box[0] + box[2] - left, box[1] + box[3] - top], outline=colour)
    draw.text((3, 2), label, fill=(255, 255, 0))
    return tile


def crop_sheets(rows, dataset, data_root, pairs, out):
    files = {im['id']: im['file_name'] for im in dataset['images']}
    cache, written = {}, []
    out.mkdir(parents=True, exist_ok=True)
    for a, b in pairs:
        groups = ((f'{a} -> {b}', [r for r in rows if r['gt'] == a and r['predicted'] == b], (255, 64, 64)),
                  (f'{a} correct', [r for r in rows if r['gt'] == a and r['predicted'] == a], (64, 255, 64)),
                  (f'{b} correct', [r for r in rows if r['gt'] == b and r['predicted'] == b], (64, 160, 255)))
        sheet = Image.new('RGB', (8 * CROP_WINDOW, sum((14 + CROP_WINDOW * -(-min(len(g), CROP_SAMPLES) // 8))
                                                       for _, g, _ in groups)), (0, 0, 0))
        draw, y = ImageDraw.Draw(sheet), 0
        for title, group, colour in groups:
            draw.text((4, y + 1), f'{title}  (n={len(group)}, window {CROP_WINDOW}px source)', fill=(255, 255, 255))
            y += 14
            chosen = sorted(random.Random(0).sample(group, min(len(group), CROP_SAMPLES)),
                            key=lambda r: (r['image_id'], r['bbox']))
            for i, r in enumerate(chosen):
                if r['image_id'] not in cache:
                    cache[r['image_id']] = to_8bit(Image.open(data_root / files[r['image_id']]))
                rel = '-' if r['relative_gt'] is None else f"{r['relative_gt']:.2f}"
                tile = crop(cache[r['image_id']], r['bbox'], f"#{r['image_id']} s{r['size']:.0f} r{rel}", colour)
                sheet.paste(tile, ((i % 8) * CROP_WINDOW, y + (i // 8) * CROP_WINDOW))
            y += CROP_WINDOW * -(-len(chosen) // 8)
        path = out / f"{a}_as_{b}.png".replace('/', '-')
        sheet.save(path)
        written.append(str(path))
    return written


def fmt(value, digits=2):
    return '—' if value is None else f'{value:.{digits}f}'


def analyze(diag, crops=True):
    diag = Path(diag).resolve()
    annotation, split = split_annotation(diag)
    dataset = load_json(annotation)
    manifest = load_json(diag / 'manifest.json')
    rows = {role: records(dataset, load_json(diag / role / 'predictions.json.gz')) for role in ROLES}
    pairs = top_pairs(rows['baseline'])
    classes = list(dict.fromkeys(name for pair in pairs for name in pair))
    out = diag / 'confusion_cues'
    summary = dict(split=split, focus_pairs=[list(p) for p in pairs],
                   ranking={role: ranking_table(rows[role], classes) for role in ROLES},
                   geometry=geometry_table(rows['baseline'], pairs),
                   clustering={role: clustering_table(rows[role], classes) for role in ROLES},
                   neighbour_gap={role: binned(rows[role], 'neighbour_gap', NN_BINS) for role in ROLES},
                   image_gt={role: binned(rows[role], 'image_gt', COUNT_BINS) for role in ROLES},
                   definitions=dict(located=f'any-class prediction, score>={SCORE_THRESHOLD}, IoU>=0.5; not one-to-one',
                                    true_rank='rank of the GT class among the exported class scores of the same box; '
                                              'absent = not among the top-300 query x class exports',
                                    relative_gt='max(w,h) / median max(w,h) of the other GT in the image (>=2 others)',
                                    relative_pred='same for the matched prediction against the other confident '
                                                  'predictions (usable at inference)',
                                    neighbour_gap='nearest GT centre distance / own max(w,h)',
                                    dispersion='Pearson chi2 / binomial variance over images with >=2 located GT; '
                                               '1 = errors spread like independent draws, >1 = clustered',
                                    separability='max(AUC, 1-AUC) of the feature for GT class a vs b'))
    if crops:
        summary['crops'] = crop_sheets(rows['baseline'], dataset, Path(manifest['data_root']), pairs, out / 'crops')
    save_json(out / 'summary.json', summary)

    lines = [f'# 型号混淆线索（{split}）', '', '焦点混淆对（baseline 最多的 4 对）：' +
             '，'.join(f'{a}→{b}' for a, b in pairs), '',
             '## 1. 定位后的类别分数排序', '',
             '| 模型 | 类别 | 定位 | top1 | top2 | 判错 | 判错中真类第 2 | 判错中真类缺失 | 判错分数差中位数 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for role in ROLES:
        lines += [f"| {role} | {r['category']} | {r['located']} | {fmt(r['top1'], 3)} | {fmt(r['top2'], 3)} | "
                  f"{r['wrong']} | {fmt(r['wrong_true_second'], 3)} | {fmt(r['wrong_true_absent'], 3)} | "
                  f"{fmt(r['wrong_median_margin'], 3)} |" for r in summary['ranking'][role]]
    lines += ['', '## 2. 几何线索（baseline，GT 全体）', '',
              '可分性 = max(AUC, 1−AUC)，0.5 为无法区分。相对尺寸 = 本机 max(w,h) / 同图其他飞机中位数。', '',
              '| a→b | 源图尺寸 | 640 尺寸 | 同图相对(GT) | 同图相对(预测) | 长宽比 | '
              'a 正确 相对/源图 | a→b 相对/源图 | b 正确 相对/源图 |', '|---|---:|---:|---:|---:|---:|---|---|---|']
    for r in summary['geometry']:
        lines.append(f"| {r['gt']}→{r['predicted']} | " + ' | '.join(fmt(r[f'{f}_separability'], 3) for f in FEATURES) +
                     ' | ' + ' | '.join(f"{fmt(r[f'{g}_relative_gt'])} / {fmt(r[f'{g}_size'], 0)} (n={r[f'{g}_n']})"
                                        for g in ('a_correct', 'a_as_b', 'b_correct')) + ' |')
    lines += ['', '## 3. 错误是否集中在少数图像', '',
              '| 模型 | 类别 | 含该类的图 | 有错的图 | 判错数 | 一半错误所在图数 | 离散度 | 错误中来自"≥80% 判错"图的比例 |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for role in ROLES:
        lines += [f"| {role} | {r['category']} | {r['images']} | {r['images_with_error']} | {r['errors']} | "
                  f"{r['images_for_half_errors'] or '—'} | {fmt(r['dispersion'])} | "
                  f"{fmt(r['errors_in_mostly_wrong_images'], 3)} |" for r in summary['clustering'][role]]
    for key, title in (('neighbour_gap', '4a. 拥挤程度：最近邻距离 / 自身尺寸'), ('image_gt', '4b. 拥挤程度：同图 GT 数')):
        lines += ['', f'## {title}（全部类别）', '',
                  '| 模型 | 区间 | GT | 定位率 | 定位后型号正确率 |', '|---|---|---:|---:|---:|']
        for role in ROLES:
            lines += [f"| {role} | {r['bin']} | {r['gt']} | {fmt(r['located_rate'], 3)} | "
                      f"{fmt(r['correct_given_located'], 3)} |" for r in summary[key][role]]
    if crops:
        lines += ['', '## 5. 裁剪图', '',
                  f'每对一张：红 = a 判成 b，绿 = a 正确，蓝 = b 正确；固定 {CROP_WINDOW} 源图像素窗口、不缩放，'
                  '标注 `#图像 id s源图尺寸 r同图相对尺寸`，每组最多随机 16 个（seed 0）。', '']
        lines += [f'- `{Path(p).relative_to(diag).as_posix()}`' for p in summary['crops']]
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(out / 'report.md', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('diag', nargs='+', type=Path, help='finished compare_rem_102e.py output directory')
    parser.add_argument('--no-crops', action='store_true', help='skip the crop sheets (no image reads)')
    args = parser.parse_args()
    for diag in args.diag:
        analyze(diag, crops=not args.no_crops)


if __name__ == '__main__':
    main()
