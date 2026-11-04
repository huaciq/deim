#!/usr/bin/env python3
"""CPU follow-up to diagnose_detection_errors.py: why are targets missed, and what are the false positives?

Every test GT gets a status from the exported candidates (any class):
located (IoU>=0.5, score>=0.3), low_score (IoU>=0.5 below that), poor_box (best IoU in [0.1, 0.5)),
no_candidate (best IoU<0.1). The located rate is then split by match score, truncation at the tile
edge, contrast, size, source image size and whether the image holds any seen target, to check whether
"unseen" really means an unknown aircraft or a weak or cut target the matcher could not match.
Crop sheets show missed unseen targets per class and the top background false positives.

The export keeps the top 300 of query x class scores, so no_candidate means no box among those.
Diagnostic only: selects nothing and changes no metric.
Writes OUTPUT/{report.md,summary.json,targets.csv,sheets/*.png}; no inference or training.
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

from tools.analyze_confusion_cues import CROP_WINDOW, crop, to_8bit
from tools.analyze_sar_characteristics import measure_split
from tools.compare_rem_102e import write_csv
from tools.diagnose_detection_errors import SCORE, SEEN_THRESHOLD, annotation_ids, best_copies, fmt, seen_scores
from tools.sar_analysis_core import iou_matrix, load_json, save_json

STATUSES = ('located', 'low_score', 'poor_box', 'no_candidate')
BACKGROUND_IOU = .1
EDGE_MARGIN = 2  # source pixels from the tile border
MATCH_BINS = ((-1., .8), (.8, SEEN_THRESHOLD), (SEEN_THRESHOLD, 1.01))
SIZE_BINS = ((0, 40), (40, 60), (60, 80), (80, 100), (100, 10000))
SHEET_SAMPLES = 32
SHEET_COLUMNS = 8


def status_of(box, candidates, threshold=SCORE):
    """Status, best IoU over all candidates, and the top score among candidates with IoU>=0.5."""
    if not candidates:
        return 'no_candidate', 0., None
    overlap = iou_matrix([box], [d['bbox'] for d in candidates])[0]
    scores = [d['score'] for d, value in zip(candidates, overlap) if value >= .5]
    best = float(overlap.max())
    if scores:
        return ('located' if max(scores) >= threshold else 'low_score'), best, float(max(scores))
    return ('poor_box' if best >= BACKGROUND_IOU else 'no_candidate'), best, None


def target_rows(dataset, predictions, scores, measures, threshold):
    """One row per GT, in annotation order of the image list."""
    names = {c['id']: c['name'] for c in dataset['categories']}
    images = {im['id']: im for im in dataset['images']}
    candidates = defaultdict(list)
    for d in best_copies(predictions):
        candidates[d['image_id']].append(d)
    annotations = {a['id']: a for a in dataset['annotations']}
    seen_images = {a['image_id'] for a in dataset['annotations'] if scores[a['id']] >= threshold}
    counts = Counter(a['image_id'] for a in dataset['annotations'])
    rows = []
    for annotation_id, measure in zip(annotation_ids(dataset), measures):
        a = annotations[annotation_id]
        image = images[a['image_id']]
        x, y, w, h = a['bbox']
        status, best_iou, best_score = status_of(a['bbox'], candidates[a['image_id']])
        rows.append(dict(annotation_id=annotation_id, image_id=a['image_id'], category=names[a['category_id']],
                         bbox=a['bbox'], seen=scores[annotation_id] >= threshold, match_score=scores[annotation_id],
                         status=status, best_iou=best_iou, best_score=best_score, size=max(w, h),
                         at_edge=(x <= EDGE_MARGIN or y <= EDGE_MARGIN or x + w >= image['width'] - EDGE_MARGIN
                                  or y + h >= image['height'] - EDGE_MARGIN),
                         tcr_db=measure['tcr_db'], source_size=measure['source_size'], image_gt=counts[a['image_id']],
                         image_has_seen=a['image_id'] in seen_images))
    return rows


def rate(rows):
    return sum(r['status'] == 'located' for r in rows) / len(rows) if rows else None


def split_rates(rows, groups):
    """Located rate of seen and unseen targets per group; groups maps a label to a predicate."""
    table = []
    for label, keep in groups:
        chosen = [r for r in rows if keep(r)]
        seen = [r for r in chosen if r['seen']]
        unseen = [r for r in chosen if not r['seen']]
        table.append(dict(group=label, seen_gt=len(seen), seen_rate=rate(seen), unseen_gt=len(unseen), unseen_rate=rate(unseen)))
    return table


def status_table(rows, labels):
    table = []
    for label, keep in labels:
        chosen = [r for r in rows if keep(r)]
        row = dict(group=label, gt=len(chosen))
        row.update({s: sum(r['status'] == s for r in chosen) for s in STATUSES})
        table.append(row)
    return table


def concentration(values):
    """How many images hold half of a per-image count."""
    counts = sorted(values, reverse=True)
    total = sum(counts)
    half = next((i + 1 for i, c in enumerate(np.cumsum(counts)) if c >= total / 2), None) if total else None
    return dict(total=total, images=len(counts), images_for_half=half, top10_share=sum(counts[:10]) / total if total else None)


def false_positives(dataset, predictions, threshold=SCORE):
    gts = defaultdict(list)
    for a in dataset['annotations']:
        gts[a['image_id']].append(a['bbox'])
    names = {c['id']: c['name'] for c in dataset['categories']}
    rows = []
    for d in best_copies(predictions):
        if d['score'] < threshold:
            continue
        boxes = gts[d['image_id']]
        if not boxes or float(iou_matrix([d['bbox']], boxes)[0].max()) < BACKGROUND_IOU:
            rows.append(dict(image_id=d['image_id'], category=names[d['category_id']], bbox=d['bbox'], score=d['score']))
    return sorted(rows, key=lambda r: -r['score'])


def contact_sheet(path, title, items, files, data_root, colour, label):
    """Fixed source-pixel window, no rescaling, random sample in a stable order given by the caller."""
    rows = -(-len(items) // SHEET_COLUMNS)
    sheet = Image.new('RGB', (SHEET_COLUMNS * CROP_WINDOW, 14 + rows * CROP_WINDOW), (0, 0, 0))
    ImageDraw.Draw(sheet).text((4, 1), title, fill=(255, 255, 255))
    cache = {}
    for i, item in enumerate(items):
        if item['image_id'] not in cache:
            cache[item['image_id']] = to_8bit(Image.open(data_root / files[item['image_id']]))
        tile = crop(cache[item['image_id']], item['bbox'], label(item), colour)
        sheet.paste(tile, ((i % SHEET_COLUMNS) * CROP_WINDOW, 14 + (i // SHEET_COLUMNS) * CROP_WINDOW))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return str(path)


def sheets(rows, positives, dataset, data_root, output):
    files = {im['id']: im['file_name'] for im in dataset['images']}
    written = []
    for name in [c['name'] for c in dataset['categories']]:
        for kind, keep, colour in (('missed', lambda r: r['status'] != 'located', (255, 64, 64)),
                                   ('located', lambda r: r['status'] == 'located', (64, 255, 64))):
            group = [r for r in rows if r['category'] == name and not r['seen'] and keep(r)]
            if not group:
                continue
            chosen = sorted(random.Random(0).sample(group, min(len(group), SHEET_SAMPLES)),
                            key=lambda r: (r['image_id'], r['annotation_id']))
            title = f'unseen {name} {kind} (n={len(group)}, window {CROP_WINDOW}px source, match = best correlation in training set)'
            written.append(contact_sheet(output / f"sheets/unseen_{kind}_{name}.png".replace('/', '-').replace('sheets-', 'sheets/'),
                                         title, chosen, files, data_root, colour,
                                         lambda r: f"#{r['image_id']} m{r['match_score']:.2f} {r['status'][:4]}"))
    if positives:
        title = f'top background false positives (n={len(positives)} at score>={SCORE}): may include unlabelled aircraft'
        written.append(contact_sheet(output / 'sheets/false_positives.png', title, positives[:SHEET_SAMPLES], files, data_root,
                                     (255, 200, 0), lambda r: f"#{r['image_id']} {r['category'][:7]} {r['score']:.2f}"))
    return written


def analyze(predictions_path, data_root, output, reference='trainval', split='test', threshold=SEEN_THRESHOLD, make_sheets=True):
    data_root, output = Path(data_root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    dataset = load_json(data_root / f'COCO/{split}.json')
    predictions = load_json(predictions_path)
    scores = seen_scores(dataset, load_json(data_root / f'COCO/{reference}.json'), data_root)
    _, measures, _ = measure_split(dataset, data_root, split)
    rows = target_rows(dataset, predictions, scores, measures, threshold)
    unseen = [r for r in rows if not r['seen']]
    names = [c['name'] for c in dataset['categories']]
    edges = np.quantile([r['tcr_db'] for r in rows], [1 / 3, 2 / 3]).tolist()
    contrast = ((f'低对比（<{edges[0]:.1f} dB）', lambda r: r['tcr_db'] < edges[0]),
                (f'中（{edges[0]:.1f}–{edges[1]:.1f} dB）', lambda r: edges[0] <= r['tcr_db'] < edges[1]),
                (f'高（≥{edges[1]:.1f} dB）', lambda r: r['tcr_db'] >= edges[1]))
    sources = sorted({r['source_size'] for r in rows}, key=lambda s: int(s.split('x')[0]))
    positives = false_positives(dataset, predictions)
    missed = Counter(r['image_id'] for r in unseen if r['status'] != 'located')
    fp_counts = Counter(p['image_id'] for p in positives)
    all_unseen_images = {r['image_id'] for r in rows if not r['image_has_seen']}
    summary = dict(split=split, reference=reference, seen_threshold=threshold, gt=len(rows), unseen=len(unseen),
                   status=status_table(rows, [('见过', lambda r: r['seen']), ('没见过', lambda r: not r['seen'])] +
                                       [(f'没见过 {c}', lambda r, c=c: not r['seen'] and r['category'] == c) for c in names]),
                   match=split_rates(rows, [(f'[{a:g}, {b:g})', lambda r, a=a, b=b: a <= r['match_score'] < b) for a, b in MATCH_BINS]),
                   edge=split_rates(rows, [('切片边缘', lambda r: r['at_edge']), ('不在边缘', lambda r: not r['at_edge'])]),
                   contrast=split_rates(rows, contrast),
                   size=split_rates(rows, [(f'[{a}, {b})', lambda r, a=a, b=b: a <= r['size'] < b) for a, b in SIZE_BINS]),
                   source=split_rates(rows, [(s, lambda r, s=s: r['source_size'] == s) for s in sources]),
                   image_kind=split_rates(rows, [('图中有见过的目标', lambda r: r['image_has_seen']),
                                                 ('图中全部没见过', lambda r: not r['image_has_seen'])]),
                   unseen_misses_by_image=concentration(missed.values()),
                   false_positives=dict(concentration(fp_counts.values()), by_class=dict(Counter(p['category'] for p in positives)),
                                        in_all_unseen_images=sum(p['image_id'] in all_unseen_images for p in positives),
                                        p50_score=float(np.median([p['score'] for p in positives])) if positives else None),
                   images=dict(total=len(dataset['images']), all_unseen=len(all_unseen_images)),
                   definitions=dict(located=f'any-class candidate with IoU>=0.5 and score>={SCORE}',
                                    low_score='a candidate with IoU>=0.5 exists, all below that score',
                                    poor_box=f'best candidate IoU in [{BACKGROUND_IOU}, 0.5)',
                                    no_candidate=f'best candidate IoU<{BACKGROUND_IOU}; candidates are the top 300 query x class exports',
                                    at_edge=f'box within {EDGE_MARGIN} px of the tile border',
                                    tcr_db='brightest 10% of box pixels against the clutter ring, as in analyze_sar_characteristics',
                                    false_positive=f'score>={SCORE}, IoU<{BACKGROUND_IOU} to every GT'))
    if make_sheets:
        summary['sheets'] = sheets(rows, positives, dataset, data_root, output)
    save_json(output / 'summary.json', summary)
    write_csv(output / 'targets.csv', [dict(r, bbox=' '.join(f'{v:.1f}' for v in r['bbox'])) for r in rows])
    (output / 'report.md').write_text(render(summary), encoding='utf-8')
    print(output / 'report.md', flush=True)
    return summary


def rate_table(title, rows):
    lines = ['', f'## {title}', '', '| 分组 | 见过 GT | 见过定位率 | 没见过 GT | 没见过定位率 |', '|---|---:|---:|---:|---:|']
    return lines + [f"| {r['group']} | {r['seen_gt']} | {fmt(r['seen_rate'], 3)} | {r['unseen_gt']} | {fmt(r['unseen_rate'], 3)} |" for r in rows]


def render(s):
    lines = [f"# {s['split']} 漏检原因（GT {s['gt']}，没见过 {s['unseen']}）", '',
             '## 1. 候选框状态', '', '| 分组 | GT | 已定位 | 有框但分数低 | 框不准 | 没有候选 |', '|---|---:|---:|---:|---:|---:|']
    lines += [f"| {r['group']} | {r['gt']} | " + ' | '.join(str(r[k]) for k in STATUSES) + ' |' for r in s['status']]
    lines += ['', '判读：有框但分数低多，是排序或置信度问题；框不准多，是回归问题；没有候选多，才说明 query 选择或特征本身有问题。']
    for title, key in (('2. 按匹配分（接近 0.9 的可能是见过的飞机被弱化或截断）', 'match'), ('3. 按是否在切片边缘', 'edge'),
                       ('4. 按对比度', 'contrast'), ('5. 按框尺寸（max(w,h)，源图像素）', 'size'), ('6. 按源图尺寸', 'source'),
                       ('7. 按图内是否有见过的目标（全部没见过的图可能是训练没覆盖的场景）', 'image_kind')):
        lines += rate_table(title, s[key])
    m, f = s['unseen_misses_by_image'], s['false_positives']
    lines += ['', '## 8. 集中程度', '', '| 项目 | 总数 | 涉及图数 | 占一半所需图数 | 前 10 张占比 |', '|---|---:|---:|---:|---:|',
              f"| 没见过目标的漏检 | {m['total']} | {m['images']} | {m['images_for_half'] or '—'} | {fmt(m['top10_share'], 3)} |",
              f"| 背景误检 | {f['total']} | {f['images']} | {f['images_for_half'] or '—'} | {fmt(f['top10_share'], 3)} |", '',
              f"测试图 {s['images']['total']} 张，其中全部目标都没见过的 {s['images']['all_unseen']} 张；"
              f"背景误检有 {f['in_all_unseen_images']} 个落在这些图上，分数中位数 {fmt(f['p50_score'])}。", '',
              '背景误检按类别：' + '，'.join(f'{k} {v}' for k, v in sorted(f['by_class'].items(), key=lambda kv: -kv[1])), '']
    if s.get('sheets'):
        lines += ['## 9. 裁剪图', '', '`m` 为在训练集中的最佳匹配相关，`loca/low_/poor/no_c` 为状态。false_positives 可能包含漏标的飞机。', '']
        lines += [f'- `{Path(p).parent.name}/{Path(p).name}`' for p in s['sheets']]
        lines.append('')
    lines += ['## 定义', ''] + [f'- {k}：{v}' for k, v in s['definitions'].items()]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--predictions', type=Path, required=True, help='predictions.json.gz of the run on --split')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new output directory (must not exist)')
    parser.add_argument('--reference', default='trainval', help='split the run was trained on')
    parser.add_argument('--split', default='test')
    parser.add_argument('--no-sheets', action='store_true')
    args = parser.parse_args()
    analyze(args.predictions, args.data_root, args.output, args.reference, args.split, make_sheets=not args.no_sheets)


if __name__ == '__main__':
    main()
