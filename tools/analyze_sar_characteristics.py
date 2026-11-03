#!/usr/bin/env python3
"""CPU check of SAR imaging characteristics behind the narrow-body type confusion.

Reads the original COCO annotations and images of SAR-AIRcraft-1.0 (optionally the
baseline/REM predictions of finished compare_rem_102e.py diagnoses) and reports:

1. co-occurrence size ratio: in images holding both classes, does the box size
   ratio follow the physical size (Boeing787/A330 are wide-bodies, ~1.6x 737/A320)?
2. per-image radiometry: background level, high percentile, saturated/zero pixel
   fraction and speckle equivalent number of looks per split and source size;
3. per-target target-to-clutter ratio (TCR) per split and class, and for the
   diagnosed splits correct versus wrong type;
4. principal axis orientation of the strong scatterers versus the type accuracy;
5. near-duplicate targets (overlapping tiles) within a split and against train.

Writes OUTPUT/{report.md,summary.json,targets.csv,images.csv}; no inference.
Pixel values are taken as stored (8-bit amplitude products); dB values use
20*log10 and therefore assume amplitude, not intensity.
"""

import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.analyze_confusion_cues import records
from tools.compare_rem_102e import ROLES, split_annotation
from tools.sar_analysis_core import load_json, save_json

SIZE_PAIRS = (('Boeing787', 'Boeing737'), ('A330', 'A320/321'), ('A330', 'Boeing737'),
              ('Boeing787', 'A330'), ('A220', 'ARJ21'), ('Boeing737', 'A320/321'))
BOX_MARGIN = .25     # GT boxes grown by this fraction of max(w,h) before masking clutter
RING = .5            # clutter ring width around a target, fraction of max(w,h)
ENL_BLOCK = 16       # pixels; homogeneous-block size for the speckle look estimate
TOP_FRACTION = .1    # brightest share of box pixels taken as the target response
ELONGATED = 1.5      # axis ratio above which the orientation is considered defined
DUP_SIDE = 24        # crops are resized to DUP_SIDE x DUP_SIDE for normalized correlation
DUP_THRESHOLDS = (.8, .9, .95)
DUP_SIZE_TOLERANCE = .1


def side(box):
    return max(box[2], box[3])


def grey(path):
    image = Image.open(path)
    values = np.asarray(image, dtype=np.float32)
    return (values.mean(axis=2) if values.ndim == 3 else values), image.mode


def grown(box, fraction, width, height):
    pad = side(box) * fraction
    return (max(int(box[0] - pad), 0), max(int(box[1] - pad), 0),
            min(int(np.ceil(box[0] + box[2] + pad)), width), min(int(np.ceil(box[1] + box[3] + pad)), height))


def clutter_mask(shape, boxes):
    mask = np.ones(shape, bool)
    for box in boxes:
        x0, y0, x1, y1 = grown(box, BOX_MARGIN, shape[1], shape[0])
        mask[y0:y1, x0:x1] = False
    return mask


def enl(values, mask):
    """Median (mean/std)^2 over fully clutter, non-constant ENL_BLOCK blocks."""
    h, w = (s - s % ENL_BLOCK for s in values.shape)
    if not h or not w:
        return None
    blocks = values[:h, :w].reshape(h // ENL_BLOCK, ENL_BLOCK, w // ENL_BLOCK, ENL_BLOCK).swapaxes(1, 2)
    keep = mask[:h, :w].reshape(h // ENL_BLOCK, ENL_BLOCK, w // ENL_BLOCK, ENL_BLOCK).swapaxes(1, 2).all(axis=(2, 3))
    blocks = blocks[keep].reshape(-1, ENL_BLOCK * ENL_BLOCK)
    mean, std = blocks.mean(axis=1), blocks.std(axis=1)
    good = (std > 0) & (mean > 0)
    return float(np.median((mean[good] / std[good]) ** 2)) if good.any() else None


def orientation(patch):
    """Principal axis (degrees from image x, [0, 180)) and axis ratio of the strong scatterers."""
    threshold = np.quantile(patch, 1 - 2 * TOP_FRACTION)
    weights = np.clip(patch - threshold, 0, None)
    if weights.sum() <= 0:
        return None, None
    y, x = np.mgrid[:patch.shape[0], :patch.shape[1]]
    total = weights.sum()
    cx, cy = (weights * x).sum() / total, (weights * y).sum() / total
    mu20, mu02 = (weights * (x - cx) ** 2).sum() / total, (weights * (y - cy) ** 2).sum() / total
    mu11 = (weights * (x - cx) * (y - cy)).sum() / total
    spread = np.sqrt(((mu20 - mu02) / 2) ** 2 + mu11 ** 2)
    major, minor = (mu20 + mu02) / 2 + spread, (mu20 + mu02) / 2 - spread
    angle = float(np.degrees(0.5 * np.arctan2(2 * mu11, mu20 - mu02)) % 180)
    return angle, float(np.sqrt(major / minor)) if minor > 1e-9 else float('inf')


def crop_vector(values, box):
    x0, y0 = int(box[0]), int(box[1])
    x1, y1 = max(int(np.ceil(box[0] + box[2])), x0 + 1), max(int(np.ceil(box[1] + box[3])), y0 + 1)
    patch = Image.fromarray(values[y0:y1, x0:x1]).resize((DUP_SIDE, DUP_SIDE), Image.BILINEAR)
    vector = np.array(patch, np.float32).ravel()
    vector -= vector.mean()
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def measure_split(dataset, data_root, split):
    names = {c['id']: c['name'] for c in dataset['categories']}
    by_image = defaultdict(list)
    for a in dataset['annotations']:
        by_image[a['image_id']].append(a)
    image_rows, target_rows, vectors = [], [], []
    for im in dataset['images']:
        values, mode = grey(data_root / im['file_name'])
        annotations = by_image[im['id']]
        mask = clutter_mask(values.shape, [a['bbox'] for a in annotations])
        clutter = values[mask] if mask.any() else values.ravel()
        peak = float(values.max())
        image_rows.append(dict(split=split, image_id=im['id'], file_name=im['file_name'], mode=mode,
                               source_size=f"{im['width']}x{im['height']}", gt=len(annotations),
                               background=float(np.median(clutter)), p99=float(np.percentile(values, 99)),
                               saturated=float((values >= (255 if mode in ('L', 'RGB') else peak)).mean()),
                               zero=float((values == 0).mean()), enl=enl(values, mask)))
        for a in annotations:
            x0, y0, x1, y1 = grown(a['bbox'], 0, values.shape[1], values.shape[0])
            patch = values[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)]
            rx0, ry0, rx1, ry1 = grown(a['bbox'], RING, values.shape[1], values.shape[0])
            ring = values[ry0:ry1, rx0:rx1][mask[ry0:ry1, rx0:rx1]]
            ring_level = float(np.median(ring)) if ring.size else image_rows[-1]['background']
            flat = np.sort(patch.ravel())
            top = float(flat[-max(1, int(len(flat) * TOP_FRACTION)):].mean())
            angle, ratio = orientation(patch)
            target_rows.append(dict(split=split, image_id=im['id'], bbox=a['bbox'], category=names[a['category_id']],
                                    size=side(a['bbox']), source_size=image_rows[-1]['source_size'],
                                    background=image_rows[-1]['background'], ring=ring_level, top=top,
                                    tcr_db=float(20 * np.log10((top + 1) / (ring_level + 1))),
                                    angle=angle, axis_ratio=ratio))
            vectors.append(crop_vector(values, a['bbox']))
    return image_rows, target_rows, np.array(vectors, np.float32).reshape(-1, DUP_SIDE * DUP_SIDE)


def quantiles(values, qs=(.1, .5, .9)):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return [float(q) for q in np.quantile(values, qs)] if values else [None] * len(qs)


def size_ratios(rows, split):
    by_image = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_image[r['image_id']][r['category']].append(r['size'])
    table = []
    for a, b in SIZE_PAIRS:
        ratios = [np.median(c[a]) / np.median(c[b]) for c in by_image.values() if c[a] and c[b]]
        table.append(dict(split=split, a=a, b=b, images=len(ratios), ratio=quantiles(ratios, (.25, .5, .75))))
    return table


def similar(left, right, sizes_left, sizes_right, same=False, chunk=2048):
    """Pairs (i, j, correlation) above the lowest threshold with box sizes within tolerance."""
    pairs = []
    for start in range(0, len(left), chunk):
        scores = left[start:start + chunk] @ right.T
        ratio = sizes_left[start:start + chunk, None] / sizes_right[None, :]
        hit = (scores >= min(DUP_THRESHOLDS)) & (np.abs(ratio - 1) <= DUP_SIZE_TOLERANCE)
        for i, j in zip(*np.nonzero(hit)):
            if not same or start + i < j:
                pairs.append((start + i, j, float(scores[i, j])))
    return pairs


def groups(count, pairs, threshold, image_of):
    parent = list(range(count))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j, score in pairs:
        if score >= threshold and image_of[i] != image_of[j]:
            parent[find(i)] = find(j)
    return [find(i) for i in range(count)]


def duplicate_tables(targets, vectors, split):
    rows, vec = targets[split], vectors[split]
    sizes = np.array([r['size'] for r in rows], float)
    image_of = [r['image_id'] for r in rows]
    within = similar(vec, vec, sizes, sizes, same=True)
    table = []
    for threshold in DUP_THRESHOLDS:
        root = groups(len(rows), within, threshold, image_of)
        members = Counter(root)
        per_class = defaultdict(lambda: [0, set(), 0])
        for r, g in zip(rows, root):
            per_class[r['category']][0] += 1
            per_class[r['category']][1].add(g)
            per_class[r['category']][2] += members[g] > 1
        table.append(dict(split=split, threshold=threshold, gt=len(rows), unique=len(members),
                          in_duplicate_groups=sum(members[g] > 1 for g in root),
                          per_class={k: dict(gt=v[0], unique=len(v[1]), duplicated=v[2]) for k, v in per_class.items()}))
        for r, g in zip(rows, root):
            r[f'group_{threshold}'] = int(g)
    leak = []
    if split != 'train' and 'train' in vectors:
        train_sizes = np.array([r['size'] for r in targets['train']], float)
        cross = similar(vec, vectors['train'], sizes, train_sizes)
        best = defaultdict(float)
        for i, _, score in cross:
            best[i] = max(best[i], score)
        for threshold in DUP_THRESHOLDS:
            leak.append(dict(split=split, threshold=threshold, gt=len(rows),
                             matched_in_train=sum(score >= threshold for score in best.values())))
        for i, r in enumerate(rows):
            r['train_match'] = best.get(i)
    return table, leak


def attach_predictions(targets, diag):
    """Mark each GT of a diagnosed split with its located/predicted type per role."""
    annotation, split = split_annotation(Path(diag).resolve())
    dataset = load_json(annotation)
    index = {(r['image_id'], tuple(r['bbox'])): r for r in targets[split]}
    for role in ROLES:
        for row in records(dataset, load_json(Path(diag) / role / 'predictions.json.gz')):
            target = index[(row['image_id'], tuple(row['bbox']))]
            target[f'{role}_located'] = row['located']
            target[f'{role}_predicted'] = row['predicted']
    return split


def accuracy(rows, role):
    located = [r for r in rows if r.get(f'{role}_located')]
    return dict(gt=len(rows), located=len(located),
                correct_given_located=(sum(r[f'{role}_predicted'] == r['category'] for r in located) / len(located)
                                       if located else None))


def binned_accuracy(rows, classes, key, edges, labels, role='baseline'):
    table = []
    for name in classes:
        for low, high, label in zip(edges[:-1], edges[1:], labels):
            subset = [r for r in rows if r['category'] == name and r[key] is not None and low <= r[key] < high]
            table.append(dict(category=name, bin=label, **accuracy(subset, role)))
    return table


def fmt(value, digits=2):
    return '—' if value is None else f'{value:.{digits}f}'


def analyze(data_root, output, diags=(), splits=('train', 'val', 'test')):
    data_root, output = Path(data_root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    images, targets, vectors = {}, {}, {}
    for split in splits:
        dataset = load_json(data_root / f'COCO/{split}.json')
        images[split], targets[split], vectors[split] = measure_split(dataset, data_root, split)
        print(f'measured {split}: {len(images[split])} images, {len(targets[split])} targets', flush=True)
    diagnosed = [attach_predictions(targets, d) for d in diags]
    classes = sorted({r['category'] for rows in targets.values() for r in rows})

    summary = dict(splits=list(splits), diagnosed=diagnosed, classes=classes)
    summary['size_ratio'] = [row for s in splits for row in size_ratios(targets[s], s)]
    summary['radiometry'] = []
    for split in splits:
        for key in ['all'] + sorted({r['source_size'] for r in images[split]}, key=lambda k: int(k.split('x')[0])):
            subset = [r for r in images[split] if key in ('all', r['source_size'])]
            summary['radiometry'].append(dict(split=split, source_size=key, images=len(subset),
                                              modes=dict(Counter(r['mode'] for r in subset)),
                                              **{k: quantiles([r[k] for r in subset])
                                                 for k in ('background', 'p99', 'saturated', 'zero', 'enl')}))
    summary['tcr'] = [dict(split=s, category=c, gt=len(v), tcr_db=quantiles([r['tcr_db'] for r in v]),
                           ring=quantiles([r['ring'] for r in v]), top=quantiles([r['top'] for r in v]))
                      for s in splits for c in classes for v in [[r for r in targets[s] if r['category'] == c]]]
    # Fixed bins from train (or the first split) so every split is cut at the same values.
    reference = targets['train'] if 'train' in targets else targets[splits[0]]
    background_edges = [-np.inf] + quantiles([r['background'] for r in reference], (1 / 3, 2 / 3)) + [np.inf]
    tcr_edges = [-np.inf] + quantiles([r['tcr_db'] for r in reference], (1 / 3, 2 / 3)) + [np.inf]
    angle_edges, angle_labels = [0, 45, 90, 135, 180], ['0°±22.5', '45°±22.5', '90°±22.5', '135°±22.5']
    summary['bins'] = dict(background=background_edges[1:3], tcr_db=tcr_edges[1:3])
    summary['diagnosed_tables'] = {}
    for split in diagnosed:
        rows = targets[split]
        for r in rows:
            # Shifted by 22.5 so each 45° bin is centred on 0/45/90/135 and 180 folds onto 0.
            r['axis_angle'] = ((r['angle'] + 22.5) % 180 if r['axis_ratio'] is not None
                               and r['axis_ratio'] >= ELONGATED else None)
        tables = dict(
            tcr_by_outcome=[dict(category=c, role=role, outcome=outcome, gt=len(v),
                                 tcr_db=quantiles([r['tcr_db'] for r in v]),
                                 background=quantiles([r['background'] for r in v]))
                            for c in classes for role in ROLES
                            for outcome, v in (('correct', [r for r in rows if r['category'] == c and r.get(f'{role}_located')
                                                            and r[f'{role}_predicted'] == c]),
                                               ('wrong', [r for r in rows if r['category'] == c and r.get(f'{role}_located')
                                                          and r[f'{role}_predicted'] != c]),
                                               ('missed', [r for r in rows if r['category'] == c
                                                           and not r.get(f'{role}_located')]))],
            by_background=binned_accuracy(rows, classes, 'background', background_edges, ['暗', '中', '亮']),
            by_tcr=binned_accuracy(rows, classes, 'tcr_db', tcr_edges, ['低', '中', '高']),
            by_angle=binned_accuracy(rows, classes, 'axis_angle', angle_edges, angle_labels),
            elongated_fraction={c: (sum(r['axis_angle'] is not None for r in rows if r['category'] == c)
                                    / max(1, sum(r['category'] == c for r in rows))) for c in classes})
        summary['diagnosed_tables'][split] = tables
    summary['duplicates'], summary['train_leak'] = [], []
    for split in splits:
        table, leak = duplicate_tables(targets, vectors, split)
        summary['duplicates'] += table
        summary['train_leak'] += leak
    for split in diagnosed:  # how many distinct aircraft the baseline type errors are
        rows = targets[split]
        summary.setdefault('unique_errors', [])
        for c in classes:
            wrong = [r for r in rows if r['category'] == c and r.get('baseline_located') and r['baseline_predicted'] != c]
            summary['unique_errors'].append(dict(split=split, category=c, wrong=len(wrong),
                                                 unique_at_0_9=len({r['group_0.9'] for r in wrong}),
                                                 matched_in_train_0_9=sum((r.get('train_match') or 0) >= .9
                                                                          for r in wrong)))
    summary['definitions'] = dict(
        size='max(w,h) of the GT box in source pixels',
        background=f'median of pixels outside all GT boxes grown by {BOX_MARGIN} x max(w,h)',
        tcr_db=f'20*log10((mean of brightest {TOP_FRACTION:.0%} box pixels + 1) / (median clutter ring + 1)); '
               f'ring = box grown by {RING} x max(w,h) minus all grown GT boxes',
        enl=f'median (mean/std)^2 over clutter-only {ENL_BLOCK}x{ENL_BLOCK} blocks (amplitude values as stored)',
        axis_angle=f'principal axis of box pixels above the {1 - 2 * TOP_FRACTION:.0%} quantile, degrees from image x; '
                   f'kept only when the axis ratio >= {ELONGATED} (cross-shaped aircraft are often ambiguous)',
        duplicates=f'normalized correlation of {DUP_SIDE}x{DUP_SIDE} box crops, box sizes within '
                   f'{DUP_SIZE_TOLERANCE:.0%}, different images; grouped transitively',
        located='baseline/REM prediction of any class, score>=0.3, IoU>=0.5 (compare_rem_102e.confusion rule)')
    save_json(output / 'summary.json', summary)
    write_csv(output / 'images.csv', [r for s in splits for r in images[s]])
    write_csv(output / 'targets.csv', [dict(r, bbox=' '.join(f'{v:g}' for v in r['bbox'])) for s in splits for r in targets[s]])
    write_report(output, summary)
    print(output / 'report.md', flush=True)
    return summary


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def q3(values, digits=1):
    return ' / '.join(fmt(v, digits) for v in values)


def write_report(output, s):
    lines = ['# SAR 成像特性检查', '', '数值为 p10 / p50 / p90（尺寸比为 p25 / p50 / p75）。dB 按幅度计算（20·log10）。', '',
             '## 1. 同图尺寸比（max(w,h)，同时含两类的图像）', '',
             '物理参照：787、A330 为宽体机，机长约为 737、A320 的 1.5–1.7 倍；A220 与 ARJ21 相近（约 1.1）。', '',
             '| 划分 | a / b | 图像数 | 尺寸比 |', '|---|---|---:|---|']
    lines += [f"| {r['split']} | {r['a']} / {r['b']} | {r['images']} | {q3(r['ratio'], 2)} |" for r in s['size_ratio']]
    lines += ['', '## 2. 每图辐射统计', '',
              '| 划分 | 源图 | 图 | 模式 | 背景 | p99 | 饱和比例 | 零值比例 | 等效视数 |', '|---|---|---:|---|---|---|---|---|---|']
    lines += [f"| {r['split']} | {r['source_size']} | {r['images']} | "
              f"{', '.join(f'{k}:{v}' for k, v in r['modes'].items())} | {q3(r['background'])} | {q3(r['p99'])} | "
              f"{q3(r['saturated'], 4)} | {q3(r['zero'], 4)} | {q3(r['enl'])} |" for r in s['radiometry']]
    lines += ['', '## 3. 目标/杂波对比度（TCR，dB）', '',
              '| 划分 | 类别 | GT | TCR | 杂波环 | 目标强响应 |', '|---|---|---:|---|---|---|']
    lines += [f"| {r['split']} | {r['category']} | {r['gt']} | {q3(r['tcr_db'])} | {q3(r['ring'])} | {q3(r['top'])} |"
              for r in s['tcr']]
    edges = s['bins']
    for split, t in s['diagnosed_tables'].items():
        lines += ['', f'## 4. {split}：判对 / 判错 / 漏检的辐射特征', '',
                  '| 类别 | 模型 | 结果 | GT | TCR | 背景 |', '|---|---|---|---:|---|---|']
        lines += [f"| {r['category']} | {r['role']} | {r['outcome']} | {r['gt']} | {q3(r['tcr_db'])} | {q3(r['background'])} |"
                  for r in t['tcr_by_outcome']]
        for key, title, unit in (('by_background', '按整图背景分组', f"train 三分位 {fmt(edges['background'][0], 1)} / {fmt(edges['background'][1], 1)}"),
                                 ('by_tcr', '按目标 TCR 分组', f"train 三分位 {fmt(edges['tcr_db'][0], 1)} / {fmt(edges['tcr_db'][1], 1)} dB"),
                                 ('by_angle', f'按强散射主轴方向分组（轴比 ≥ {ELONGATED}）', '相对图像 x 轴')):
            lines += ['', f'### {split}：{title}（baseline，{unit}）', '',
                      '| 类别 | 区间 | GT | 定位 | 定位后型号正确率 |', '|---|---|---:|---:|---:|']
            lines += [f"| {r['category']} | {r['bin']} | {r['gt']} | {r['located']} | {fmt(r['correct_given_located'], 3)} |"
                      for r in t[key]]
        lines += ['', '主轴方向可判定的比例：' + '，'.join(f'{k} {v:.2f}' for k, v in t['elongated_fraction'].items())]
    lines += ['', '## 5. 重叠切片 / 重复目标', '',
              '| 划分 | 相关阈值 | GT | 去重后 | 处于重复组的 GT |', '|---|---:|---:|---:|---:|']
    lines += [f"| {r['split']} | {r['threshold']} | {r['gt']} | {r['unique']} | {r['in_duplicate_groups']} |"
              for r in s['duplicates']]
    if s['train_leak']:
        lines += ['', '与 train 中目标高度相似的数量：', '', '| 划分 | 相关阈值 | GT | 在 train 中有匹配 |', '|---|---:|---:|---:|']
        lines += [f"| {r['split']} | {r['threshold']} | {r['gt']} | {r['matched_in_train']} |" for r in s['train_leak']]
    if s.get('unique_errors'):
        lines += ['', 'baseline 判错实例去重（阈值 0.9）：', '',
                  '| 划分 | 类别 | 判错 | 去重后 | 在 train 中有匹配 |', '|---|---|---:|---:|---:|']
        lines += [f"| {r['split']} | {r['category']} | {r['wrong']} | {r['unique_at_0_9']} | {r['matched_in_train_0_9']} |"
                  for r in s['unique_errors']]
    lines += ['', '## 定义', ''] + [f'- {k}：{v}' for k, v in s['definitions'].items()]
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data-root', type=Path, required=True, help='SAR-AIRcraft-1.0 root (COCO/{split}.json, images)')
    parser.add_argument('--output', type=Path, required=True, help='new output directory (must not exist)')
    parser.add_argument('--diag', type=Path, action='append', default=[],
                        help='finished compare_rem_102e.py output whose split is analysed with predictions')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    args = parser.parse_args()
    analyze(args.data_root, args.output, args.diag, tuple(args.splits))


if __name__ == '__main__':
    main()
