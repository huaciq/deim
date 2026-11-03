#!/usr/bin/env python3
"""CPU check of the label-free structure priors proposed in the mechanism survey (section 7.5).

1. Mirror-symmetry axis pseudo-labels (S1): for each GT, the axis through the box centre
   about which the log-amplitude crop is most self-similar. Reports peak sharpness, how well
   the axis is separated from the perpendicular one (fuselage vs wing line), stability under
   box jitter, agreement with the second-moment principal axis, and contact sheets for a
   visual check.
2. Radar look direction (S2): median clutter in four strips just outside each box. The
   shadow side is darker; a consistent darker side over the dataset (or per image) tells
   whether horizontal flips reverse the illumination. Random background boxes of the same
   size are the no-target control.
3. Speckle looks at the working resolution of protocol B (1500 canvas resized to 1024),
   needed to parameterise speckle resampling.

Reads original COCO annotations and images only; writes OUTPUT/{report.md,summary.json,
targets.csv,sheets/*.png}. No inference, no change to annotations.
"""

import argparse
from collections import defaultdict
import csv
from multiprocessing import Pool
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.analyze_sar_characteristics import ELONGATED, clutter_mask, enl, grey, grown, orientation, side
from tools.sar_analysis_core import load_json, save_json

SYM_SIDE = 48          # symmetry window is resampled to SYM_SIDE x SYM_SIDE
SYM_WINDOW = 1.2       # window side as a fraction of max(w, h), centred on the box
SYM_STEP = 2           # degrees between candidate axes
SYM_OFFSETS = (-3, -1.5, 0, 1.5, 3)  # perpendicular shifts of the mirror line from the centroid, window pixels
JITTER = .1            # box centre shift / size change, fraction of max(w, h)
STRIP = .5             # look-direction strip depth, fraction of max(w, h)
CANVAS, WORK = 1500, 1024
ANGLES = np.arange(0, 180, SYM_STEP)


def bilinear(values, x, y):
    """Sample values at float coordinates; outside the array counts as NaN."""
    h, w = values.shape
    inside = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    x, y = np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)
    x0, y0 = np.minimum(x.astype(int), w - 2), np.minimum(y.astype(int), h - 2)
    fx, fy = x - x0, y - y0
    out = (values[y0, x0] * (1 - fx) * (1 - fy) + values[y0, x0 + 1] * fx * (1 - fy)
           + values[y0 + 1, x0] * (1 - fx) * fy + values[y0 + 1, x0 + 1] * fx * fy)
    return np.where(inside, out, np.nan)


def symmetry_window(log_values, cx, cy, size):
    """SYM_SIDE x SYM_SIDE window of side `size` centred at (cx, cy); NaN outside the image."""
    offsets = (np.arange(SYM_SIDE) + .5) / SYM_SIDE * size - size / 2
    gx, gy = np.meshgrid(cx + offsets, cy + offsets)
    return bilinear(log_values, gx, gy)


def symmetry_curve(window):
    """Best normalized correlation between the window and its mirror about a line at each angle in ANGLES.

    The box centre is generally off the fuselage (swept wings, rotation), so the mirror lines pass
    near the brightness centroid, which lies on the axis of a mirror-symmetric target, and are
    shifted perpendicular to themselves by SYM_OFFSETS window pixels; the best shift is kept.
    """
    c = (SYM_SIDE - 1) / 2
    gy, gx = np.mgrid[:SYM_SIDE, :SYM_SIDE] - c
    disk = gx ** 2 + gy ** 2 <= c ** 2
    filled = np.where(np.isnan(window), np.nanmean(window), window)
    weights = np.clip(filled - np.median(filled[disk]), 0, None) * disk
    qx, qy = ((weights * gx).sum() / weights.sum(), (weights * gy).sum() / weights.sum()) if weights.sum() > 0 else (0., 0.)
    scores = []
    for angle in np.radians(ANGLES):
        cos2, sin2 = np.cos(2 * angle), np.sin(2 * angle)
        nx, ny = -np.sin(angle), np.cos(angle)
        best = -1.
        for d in SYM_OFFSETS:
            px, py = qx + d * nx, qy + d * ny
            # reflection across the line through (px, py) at `angle` (image x to the right, y down)
            dx, dy = gx - px, gy - py
            mirrored = bilinear(filled, cos2 * dx + sin2 * dy + px + c, sin2 * dx - cos2 * dy + py + c)
            keep = disk & ~np.isnan(mirrored) & ~np.isnan(window)
            a, b = window[keep] - window[keep].mean(), mirrored[keep] - mirrored[keep].mean()
            denom = np.sqrt((a * a).sum() * (b * b).sum())
            best = max(best, float((a * b).sum() / denom) if denom > 0 else 0.)
        scores.append(best)
    return np.array(scores)


def axis_from_curve(curve):
    best = int(np.argmax(curve))
    perpendicular = curve[(best + len(ANGLES) // 2) % len(ANGLES)]
    return float(ANGLES[best]), float(curve[best]), float(curve[best] - np.median(curve)), float(curve[best] - perpendicular)


def axial_difference(a, b):
    d = abs(a - b) % 180
    return min(d, 180 - d)


def smooth(values):
    """3x3 box filter on log amplitude: weak speckle suppression before the symmetry search."""
    padded = np.pad(np.log1p(values), 1, mode='edge')
    return sum(padded[i:i + values.shape[0], j:j + values.shape[1]] for i in range(3) for j in range(3)) / 9


def strips(values, box, others):
    """Median of the four outside strips (left, right, top, bottom); pixels inside any GT box are excluded."""
    height, width = values.shape
    depth = max(int(round(side(box) * STRIP)), 2)
    x0, y0 = int(box[0]), int(box[1])
    x1, y1 = int(np.ceil(box[0] + box[2])), int(np.ceil(box[1] + box[3]))
    regions = dict(left=(x0 - depth, y0, x0, y1), right=(x1, y0, x1 + depth, y1),
                   top=(x0, y0 - depth, x1, y0), bottom=(x0, y1, x1, y1 + depth))
    out = {}
    for name, (a, b, c, d) in regions.items():
        a, b, c, d = max(a, 0), max(b, 0), min(c, width), min(d, height)
        if c - a < 2 or d - b < 2:
            out[name] = None
            continue
        keep = others[b:d, a:c]
        out[name] = float(np.median(values[b:d, a:c][keep])) if keep.sum() >= 8 else None
    return out


def log_ratio(a, b):
    return None if a is None or b is None else float(np.log((a + 1) / (b + 1)))


def random_boxes(rng, mask, boxes, count):
    """Background boxes with the sizes of the GT boxes, fully inside clutter (no-target control)."""
    height, width = mask.shape
    out = []
    for box in boxes[:count]:
        w, h = int(np.ceil(box[2])), int(np.ceil(box[3]))
        for _ in range(20):
            if width - w < 2 or height - h < 2:
                break
            x, y = int(rng.integers(0, width - w)), int(rng.integers(0, height - h))
            if mask[y:y + h, x:x + w].all():
                out.append([x, y, w, h])
                break
    return out


def working_enl(values, mask):
    """ENL after the protocol-B geometry: pad to CANVAS (top-left) and resize the canvas to WORK."""
    scale = WORK / CANVAS
    size = (max(int(round(values.shape[1] * scale)), 1), max(int(round(values.shape[0] * scale)), 1))
    resized = np.asarray(Image.fromarray(values).resize(size, Image.BILINEAR), np.float32)
    small = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 0
    return enl(resized, small)


def measure_image(job):
    """Rows, image row and symmetry windows of one image (worker entry point)."""
    split, data_root, im, anns, names, seed = job
    rng = np.random.default_rng([seed, im['id']])
    rows, crops = [], []
    values, _ = grey(data_root / im['file_name'])
    boxes = [a['bbox'] for a in anns]
    mask = clutter_mask(values.shape, boxes)
    outside = np.ones(values.shape, bool)  # raw GT boxes only: the shadow lies right next to the box
    for box in boxes:
        x0, y0, x1, y1 = grown(box, 0, values.shape[1], values.shape[0])
        outside[y0:y1, x0:x1] = False
    log_values = smooth(values)
    image_row = dict(split=split, image_id=im['id'], gt=len(anns), enl_native=enl(values, mask),
                     enl_work=working_enl(values, mask))
    for kind, kind_boxes in (('gt', boxes), ('control', random_boxes(rng, mask, boxes, len(boxes)))):
        for k, box in enumerate(kind_boxes):
            s = strips(values, box, outside)
            row = dict(split=split, image_id=im['id'], kind=kind, bbox=' '.join(f'{v:.1f}' for v in box),
                       size=side(box), lr=log_ratio(s['left'], s['right']), tb=log_ratio(s['top'], s['bottom']))
            if kind == 'gt':
                a = anns[k]
                cx, cy, size = box[0] + box[2] / 2, box[1] + box[3] / 2, side(box) * SYM_WINDOW
                window = symmetry_window(log_values, cx, cy, size)
                axis, peak, sharp, margin = axis_from_curve(symmetry_curve(window))
                jitter = []
                for dx, dy, ds in ((JITTER, 0, 0), (0, -JITTER, 0), (0, 0, JITTER), (-JITTER, JITTER, -JITTER)):
                    j = axis_from_curve(symmetry_curve(symmetry_window(
                        log_values, cx + dx * side(box), cy + dy * side(box), size * (1 + ds))))[0]
                    jitter.append(axial_difference(axis, j))
                x0, y0, x1, y1 = grown(box, 0, values.shape[1], values.shape[0])
                principal, ratio = orientation(values[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)])
                row.update(category=names[a['category_id']], ann_id=a['id'], sym_axis=axis, sym_peak=peak,
                           sym_sharpness=sharp, sym_margin=margin, jitter_max=max(jitter), principal_axis=principal,
                           axis_ratio=ratio, sym_vs_principal=None if principal is None else axial_difference(axis, principal))
                crops.append((row, window.astype(np.float32)))
            rows.append(row)
    return rows, image_row, crops


def measure(dataset, data_root, split, seed, workers):
    names = {c['id']: c['name'] for c in dataset['categories']}
    by_image = defaultdict(list)
    for a in dataset['annotations']:
        by_image[a['image_id']].append(a)
    jobs = [(split, data_root, im, by_image[im['id']], names, seed) for im in dataset['images']]
    if workers > 1:
        with Pool(workers) as pool:
            results = pool.map(measure_image, jobs, chunksize=8)
    else:
        results = [measure_image(job) for job in jobs]
    rows = [r for result in results for r in result[0]]
    images = [result[1] for result in results]
    crops = [c for result in results for c in result[2]]
    print(f'measured {split}: {len(images)} images, {sum(r["kind"] == "gt" for r in rows)} targets', flush=True)
    return rows, images, crops


def quantiles(values, qs=(.1, .5, .9)):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return [float(np.quantile(values, q)) for q in qs] if values else None


def share(values, predicate):
    values = [v for v in values if v is not None]
    return (sum(predicate(v) for v in values) / len(values)) if values else None


def symmetry_summary(rows, threshold):
    gt = [r for r in rows if r['kind'] == 'gt']
    confident = [r for r in gt if r['sym_margin'] >= threshold]
    elongated = [r for r in gt if r['principal_axis'] is not None and r['axis_ratio'] >= ELONGATED]
    return dict(
        targets=len(gt), margin=quantiles([r['sym_margin'] for r in gt]), peak=quantiles([r['sym_peak'] for r in gt]),
        confident_share=len(confident) / len(gt) if gt else None,
        jitter_within_10=share([r['jitter_max'] for r in gt], lambda v: v <= 10),
        jitter_within_10_confident=share([r['jitter_max'] for r in confident], lambda v: v <= 10),
        principal_elongated=len(elongated),
        # 0 deg: the symmetry axis is the long bright axis (fuselage); 90 deg: it is perpendicular (wing line)
        vs_principal_within_20=share([r['sym_vs_principal'] for r in elongated], lambda v: v <= 20),
        vs_principal_beyond_70=share([r['sym_vs_principal'] for r in elongated], lambda v: v >= 70))


def look_summary(rows, images):
    out = {}
    for kind in ('gt', 'control'):
        part = [r for r in rows if r['kind'] == kind]
        out[kind] = dict(boxes=len(part), lr=quantiles([r['lr'] for r in part]), tb=quantiles([r['tb'] for r in part]),
                         left_darker=share([r['lr'] for r in part], lambda v: v < 0),
                         top_darker=share([r['tb'] for r in part], lambda v: v < 0))
    per_image = defaultdict(lambda: dict(lr=[], tb=[]))
    for r in rows:
        if r['kind'] == 'gt':
            for key in ('lr', 'tb'):
                if r[key] is not None:
                    per_image[r['image_id']][key].append(r[key])
    for key in ('lr', 'tb'):
        medians = [float(np.median(v[key])) for v in per_image.values() if len(v[key]) >= 3]
        majority = np.sign(np.median(medians)) if medians else 0
        out[f'images_{key}'] = dict(images=len(medians), median=quantiles(medians),
                                    agree_with_majority=share(medians, lambda v: np.sign(v) == majority))
    out['enl_native'] = quantiles([i['enl_native'] for i in images])
    out['enl_work'] = quantiles([i['enl_work'] for i in images])
    return out


def contact_sheet(path, crops, rng, count=64, cell=96):
    pick = [crops[i] for i in rng.choice(len(crops), size=min(count, len(crops)), replace=False)]
    columns = 8
    sheet = Image.new('RGB', (columns * cell, -(-len(pick) // columns) * (cell + 14)), 'black')
    draw = ImageDraw.Draw(sheet)
    for k, (row, window) in enumerate(pick):
        w = np.nan_to_num(window, nan=np.nanmin(window))
        w = (w - w.min()) / max(w.max() - w.min(), 1e-6) * 255
        tile = Image.fromarray(w.astype(np.uint8)).resize((cell, cell), Image.NEAREST).convert('RGB')
        x, y = k % columns * cell, k // columns * (cell + 14)
        sheet.paste(tile, (x, y))
        c = cell / 2
        for angle, colour in ((row['sym_axis'], (255, 40, 40)), (row['principal_axis'], (60, 140, 255))):
            if angle is None:
                continue
            dx, dy = np.cos(np.radians(angle)) * c, np.sin(np.radians(angle)) * c
            draw.line([(x + c - dx, y + c - dy), (x + c + dx, y + c + dy)], fill=colour, width=1)
        draw.text((x + 2, y + cell), f"{row['category'][:9]} m{row['sym_margin']:.2f}", fill=(255, 255, 255))
    sheet.save(path)


def fmt(values, digits=2):
    if values is None:
        return '—'
    if isinstance(values, list):
        return ' / '.join(f'{v:.{digits}f}' for v in values)
    return f'{values:.{digits}f}'


def write_report(output, summary):
    lines = ['# 结构先验 CPU 检查', '', '数值为 p10 / p50 / p90。', '',
             '## 1. 对称轴伪标签', '',
             f"margin = 峰值相关 − 中位数；高置信阈值 {summary['threshold']}。与主轴比较只用轴比 ≥ {ELONGATED} 的目标："
             '≤20° 表示对称轴就是亮像素长轴（机身），≥70° 表示对称轴与长轴垂直（长轴其实是机翼）。', '',
             '| 划分 | GT | 峰值相关 | margin | 高置信比例 | 抖动后变化 ≤10°（全部 / 高置信） | 主轴可判定 | 与主轴 ≤20° | 与主轴 ≥70° |',
             '|---|---:|---|---|---:|---|---:|---:|---:|']
    for split, s in summary['symmetry'].items():
        lines.append(f"| {split} | {s['targets']} | {fmt(s['peak'])} | {fmt(s['margin'])} | {fmt(s['confident_share'])} | "
                     f"{fmt(s['jitter_within_10'])} / {fmt(s['jitter_within_10_confident'])} | {s['principal_elongated']} | "
                     f"{fmt(s['vs_principal_within_20'])} | {fmt(s['vs_principal_beyond_70'])} |")
    lines += ['', '各类高置信比例：', '', '| 划分 | 类别 | GT | 高置信比例 | margin p50 |', '|---|---|---:|---:|---:|']
    lines += [f"| {r['split']} | {r['category']} | {r['gt']} | {fmt(r['confident_share'])} | {fmt(r['margin_p50'])} |"
              for r in summary['symmetry_by_class']]
    lines += ['', f"目视检查：`sheets/` 中红线为对称轴，蓝线为亮像素主轴（不可判定时不画）；窗口为框的 {SYM_WINDOW} 倍。", '',
              '## 2. 照射方向（框外条带的中位数）', '',
              'lr = ln((左+1)/(右+1))，tb = ln((上+1)/(下+1))；负值表示左 / 上更暗。control 为同尺寸的随机背景框。', '',
              '| 划分 | 类型 | 框数 | lr | 左更暗比例 | tb | 上更暗比例 |', '|---|---|---:|---|---:|---|---:|']
    for split, s in summary['look'].items():
        for kind in ('gt', 'control'):
            k = s[kind]
            lines.append(f"| {split} | {kind} | {k['boxes']} | {fmt(k['lr'], 3)} | {fmt(k['left_darker'])} | "
                         f"{fmt(k['tb'], 3)} | {fmt(k['top_darker'])} |")
    lines += ['', '按图（≥3 个目标的图取中位数）：', '', '| 划分 | 方向 | 图数 | 图中位数 | 与全局多数同号比例 |', '|---|---|---:|---|---:|']
    for split, s in summary['look'].items():
        for key in ('lr', 'tb'):
            k = s[f'images_{key}']
            lines.append(f"| {split} | {key} | {k['images']} | {fmt(k['median'], 3)} | {fmt(k['agree_with_majority'])} |")
    lines += ['', '## 3. 斑点等效视数', '', f'工作分辨率：补边到 {CANVAS} 后缩放到 {WORK}（B 协议），双线性。', '',
              '| 划分 | 原图 ENL | 工作分辨率 ENL |', '|---|---|---|']
    lines += [f"| {split} | {fmt(s['enl_native'])} | {fmt(s['enl_work'])} |" for split, s in summary['look'].items()]
    lines += ['', '## 判读', '',
              '- S1 继续的条件：高置信比例过半、高置信目标抖动后 ≤10° 的比例高、目视对称轴落在机身上；'
              '"与主轴 ≥70°" 占比高说明二阶矩主轴常把机翼当长轴，正是对称轴要解决的问题。',
              '- S2 继续的条件：GT 的某一方向（lr 或 tb）明显偏离 0 且同号比例高，而 control 接近 0、约一半；'
              '按图同号比例低则说明照射方向按图不同，需要先按图规范化。',
              '- 工作分辨率 ENL 用作斑点重采样的 L。']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_csv(path, rows):
    keys = sorted({k for r in rows for k in r})
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def analyze(data_root, output, splits=('train', 'val', 'test'), threshold=.15, seed=0, sheets=True, workers=1):
    data_root, output = Path(data_root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    summary = dict(threshold=threshold, splits=list(splits), symmetry={}, symmetry_by_class=[], look={})
    all_rows = []
    for split in splits:
        rows, images, crops = measure(load_json(data_root / f'COCO/{split}.json'), data_root, split, seed, workers)
        summary['symmetry'][split] = symmetry_summary(rows, threshold)
        summary['look'][split] = look_summary(rows, images)
        for c in sorted({r['category'] for r in rows if r['kind'] == 'gt'}):
            part = [r for r in rows if r['kind'] == 'gt' and r['category'] == c]
            summary['symmetry_by_class'].append(dict(split=split, category=c, gt=len(part),
                                                     confident_share=share([r['sym_margin'] for r in part], lambda v: v >= threshold),
                                                     margin_p50=float(np.median([r['sym_margin'] for r in part]))))
        if sheets and crops:
            (output / 'sheets').mkdir(exist_ok=True)
            contact_sheet(output / f'sheets/{split}_random.png', crops, rng)
            for name, keep in (('low_margin', lambda r: r['sym_margin'] < threshold),
                               ('high_margin', lambda r: r['sym_margin'] >= threshold)):
                part = [c for c in crops if keep(c[0])]
                if part:
                    contact_sheet(output / f'sheets/{split}_{name}.png', part, rng)
        all_rows += rows
    save_json(output / 'summary.json', summary)
    write_csv(output / 'targets.csv', all_rows)
    write_report(output, summary)
    print(output / 'report.md', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new output directory (must not exist)')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--threshold', type=float, default=.15, help='symmetry margin counted as confident')
    parser.add_argument('--no-sheets', action='store_true')
    parser.add_argument('--workers', type=int, default=8, help='CPU processes over images')
    args = parser.parse_args()
    analyze(args.data_root, args.output, args.splits, args.threshold, sheets=not args.no_sheets, workers=args.workers)


if __name__ == '__main__':
    main()
