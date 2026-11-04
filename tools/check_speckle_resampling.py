#!/usr/bin/env python3
"""CPU check of SpeckleResample on real training images; fixes its noise_sigma before any GPU run.

For a random sample of images it compares stored and resampled values: speckle looks at native and
working resolution, clutter autocorrelation at lags 1-3, quantiles of value / local mean, and how
much of the original pattern survives (pixel correlation in clutter and in GT boxes; crop correlation
of GT boxes with the matcher of check_split_overlap.py). Uses only --split, never test.
Writes OUTPUT/{report.md,summary.json,sheet.png}; no training, no change to images or annotations.
"""

import argparse
from multiprocessing import Pool
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as NF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.data.transforms._transforms import SpeckleResample
from tools.analyze_sar_characteristics import clutter_mask, crop_vector, enl, grey, grown
from tools.check_structure_priors import working_enl
from tools.sar_analysis_core import load_json, save_json

QUANTILES = (.01, .1, .5, .9, .99)
LAGS = (1, 2, 3)
SHEET_TARGETS, SHEET_WINDOW = 8, 128


def local_mean(values, window=7):
    return NF.avg_pool2d(torch.from_numpy(values)[None, None], window, stride=1, padding=window // 2,
                         count_include_pad=False)[0, 0].numpy()


def autocorrelation(residual, mask, lag):
    """Horizontal and vertical autocorrelation of the clutter residual at one lag, averaged."""
    values = []
    for a, b, keep in ((residual[:, :-lag], residual[:, lag:], mask[:, :-lag] & mask[:, lag:]),
                       (residual[:-lag], residual[lag:], mask[:-lag] & mask[lag:])):
        if keep.sum() > 100:
            values.append(float(np.corrcoef(a[keep], b[keep])[0, 1]))
    return float(np.mean(values)) if values else None


def statistics(values, mask):
    mean = local_mean(values)
    keep = mask & (mean > 1)  # very dark pixels are dominated by 8-bit rounding
    residual = np.divide(values, mean, out=np.ones_like(values), where=mean > 0)
    row = dict(enl_native=enl(values, mask), enl_work=working_enl(values, mask),
               mean=float(values[mask].mean()) if mask.any() else None)
    for lag in LAGS:
        row[f'lag{lag}'] = autocorrelation(residual, keep, lag)
    if keep.any():
        for q, v in zip(QUANTILES, np.quantile(residual[keep], QUANTILES)):
            row[f'q{q}'] = float(v)
    return row


def pixel_correlation(a, b, mask):
    return float(np.corrcoef(a[mask], b[mask])[0, 1]) if mask.sum() > 100 else None


def measure_image(job):
    path, boxes, sigmas, gain, seed = job
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    values, _ = grey(path)
    values = values.astype(np.float32)
    mask = clutter_mask(values.shape, boxes)
    inside = np.zeros(values.shape, bool)
    for box in boxes:
        x0, y0, x1, y1 = grown(box, 0, values.shape[1], values.shape[0])
        inside[y0:y1, x0:x1] = True
    rows = dict(stored=statistics(values, mask))
    for sigma in sigmas:
        op = SpeckleResample(p=1., noise_sigma=sigma, structure_gain=gain)
        start = time.perf_counter()
        first = op.resample(torch.from_numpy(values))
        elapsed = time.perf_counter() - start
        if first is None:
            continue
        first = first.round().numpy().astype(np.float32)
        second = op.resample(torch.from_numpy(values)).round().numpy().astype(np.float32)
        row = statistics(first, mask)
        crops = [float(crop_vector(first, b) @ crop_vector(values, b)) for b in boxes if b[2] >= 1 and b[3] >= 1]
        pairs = [float(crop_vector(first, b) @ crop_vector(second, b)) for b in boxes if b[2] >= 1 and b[3] >= 1]
        row.update(seconds=elapsed, clutter_correlation=pixel_correlation(values, first, mask),
                   target_correlation=pixel_correlation(values, first, inside),
                   crop_vs_stored=crops, crop_between_draws=pairs)
        rows[f'sigma={sigma:g}'] = row
    return rows


def aggregate(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return float(np.median(values)) if values else None


def summarize(results, sigmas):
    summary = dict(images=len(results), variants={})
    for name in ['stored'] + [f'sigma={s:g}' for s in sigmas]:
        rows = [r[name] for r in results if name in r]
        keys = ['enl_native', 'enl_work', 'mean'] + [f'lag{l}' for l in LAGS] + [f'q{q}' for q in QUANTILES]
        item = {k: aggregate(rows, k) for k in keys}
        if name != 'stored':
            crops = np.array([v for r in rows for v in r['crop_vs_stored']])
            pairs = np.array([v for r in rows for v in r['crop_between_draws']])
            item.update(seconds=aggregate(rows, 'seconds'), clutter_correlation=aggregate(rows, 'clutter_correlation'),
                        target_correlation=aggregate(rows, 'target_correlation'),
                        crop_vs_stored_p50=float(np.median(crops)), crop_vs_stored_above_09=float((crops >= .9).mean()),
                        crop_between_draws_p50=float(np.median(pairs)), crop_between_draws_above_09=float((pairs >= .9).mean()))
        summary['variants'][name] = item
    stored = summary['variants']['stored']
    distance = {n: sum(abs(v[f'lag{l}'] - stored[f'lag{l}']) for l in LAGS)
                for n, v in summary['variants'].items() if n != 'stored'}
    summary['closest_correlation'] = min(distance, key=distance.get)
    return summary


def fmt(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def write_report(output, summary):
    variants = summary['variants']
    lines = [f"# 斑点重采样统计检查（{summary['split']}，{summary['images']} 张，structure_gain={summary['structure_gain']:g}）", '',
             '各量为逐图中位数。值 / 局部均值的分位数与自相关只统计杂波且局部均值 > 1 的像素。', '',
             '| 版本 | 原图 ENL | 工作分辨率 ENL | 杂波均值 | lag1 | lag2 | lag3 | ' + ' | '.join(f'q{q}' for q in QUANTILES) + ' |',
             '|---|---:|---:|---:|---:|---:|---:|' + '---:|' * len(QUANTILES)]
    for name, v in variants.items():
        lines.append(f"| {name} | {fmt(v['enl_native'], 2)} | {fmt(v['enl_work'], 2)} | {fmt(v['mean'], 2)} | " +
                     ' | '.join(fmt(v[f'lag{l}']) for l in LAGS) + ' | ' + ' | '.join(fmt(v[f'q{q}']) for q in QUANTILES) + ' |')
    lines += ['', f"自相关最接近原图：{summary['closest_correlation']}。", '', '## 原有图案保留了多少', '',
              '| 版本 | 杂波像素相关 | 目标框像素相关 | 目标裁剪与原图相关 p50 | 与原图 ≥0.9 比例 | 两次重采样相关 p50 | 两次重采样 ≥0.9 比例 | 单张耗时（秒） |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, v in variants.items():
        if name != 'stored':
            lines.append(f"| {name} | {fmt(v['clutter_correlation'])} | {fmt(v['target_correlation'])} | {fmt(v['crop_vs_stored_p50'])} | "
                         f"{fmt(v['crop_vs_stored_above_09'])} | {fmt(v['crop_between_draws_p50'])} | {fmt(v['crop_between_draws_above_09'])} | "
                         f"{fmt(v['seconds'])} |")
    lines += ['', '通过条件：ENL 与分位数接近原图，自相关接近原图，杂波像素相关明显低于 1；目标框相关应高于杂波（强散射保留）。',
              '`sheet.png` 每列上为原图、下为重采样，窗口 128 源图像素。', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def contact_sheet(path, picks, sigma, gain, seed):
    torch.manual_seed(seed)
    op = SpeckleResample(p=1., noise_sigma=sigma, structure_gain=gain)
    sheet = Image.new('L', (SHEET_WINDOW * len(picks), SHEET_WINDOW * 2 + 12), 0)
    draw = ImageDraw.Draw(sheet)
    draw.text((2, 0), f'top: stored, bottom: resampled, sigma={sigma:g}', fill=255)
    for i, (image_path, box) in enumerate(picks):
        values, _ = grey(image_path)
        values = values.astype(np.float32)
        new = op.resample(torch.from_numpy(values))
        new = values if new is None else new.round().numpy()
        cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
        left, top = int(cx - SHEET_WINDOW / 2), int(cy - SHEET_WINDOW / 2)
        for row, data in enumerate((values, new)):
            tile = Image.fromarray(data.astype(np.uint8)).crop((left, top, left + SHEET_WINDOW, top + SHEET_WINDOW))
            sheet.paste(tile, (i * SHEET_WINDOW, 12 + row * SHEET_WINDOW))
    sheet.save(path)


def analyze(data_root, output, split='trainval', images=150, sigmas=(.5, .7, .9, 1.1), gain=2., seed=0, workers=1):
    data_root, output = Path(data_root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    dataset = load_json(data_root / f'COCO/{split}.json')
    boxes = {}
    for a in dataset['annotations']:
        boxes.setdefault(a['image_id'], []).append(a['bbox'])
    chosen = random.Random(seed).sample(dataset['images'], min(images, len(dataset['images'])))
    jobs = [(data_root / im['file_name'], boxes.get(im['id'], []), tuple(sigmas), gain, seed + i) for i, im in enumerate(chosen)]
    if workers > 1:
        with Pool(workers) as pool:
            results = pool.map(measure_image, jobs)
    else:
        results = [measure_image(job) for job in jobs]
    summary = dict(split=split, seed=seed, sigmas=list(sigmas), structure_gain=gain, **summarize(results, sigmas))
    save_json(output / 'summary.json', summary)
    write_report(output, summary)
    picks = [(job[0], job[1][0]) for job in jobs if job[1]][:SHEET_TARGETS]
    contact_sheet(output / 'sheet.png', picks, float(summary['closest_correlation'].split('=')[1]), gain, seed)
    print(output / 'report.md', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new output directory (must not exist)')
    parser.add_argument('--split', default='trainval')
    parser.add_argument('--images', type=int, default=150)
    parser.add_argument('--sigmas', type=float, nargs='+', default=[.5, .7, .9, 1.1])
    parser.add_argument('--structure-gain', type=float, default=2.)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    analyze(args.data_root, args.output, args.split, args.images, args.sigmas, args.structure_gain, workers=args.workers)


if __name__ == '__main__':
    main()
