"""Replay DEIM's training Mosaic on SARDet-100K and measure what the stitching does to SAR scenes.

Usage: python tools/sardet100k/mosaic_seam_audit.py OUT_DIR [--config CFG] [--n 4000] [--seed 0] [--per-gallery 16]

Question: Dense O2O gets its extra positives from Mosaic. Mosaic pastes four images from possibly different sensors,
scene types and resize factors onto one canvas with hard straight seams and zero fill. Does that create scene
structure that never occurs in a real SAR image, and how many of the added positives sit inside it?

What it does, per sample:
- Runs the configured Mosaic op itself (same class, same parameters, `use_cache=False` path), and applies the same
  RandomAffine draw to a quadrant-label image so every output pixel knows which source image (or the zero fill) it
  came from. RandomPhotometricDistort/flip/Resize are not applied: they do not change the stitching.
- Scene type per quadrant from its own resized image: fraction of 5x5-smoothed gray below --dark (sea-like dark
  homogeneous backscatter). sea >= --sea-frac, land <= --land-frac, otherwise coast. Gray levels are uncalibrated
  8-bit products, so this is a heuristic; the gallery is the check.
- Physical seam: for each adjacent quadrant pair, the |log-ratio| of mean gray (dB of gray, not backscatter) and the
  CV ratio in the --strip pixels on each side. Natural reference: the same measurement across a virtual seam shifted
  --shift pixels inside one quadrant, i.e. a boundary of the same length and orientation inside one real image.
- Target context: ring of one box long side around each GT; share that is another quadrant or the zero fill.

Source groups come from the dataset profile cache (pixel-matched OGSOD/MSAR, others grouped by image size).
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

REPO = Path(os.environ.get('DEIM_REPO', Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(REPO))
from engine.core import YAMLConfig  # noqa: E402
from engine.data._misc import convert_to_tv_tensor  # noqa: E402
from engine.data.dataset.coco_dataset import CocoDetection  # noqa: E402
from engine.data.transforms.mosaic import Mosaic  # noqa: E402

DEFAULT_CONFIG = 'configs/deim_dfine/deim_hgnetv2_m_sardet100k_36e_b32_nozoom_nocrop.yml'
DEFAULT_SOURCES = '/root/zxy/experiments/diagnostics/sardet100k_dataset_profile_20261106/source_train.json'
PANEL = 640
QUAD_COLORS = [(255, 90, 90), (90, 200, 255), (120, 230, 120), (255, 200, 60)]
FONT = ImageFont.load_default(size=13)
STATE = {}


def group_of(source, width, height):
    """Same grouping as dataset_profile.py."""
    if source != 'other':
        return source
    return f'other_{width}' if width == height and width in (224, 256, 512, 800, 1000) else 'other_misc'


def init_worker(args, mosaic_kwargs, img_folder, ann_file, groups):
    torch.set_num_threads(1)
    STATE['args'] = args
    STATE['mosaic'] = Mosaic(**mosaic_kwargs)
    STATE['dataset'] = CocoDetection(img_folder, ann_file, transforms=None, remap_mscoco_category=False)
    STATE['groups'] = groups


def scene_of(gray, args):
    dark = ndimage.uniform_filter(gray, 5) < args.dark
    frac = float(dark.mean())
    kind = 'sea' if frac >= args.sea_frac else 'land' if frac <= args.land_frac else 'coast'
    return kind, frac


def side_stats(gray, mask):
    values = gray[mask]
    if values.size < 16:
        return None
    mean = float(values.mean())
    return mean, float(values.std() / max(mean, 1e-3))


def step(gray, a_mask, b_mask):
    a, b = side_stats(gray, a_mask), side_stats(gray, b_mask)
    if a is None or b is None:
        return None
    db = abs(10 * np.log10((a[0] + 1) / (b[0] + 1)))
    cv = abs(np.log((a[1] + 1e-3) / (b[1] + 1e-3)))
    return db, cv


def run_sample(k):
    args, mosaic, dataset, groups = STATE['args'], STATE['mosaic'], STATE['dataset'], STATE['groups']
    random.seed(args.seed * 1000003 + k)
    torch.manual_seed(args.seed * 1000003 + k)
    main = random.randrange(len(dataset))

    loaded = []
    original_load = dataset.load_item

    def logged(idx):
        loaded.append(idx)
        return original_load(idx)

    dataset.load_item = logged
    image, target = dataset.load_item(main)
    loaded.clear()
    images, targets, max_h, max_w = mosaic.load_samples_from_dataset(image, target, dataset)
    dataset.load_item = original_load
    indices = [main] + loaded
    counts = [len(t['labels']) for t in targets]
    merged, merged_t = mosaic.create_mosaic_from_dataset(images, targets, max_h, max_w)
    merged_t['boxes'] = convert_to_tv_tensor(merged_t['boxes'], 'boxes', box_format='xyxy', spatial_size=merged.size[::-1])

    label = Image.new('L', merged.size, 0)
    for q, (img, offset) in enumerate(zip(images, [(0, 0), (max_w, 0), (0, max_h), (max_w, max_h)])):
        label.paste(Image.new('L', img.size, q + 1), offset)
    affine_seed = int(torch.randint(0, 2 ** 31 - 1, (1,)))
    torch.manual_seed(affine_seed)
    out, out_t = mosaic.affine_transform(merged, merged_t)
    torch.manual_seed(affine_seed)
    quad = np.asarray(mosaic.affine_transform(label), np.uint8)  # nearest interpolation, fill 0

    gray = np.asarray(out.convert('L'), np.float32)
    H, W = gray.shape
    boxes = out_t['boxes'].as_subclass(torch.Tensor).numpy().astype(np.float32)
    labels = out_t['labels'].numpy()
    box_quad = np.repeat(np.arange(1, 5), counts)
    keep = ((boxes[:, 2] - boxes[:, 0]) >= 1) & ((boxes[:, 3] - boxes[:, 1]) >= 1)  # SanitizeBoundingBoxes(min_size=1)

    quads = []
    for q, (idx, img) in enumerate(zip(indices, images)):
        info = dataset.coco.imgs[dataset.ids[idx]]
        kind, frac = scene_of(np.asarray(img.convert('L'), np.float32), args)
        quads.append(dict(q=q + 1, idx=idx, file=info['file_name'], group=groups[idx], scene=kind, dark_frac=round(frac, 3),
                          resize=round(img.size[0] / info['width'], 4), visible=round(float((quad == q + 1).mean()), 4),
                          targets=int(keep[box_quad == q + 1].sum())))

    seams = []
    dist = {}
    for q in range(1, 5):
        if (quad == q).any():
            dist[q] = ndimage.distance_transform_edt(quad != q)
    s, d = args.strip, args.shift
    for a in range(1, 5):
        for b in range(a + 1, 5):
            if a not in dist or b not in dist:
                continue
            border = (quad == a) & (dist[b] <= 1.5)
            if border.sum() < args.min_seam:
                continue
            physical = step(gray, (quad == a) & (dist[b] <= s), (quad == b) & (dist[a] <= s))
            natural = [step(gray, (quad == x) & (dist[y] > d - s) & (dist[y] <= d), (quad == x) & (dist[y] > d) & (dist[y] <= d + s))
                       for x, y in ((a, b), (b, a))]
            if physical is None:
                continue
            qa, qb = quads[a - 1], quads[b - 1]
            seams.append(dict(a=a, b=b, length=int(border.sum()), db=round(physical[0], 3), cv=round(physical[1], 3),
                              natural=[None if n is None else [round(n[0], 3), round(n[1], 3), (qa, qb)[i]['scene']] for i, n in enumerate(natural)],
                              scenes='|'.join(sorted((qa['scene'], qb['scene']))), cross_source=qa['group'] != qb['group']))
    fill_frac = float((quad == 0).mean())

    rows = []
    for i in np.flatnonzero(keep):
        x0, y0, x1, y1 = boxes[i]
        pad = max(x1 - x0, y1 - y0)
        ox0, oy0, ox1, oy1 = int(max(x0 - pad, 0)), int(max(y0 - pad, 0)), int(min(np.ceil(x1 + pad), W)), int(min(np.ceil(y1 + pad), H))
        ring = np.ones((oy1 - oy0, ox1 - ox0), bool)
        ring[int(y0) - oy0:int(np.ceil(y1)) - oy0, int(x0) - ox0:int(np.ceil(x1)) - ox0] = False
        window = quad[oy0:oy1, ox0:ox1][ring]
        own = int(box_quad[i])
        clipped = 1 - window.size / max((x1 - x0 + 2 * pad) * (y1 - y0 + 2 * pad) - (x1 - x0) * (y1 - y0), 1)
        other = [int(v) for v in np.unique(window) if v not in (0, own)]
        rows.append(dict(cls=int(labels[i]), q=own, group=quads[own - 1]['group'], scene=quads[own - 1]['scene'],
                         long_side=round(float(pad), 1),
                         foreign=round(float(((window != own) & (window != 0)).mean()) if window.size else 0., 3),
                         fill=round(float((window == 0).mean()) if window.size else 0., 3),
                         off_canvas=round(float(clipped), 3),
                         foreign_scenes='|'.join(sorted({quads[v - 1]['scene'] for v in other})),
                         foreign_cross_source=any(quads[v - 1]['group'] != quads[own - 1]['group'] for v in other)))
    return dict(k=k, size=[W, H], fill=round(fill_frac, 4), quads=quads, seams=seams, targets=rows,
                single_targets=counts[0], mosaic_targets=int(keep.sum()), affine_seed=affine_seed,
                box=boxes[keep].round(1).tolist(), box_quad=box_quad[keep].tolist())


def render(row, out, mosaic, dataset, args, title):
    """Re-run the sample deterministically and draw raw | annotated at network resolution."""
    random.seed(args.seed * 1000003 + row['k'])
    torch.manual_seed(args.seed * 1000003 + row['k'])
    main = random.randrange(len(dataset))
    image, target = dataset.load_item(main)
    images, targets, max_h, max_w = mosaic.load_samples_from_dataset(image, target, dataset)
    merged, merged_t = mosaic.create_mosaic_from_dataset(images, targets, max_h, max_w)
    merged_t['boxes'] = convert_to_tv_tensor(merged_t['boxes'], 'boxes', box_format='xyxy', spatial_size=merged.size[::-1])
    label = Image.new('L', merged.size, 0)
    for q, (img, offset) in enumerate(zip(images, [(0, 0), (max_w, 0), (0, max_h), (max_w, max_h)])):
        label.paste(Image.new('L', img.size, q + 1), offset)
    seed = row['affine_seed']
    torch.manual_seed(seed)
    img, _ = mosaic.affine_transform(merged, merged_t)
    torch.manual_seed(seed)
    quad = np.asarray(mosaic.affine_transform(label), np.uint8)
    W, H = img.size
    sx, sy = PANEL / W, PANEL / H
    raw = img.convert('RGB').resize((PANEL, PANEL), Image.BILINEAR)
    ann = raw.copy()
    edge = (ndimage.maximum_filter(quad, 3) != ndimage.minimum_filter(quad, 3))
    edge = np.asarray(Image.fromarray(edge.astype(np.uint8) * 255).resize((PANEL, PANEL), Image.NEAREST)) > 0
    pixels = np.asarray(ann).copy()
    pixels[edge] = (255, 0, 255)
    ann = Image.fromarray(pixels)
    draw = ImageDraw.Draw(ann)
    names = {c['id']: c['name'] for c in dataset.coco.dataset['categories']}
    for j, (box, q) in enumerate(zip(row['box'], row['box_quad'])):
        t = row['targets'][j]
        bad = t['foreign'] + t['fill'] >= args.contaminated
        color = (255, 40, 40) if bad else QUAD_COLORS[q - 1]
        draw.rectangle([box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy], outline=color, width=2 if bad else 1)
    for q in row['quads']:
        ys, xs = np.nonzero(quad == q['q'])
        if not len(xs):
            continue
        x, y = xs.mean() * sx, ys.mean() * sy
        text = f"Q{q['q']} {q['group']} {q['scene']} d={q['dark_frac']:.2f} x{q['resize']:.2f}"
        draw.rectangle(draw.textbbox((x - 90, y), text, font=FONT), fill=(0, 0, 0))
        draw.text((x - 90, y), text, fill=QUAD_COLORS[q['q'] - 1], font=FONT)
    seam_text = '  '.join(f"Q{s['a']}|Q{s['b']} {s['scenes']} {s['db']:.1f}dB" for s in row['seams'])
    canvas = Image.new('RGB', (2 * PANEL + 8, PANEL + 40), (20, 20, 20))
    canvas.paste(raw, (0, 40))
    canvas.paste(ann, (PANEL + 8, 40))
    d = ImageDraw.Draw(canvas)
    d.text((6, 4), f"{title}  sample {row['k']}  targets {row['mosaic_targets']} (single {row['single_targets']})  fill {row['fill']:.2f}", fill=(255, 255, 255), font=FONT)
    d.text((6, 21), seam_text[:200], fill=(255, 180, 255), font=FONT)
    canvas.save(out)


def natural_coast(dataset, groups, args, out, count):
    """Real single images classed as coast: the natural sea/land boundary the seams are compared with."""
    rng = random.Random(args.seed)
    picks, tries = [], 0
    while len(picks) < count and tries < 4000:
        tries += 1
        idx = rng.randrange(len(dataset))
        image, target = dataset.load_item(idx)
        image, target = STATE['mosaic'].resize(image, target)
        kind, frac = scene_of(np.asarray(image.convert('L'), np.float32), args)
        if kind == 'coast':
            picks.append((image, target, groups[idx], frac))
    tile = PANEL // 2
    sheet = Image.new('RGB', (4 * tile, ((len(picks) + 3) // 4) * tile), (20, 20, 20))
    for i, (image, target, group, frac) in enumerate(picks):
        view = image.convert('RGB')
        sx, sy = tile / view.size[0], tile / view.size[1]
        view = view.resize((tile, tile), Image.BILINEAR)
        draw = ImageDraw.Draw(view)
        for box in target['boxes'].as_subclass(torch.Tensor).tolist():
            draw.rectangle([box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy], outline=(80, 255, 80))
        draw.text((4, 4), f'{group} coast d={frac:.2f}', fill=(255, 255, 0), font=FONT)
        sheet.paste(view, ((i % 4) * tile, (i // 4) * tile))
    sheet.save(out)


def quantiles(values):
    values = np.asarray(values, np.float64)
    if not values.size:
        return None
    return {f'p{p}': round(float(np.percentile(values, p)), 3) for p in (5, 25, 50, 75, 95)} | {'n': int(values.size)}


def summarise(results, names, args):
    report = {}
    natural_db = [n[0] for r in results for s in r['seams'] for n in s['natural'] if n is not None]
    natural_coast_db = [n[0] for r in results for s in r['seams'] for n in s['natural'] if n is not None and n[2] == 'coast']
    ref95 = float(np.percentile(natural_db, 95)) if natural_db else np.nan
    coast95 = float(np.percentile(natural_coast_db, 95)) if natural_coast_db else np.nan
    report['natural_seam_db'] = quantiles(natural_db)
    report['natural_seam_db_inside_coast_images'] = quantiles(natural_coast_db)
    seams = [s for r in results for s in r['seams']]
    report['physical_seam_db'] = quantiles([s['db'] for s in seams])
    by_type = defaultdict(list)
    for s in seams:
        by_type[(s['scenes'], 'cross_source' if s['cross_source'] else 'same_source')].append(s)
    report['seams_by_type'] = {f'{k[0]} {k[1]}': dict(share=round(len(v) / max(len(seams), 1), 4), db=quantiles([s['db'] for s in v]),
                                                      cv_logratio_p50=round(float(np.median([s['cv'] for s in v])), 3),
                                                      above_natural_p95=round(float(np.mean([s['db'] > ref95 for s in v])), 4),
                                                      above_natural_coast_p95=round(float(np.mean([s['db'] > coast95 for s in v])), 4))
                               for k, v in sorted(by_type.items(), key=lambda kv: -len(kv[1]))}
    report['mosaics'] = dict(
        n=len(results),
        with_sea_land_seam=round(float(np.mean([any(s['scenes'] in ('land|sea',) for s in r['seams']) for r in results])), 4),
        with_cross_source_seam=round(float(np.mean([any(s['cross_source'] for s in r['seams']) for r in results])), 4),
        distinct_groups=dict(Counter(len({q['group'] for q in r['quads']}) for r in results)),
        resize_ratio_max_over_min=quantiles([max(q['resize'] for q in r['quads']) / min(q['resize'] for q in r['quads']) for r in results]),
        fill_fraction=quantiles([r['fill'] for r in results]),
        targets_single=quantiles([r['single_targets'] for r in results]),
        targets_mosaic=quantiles([r['mosaic_targets'] for r in results]))
    targets = [t for r in results for t in r['targets']]
    bad = lambda t: t['foreign'] + t['fill'] >= args.contaminated
    report['targets'] = dict(n=len(targets), contaminated=round(float(np.mean([bad(t) for t in targets])), 4),
                             foreign_quadrant=round(float(np.mean([t['foreign'] >= args.contaminated for t in targets])), 4),
                             zero_fill=round(float(np.mean([t['fill'] >= args.contaminated for t in targets])), 4),
                             foreign_cross_source=round(float(np.mean([t['foreign_cross_source'] and t['foreign'] >= args.contaminated for t in targets])), 4))
    report['targets_by_class'] = {}
    for c in sorted({t['cls'] for t in targets}):
        rows = [t for t in targets if t['cls'] == c]
        foreign = [t for t in rows if t['foreign'] >= args.contaminated]
        report['targets_by_class'][names.get(c, str(c))] = dict(
            n=len(rows), contaminated=round(float(np.mean([bad(t) for t in rows])), 4),
            foreign_quadrant=round(len(foreign) / len(rows), 4),
            zero_fill=round(float(np.mean([t['fill'] >= args.contaminated for t in rows])), 4),
            own_scene=dict(Counter(t['scene'] for t in rows)),
            foreign_scene_when_contaminated=dict(Counter(f"{t['scene']}->{t['foreign_scenes']}" for t in foreign).most_common(6)))
    report['scene_by_group'] = {}
    for r in results:
        for q in r['quads']:
            report['scene_by_group'].setdefault(q['group'], Counter())[q['scene']] += 1
    report['scene_by_group'] = {g: dict(c) for g, c in sorted(report['scene_by_group'].items())}
    report['dark_frac_by_group'] = {g: quantiles([q['dark_frac'] for r in results for q in r['quads'] if q['group'] == g])
                                    for g in report['scene_by_group']}
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--sources', default=DEFAULT_SOURCES)
    parser.add_argument('--n', type=int, default=4000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--dark', type=float, default=40., help='gray level below which a smoothed pixel is sea-like')
    parser.add_argument('--sea-frac', type=float, default=.6)
    parser.add_argument('--land-frac', type=float, default=.15)
    parser.add_argument('--strip', type=int, default=8, help='pixels on each side of a seam')
    parser.add_argument('--shift', type=int, default=32, help='virtual natural seam offset inside one quadrant')
    parser.add_argument('--min-seam', type=int, default=32)
    parser.add_argument('--contaminated', type=float, default=.1, help='ring share from another quadrant/fill that counts')
    parser.add_argument('--per-gallery', type=int, default=16)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = YAMLConfig(str(REPO / args.config)).yaml_cfg
    data_cfg = cfg['train_dataloader']['dataset']
    ops = [op for op in data_cfg['transforms']['ops'] if op['type'] == 'Mosaic']
    mosaic_kwargs = {k: v for k, v in ops[0].items() if k != 'type'}
    mosaic_kwargs['use_cache'] = False  # the configured path; asserted below
    assert not ops[0].get('use_cache', False), 'cache path not replayed'
    ann = json.load(open(data_cfg['ann_file']))
    sources = json.load(open(args.sources))
    assert len(sources) == len(ann['images'])
    by_id = {im['id']: (s, im['width'], im['height']) for im, s in zip(ann['images'], sources)}
    probe = CocoDetection(data_cfg['img_folder'], data_cfg['ann_file'], transforms=None)
    groups = [group_of(*by_id[i]) for i in probe.ids]
    names = {c['id']: c['name'] for c in ann['categories']}
    del ann

    revision = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    meta = dict(config=args.config, git_revision=revision, mosaic=mosaic_kwargs,
                mosaic_prob=data_cfg['transforms'].get('mosaic_prob'), policy=data_cfg['transforms'].get('policy'), args=vars(args) | {'out': str(args.out)})
    print(json.dumps(meta, indent=1, default=str), flush=True)

    with mp.Pool(args.workers, init_worker, (args, mosaic_kwargs, data_cfg['img_folder'], data_cfg['ann_file'], groups)) as pool:
        results = []
        for i, r in enumerate(pool.imap(run_sample, range(args.n), chunksize=8)):
            results.append(r)
            if (i + 1) % 500 == 0:
                print(f'{i + 1}/{args.n}', flush=True)

    report = summarise(results, names, args) | {'meta': meta}
    (args.out / 'report.json').write_text(json.dumps(report, indent=1, default=str))
    with open(args.out / 'seams.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sample', 'a', 'b', 'group_a', 'group_b', 'scenes', 'cross_source', 'length', 'db', 'cv_logratio', 'natural_db_a', 'natural_db_b'])
        for r in results:
            for s in r['seams']:
                nat = [n[0] if n else '' for n in s['natural']]
                w.writerow([r['k'], s['a'], s['b'], r['quads'][s['a'] - 1]['group'], r['quads'][s['b'] - 1]['group'], s['scenes'],
                            int(s['cross_source']), s['length'], s['db'], s['cv'], *nat])
    with open(args.out / 'targets.csv', 'w', newline='') as f:
        w = csv.writer(f)
        keys = ['cls', 'q', 'group', 'scene', 'long_side', 'foreign', 'fill', 'off_canvas', 'foreign_scenes', 'foreign_cross_source']
        w.writerow(['sample', 'class'] + keys)
        for r in results:
            for t in r['targets']:
                w.writerow([r['k'], names.get(t['cls'], t['cls'])] + [t[k] for k in keys])
    (args.out / 'samples.json').write_text(json.dumps([{k: v for k, v in r.items() if k not in ('box', 'box_quad', 'targets')} for r in results]))

    # Galleries: strongest sea/land seams, strongest cross-source seams, most contaminated targets, and random controls.
    init_worker(args, mosaic_kwargs, data_cfg['img_folder'], data_cfg['ann_file'], groups)
    mosaic, dataset = STATE['mosaic'], STATE['dataset']
    sea_land = sorted([r for r in results if any(s['scenes'] == 'land|sea' for s in r['seams'])],
                      key=lambda r: -max(s['db'] for s in r['seams'] if s['scenes'] == 'land|sea'))
    cross = sorted([r for r in results if any(s['cross_source'] for s in r['seams'])],
                   key=lambda r: -max(s['db'] for s in r['seams'] if s['cross_source']))
    contaminated = sorted(results, key=lambda r: -sum(t['foreign'] + t['fill'] >= args.contaminated for t in r['targets']))
    rng = random.Random(args.seed)
    picks = dict(sea_land=sea_land, cross_source=cross, contaminated_targets=contaminated, random=rng.sample(results, len(results)))
    seen = set()
    for name, rows in picks.items():
        folder = args.out / 'gallery' / name
        folder.mkdir(parents=True, exist_ok=True)
        taken = 0
        for row in rows:
            if taken >= args.per_gallery:
                break
            if row['k'] in seen and name != 'random':
                continue
            seen.add(row['k'])
            render(row, folder / f'{taken:02d}_sample{row["k"]}.jpg', mosaic, dataset, args, name)
            taken += 1
    natural_coast(dataset, groups, args, args.out / 'gallery' / 'natural_coast_reference.jpg', args.per_gallery)
    print(json.dumps({k: v for k, v in report.items() if k != 'meta'}, indent=1, default=str))


if __name__ == '__main__':
    main()
