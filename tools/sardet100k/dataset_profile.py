"""CPU profile of SARDet-100K: class balance, target size, scattering morphology, strong background, image statistics, split shift.

Usage: python tools/sardet100k/dataset_profile.py OUT_DIR [--errors VAL_ERRORS_CSV] [--duplicates E000_REPORT_JSON]

Images are processed 8-bit products with unknown per-source radiometric scaling, so every intensity value here is a gray
level, not calibrated backscatter. Class and source are entangled, so morphology and clutter are also reported within source.
"""
import argparse
import csv
import json
import multiprocessing as mp
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from error_gallery import identify_sources, thumb  # noqa: E402

ROOT = Path('/root/zxy/TCRA-SAR/datasets/SARDet_100K')
SPLITS = ('train', 'val', 'test')
SIZE_EDGES = [0, 16, 32, 64, 128, 1 << 20]
SIZE_LABELS = ['<16', '16-32', '32-64', '64-128', '>=128']
MORPH = ['contrast_db', 'target_z', 'peak_to_mean', 'target_cv', 'bright_frac', 'components', 'largest_comp_frac', 'elongation', 'center_edge']
IMAGE_FEATURES = ['mean', 'std', 'p1', 'p50', 'p99', 'dark_frac', 'sat_frac', 'levels', 'entropy', 'lag1_x', 'lag1_y', 'enl']
EIGHT = np.ones((3, 3), int)
TEMPLATE = 24


def target_features(gray, gt_mask, box):
    H, W = gray.shape
    x, y, w, h = box
    x1, y1 = int(np.floor(max(x, 0))), int(np.floor(max(y, 0)))
    x2, y2 = int(np.ceil(min(x + w, W))), int(np.ceil(min(y + h, H)))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    crop = gray[y1:y2, x1:x2]
    short = min(x2 - x1, y2 - y1)
    gap, width = max(2, round(.25 * short)), max(3, round(.5 * short))
    ox1, oy1, ox2, oy2 = max(x1 - gap - width, 0), max(y1 - gap - width, 0), min(x2 + gap + width, W), min(y2 + gap + width, H)
    ix1, iy1, ix2, iy2 = max(x1 - gap, 0), max(y1 - gap, 0), min(x2 + gap, W), min(y2 + gap, H)
    select = ~gt_mask[oy1:oy2, ox1:ox2]
    select[iy1 - oy1:iy2 - oy1, ix1 - ox1:ix2 - ox1] = False
    ring = gray[oy1:oy2, ox1:ox2][select]
    if ring.size < 8:
        return None
    mean_t, mean_r, std_r = crop.mean(), ring.mean(), ring.std()
    bright = crop > mean_r + 2 * std_r
    count = int(bright.sum())
    labels, n = ndimage.label(bright, structure=EIGHT)
    largest = float(np.bincount(labels.ravel())[1:].max() / count) if n else np.nan
    elongation = np.nan
    if count >= 3:
        ys, xs = np.nonzero(bright)
        values = np.linalg.eigvalsh(np.cov(np.stack([xs, ys])) + 1e-6 * np.eye(2))
        elongation = float(np.sqrt(values[1] / max(values[0], 1e-6)))
    ch, cw = crop.shape
    center = np.zeros(crop.shape, bool)
    center[ch // 4:ch - ch // 4, cw // 4:cw - cw // 4] = True
    center_edge = float((crop[center].mean() + 1) / (crop[~center].mean() + 1)) if (~center).any() else np.nan
    return [10 * np.log10((mean_t + 1) / (mean_r + 1)), (np.percentile(crop, 90) - mean_r) / (std_r + 1), (crop.max() + 1) / (mean_t + 1),
            crop.std() / (mean_t + 1), count / crop.size, n, largest, elongation, center_edge], (x1, y1, x2, y2)


def profile(task):
    split, index, path, boxes = task
    with Image.open(path) as image:
        mode = image.mode
        array = np.asarray(image.convert('L'), np.float32)
    H, W = array.shape
    flat = array.ravel()
    histogram = np.bincount(flat.astype(np.uint8), minlength=256).astype(np.float64)
    share = histogram / histogram.sum()
    percent = np.percentile(flat, [1, 50, 99])
    centered = array - array.mean()
    denominator = (centered * centered).mean() + 1e-6
    tile = 16
    enl = np.nan
    if H >= tile and W >= tile:
        tiles = array[:H // tile * tile, :W // tile * tile].reshape(H // tile, tile, W // tile, tile).transpose(0, 2, 1, 3).reshape(-1, tile * tile)
        m, s = tiles.mean(1), tiles.std(1)
        valid = (m > 5) & (s > 0)
        if valid.sum() >= 4:
            m, s = m[valid], s[valid]
            calm = np.argsort(s / m)[:max(len(m) // 4, 1)]
            enl = float(np.median((m[calm] / s[calm]) ** 2))
    stats = [array.mean(), array.std(), percent[0], percent[1], percent[2], (flat <= 20).mean(), (flat >= 250).mean(), (histogram > 0).sum(),
             float(-(share[share > 0] * np.log2(share[share > 0])).sum()),
             (centered[:, :-1] * centered[:, 1:]).mean() / denominator, (centered[:-1] * centered[1:]).mean() / denominator, enl]

    gt_mask, guard = np.zeros((H, W), bool), np.zeros((H, W), bool)
    for _, x, y, w, h in boxes:
        x1, y1, x2, y2 = int(max(np.floor(x), 0)), int(max(np.floor(y), 0)), int(min(np.ceil(x + w), W)), int(min(np.ceil(y + h), H))
        gt_mask[y1:y2, x1:x2] = True
        gap = max(2, round(.25 * min(w, h)))
        guard[max(y1 - gap, 0):y2 + gap, max(x1 - gap, 0):x2 + gap] = True

    rows, levels, sides = [], [], []
    templates = {}
    for g, (c, x, y, w, h) in enumerate(boxes):
        result = target_features(array, gt_mask, (x, y, w, h))
        if result is None:
            continue
        features, (x1, y1, x2, y2) = result
        crop = array[y1:y2, x1:x2]
        levels.append(np.percentile(crop, 90))
        sides.append(max(x2 - x1, y2 - y1))
        rows.append([g, *features])
        if max(x2 - x1, y2 - y1) >= 8:
            scaled = np.asarray(Image.fromarray(crop).resize((TEMPLATE, TEMPLATE), Image.BILINEAR), np.float32)
            span = scaled.max() - scaled.min()
            if span > 0:
                total, n = templates.get(c, (np.zeros((TEMPLATE, TEMPLATE), np.float64), 0))
                templates[c] = (total + (scaled - scaled.min()) / span, n + 1)

    clutter = [np.nan] * 5
    background = ~guard
    if levels and background.any():
        level = float(np.median(levels))
        side = float(np.median(sides))
        strong = (array >= level) & background
        labels, n = ndimage.label(strong, structure=EIGHT)
        like = 0
        if n:
            for s in ndimage.find_objects(labels):
                long_side = max(s[0].stop - s[0].start, s[1].stop - s[1].start)
                like += .5 * side <= long_side <= 2 * side
        clutter = [level, strong.sum() / background.sum(), (array[background] >= 200).mean(), like, np.percentile(array[background], 99) / max(level, 1)]
    return split, index, mode, np.array(stats, np.float32), np.array(clutter, np.float32), np.array(rows, np.float32).reshape(-1, 1 + len(MORPH)), histogram, templates


def lda_fit_predict(train_x, train_y, test_x):
    """Equal-prior linear discriminant with slight shrinkage. Returns class labels, per-class scores."""
    mean, std = train_x.mean(0), train_x.std(0) + 1e-6
    a, b = (train_x - mean) / std, (test_x - mean) / std
    classes = np.unique(train_y)
    centers = np.stack([a[train_y == c].mean(0) for c in classes])
    pooled = sum(np.cov(a[train_y == c].T) for c in classes) / len(classes) + 1e-3 * np.eye(a.shape[1])
    weights = np.linalg.solve(pooled, centers.T).T
    scores = b @ weights.T - .5 * (centers * weights).sum(1)
    return classes[scores.argmax(1)], scores, classes


def balanced_accuracy(truth, predicted):
    return float(np.mean([(predicted[truth == c] == c).mean() for c in np.unique(truth)]))


def auc(positive, negative, limit=20000, seed=0):
    rng = np.random.default_rng(seed)
    positive, negative = positive[np.isfinite(positive)], negative[np.isfinite(negative)]
    if not len(positive) or not len(negative):
        return float('nan')
    positive = rng.choice(positive, limit, replace=False) if len(positive) > limit else positive
    negative = rng.choice(negative, limit, replace=False) if len(negative) > limit else negative
    joined = np.concatenate([positive, negative])
    _, inverse, counts = np.unique(joined, return_inverse=True, return_counts=True)
    rank = (counts.cumsum() - (counts - 1) / 2)[inverse]
    return float((rank[:len(positive)].sum() - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative)))


def js_divergence(p, q):
    p, q = p / max(p.sum(), 1e-12), q / max(q.sum(), 1e-12)
    m = (p + q) / 2

    def kl(a, b):
        keep = a > 0
        return float((a[keep] * np.log2(a[keep] / b[keep])).sum())
    return .5 * kl(p, m) + .5 * kl(q, m)


def js_vs_null(reference, other, edges, seed=0):
    """JS(reference, other) next to JS between a same-size random part of reference and the rest, which sets the noise floor."""
    reference, other = reference[np.isfinite(reference)], other[np.isfinite(other)]
    value = js_divergence(np.histogram(reference, edges)[0].astype(float), np.histogram(other, edges)[0].astype(float))
    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(5):
        perm = rng.permutation(len(reference))
        size = min(len(other), len(reference) // 2)
        nulls.append(js_divergence(np.histogram(reference[perm[:size]], edges)[0].astype(float), np.histogram(reference[perm[size:]], edges)[0].astype(float)))
    return round(value, 5), round(float(np.mean(nulls)), 5)


def percentiles(values, points=(5, 25, 50, 75, 95)):
    values = values[np.isfinite(values)]
    if not len(values):
        return {}
    return {f'p{p}': round(float(np.percentile(values, p)), 3) for p in points}


def eta_squared(values, groups):
    keep = np.isfinite(values)
    values, groups = values[keep], groups[keep]
    if not len(values):
        return float('nan')
    total = ((values - values.mean()) ** 2).sum()
    between = sum((groups == g).sum() * (values[groups == g].mean() - values.mean()) ** 2 for g in np.unique(groups))
    return float(between / max(total, 1e-9))


def group_of(source, width, height):
    if source != 'other':
        return source
    return f'other_{width}' if width == height and width in (224, 256, 512, 800, 1000) else 'other_misc'


def load_split(split, out):
    data = json.load(open(ROOT / f'Annotations/{split}.json'))
    folder = ROOT / 'JPEGImages' / split
    paths = [str(folder / i['file_name']) for i in data['images']]
    cache = out / f'source_{split}.json'
    if cache.exists():
        sources = json.load(open(cache))
    else:
        sources = [str(s) for s in identify_sources(paths)]
        cache.write_text(json.dumps(sources))
    return data, paths, sources


def contact_sheet(rows, out_path, per_row=14, cell=88):
    """rows: (label, mean template or None, [(path, xyxy), ...])."""
    canvas = Image.new('RGB', (150 + cell * (per_row + 1), (cell + 4) * len(rows)), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    for r, (label, template, samples) in enumerate(rows):
        y = r * (cell + 4)
        draw.text((6, y + 34), label, fill=(235, 235, 235))
        if template is not None:
            canvas.paste(Image.fromarray((template * 255).astype(np.uint8)).resize((cell, cell), Image.BICUBIC).convert('RGB'), (150, y))
        for k, (path, box) in enumerate(samples[:per_row]):
            with Image.open(path) as image:
                gray = image.convert('RGB')
            x1, y1, x2, y2 = box
            span = max(max(x2 - x1, y2 - y1) * 2, 32)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            ox, oy = cx - span / 2, cy - span / 2
            crop = gray.crop((int(ox), int(oy), int(ox + span), int(oy + span))).resize((cell, cell), Image.BICUBIC)
            scale = cell / span
            ImageDraw.Draw(crop).rectangle([(x1 - ox) * scale, (y1 - oy) * scale, (x2 - ox) * scale, (y2 - oy) * scale], outline=(60, 220, 90))
            canvas.paste(crop, (150 + (k + 1) * cell, y))
    canvas.save(out_path, quality=90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    parser.add_argument('--errors', type=Path)
    parser.add_argument('--duplicates', type=Path)
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {}
    names, images_meta, tasks = None, {}, []
    annotation_rows = {s: [] for s in SPLITS}
    for split in SPLITS:
        data, paths, sources = load_split(split, args.out)
        names = {c['id']: c['name'] for c in data['categories']}
        position = {im['id']: i for i, im in enumerate(data['images'])}
        by_image = defaultdict(list)
        for a in data['annotations']:
            by_image[position[a['image_id']]].append((a['category_id'], *a['bbox']))
        images_meta[split] = dict(images=data['images'], paths=paths, sources=sources, boxes=by_image,
                                  groups=np.array([group_of(s, i['width'], i['height']) for s, i in zip(sources, data['images'])]))
        for i, path in enumerate(paths):
            tasks.append((split, i, path, by_image.get(i, [])))
        print(split, 'source', Counter(sources), flush=True)
    classes = sorted(names)

    stats = {s: np.full((len(images_meta[s]['paths']), len(IMAGE_FEATURES)), np.nan, np.float32) for s in SPLITS}
    clutter = {s: np.full((len(images_meta[s]['paths']), 5), np.nan, np.float32) for s in SPLITS}
    morph = {s: [] for s in SPLITS}
    histograms = defaultdict(lambda: np.zeros(256))
    templates = {}
    modes = Counter()
    with mp.Pool(args.workers) as pool:
        for done, (split, i, mode, image_stats, image_clutter, rows, histogram, template) in enumerate(pool.imap_unordered(profile, tasks, chunksize=32)):
            stats[split][i], clutter[split][i] = image_stats, image_clutter
            group = images_meta[split]['groups'][i]
            modes[(group, mode)] += 1
            histograms[f'{split}|{group}'] += histogram / max(histogram.sum(), 1)
            for row in rows:
                c, x, y, w, h = images_meta[split]['boxes'][i][int(row[0])]
                morph[split].append((i, c, *row[1:]))
            if split == 'train':
                source = images_meta[split]['sources'][i]
                for c, (total, n) in template.items():
                    key = (c, source)
                    current = templates.get(key, (0, 0))
                    templates[key] = (current[0] + total, current[1] + n)
            if done % 20000 == 0:
                print('profiled', done, 'of', len(tasks), flush=True)

    # Annotation table: split, image position, class, long side, short side, area, input long side at 640, truncated.
    ann = {}
    for split in SPLITS:
        meta = images_meta[split]
        rows = []
        for i, boxes in meta['boxes'].items():
            image = meta['images'][i]
            longest = max(image['width'], image['height'])
            for c, x, y, w, h in boxes:
                out_of_image = x < -1 or y < -1 or x + w > image['width'] + 1 or y + h > image['height'] + 1
                rows.append((i, c, max(w, h), min(w, h), w * h, max(w, h) * 640 / longest, out_of_image))
        ann[split] = np.array(rows, np.float64)
        morph[split] = np.array(morph[split], np.float64)

    # ---- 1. class counts ----
    q1 = {}
    for split in SPLITS:
        table = ann[split]
        counts = {names[c]: int((table[:, 1] == c).sum()) for c in classes}
        image_counts = {names[c]: len(set(table[table[:, 1] == c, 0].astype(int).tolist())) for c in classes}
        values = np.array(list(counts.values()), float)
        share = values / values.sum()
        per_image = np.bincount(table[:, 0].astype(int), minlength=len(images_meta[split]['paths']))
        pairs = defaultdict(set)
        for i, c in table[:, :2].astype(int):
            pairs[i].add(c)
        q1[split] = dict(images=len(images_meta[split]['paths']), instances=int(values.sum()), counts=counts, share={k: round(float(v), 4) for k, v in zip(counts, share)},
                         image_counts=image_counts, instances_per_image_of_class={k: round(counts[k] / max(image_counts[k], 1), 2) for k in counts},
                         imbalance_max_over_min=round(float(values.max() / values.min()), 1), effective_classes=round(float(np.exp(-(share * np.log(share)).sum())), 2),
                         per_image=percentiles(per_image.astype(float), (50, 90, 99)) | dict(max=int(per_image.max())), images_without_gt=int((per_image == 0).sum()),
                         single_class_image_share=round(float(np.mean([len(v) == 1 for v in pairs.values()])), 4))
    report['q1_class_counts'] = q1
    report['class_by_source'] = {}
    for split in SPLITS:
        table = ann[split]
        sources = np.array(images_meta[split]['groups'])[table[:, 0].astype(int)]
        report['class_by_source'][split] = {g: {names[c]: int(((sources == g) & (table[:, 1] == c)).sum()) for c in classes} for g in sorted(set(sources))}

    # ---- 2. sizes ----
    q2 = {}
    for split in SPLITS:
        table = ann[split]
        rows = {}
        for c in classes:
            part = table[table[:, 1] == c]
            bins, _ = np.histogram(part[:, 2], SIZE_EDGES)
            rows[names[c]] = dict(n=len(part), long_side=percentiles(part[:, 2]), sqrt_area=percentiles(np.sqrt(part[:, 4])), aspect=percentiles(part[:, 2] / np.maximum(part[:, 3], 1e-3), (50, 95)),
                                  long_side_share=dict(zip(SIZE_LABELS, [round(float(v), 4) for v in bins / max(len(part), 1)])),
                                  coco_small=round(float((part[:, 4] < 32 ** 2).mean()), 4), coco_medium=round(float(((part[:, 4] >= 32 ** 2) & (part[:, 4] < 96 ** 2)).mean()), 4), coco_large=round(float((part[:, 4] >= 96 ** 2).mean()), 4),
                                  input640_long_side=percentiles(part[:, 5]), input640_lt16=round(float((part[:, 5] < 16).mean()), 4), input640_lt32=round(float((part[:, 5] < 32).mean()), 4), out_of_image=round(float(part[:, 6].mean()), 4))
        q2[split] = rows
    report['q2_sizes'] = q2
    table = ann['train']
    groups = images_meta['train']['groups'][table[:, 0].astype(int)]
    report['q2_train_long_side_by_group'] = {g: {names[c]: dict(n=int(((groups == g) & (table[:, 1] == c)).sum()), **percentiles(table[(groups == g) & (table[:, 1] == c), 2], (25, 50, 75))) for c in classes if ((groups == g) & (table[:, 1] == c)).sum() >= 30} for g in sorted(set(groups))}

    # ---- 3. morphology (train fit; val and test only for held-out accuracy) ----
    train_rows = morph['train']
    train_group = images_meta['train']['groups'][train_rows[:, 0].astype(int)]
    train_source = np.array(images_meta['train']['sources'])[train_rows[:, 0].astype(int)]
    train_class = train_rows[:, 1].astype(int)
    values = train_rows[:, 2:]
    q3 = dict(n_targets={s: len(morph[s]) for s in SPLITS})
    q3['by_class'] = {names[c]: {f: percentiles(values[train_class == c, k], (25, 50, 75)) for k, f in enumerate(MORPH)} for c in classes}
    q3['by_class_source'] = {}
    for source in ('ogsod', 'msar', 'other'):
        for c in classes:
            keep = (train_source == source) & (train_class == c)
            if keep.sum() >= 100:
                q3['by_class_source'][f'{names[c]}|{source}'] = dict(n=int(keep.sum()), **{f: round(float(np.nanmedian(values[keep, k])), 3) for k, f in enumerate(MORPH)})
    q3['eta2_class'] = {f: round(eta_squared(values[:, k], train_class), 3) for k, f in enumerate(MORPH)}
    q3['eta2_source'] = {f: round(eta_squared(values[:, k], train_source), 3) for k, f in enumerate(MORPH)}
    q3['eta2_class_within_source'] = {source: {f: round(eta_squared(values[train_source == source, k], train_class[train_source == source]), 3) for k, f in enumerate(MORPH)} for source in ('ogsod', 'msar', 'other')}
    pairs = {}
    for source in ('ogsod', 'msar', 'other'):
        present = [c for c in classes if ((train_source == source) & (train_class == c)).sum() >= 100]
        for a in range(len(present)):
            for b in range(a + 1, len(present)):
                ka, kb = (train_source == source) & (train_class == present[a]), (train_source == source) & (train_class == present[b])
                scores = {f: auc(values[ka, k], values[kb, k]) for k, f in enumerate(MORPH)}
                best = max(scores, key=lambda f: abs(scores[f] - .5))
                pairs[f'{source}:{names[present[a]]}-{names[present[b]]}'] = dict(best_feature=best, best_auc=round(max(scores[best], 1 - scores[best]), 3), median_auc=round(float(np.median([max(v, 1 - v) for v in scores.values()])), 3))
    q3['pair_auc_within_source'] = pairs
    across = {}
    for c in classes:
        present = [s for s in ('ogsod', 'msar', 'other') if ((train_source == s) & (train_class == c)).sum() >= 100]
        for a in range(len(present)):
            for b in range(a + 1, len(present)):
                ka, kb = (train_source == present[a]) & (train_class == c), (train_source == present[b]) & (train_class == c)
                scores = {f: auc(values[ka, k], values[kb, k]) for k, f in enumerate(MORPH)}
                best = max(scores, key=lambda f: abs(scores[f] - .5))
                across[f'{names[c]}:{present[a]}-{present[b]}'] = dict(best_feature=best, best_auc=round(max(scores[best], 1 - scores[best]), 3), median_auc=round(float(np.median([max(v, 1 - v) for v in scores.values()])), 3))
    q3['same_class_across_source_auc'] = across

    # Simple linear separability on hand features. Near-duplicate tiles make these upper estimates.
    def design(split, kind):
        rows = morph[split]
        image_index, cls = rows[:, 0].astype(int), rows[:, 1].astype(int)
        features = np.nan_to_num(rows[:, 2:], nan=0.)
        sides = []
        lookup = defaultdict(list)
        for i, boxes in images_meta[split]['boxes'].items():
            for c, x, y, w, h in boxes:
                lookup[(i, c)].append(max(w, h))
        # Size is summarised per image and class because morph rows skip unusable boxes.
        for i, c in zip(image_index, cls):
            sides.append(np.median(lookup[(i, c)]))
        size = np.log2(np.maximum(np.array(sides), 1))[:, None]
        src = np.array(images_meta[split]['sources'])[image_index]
        X = dict(morph=features, size=size, both=np.hstack([features, size]))[kind]
        return X, cls, src
    separability = {}
    for label, sources in [('all_sources', None), ('ogsod', 'ogsod'), ('msar', 'msar'), ('other', 'other')]:
        row = {}
        for kind in ('morph', 'size', 'both'):
            x_train, y_train, s_train = design('train', kind)
            x_val, y_val, s_val = design('val', kind)
            if sources:
                keep_train, keep_val = s_train == sources, s_val == sources
                x_train, y_train, x_val, y_val = x_train[keep_train], y_train[keep_train], x_val[keep_val], y_val[keep_val]
            present = [c for c in np.unique(y_train) if (y_train == c).sum() >= 100 and (y_val == c).sum() >= 20]
            keep_train, keep_val = np.isin(y_train, present), np.isin(y_val, present)
            predicted, _, _ = lda_fit_predict(x_train[keep_train], y_train[keep_train], x_val[keep_val])
            row[kind] = round(balanced_accuracy(y_val[keep_val], predicted), 3)
        row['classes'] = [names[int(c)] for c in present]
        row['chance'] = round(1 / len(present), 3)
        separability[label] = row
    q3['lda_balanced_accuracy_train_to_val'] = separability
    report['q3_morphology'] = q3

    # ---- 4. strong background ----
    q4 = {}
    all_groups = np.concatenate([images_meta[s]['groups'] for s in SPLITS])
    all_clutter = np.concatenate([clutter[s] for s in SPLITS])
    for group in sorted(set(all_groups)):
        part = all_clutter[all_groups == group]
        q4[group] = dict(images=len(part), target_level=percentiles(part[:, 0], (25, 50, 75)), bg_frac_ge_target_level=percentiles(part[:, 1], (25, 50, 75, 95)),
                         bg_frac_ge_200=percentiles(part[:, 2], (25, 50, 75, 95)), target_like_blobs=percentiles(part[:, 3], (25, 50, 75, 95)), bg_p99_over_target_level=percentiles(part[:, 4], (25, 50, 75, 95)),
                         share_bg_ge_target_over_1pct=round(float(np.nanmean(part[:, 1] > .01)), 4), share_bg_ge_target_over_5pct=round(float(np.nanmean(part[:, 1] > .05)), 4), share_bg_ge_target_over_20pct=round(float(np.nanmean(part[:, 1] > .2)), 4),
                         share_blobs_ge_5=round(float(np.nanmean(part[:, 3] >= 5)), 4))
    report['q4_background_by_group'] = q4
    report['q4_overall'] = dict(share_bg_ge_target_over_1pct=round(float(np.nanmean(all_clutter[:, 1] > .01)), 4), share_bg_ge_target_over_5pct=round(float(np.nanmean(all_clutter[:, 1] > .05)), 4), share_bg_ge_target_over_20pct=round(float(np.nanmean(all_clutter[:, 1] > .2)), 4),
                                bg_frac_ge_200_over_5pct=round(float(np.nanmean(all_clutter[:, 2] > .05)), 4), share_blobs_ge_5=round(float(np.nanmean(all_clutter[:, 3] >= 5)), 4), images=len(all_clutter))
    by_class = {}
    table = ann['train']
    for c in classes:
        idx = np.unique(table[table[:, 1] == c, 0].astype(int))
        for source in ('ogsod', 'msar', 'other'):
            keep = idx[np.array(images_meta['train']['sources'])[idx] == source]
            if len(keep) >= 100:
                part = clutter['train'][keep]
                by_class[f'{names[c]}|{source}'] = dict(images=len(keep), bg_frac_ge_target_level=percentiles(part[:, 1], (50, 75)), bg_frac_ge_200=percentiles(part[:, 2], (50, 75)), target_like_blobs=percentiles(part[:, 3], (50, 75)))
    report['q4_train_by_class_source'] = by_class
    if args.errors and args.errors.exists():
        file_index = {im['file_name']: i for i, im in enumerate(images_meta['val']['images'])}
        fp, fn = np.zeros(len(file_index)), np.zeros(len(file_index))
        for row in csv.DictReader(open(args.errors)):
            (fp if row['group'] == 'FP' else fn)[file_index[row['file_name']]] += 1
        gt = np.array([len(images_meta['val']['boxes'].get(i, [])) for i in range(len(file_index))], float)
        link = {}
        source_val = np.array(images_meta['val']['sources'])
        for source in ('ogsod', 'msar', 'other', 'all'):
            keep = np.ones(len(gt), bool) if source == 'all' else source_val == source
            metric = clutter['val'][:, 1]
            keep = keep & np.isfinite(metric)
            edges = np.quantile(metric[keep], [0, .25, .5, .75, 1])
            rows = []
            for a, b, last in zip(edges[:-1], edges[1:], [False, False, False, True]):
                part = keep & (metric >= a) & ((metric <= b) if last else (metric < b))
                rows.append(dict(bg_frac=[round(float(a), 4), round(float(b), 4)], images=int(part.sum()), gt=int(gt[part].sum()), miss_rate=round(float(fn[part].sum() / max(gt[part].sum(), 1)), 4), fp_per_image=round(float(fp[part].mean()), 3), fp_per_gt=round(float(fp[part].sum() / max(gt[part].sum(), 1)), 3)))
            link[source] = rows
        report['q4_val_errors_by_clutter_quartile'] = link

    # ---- 5. image statistics ----
    q5 = dict(pixel_sizes={s: {f'{w}x{h}': n for (w, h), n in Counter((i['width'], i['height']) for i in images_meta[s]['images']).most_common(8)} for s in SPLITS},
              modes={f'{g}|{m}': n for (g, m), n in sorted(modes.items())}, formats={s: dict(Counter(Path(i['file_name']).suffix for i in images_meta[s]['images'])) for s in SPLITS})
    all_stats = np.concatenate([stats[s] for s in SPLITS])
    q5['by_group'] = {g: dict(images=int((all_groups == g).sum()), **{f: percentiles(all_stats[all_groups == g, k], (10, 50, 90)) for k, f in enumerate(IMAGE_FEATURES)}) for g in sorted(set(all_groups))}
    q5['eta2_group'] = {f: round(eta_squared(all_stats[:, k], all_groups), 3) for k, f in enumerate(IMAGE_FEATURES)}
    x, y = np.nan_to_num(all_stats, nan=0.), all_groups
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(x))
    cut = len(x) // 2
    predicted, _, _ = lda_fit_predict(x[perm[:cut]], y[perm[:cut]], x[perm[cut:]])
    q5['lda_group_from_image_stats'] = dict(balanced_accuracy=round(balanced_accuracy(y[perm[cut:]], predicted), 3), chance=round(1 / len(set(y)), 3), note='random image split; near-duplicates inflate this')
    report['q5_images'] = q5
    report['gray_hist'] = {k: [round(float(v), 6) for v in (h / max(h.sum(), 1e-12))] for k, h in histograms.items()}

    # ---- 6. split shift ----
    q6 = {}
    def shares(split, key):
        values = Counter(key(split).tolist())
        total = sum(values.values())
        return {k: round(v / total, 4) for k, v in sorted(values.items())}
    q6['group_share_by_image'] = {s: shares(s, lambda s: images_meta[s]['groups']) for s in SPLITS}
    q6['class_share_js'] = {}
    class_counts = {s: np.array([q1[s]['counts'][names[c]] for c in classes], float) for s in SPLITS}
    for s in ('val', 'test'):
        q6['class_share_js'][f'train_{s}'] = round(js_divergence(class_counts['train'], class_counts[s]), 5)
    q6['class_share_js']['val_test'] = round(js_divergence(class_counts['val'], class_counts['test']), 5)
    edges = np.logspace(np.log10(2), np.log10(2048), 25)
    q6['size_js_by_class'] = {}
    for c in classes:
        reference = ann['train'][ann['train'][:, 1] == c, 2]
        row = {}
        for s in ('val', 'test'):
            value, null = js_vs_null(reference, ann[s][ann[s][:, 1] == c, 2], edges)
            row[s] = dict(js=value, null=null)
        row['val_test'] = round(js_divergence(np.histogram(ann['val'][ann['val'][:, 1] == c, 2], edges)[0].astype(float), np.histogram(ann['test'][ann['test'][:, 1] == c, 2], edges)[0].astype(float)), 5)
        q6['size_js_by_class'][names[c]] = row
    q6['image_feature_js'] = {}
    for k, f in enumerate(IMAGE_FEATURES):
        finite = stats['train'][:, k][np.isfinite(stats['train'][:, k])]
        feature_edges = np.unique(np.quantile(finite, np.linspace(0, 1, 33)))
        if len(feature_edges) < 3:
            feature_edges = np.linspace(finite.min(), finite.max() + 1, 33)
        feature_edges[0], feature_edges[-1] = -np.inf, np.inf
        row = {}
        for s in ('val', 'test'):
            value, null = js_vs_null(stats['train'][:, k], stats[s][:, k], feature_edges)
            spread = np.nanstd(stats['train'][:, k])
            row[s] = dict(js=value, null=null, smd=round(float((np.nanmean(stats[s][:, k]) - np.nanmean(stats['train'][:, k])) / max(spread, 1e-6)), 3))
        q6['image_feature_js'][f] = row
    domain = {}
    for s in ('val', 'test'):
        rng = np.random.default_rng(0)
        a, b = np.nan_to_num(stats['train']), np.nan_to_num(stats[s])
        pick = rng.permutation(len(a))[:len(b) * 2]
        a, b = a[pick], b[rng.permutation(len(b))]
        xa, xb = a[:len(a) // 2], a[len(a) // 2:]
        ya, yb = b[:len(b) // 2], b[len(b) // 2:]
        fit_x = np.vstack([xa, ya])
        fit_y = np.r_[np.zeros(len(xa)), np.ones(len(ya))]
        _, scores, _ = lda_fit_predict(fit_x, fit_y, np.vstack([xb, yb]))
        margin = scores[:, 1] - scores[:, 0]
        domain[f'train_vs_{s}_auc'] = round(auc(margin[len(xb):], margin[:len(xb)]), 3)
        shuffled = a[rng.permutation(len(a))]
        half = len(shuffled) // 2
        quarter = half // 2
        fit_x = np.vstack([shuffled[:quarter], shuffled[half:half + quarter]])
        fit_y = np.r_[np.zeros(quarter), np.ones(quarter)]
        _, scores, _ = lda_fit_predict(fit_x, fit_y, np.vstack([shuffled[quarter:half], shuffled[half + quarter:]]))
        margin = scores[:, 1] - scores[:, 0]
        domain[f'null_train_vs_train_auc_for_{s}'] = round(auc(margin[half - quarter:], margin[:half - quarter]), 3)
    q6['domain_lda'] = domain
    # Within-group test: composition removed.
    within = {}
    for group in sorted(set(all_groups)):
        row = {}
        for k, f in enumerate(IMAGE_FEATURES):
            base = stats['train'][images_meta['train']['groups'] == group, k]
            if len(base) < 50:
                continue
            spread = np.nanstd(base)
            values_for = {}
            for s in ('val', 'test'):
                other = stats[s][images_meta[s]['groups'] == group, k]
                if len(other) >= 50:
                    values_for[s] = round(float((np.nanmean(other) - np.nanmean(base)) / max(spread, 1e-6)), 3)
            row[f] = values_for
        within[group] = row
    q6['within_group_smd'] = within
    aircraft = {}
    aircraft_id = next(c for c in classes if names[c] == 'aircraft')
    for s in SPLITS:
        part = ann[s][ann[s][:, 1] == aircraft_id]
        widths = np.array([images_meta[s]['images'][int(i)]['width'] for i in part[:, 0]])
        aircraft[s] = dict(n=len(part), images=len(set(part[:, 0].astype(int).tolist())), per_image=round(len(part) / max(len(set(part[:, 0].astype(int).tolist())), 1), 2), image_width={str(k): int(v) for k, v in Counter(widths.tolist()).most_common(6)},
                           long_side=percentiles(part[:, 2]), out_of_image=round(float(part[:, 6].mean()), 4), aspect=percentiles(part[:, 2] / np.maximum(part[:, 3], 1e-3), (50, 95)),
                           groups={k: int(v) for k, v in Counter(images_meta[s]['groups'][part[:, 0].astype(int)].tolist()).most_common()})
        rows = morph[s][morph[s][:, 1] == aircraft_id]
        aircraft[s]['morph_median'] = {f: round(float(np.nanmedian(rows[:, 2 + k])), 3) for k, f in enumerate(MORPH)}
    q6['aircraft_by_split'] = aircraft
    with mp.Pool(args.workers) as pool:
        thumbs = {s: np.stack([r[0] for r in pool.map(thumb, images_meta[s]['paths'], chunksize=64)]) for s in ('val', 'test')}
    best = np.concatenate([(thumbs['test'][start:start + 512] @ thumbs['val'].T).max(1) for start in range(0, len(thumbs['test']), 512)])
    q6['test_images_matching_val_thumbnail'] = {str(t): round(float((best >= t).mean()), 4) for t in (.9, .95, .98, .99)}
    if args.duplicates and args.duplicates.exists():
        previous = json.load(open(args.duplicates))
        q6['train_match_rates'] = {s: dict(whole_image=previous[s]['whole_image_match_rate'], target_crop=previous[s]['crop_match_rate']) for s in ('val', 'test') if s in previous}
    report['q6_split_shift'] = q6
    (args.out / 'report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print('report written', flush=True)

    # ---- montages ----
    rng = np.random.default_rng(0)
    rows = []
    meta = images_meta['train']
    table = []
    for i, boxes in meta['boxes'].items():
        for c, x, y, w, h in boxes:
            if max(w, h) >= 8:
                table.append((c, meta['sources'][i], i, x, y, x + w, y + h))
    by_key = defaultdict(list)
    for row in table:
        by_key[(row[0], row[1])].append(row)
    for c in classes:
        for source in ('ogsod', 'msar', 'other'):
            samples = by_key.get((c, source), [])
            if len(samples) < 100:
                continue
            picks = [samples[j] for j in rng.choice(len(samples), 14, replace=False)]
            total, n = templates.get((c, source), (None, 0))
            rows.append((f'{names[c]} | {source} n={len(samples)}', total / n if n else None, [(meta['paths'][p[2]], p[3:]) for p in picks]))
    contact_sheet(rows, args.out / 'targets_by_class_source.jpg')
    tiles = []
    for group in sorted(set(meta['groups'])):
        idx = np.flatnonzero(meta['groups'] == group)
        tiles.append((group, rng.choice(idx, min(10, len(idx)), replace=False)))
    cell = 160
    canvas = Image.new('RGB', (110 + cell * 10, (cell + 4) * len(tiles)), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    for r, (group, picks) in enumerate(tiles):
        draw.text((6, r * (cell + 4) + 70), f'{group}\n{(meta["groups"] == group).sum()}', fill=(235, 235, 235))
        for k, i in enumerate(picks):
            with Image.open(meta['paths'][i]) as image:
                canvas.paste(image.convert('RGB').resize((cell, cell), Image.BILINEAR), (110 + k * cell, r * (cell + 4)))
    canvas.save(args.out / 'scenes_by_group.jpg', quality=88)
    print('montages written', flush=True)


if __name__ == '__main__':
    main()
