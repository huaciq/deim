"""Annotation-only region selection for scale-preserving dense O2O training.

All coordinates are in the ordinary square training-resize frame. A primary
image keeps ALL its targets. Donor crops keep every intersecting target and its
local context; a crop cutting either is inadmissible. No validation data is used.
"""

import json
import random
from collections import defaultdict

import numpy as np
from scipy.optimize import nnls


def tile_sizes(size):
    if size % 2:
        raise ValueError('RegionDense requires an even canvas size')
    return ((size // 2, size // 2), (size // 2, size), (size, size // 2))


def context_boxes(boxes, size, margin):
    longest = np.max(boxes[:, 2:] - boxes[:, :2], axis=1) * margin
    return np.clip(boxes + np.stack((-longest, -longest, longest, longest), 1), 0, size)


def origin_range(region, tile, size):
    lo = np.maximum(0, np.ceil(region[2:] - tile)).astype(np.int64)
    hi = np.minimum(np.array([size, size]) - tile, np.floor(region[:2])).astype(np.int64)
    return lo, hi


def visible_and_preserved(boxes, contexts, crop, tolerance=1e-6):
    crop = np.asarray(crop)
    visible = np.all(np.minimum(boxes[:, 2:], crop[2:]) >
                     np.maximum(boxes[:, :2], crop[:2]), axis=1)
    preserved = (np.all(contexts[:, :2] >= crop[:2] - tolerance, axis=1) &
                 np.all(contexts[:, 2:] <= crop[2:] + tolerance, axis=1))
    return visible, preserved


def candidate_crops(boxes, labels, size, margin, tile, rng, classes,
                    max_anchors=16, max_regions=8):
    """Bounded deterministic search; search limits are CPU budgets, not AP tuning."""
    contexts = context_boxes(boxes, size, margin)
    eligible = np.flatnonzero(np.all(contexts[:, 2:] - contexts[:, :2] <= tile, axis=1))
    rng.shuffle(eligible)
    found = {}
    for index in eligible[:max_anchors]:
        lo, hi = origin_range(contexts[index], tile, size)
        if np.any(lo > hi):
            continue
        xs = np.unique([lo[0], (lo[0] + hi[0]) // 2, hi[0]])
        ys = np.unique([lo[1], (lo[1] + hi[1]) // 2, hi[1]])
        for left in xs:
            for top in ys:
                crop = np.array([left, top, left + tile[0], top + tile[1]])
                visible, preserved = visible_and_preserved(boxes, contexts, crop)
                if visible.any() and np.all(preserved[visible]):
                    found[(int(left), int(top))] = np.bincount(labels[visible], minlength=classes)
        if len(found) >= max_regions:
            break
    keys = list(found)
    if len(keys) > max_regions:
        keys = [keys[i] for i in rng.choice(len(keys), max_regions, replace=False)]
    return [(x, y, found[(x, y)]) for x, y in keys]


def sampling_groups(records, classes):
    """Choose a bucket, an image uniformly, then one of that image's crops."""
    _, starts, lengths = np.unique(records[:, 0], return_index=True, return_counts=True)
    groups = [np.flatnonzero(records[starts, 3] == k) for k in range(classes)]
    means = np.zeros((classes, classes), dtype=np.float64)
    for k, group in enumerate(groups):
        if len(group):
            means[k] = np.mean([records[s:s + n, 4:].mean(0)
                                for s, n in zip(starts[group], lengths[group])], axis=0)
    return starts, lengths, groups, means


def build_bank(annotation, size=640, margin=1.0, seed=20261110,
               max_anchors=16, max_regions=8, progress=None):
    """Return portable NumPy arrays and a human-readable construction report."""
    classes = len(annotation['categories'])
    category_ids = sorted(c['id'] for c in annotation['categories'])
    if category_ids != list(range(classes)):
        raise ValueError('RegionDense expects zero-based contiguous category IDs')
    images = sorted(annotation['images'], key=lambda x: x['id'])
    annotations = defaultdict(list)
    for item in annotation['annotations']:
        if not item.get('iscrowd', 0):
            annotations[item['image_id']].append(item)
    rng = np.random.default_rng(seed)
    counts = np.zeros((len(images), classes), dtype=np.int64)
    regions = np.zeros((len(images), 4), dtype=np.float64)
    primary = np.full(len(images), -1, dtype=np.int32)
    records = [[] for _ in tile_sizes(size)]
    for index, image in enumerate(images):
        items = annotations[image['id']]
        boxes = np.array([a['bbox'] for a in items], dtype=np.float64).reshape(-1, 4)
        labels = np.array([a['category_id'] for a in items], dtype=np.int64)
        boxes[:, 2:] += boxes[:, :2]
        boxes *= np.array([size / image['width'], size / image['height']] * 2)
        boxes = np.clip(boxes, 0, size)
        valid = np.all(boxes[:, 2:] > boxes[:, :2], axis=1)
        boxes, labels = boxes[valid], labels[valid]
        counts[index] = np.bincount(labels, minlength=classes)
        if not len(boxes):
            continue
        contexts = context_boxes(boxes, size, margin)
        region = np.r_[contexts[:, :2].min(0), contexts[:, 2:].max(0)]
        regions[index] = region
        bucket = int(counts[index].argmax())
        for layout, tile in enumerate(tile_sizes(size)):
            lo, hi = origin_range(region, tile, size)
            if primary[index] < 0 and np.all(lo <= hi):
                primary[index] = layout
            for x, y, histogram in candidate_crops(
                    boxes, labels, size, margin, tile, rng, classes, max_anchors, max_regions):
                records[layout].append([index, x, y, bucket, *histogram])
        if progress and (index + 1) % 20000 == 0:
            progress(index + 1)
    desired = counts.sum(0) / counts.sum()
    probabilities, pools = [], []
    arrays = dict(image_ids=np.array([x['id'] for x in images], dtype=np.int64),
                  counts=counts, regions=regions, primary=primary)
    for layout, rows in enumerate(records):
        rows = np.array(rows, dtype=np.int32).reshape(-1, classes + 4)
        starts, lengths, groups, means = sampling_groups(rows, classes)
        weights, residual = nnls(means.T, desired)
        if residual > 1e-6 or weights.sum() == 0:
            raise ValueError(f'Layout {layout} cannot preserve donor class proportions: residual={residual}')
        weights /= weights.sum()
        probabilities.append(weights)
        expected = weights @ means
        pools.append(dict(images=len(starts), regions=len(rows),
                          images_per_bucket=[len(g) for g in groups],
                          expected_objects=float(expected.sum()),
                          expected_class_share=(expected / expected.sum()).tolist(),
                          sampling_probability=weights.tolist()))
        arrays[f'records_{layout}'] = rows
    metadata = dict(version=1, size=size, margin=margin, seed=seed,
                    max_anchors=max_anchors, max_regions=max_regions,
                    category_ids=category_ids,
                    category_names=[next(c['name'] for c in annotation['categories'] if c['id'] == k)
                                    for k in category_ids])
    arrays['probabilities'] = np.array(probabilities)
    arrays['metadata'] = np.array(json.dumps(metadata))
    routing = dict(zip(('whole', 'four', 'two_vertical', 'two_horizontal'),
                       np.bincount(primary + 1, minlength=4).tolist()))
    report = dict(metadata=metadata, images=len(images), targets=int(counts.sum()),
                  routing=routing, original_class_share=desired.tolist(), pools=pools)
    return arrays, report


class RegionBank:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as archive:
            self.metadata = json.loads(str(archive['metadata']))
            self.image_ids = archive['image_ids'].copy()
            self.counts = archive['counts'].copy()
            self.regions = archive['regions'].copy()
            self.primary = archive['primary'].copy()
            self.probabilities = archive['probabilities'].copy()
            self.records = [archive[f'records_{k}'].copy() for k in range(3)]
        self.size = self.metadata['size']
        self.margin = self.metadata['margin']
        self.classes = len(self.metadata['category_ids'])
        self.id_to_index = {int(v): k for k, v in enumerate(self.image_ids)}
        self.groups = [sampling_groups(r, self.classes)[:3] for r in self.records]

    def plan(self, image_id, rng=random, max_targets=300):
        """None means the unchanged ordinary full-image path."""
        index = self.id_to_index[int(image_id)]
        layout = int(self.primary[index])
        if layout < 0:
            return None
        width, height = tile_sizes(self.size)[layout]
        lo, hi = origin_range(self.regions[index], (width, height), self.size)
        left, top = (rng.randint(int(a), int(b)) for a, b in zip(lo, hi))
        samples = [dict(index=index, image_id=int(image_id), primary=True,
                        crop=(left, top, left + width, top + height),
                        counts=self.counts[index].copy())]
        seen = {index}
        starts, lengths, groups = self.groups[layout]
        rows = self.records[layout]
        for _ in range(3 if layout == 0 else 1):
            selected = None
            for attempt in range(40):
                bucket = rng.choices(range(self.classes), weights=self.probabilities[layout], k=1)[0]
                group = rng.choice(groups[bucket])
                row = rows[rng.randrange(int(starts[group]), int(starts[group] + lengths[group]))]
                if int(row[0]) not in seen:
                    selected = row
                    break
            if selected is None:
                return None
            index, left, top = map(int, selected[:3])
            seen.add(index)
            samples.append(dict(index=index, image_id=int(self.image_ids[index]), primary=False,
                                crop=(left, top, left + width, top + height), counts=selected[4:].copy()))
        if sum(int(s['counts'].sum()) for s in samples) > max_targets:
            return None
        rng.shuffle(samples)
        for sample, (x, y) in zip(samples, [(x, y) for y in range(0, self.size, height)
                                          for x in range(0, self.size, width)]):
            sample['offset'] = (x, y)
        return samples
