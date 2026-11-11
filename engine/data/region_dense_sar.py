"""Background-compatible RegionDense with conserved source-image passes.

Planning uses training pixels/annotations only and runs before training. Saved
epoch plans make source scheduling independent of DDP rank, worker and resume.
"""

import random

import numpy as np
from PIL import Image


def background_features(image, boxes, crops, mask_margin=2., min_pixels=256):
    """Display-gray/texture proxies, never calibrated SAR or physical ENL."""
    gray = np.asarray(image.convert('L'), dtype=np.float32)
    mask = np.ones(gray.shape, dtype=bool)
    height, width = gray.shape
    for box in boxes:
        left, top = np.maximum(0, np.floor(box[:2] - mask_margin)).astype(int)
        right, bottom = np.minimum([width, height], np.ceil(box[2:] + mask_margin)).astype(int)
        mask[top:bottom, left:right] = False
    features = np.zeros((3, 5), dtype=np.float64)
    valid = np.zeros(3, dtype=bool)
    for layout, (left, top, right, bottom) in enumerate(crops):
        if left < 0:
            continue
        a, m = gray[top:bottom, left:right], mask[top:bottom, left:right]
        values = a[m]
        dx = np.abs(np.diff(a, axis=1))[m[:, :-1] & m[:, 1:]]
        dy = np.abs(np.diff(a, axis=0))[m[:-1, :] & m[1:, :]]
        differences = np.concatenate((dx, dy))
        if len(values) < min_pixels or len(differences) < min_pixels:
            continue
        features[layout] = np.r_[np.quantile(values, [.25, .5, .75]),
                                 np.quantile(differences, [.5, .9])]
        valid[layout] = True
    return features, valid


def extract_image_features(task):
    index, filename, size, boxes, crops, margin, minimum = task
    with Image.open(filename) as source:
        image = source.convert('RGB').resize((size, size), Image.Resampling.BILINEAR)
    features, valid = background_features(image, boxes, crops, margin, minimum)
    return index, features, valid


class SourceQueuePlanner:
    """Finite FIFO queue. Never draw new donors with replacement or drop a GT."""

    def __init__(self, bank, probability=.5, queue_capacity=64,
                 nearest_candidates=4, max_targets=300, seed=0):
        self.bank = bank
        self.n = len(bank['image_ids'])
        self.probability = probability
        self.queue_capacity = queue_capacity
        self.nearest_candidates = nearest_candidates
        self.max_targets = max_targets
        self.rng = random.Random(seed)
        self.eligible = bank['crops'][:, :, 0] >= 0
        self.counts = bank['counts'].sum(1)
        self.features = bank['features']
        self.valid = bank['feature_valid']
        self.pending, self.order = [], []
        self.cursor = 0
        self.completed_passes = []
        self.usage = np.zeros(self.n, dtype=np.int64)
        self.pass_usage = np.zeros(self.n, dtype=np.int32)
        self.pass_canvases = 0
        self.feature_fallbacks = 0

    def _fill(self):
        if not self.pending and self.cursor == len(self.order):
            self.order = list(range(self.n))
            self.rng.shuffle(self.order)
            self.cursor = 0
            self.pass_usage.fill(0)
            self.pass_canvases = 0
        while len(self.pending) < self.queue_capacity and self.cursor < self.n:
            self.pending.append(self.order[self.cursor])
            self.cursor += 1

    def select(self, primary, choices, layout):
        """Uniform among the nearest eligible candidates; random if unavailable."""
        if self.nearest_candidates > 0 and self.valid[primary, layout]:
            candidates = np.array([v for v in choices if self.valid[v, layout]], dtype=np.int64)
            if len(candidates):
                distance = np.abs(self.features[candidates, layout] - self.features[primary, layout]).mean(1)
                ranked = np.argsort(distance, kind='stable')[:self.nearest_candidates]
                return int(candidates[self.rng.choice(ranked.tolist())])
        if self.nearest_candidates > 0:
            self.feature_fallbacks += 1
        return self.rng.choice(choices)

    def next(self):
        self._fill()
        primary = self.pending.pop(0)
        layout, selected = -1, []
        if self.rng.random() < self.probability:
            for k, needed in enumerate((3, 1, 1)):
                if not self.eligible[primary, k]:
                    continue
                choices = [v for v in self.pending if self.eligible[v, k]]
                if len(choices) < needed:
                    continue
                proposed, total = [], int(self.counts[primary])
                for _ in range(needed):
                    allowed = [v for v in choices if v not in proposed and total + self.counts[v] <= self.max_targets]
                    if not allowed:
                        break
                    chosen = self.select(primary, allowed, k)
                    proposed.append(chosen)
                    total += int(self.counts[chosen])
                if len(proposed) == needed:
                    layout, selected = k, proposed
                    for source in selected:
                        self.pending.remove(source)
                    break
        group = [primary] + selected
        self.usage[group] += 1
        self.pass_usage[group] += 1
        self.pass_canvases += 1
        if not self.pending and self.cursor == self.n:
            if not np.all(self.pass_usage == 1):
                raise AssertionError('Each complete source pass must use every image exactly once')
            self.completed_passes.append(dict(source_images=self.n, gt=int(self.counts.sum()),
                                             canvases=self.pass_canvases, every_source_exactly_once=True))
        # Randomize slot assignment, without altering selected sources or pixels.
        self.rng.shuffle(group)
        return [layout, *group, *([-1] * (4 - len(group)))]


def epoch_plan_summary(rows, bank):
    sources = rows[:, 1:].reshape(-1)
    sources = sources[sources >= 0]
    usage = np.bincount(sources, minlength=len(bank['image_ids']))
    histogram = usage @ bank['counts']
    return dict(canvases=len(rows), source_images_consumed=len(sources),
                gt=int(histogram.sum()), gt_per_canvas=float(histogram.sum() / len(rows)),
                class_counts=histogram.tolist(), native_size_counts=(usage @ bank['native_size_counts']).tolist(),
                input_size_counts=(usage @ bank['input_size_counts']).tolist(),
                routes={str(k): int((rows[:, 0] == k).sum()) for k in (-1, 0, 1, 2)})
