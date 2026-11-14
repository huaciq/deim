"""Scale-preserving unequal partitions after the E010 fixed-layout routes."""

import numpy as np
from PIL import Image

from .region_bank import context_boxes, origin_range
from .region_dense_sar import SourceQueuePlanner, epoch_plan_summary


def protected_geometry(boxes, size, margin):
    regions = np.zeros((len(boxes), 4), dtype=np.float64)
    required = np.full((len(boxes), 2), size + 1, dtype=np.int32)
    for i, b in enumerate(boxes):
        if not len(b):
            continue
        c = context_boxes(b, size, margin)
        regions[i] = np.r_[c[:, :2].min(0), c[:, 2:].max(0)]
        required[i] = np.ceil(regions[i, 2:]).astype(int) - np.floor(regions[i, :2]).astype(int)
    return regions, required


def centered_crop(region, width, height, size):
    tile = np.array([width, height], dtype=np.int64)
    lo, hi = origin_range(region, tile, size)
    if np.any(lo > hi):
        raise ValueError('Partition cannot contain all protected targets')
    start = (lo + hi) // 2
    return tuple(map(int, np.r_[start, start + tile]))


def _histogram_prefix(values, valid, axis, size):
    """Histograms per column (axis 0) or row (axis 1), including zero padding."""
    coordinate = (np.arange(values.shape[1])[None, :] if axis == 0
                  else np.arange(values.shape[0])[:, None])
    bins = np.broadcast_to(coordinate, values.shape) * 256 + values
    hist = np.bincount(bins[valid], minlength=size * 256).reshape(size, 256)
    return np.concatenate((np.zeros((1, 256), dtype=np.int64), hist.cumsum(0)), axis=0)


def _histogram_quantiles(hist, quantiles):
    """Exactly the linear interpolation used by np.quantile on integer pixels."""
    cumulative = hist.cumsum(1)
    count = cumulative[:, -1]
    result = []
    for q in quantiles:
        position = (count - 1) * q
        low, high = np.floor(position), np.ceil(position)
        a = (cumulative > low[:, None]).argmax(1)
        b = (cumulative > high[:, None]).argmax(1)
        result.append(a + (b - a) * (position - low))
    return np.stack(result, axis=1)


def strip_background_features(image, boxes, region, mask_margin=2., min_pixels=256):
    """All legal centered full-height/full-width crops, indexed by integer length.

    Prefix histograms avoid revisiting pixels for each possible unequal split.
    Differences crossing the crop boundary are excluded, exactly as in E010.
    """
    size = image.width
    if image.height != size:
        raise ValueError('Descriptors require the ordinary square resize frame')
    features = np.zeros((2, size + 1, 5), dtype=np.float64)
    valid = np.zeros((2, size + 1), dtype=bool)
    if not len(boxes):
        return features, valid
    gray = np.asarray(image.convert('L'), dtype=np.int16)
    mask = np.ones(gray.shape, dtype=bool)
    for box in boxes:
        left, top = np.maximum(0, np.floor(box[:2] - mask_margin)).astype(int)
        right, bottom = np.minimum([size, size], np.ceil(box[2:] + mask_margin)).astype(int)
        mask[top:bottom, left:right] = False
    dx, dy = np.abs(np.diff(gray, axis=1)), np.abs(np.diff(gray, axis=0))
    mx, my = mask[:, :-1] & mask[:, 1:], mask[:-1, :] & mask[1:, :]
    for axis in (0, 1):
        lengths = np.arange(1, size + 1)
        lo = np.maximum(0, np.ceil(region[axis + 2] - lengths)).astype(int)
        hi = np.minimum(size - lengths, np.floor(region[axis])).astype(int)
        lengths, lo, hi = lengths[lo <= hi], lo[lo <= hi], hi[lo <= hi]
        if not len(lengths):
            continue
        starts = (lo + hi) // 2
        ends = starts + lengths
        pg = _histogram_prefix(gray, mask, axis, size)
        px = _histogram_prefix(dx, mx, axis, size)
        py = _histogram_prefix(dy, my, axis, size)
        gray_hist = pg[ends] - pg[starts]
        # Horizontal differences span two columns; vertical differences span two rows.
        diff_hist = (px[ends - (axis == 0)] - px[starts]
                     + py[ends - (axis == 1)] - py[starts])
        available = (gray_hist.sum(1) >= min_pixels) & (diff_hist.sum(1) >= min_pixels)
        selected = lengths[available]
        if len(selected):
            features[axis, selected] = np.concatenate((
                _histogram_quantiles(gray_hist[available], [.25, .5, .75]),
                _histogram_quantiles(diff_hist[available], [.5, .9])), axis=1)
            valid[axis, selected] = True
    return features, valid


def extract_strip_features(task):
    index, filename, size, boxes, region, margin, minimum, center, scale = task
    with Image.open(filename) as source:
        image = source.convert('RGB').resize((size, size), Image.Resampling.BILINEAR)
    features, valid = strip_background_features(image, boxes, region, margin, minimum)
    features[valid] = (features[valid] - center) / scale
    return index, features, valid


class AdaptiveSourceQueuePlanner(SourceQueuePlanner):
    def __init__(self, bank, strips, strip_valid, size=640, adaptive=True, **kwargs):
        super().__init__(bank, **kwargs)
        self.strips, self.strip_valid = strips, strip_valid
        self.required = bank['required_wh']
        self.size, self.adaptive = size, adaptive
        self.adaptive_feature_fallbacks = 0

    def select_adaptive(self, primary):
        if self.counts[primary] == 0:
            return None
        choices = np.array([i for i in self.pending
                            if 0 < self.counts[i] <= self.max_targets - self.counts[primary]], dtype=int)
        if not len(choices):
            return None
        candidates = {}
        for axis in (0, 1):
            feasible = self.required[primary, axis] + self.required[choices, axis] <= self.size
            donors = choices[feasible]
            if not len(donors):
                continue
            cuts = (self.required[primary, axis] + self.size - self.required[donors, axis]) // 2
            available = (self.strip_valid[primary, axis, cuts]
                         & self.strip_valid[donors, axis, self.size - cuts])
            distances = np.abs(self.strips[primary, axis, cuts]
                               - self.strips[donors, axis, self.size - cuts]).mean(1)
            for donor, cut, ok, distance in zip(donors, cuts, available, distances):
                candidates.setdefault(int(donor), []).append(
                    (axis + 3, int(cut), bool(ok), float(distance)))
        if not candidates:
            return None
        ranked = []
        if self.nearest_candidates > 0:
            for donor, layouts in candidates.items():
                valid = [v for v in layouts if v[2]]
                if valid:
                    best = min(valid, key=lambda x: x[3])
                    ranked.append((donor, best))
        if ranked:
            ranked.sort(key=lambda x: x[1][3])
            donor, layout = self.rng.choice(ranked[:self.nearest_candidates])
        else:
            self.adaptive_feature_fallbacks += int(self.nearest_candidates > 0)
            donor = self.rng.choice(list(candidates))
            layout = self.rng.choice(candidates[donor])
        return layout[0], donor, layout[1]

    def next(self):
        self._fill()
        primary = self.pending.pop(0)
        layout, selected, cut = -1, [], -1
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
                    break
            if not selected and self.adaptive:
                result = self.select_adaptive(primary)
                if result is not None:
                    layout, donor, cut = result
                    selected = [donor]
        for source in selected:
            self.pending.remove(source)
        group = [primary] + selected
        self.usage[group] += 1
        self.pass_usage[group] += 1
        self.pass_canvases += 1
        if not self.pending and self.cursor == self.n:
            if not np.all(self.pass_usage == 1):
                raise AssertionError('Each complete source pass must use every image exactly once')
            self.completed_passes.append(dict(source_images=self.n, gt=int(self.counts.sum()),
                                             canvases=self.pass_canvases, every_source_exactly_once=True))
        self.rng.shuffle(group)
        if layout >= 3 and group[0] != primary:
            cut = self.size - cut
        return [layout, *group, *([-1] * (4 - len(group))), cut]


def adaptive_epoch_summary(rows, bank):
    report = epoch_plan_summary(rows[:, :5], bank)
    report['routes'] = {str(k): int((rows[:, 0] == k).sum()) for k in (-1, 0, 1, 2, 3, 4)}
    sources = rows[rows[:, 0] >= 0, 1:5].reshape(-1)
    used = sources[sources >= 0]
    packed = np.bincount(used, minlength=len(bank['image_ids']))
    report['packed_class_counts'] = (packed @ bank['counts']).tolist()
    eligible = (bank['crops'][:, :, 0] >= 0).any(1)
    report['formerly_ineligible_packed_exposures'] = int(packed[~eligible].sum())
    return report
