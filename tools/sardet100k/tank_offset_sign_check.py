"""Signed content offset for val copies under flips, using a stack of isolated targets only.

Usage: python tools/sardet100k/tank_offset_sign_check.py OUT

Between mirrored copies a and b, labels satisfy d_a + d_b = -2 on the flipped axis (from tank_orientation_recall.py), where d is
box minus target in each image's frame. So content offsets c = -d sum to +2. Constant-in-frame offset predicts c_a = c_b = +1;
an error only in the flipped copy predicts offsets of 0 and +2. Isolated targets reduce contamination from neighbours.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from error_gallery import ROOT  # noqa: E402

OUT = Path(sys.argv[1])
REACH = 9
SIZE = 256
rows = [r for r in json.load(open(OUT / 'gt_rows.json')) if r['native_gw'] > 0 and r['native_gh'] > 0]
pairs = json.load(open(OUT / 'orientation_recall.json'))['pairs']
by = defaultdict(list)
for r in rows:
    by[r['file_name']].append(r)


def offsets(name, min_nn):
    """Peak of a smoothed target stack, sub-pixel, relative to box centres."""
    values = np.asarray(Image.open(ROOT / 'JPEGImages/val' / name).convert('L'), np.float32)
    stack, count = np.zeros((2 * REACH + 1, 2 * REACH + 1)), 0
    for r in by[name]:
        if r['nn_input'] / 2.5 < min_nn:
            continue
        cx, cy = (r['gt_x1'] + r['gt_x2']) / 2, (r['gt_y1'] + r['gt_y2']) / 2
        # Half-integer centres are averaged over both neighbouring pixels so the box centre itself sits at zero.
        xs, ys = sorted({int(np.floor(cx)), int(np.ceil(cx))}), sorted({int(np.floor(cy)), int(np.ceil(cy))})
        for ix in xs:
            for iy in ys:
                if REACH <= ix < SIZE - REACH and REACH <= iy < SIZE - REACH:
                    crop = values[iy - REACH:iy + REACH + 1, ix - REACH:ix + REACH + 1]
                    stack += (crop - crop.mean()) / (crop.std() + 1e-6) / (len(xs) * len(ys))
        count += 1
    if count < 3:
        return None
    stack /= count
    grid = np.arange(-REACH, REACH + 1)
    profile_x, profile_y = stack[REACH - 3:REACH + 4].mean(0), stack[:, REACH - 3:REACH + 4].mean(1)

    def centre(profile):
        weights = np.clip(profile - np.percentile(profile, 60), 0, None)
        mask = np.abs(grid) <= 6
        return float((weights[mask] * grid[mask]).sum() / max(weights[mask].sum(), 1e-9))

    return centre(profile_x), centre(profile_y), count


report = {}
for min_nn in (0, 8):
    cache = {name: offsets(name, min_nn) for name in by}
    ogsod = np.array([cache[n][:2] for n, p in by.items() if p[0]['source'] == 'ogsod' and cache[n]])
    report[f'nn_ge_{min_nn}'] = dict(images=len(ogsod), median_dx=round(float(np.median(ogsod[:, 0])), 3), median_dy=round(float(np.median(ogsod[:, 1])), 3), histogram_dx=np.histogram(ogsod[:, 0], bins=[-7, -3, -2, -1, 0, 1, 2, 3, 7])[0].tolist(), histogram_dy=np.histogram(ogsod[:, 1], bins=[-7, -3, -2, -1, 0, 1, 2, 3, 7])[0].tolist())
    signed = defaultdict(list)
    for p in pairs:
        a, b = cache.get(p['a']), cache.get(p['b'])
        if a and b and p['matched'] >= .8 * p['of'] and p['orientation'] in ('flip_x', 'flip_y', 'rot180', 'identity'):
            signed[p['orientation']].append((a[0] + b[0], a[0] - b[0], a[1] + b[1], a[1] - b[1]))
    report[f'nn_ge_{min_nn}']['pairs'] = {k: dict(pairs=len(v), median_sum_dx=round(float(np.median([x[0] for x in v])), 2), median_abs_diff_dx=round(float(np.median(np.abs([x[1] for x in v]))), 2), median_sum_dy=round(float(np.median([x[2] for x in v])), 2), median_abs_diff_dy=round(float(np.median(np.abs([x[3] for x in v]))), 2)) for k, v in signed.items()}
(OUT / 'offset_sign_check.json').write_text(json.dumps(report, indent=1))
print(json.dumps(report))
