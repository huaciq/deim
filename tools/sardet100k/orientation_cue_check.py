"""D18: is the up/down asymmetry that the detector relies on a property of the targets, is it stable in val, and is it
needed to tell targets from strong clutter?

D17 showed that a vertical flip of the val images costs the frozen detector 27 AP. Either the data carry a stable vertical
asymmetry that is a useful cue (then sharing orientations could cost accuracy), or the detector simply never learned the
other orientations. Three checks on 24x24 crops (D12's encoding: zero mean, unit norm), independent of the detector:

  A  pixel statistics per GT: (top half - bottom half) / sum of the box, and the same for a band above / below the box;
     left / right as the comparison. Share of targets with the brighter half on top, per source and class, train and val.
  B  flip detection: a small CNN is trained on train crops to tell a vertically flipped crop from an unflipped one
     (and, separately, horizontally flipped). Its val accuracy says whether the asymmetry is there and stable in val.
     Run on target crops and on strong-clutter crops (the brightest background box of the target's size).
  C  usefulness: the same CNN is trained to tell targets from strong clutter with (i) horizontal flips only, as the
     detector was, (ii) horizontal and vertical flips, (iii) all 8 orientations, and evaluated on val in the original
     orientation and vertically flipped. If (ii) and (iii) match (i) on the original orientation, the orientation cue
     is not needed at this level and sharing orientations costs nothing; if (i) is clearly better, the cue is useful.

Reading, fixed before running (C, recall with 1% of val clutter passing, original orientation): a loss of more than
2 points from (i) to (ii) on the pooled val targets means the cue matters at crop level; within 1 point means it does not.

  python tools/sardet100k/orientation_cue_check.py extract OUT
  python tools/sardet100k/orientation_cue_check.py analyse OUT
"""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_report import r3  # noqa: E402
from generalisation_gap import CROP, unit  # noqa: E402


def box_sum(integral, x0, y0, x1, y1):
    return integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]


def work(task):
    from PIL import Image
    path, boxes = task
    values = np.asarray(Image.open(path), np.float32)
    if values.ndim == 3:
        values = values.mean(2)
    height, width = values.shape
    integral = np.pad(values.astype(np.float64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    blocked = np.zeros((height, width), bool)
    rects = []
    for x, y, w, h in boxes:
        x0, y0 = min(max(int(x), 0), width - 1), min(max(int(y), 0), height - 1)
        x1, y1 = min(max(int(np.ceil(x + w)), x0 + 1), width), min(max(int(np.ceil(y + h)), y0 + 1), height)
        rects.append((x0, y0, x1, y1))
        blocked[y0:y1, x0:x1] = True
    crop = lambda r: unit(Image.fromarray(values[r[1]:r[3], r[0]:r[2]]).resize((CROP, CROP), Image.BILINEAR))  # noqa: E731
    pos, neg, stats = [], [], []
    taken = blocked.copy()
    for x0, y0, x1, y1 in rects:
        w, h = x1 - x0, y1 - y0
        pos.append(crop((x0, y0, x1, y1)))
        ym, xm = (y0 + y1) // 2, (x0 + x1) // 2
        ratio = lambda a, b: (a - b) / (a + b + 1e-6)  # noqa: E731
        # Compare means: for odd extents the bottom/right half has one more
        # pixel. Comparing sums falsely labels a uniform odd crop asymmetric.
        top_end, bottom_start = max(ym, y0 + 1), min(ym, y1 - 1)
        left_end, right_start = max(xm, x0 + 1), min(xm, x1 - 1)
        top = box_sum(integral, x0, y0, x1, top_end) / (w * (top_end - y0))
        bottom = box_sum(integral, x0, bottom_start, x1, y1) / (w * (y1 - bottom_start))
        left = box_sum(integral, x0, y0, left_end, y1) / (h * (left_end - x0))
        right = box_sum(integral, right_start, y0, x1, y1) / (h * (x1 - right_start))
        band_v, band_h = max(4, h // 2), max(4, w // 2)
        above = values[max(y0 - band_v, 0):y0, x0:x1]
        below = values[y1:min(y1 + band_v, height), x0:x1]
        west = values[y0:y1, max(x0 - band_h, 0):x0]
        east = values[y0:y1, x1:min(x1 + band_h, width)]
        ok = lambda a, b: a.size > 0 and b.size > 0  # noqa: E731
        stats.append((ratio(top, bottom), ratio(left, right), ratio(above.mean(), below.mean()) if ok(above, below) else np.nan,
                      ratio(west.mean(), east.mean()) if ok(west, east) else np.nan))
        # strong clutter: the brightest box of this size whose pixels are all outside every GT and earlier pick
        best = None
        if h < height and w < width:
            mean = (integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]) / (h * w)
            occupied = np.pad(taken.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
            free = (occupied[h:, w:] - occupied[:-h, w:] - occupied[h:, :-w] + occupied[:-h, :-w]) == 0
            if free.any():
                k = int(np.argmax(np.where(free, mean, -1)))
                by, bx = divmod(k, mean.shape[1])
                best = (bx, by, bx + w, by + h)
        if best is None:
            neg.append(np.full(CROP * CROP, np.nan, np.float32))
        else:
            neg.append(crop(best))
            taken[best[1]:best[3], best[0]:best[2]] = True
    empty = np.zeros((0, CROP * CROP), np.float32)
    return (np.stack(pos) if pos else empty, np.stack(neg) if neg else empty, np.array(stats, np.float32).reshape(-1, 4))


def extract(args):
    from error_gallery import ROOT
    for split in ('train', 'val'):
        data = json.load(open(ROOT / f'Annotations/{split}.json'))
        names = {c['id']: c['name'].lower() for c in data['categories']}
        by_image = {}
        for a in data['annotations']:
            by_image.setdefault(a['image_id'], []).append(a)
        images = [i for i in data['images'] if i['id'] in by_image]
        tasks = [(str(ROOT / f'JPEGImages/{split}' / i['file_name']), [a['bbox'] for a in by_image[i['id']]]) for i in images]
        with mp.Pool(32) as pool:
            parts = pool.map(work, tasks, chunksize=64)

        def group(i):
            if i['width'] == 256:
                return 'ogsod' if i['file_name'].endswith('.png') else '256jpg'
            return str(i['width']) if i['width'] in (224, 512, 800) else 'misc'
        cell = np.array([f"{group(i)}/{names[a['category_id']]}" for i in images for a in by_image[i['id']]])
        np.savez_compressed(args.out / f'crops_{split}.npz', pos=np.concatenate([p[0] for p in parts]), neg=np.concatenate([p[1] for p in parts]),
                            stats=np.concatenate([p[2] for p in parts]), cell=cell, statistics_version=np.array('half_means_v2'))
        print(split, 'targets', len(cell), flush=True)


def flip(x, kind):
    """x: (N, 1, H, W) tensor; kind 0..7 over the 8 orientations (bit 0 transpose, bit 1 width flip, bit 2 height flip)."""
    if kind & 1:
        x = x.transpose(-1, -2)
    if kind & 2:
        x = x.flip(-1)
    if kind & 4:
        x = x.flip(-2)
    return x


class MiniFusion:
    """The E013 mechanism in miniature: one trunk encodes the crop and its flipped views, the feature maps are flipped
    back, a gate gives each channel group a softmax weight over the views, the fused map is classified."""

    @staticmethod
    def build(views, width=1, groups=8):
        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                c = [32 * width, 64 * width, 128 * width]
                self.trunk = nn.Sequential(nn.Conv2d(1, c[0], 3, padding=1), nn.BatchNorm2d(c[0]), nn.ReLU(), nn.MaxPool2d(2),
                                           nn.Conv2d(c[0], c[1], 3, padding=1), nn.BatchNorm2d(c[1]), nn.ReLU(), nn.MaxPool2d(2),
                                           nn.Conv2d(c[1], c[2], 3, padding=1), nn.BatchNorm2d(c[2]), nn.ReLU())
                self.views, self.groups, self.c = views, groups, c[2]
                k = len(views) + 1
                if views:
                    self.gate = nn.Sequential(nn.Conv2d(k * c[2], c[2] // 4, 1), nn.ReLU(), nn.Conv2d(c[2] // 4, k * groups, 1))
                    nn.init.zeros_(self.gate[-1].weight)
                    with torch.no_grad():
                        self.gate[-1].bias.zero_()
                        self.gate[-1].bias[:groups] = 2.
                self.head = nn.Sequential(nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Linear(4 * c[2], 128), nn.ReLU(), nn.Linear(128, 1))

            def forward(self, x):
                f = self.trunk(x)
                if self.views:
                    parts = [f] + [flip(self.trunk(flip(x, v)), v) for v in self.views]   # flips are their own inverse
                    b, c, h, w = f.shape
                    k = len(parts)
                    weight = self.gate(torch.cat(parts, 1)).view(b, k, self.groups, 1, h, w).softmax(1)
                    f = (weight * torch.stack(parts, 1).view(b, k, self.groups, c // self.groups, h, w)).sum(1).view(b, c, h, w)
                return self.head(f)
        return Net()


def train_cnn(x, y, device, augment, epochs=6, seed=0, net=None):
    """augment: callable(batch) -> batch, applied on the fly."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    net = net.to(device) if net is not None else nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(2),
                        nn.Flatten(), nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    steps = epochs * (len(x) // 512 + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(len(x), device=device)
        for k in range(0, len(x), 512):
            idx = perm[k:k + 512]
            xb, yb = augment(x[idx], y[idx])
            loss = nn.functional.binary_cross_entropy_with_logits(net(xb)[:, 0], yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    net.eval()

    def score(z):
        with torch.no_grad():
            return torch.cat([net(z[k:k + 4096])[:, 0] for k in range(0, len(z), 4096)])
    return score


def analyse(args):
    import torch
    dev = args.device
    d = {s: dict(np.load(args.out / f'crops_{s}.npz')) for s in ('train', 'val')}
    report = dict(statistics={}, flip_detection={}, usefulness={},
                  statistics_version={s: str(d[s].get('statistics_version', 'legacy_half_sums_v1')) for s in d})
    cells = [c for c in np.unique(d['val']['cell']) if (d['val']['cell'] == c).sum() >= 100 and (d['train']['cell'] == c).sum() >= 300]

    # A
    labels = ['box: top brighter', 'box: left brighter', 'band above brighter', 'band left brighter']
    for c in cells:
        row = {}
        for split in ('train', 'val'):
            s = d[split]['stats'][d[split]['cell'] == c]
            row[split] = {lab: dict(share=r3(np.nanmean(s[:, k] > 0) * 100), mean=r3(np.nanmean(s[:, k])), n=int(np.isfinite(s[:, k]).sum())) for k, lab in enumerate(labels)}
        report['statistics'][c] = row
        print('A', f'{c:18s}', ' | '.join(f"{lab} {row['train'][lab]['share']:5.1f}/{row['val'][lab]['share']:5.1f}" for lab in labels), flush=True)

    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev).view(-1, 1, CROP, CROP)  # noqa: E731
    pos = {s: t(d[s]['pos']) for s in d}
    has_neg = {s: np.isfinite(d[s]['neg']).all(1) for s in d}
    neg = {s: t(d[s]['neg'][has_neg[s]]) for s in d}
    neg_cell = {s: d[s]['cell'][has_neg[s]] for s in d}

    # B
    for what, data, cell_of in (('targets', pos, {s: d[s]['cell'] for s in d}), ('strong clutter', neg, neg_cell)):
        for axis, bit in (('vertical', 4), ('horizontal', 2)):
            def augment(xb, yb, bit=bit):
                flipped = torch.rand(len(xb), device=dev) < .5
                return torch.where(flipped.view(-1, 1, 1, 1), flip(xb, bit), xb), flipped.float()
            score = train_cnn(data['train'], torch.zeros(len(data['train']), device=dev), dev, augment)
            plain, turned = score(data['val']).cpu().numpy(), score(flip(data['val'], bit)).cpu().numpy()
            table = {'all': r3(((plain < 0).mean() + (turned > 0).mean()) * 50)}
            for c in cells:
                m = cell_of['val'] == c
                if m.sum() >= 100:
                    table[c] = r3(((plain[m] < 0).mean() + (turned[m] > 0).mean()) * 50)
            report['flip_detection'][f'{what}, {axis}'] = table
            print('B', what, axis, table, flush=True)

    # C
    x = torch.cat([pos['train'], neg['train']])
    y = torch.cat([torch.ones(len(pos['train']), device=dev), torch.zeros(len(neg['train']), device=dev)])
    regimes = {'horizontal flips only': [0, 2], 'horizontal and vertical flips': [0, 2, 4, 6], 'all 8 orientations': list(range(8))}
    for name, kinds in regimes.items():
        kinds_t = torch.as_tensor(kinds, device=dev)

        def augment(xb, yb, kinds_t=kinds_t):
            pick = kinds_t[torch.randint(len(kinds_t), (len(xb),), device=dev)]
            out = xb.clone()
            for k in kinds_t.tolist():
                m = pick == k
                if m.any():
                    out[m] = flip(xb[m], k)
            return out, yb
        score = train_cnn(x, y, dev, augment)
        table = {}
        for view, bit in (('original', 0), ('vertically flipped', 4), ('transposed', 1)):
            sp, sn = score(flip(pos['val'], bit)).cpu().numpy(), score(flip(neg['val'], bit)).cpu().numpy()
            thr = np.quantile(sn, .99)
            row = {'all': r3((sp > thr).mean() * 100)}
            for c in cells:
                m = d['val']['cell'] == c
                row[c] = r3((sp[m] > thr).mean() * 100)
            table[view] = row
        report['usefulness'][name] = table
        print('C', f'{name:30s}', {v: table[v]['all'] for v in table}, flush=True)
        print('    original, per cell:', {c: table['original'][c] for c in cells}, flush=True)
    # D: the fusion structure itself, trained as the detector is (horizontal flips only), against single-view networks
    # of the same and of doubled width. Flips 4 = height flip, 6 = both flips (kept un-transposed, so maps stay aligned).
    kinds_t = torch.as_tensor([0, 2], device=dev)

    def hflip_only(xb, yb):
        m = torch.rand(len(xb), device=dev) < .5
        out = xb.clone()
        out[m] = flip(xb[m], 2)
        return out, yb
    report['structure'] = {}
    for name, make in (('single view', lambda: MiniFusion.build([])), ('single view, double width', lambda: MiniFusion.build([], width=2)),
                       ('image + vertical flip, fused', lambda: MiniFusion.build([4])), ('image + 3 flips, fused', lambda: MiniFusion.build([2, 4, 6]))):
        rows = []
        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            score = train_cnn(x, y, dev, hflip_only, seed=seed, net=make())
            sp, sn = score(pos['val']).cpu().numpy(), score(neg['val']).cpu().numpy()
            thr = np.quantile(sn, .99)
            rows.append({'all': (sp > thr).mean() * 100, **{c: (sp[d['val']['cell'] == c] > thr).mean() * 100 for c in cells}})
        table = {k: dict(mean=r3(np.mean([r[k] for r in rows])), runs=[r3(r[k]) for r in rows]) for k in rows[0]}
        report['structure'][name] = table
        print('D', f'{name:30s}', 'all', table['all'], {c: table[c]['mean'] for c in cells}, flush=True)
    (args.out / 'orientation_cue.json').write_text(json.dumps(report, indent=1))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('extract', 'analyse'):
        p = sub.add_parser(name)
        p.add_argument('out', type=Path)
        p.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    {'extract': extract, 'analyse': analyse}[args.command](args)


if __name__ == '__main__':
    main()
