"""D29: can SAR scatterer peaks supply candidate positions for the tiny targets the encoder cannot cover?

D26 found 245 val GT for which none of the 8,400 encoder positions (stride 8 / 16 / 32 at 640) yields an initial box
with IoU >= .5, almost all OGSOD tank and bridge shorter than 16 input pixels; tank AP below 16 px is 28.7 against
88 above. A dense stride-4 level would add 25,600 tokens per image. A SAR-specific alternative is to add stride-4
tokens only where the image has a point-scatterer peak, the classical CFAR cue: a local maximum that stands out from
its clutter ring. This tool asks whether such peaks land on those targets, at what budget and how selectively.

Peaks are found on the original image (grayscale, native resolution): local maxima of the 3x3 mean, scored by the
CFAR ratio  mean3x3 / (mean of the 15x15 window outside the central 5x5 + 1). The top K by ratio are kept and
mapped to the 640 input, one per stride-4 cell. A GT is hit when a kept peak lies inside its box grown by 1 pixel.
Compared against one random same-size background box per GT (IoU < .1 with every GT) and the detector's own
background false positives (score >= .1, IoU < .1).

Reading, fixed before running:
  supported   at K = 256 the peaks hit >= 90% of GT with input long side < 16, >= 80% of the GT no encoder position
              covers (D26), and the best-peak ratio separates tiny GT from random background with AUC >= .8.
              Then peak-guided stride-4 tokens are a feasible, SAR-specific candidate source.
  refuted     hit rate below 75% for tiny GT at K = 256: the peaks miss what the encoder misses, and only a dense
              stride-4 level (a standard structure) remains.
  It does not say whether the decoder would then detect or localise those targets; only training can.

  python tools/sardet100k/peak_token_coverage.py OUT --d26 D26 --d12 D12 --pred V.npz [--workers 32]
"""
import argparse
import json
import multiprocessing as mp
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_report import iou_xyxy, r3  # noqa: E402

BUDGETS = (64, 128, 256, 512, 1024)
SIZE, STRIDE = 640, 4
STATE = {}


def peaks(path):
    from PIL import Image
    from scipy import ndimage
    v = np.asarray(Image.open(path).convert('L'), np.float32)
    m3 = ndimage.uniform_filter(v, 3, mode='reflect')
    s15 = ndimage.uniform_filter(v, 15, mode='reflect') * 225
    s5 = ndimage.uniform_filter(v, 5, mode='reflect') * 25
    ring = (s15 - s5) / 200
    ratio = m3 / (ring + 1)
    is_max = (m3 == ndimage.maximum_filter(m3, 3, mode='reflect')) & (m3 > 0)
    y, x = np.nonzero(is_max)
    r = ratio[y, x]
    h, w = v.shape
    cell = (np.floor(y * SIZE / h / STRIDE) * (SIZE // STRIDE) + np.floor(x * SIZE / w / STRIDE)).astype(np.int64)   # EvalImages resizes to 640 x 640
    order = np.argsort(-r, kind='stable')
    _, first = np.unique(cell[order], return_index=True)           # best peak per stride-4 cell
    keep = order[np.sort(first)]
    keep = keep[np.argsort(-r[keep], kind='stable')]
    return x[keep].astype(np.float32), y[keep].astype(np.float32), r[keep].astype(np.float32)


def per_image(image_id):
    info = STATE['info'][image_id]
    px, py, pr = peaks(STATE['root'] / info['file_name'])
    out = []
    for kind, boxes, ids in STATE['boxes'][image_id]:
        for b, i in zip(boxes, ids):
            inside = (px >= b[0] - 1) & (px <= b[2] + 1) & (py >= b[1] - 1) & (py <= b[3] + 1)
            rank = int(np.flatnonzero(inside)[0]) if inside.any() else 1 << 30   # peaks are sorted by ratio
            best = float(pr[inside].max()) if inside.any() else 0.
            out.append((kind, i, rank, best))
    return image_id, len(px), out


def random_boxes(gt, w, h, rng):
    out = []
    for b in gt:
        bw, bh = b[2] - b[0], b[3] - b[1]
        for _ in range(50):
            x0, y0 = rng.uniform(0, max(w - bw, 1)), rng.uniform(0, max(h - bh, 1))
            c = np.array([[x0, y0, x0 + bw, y0 + bh]])
            if iou_xyxy(c, gt).max() < .1:
                out.append(c[0])
                break
    return np.array(out).reshape(-1, 4)


def auc(pos, neg):
    s = np.concatenate([pos, neg])
    rank = np.argsort(np.argsort(s, kind='mergesort'), kind='mergesort') + 1
    return float((rank[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))) if len(pos) and len(neg) else float('nan')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    parser.add_argument('--d26', type=Path, required=True)
    parser.add_argument('--d12', type=Path, required=True)
    parser.add_argument('--pred', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    from error_gallery import ROOT
    from novel_box_anatomy import cells_of
    val = json.load(open(ROOT / 'Annotations/val.json'))
    names, cell_of = cells_of(val)
    info = {i['id']: i for i in val['images']}
    sim = np.load(args.d12 / 'similarity.npz')
    novel_of = {int(a): bool(s < .9) for a, s in zip(sim['val_ann'], sim['val_sim'])}
    rows = np.load(args.d26 / 'gt_rows.npy')
    no_position = {int(r[0]): bool(r[2] < .5) for r in rows}           # best IoU over all 8,400 positions
    gts = defaultdict(list)
    for a in val['annotations']:
        x, y, w, h = a['bbox']
        gts[a['image_id']].append(([x, y, x + w, y + h], a['id'], a['category_id']))
    with np.load(args.pred) as z:
        order = np.argsort(z['image_id'], kind='stable')
        P = {k: z[k][order] for k in ('image_id', 'label', 'score', 'box')}
    ids, first = np.unique(P['image_id'], return_index=True)
    bounds = dict(zip(ids.tolist(), zip(first.tolist(), list(first[1:]) + [len(P['image_id'])])))
    boxes, missed = {}, {}
    for image_id, im in info.items():
        g = np.array([t[0] for t in gts[image_id]], np.float64).reshape(-1, 4)
        s, e = bounds.get(image_id, (0, 0))
        pb, ps, pl = P['box'][s:e].astype(np.float64), P['score'][s:e], P['label'][s:e]
        entries = []
        if len(g):
            gid = [t[1] for t in gts[image_id]]
            entries.append(('gt', g, gid))
            same = iou_xyxy(pb, g) * (pl[:, None] == np.array([t[2] for t in gts[image_id]])[None]) if len(pb) else np.zeros((0, len(g)))
            for j, a in enumerate(gid):
                missed[a] = not bool(((same[:, j] >= .5) & (ps >= .3)).any()) if len(pb) else True
            entries.append(('random', random_boxes(g, im['width'], im['height'], np.random.default_rng(image_id)), [f'r{a}' for a in gid]))
        bg = (ps >= .1) & ((iou_xyxy(pb, g).max(1) < .1) if len(g) and len(pb) else np.ones(len(pb), bool))
        entries.append(('clutter', pb[bg], [f'c{image_id}_{k}' for k in range(int(bg.sum()))]))
        boxes[image_id] = entries
    STATE.update(info=info, root=ROOT / 'JPEGImages/val', boxes=boxes)
    with mp.get_context('fork').Pool(args.workers) as pool:
        result = pool.map(per_image, sorted(info), chunksize=16)
    n_peaks = np.array([r[1] for r in result])
    rec = defaultdict(list)
    for image_id, _, out in result:
        im = info[image_id]
        for kind, i, rank, best in out:
            rec[kind].append((i, rank, best, image_id))
    long_side = {}
    for image_id, entries in boxes.items():
        im = info[image_id]
        for kind, b, i in entries:
            for bb, ii in zip(b, i):
                long_side[ii] = max(bb[2] - bb[0], bb[3] - bb[1]) * SIZE / max(im['width'], im['height'])
    gt_rows = rec['gt']
    gid = [r[0] for r in gt_rows]
    rank = np.array([r[1] for r in gt_rows])
    best = np.array([r[2] for r in gt_rows])
    ls = np.array([long_side[i] for i in gid])
    cell = np.array([cell_of[i] for i in gid])
    nov = np.array([novel_of[i] for i in gid])
    nopos = np.array([no_position.get(i, False) for i in gid])
    miss = np.array([missed[i] for i in gid])
    r_rank = np.array([r[1] for r in rec['random']])
    r_best = np.array([r[2] for r in rec['random']])
    r_ls = np.array([long_side[r[0]] for r in rec['random']])
    c_best = np.array([r[2] for r in rec['clutter']])
    c_ls = np.array([long_side[r[0]] for r in rec['clutter']])
    report = dict(peaks_per_image=dict(mean=r3(n_peaks.mean()), median=r3(np.median(n_peaks)), p90=r3(np.percentile(n_peaks, 90))))
    print('distinct stride-4 peak cells per image', report['peaks_per_image'], flush=True)
    groups = [('all GT', np.ones(len(gid), bool)), ('input long side < 16', ls < 16), ('16-32', (ls >= 16) & (ls < 32)), ('>= 32', ls >= 32),
              ('no encoder position covers (D26)', nopos), ('final miss', miss), ('final miss, long side < 16', miss & (ls < 16)),
              ('novel, long side < 16', nov & (ls < 16))] + [(c, cell == c) for c in np.unique(cell) if (cell == c).sum() >= 300]
    report['hit'] = {}
    for label, m in groups:
        row = dict(n=int(m.sum()), **{f'K{k}': r3((rank[m] < k).mean() * 100) for k in BUDGETS}, any=r3((rank[m] < (1 << 29)).mean() * 100))
        report['hit'][label] = row
        print(f'{label:34s}', row, flush=True)
    tiny_r = r_ls < 16
    report['random_background_hit'] = {'input long side < 16': {f'K{k}': r3((r_rank[tiny_r] < k).mean() * 100) for k in BUDGETS},
                                       'all': {f'K{k}': r3((r_rank < k).mean() * 100) for k in BUDGETS}}
    report['auc_best_ratio'] = {'tiny GT vs tiny random background': r3(auc(best[ls < 16], r_best[tiny_r])),
                                'tiny GT vs tiny detector clutter': r3(auc(best[ls < 16], c_best[c_ls < 16])),
                                'all GT vs random background': r3(auc(best, r_best))}
    print('random background hit', report['random_background_hit'], flush=True)
    print('AUC of best peak ratio', report['auc_best_ratio'], flush=True)
    (args.out / 'peak_token_coverage.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
