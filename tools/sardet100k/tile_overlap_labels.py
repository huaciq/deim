"""D15.2: do the annotated 512-px aircraft also appear, unlabelled, in overlapping TRAIN tiles?

D15 (label_consistency_check.py) found that most high-score background detections on 512-px aircraft images are objects
annotated in another training tile. This tool asks the question from the annotation side and without a detector:
for every annotated aircraft of a query image, find the training tiles showing the same ground, and see whether the
aircraft is annotated there.

  1. Overlapping tiles: phase correlation of 4x-downsampled images between the query image and every 512x512 training
     image; the 16 best training tiles are kept as candidates, with the shift of the correlation peak.
  2. Appearance: the GT crop is compared with the candidate tile at the shifted position (both signs, +-6 px search) by
     normalised cross-correlation at full resolution. NCC >= .9 with the box fully inside the tile is an appearance.
  3. Label: the appearance is `labelled` if a GT of that training tile has IoU >= .5 with the shifted box.

Query sets: every val image and the D12 sample of train images that are 512x512 and contain an aircraft.

  python tools/sardet100k/tile_overlap_labels.py OUT --d12 D12 --val-pred V.npz
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

SEARCH = 6


def load_gray(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert('L'))


def whitened(images):
    import torch
    import torch.nn.functional as F
    out = []
    for k in range(0, len(images), 1024):
        x = F.avg_pool2d(images[k:k + 1024, None].float(), 4)[:, 0]
        f = torch.fft.rfft2(x - x.mean((1, 2), keepdim=True))
        out.append(f / f.abs().clamp(min=1e-6))
    return torch.cat(out)


def ncc_search(template, tile, x0, y0):
    """Best NCC of `template` (h, w) inside `tile` around (x0, y0), +-SEARCH px. Returns (ncc, x, y) or None if outside."""
    h, w = template.shape
    xa, ya, xb, yb = x0 - SEARCH, y0 - SEARCH, x0 + w + SEARCH, y0 + h + SEARCH
    if xa < 0 or ya < 0 or xb > tile.shape[1] or yb > tile.shape[0]:
        return None
    patches = tile[ya:yb, xa:xb].float().unfold(0, h, 1).unfold(1, w, 1).reshape(-1, h * w)
    patches = patches - patches.mean(1, keepdim=True)
    t = template.float().reshape(-1)
    t = t - t.mean()
    score = (patches @ t) / (patches.norm(dim=1) * t.norm()).clamp(min=1e-6)
    best = int(score.argmax())
    side = 2 * SEARCH + 1
    return float(score[best]), xa + best % side, ya + best // side


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    parser.add_argument('--d12', type=Path, required=True)
    parser.add_argument('--val-pred', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    from error_gallery import ROOT
    train = json.load(open(ROOT / 'Annotations/train.json'))
    val = json.load(open(ROOT / 'Annotations/val.json'))
    aircraft = next(c['id'] for c in train['categories'] if c['name'].lower() == 'aircraft')
    sim = np.load(args.d12 / 'similarity.npz')
    sim_of = {'val': dict(zip(sim['val_ann'].tolist(), sim['val_sim'].tolist())), 'train': dict(zip(sim['train_ann'].tolist(), sim['train_sim'].tolist()))}
    chosen = set(sim['train_images'].tolist())

    ref_images = [i for i in train['images'] if i['width'] == 512 and i['height'] == 512]
    with mp.Pool(32) as pool:
        ref = torch.from_numpy(np.stack(pool.map(load_gray, [str(ROOT / 'JPEGImages/train' / i['file_name']) for i in ref_images], chunksize=64))).to(args.device)
    ref_fft = whitened(ref)
    ref_id = np.array([i['id'] for i in ref_images])
    ref_index = {int(i): k for k, i in enumerate(ref_id)}
    ref_gt = defaultdict(list)
    for a in train['annotations']:
        if a['image_id'] in ref_index:
            ref_gt[a['image_id']].append([a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]])
    print('reference train tiles', len(ref_images), flush=True)

    with np.load(args.val_pred) as z:
        keep = z['score'] >= .3
        p_img, p_box = z['image_id'][keep], z['box'][keep].astype(np.float64)

    rows = []
    for split, data in (('val', val), ('train', train)):
        by_image = defaultdict(list)
        for a in data['annotations']:
            if a['category_id'] == aircraft:
                by_image[a['image_id']].append(a)
        todo = [i for i in data['images'] if i['width'] == 512 and i['height'] == 512 and i['id'] in by_image and (split == 'val' or i['id'] in chosen)]
        print(split, 'query images', len(todo), 'aircraft', sum(len(by_image[i['id']]) for i in todo), flush=True)
        for n, image in enumerate(todo):
            if split == 'train':
                q = ref[ref_index[image['id']]]
            else:
                q = torch.from_numpy(load_gray(str(ROOT / 'JPEGImages/val' / image['file_name']))).to(args.device)
            corr = torch.fft.irfft2(whitened(q[None]) * ref_fft.conj(), s=(128, 128)).flatten(1)
            peak, where = corr.max(1)
            if split == 'train':
                peak[ref_index[image['id']]] = -1
            top = peak.topk(16).indices.tolist()
            for a in by_image[image['id']]:
                x, y, w, h = (int(round(v)) for v in a['bbox'])
                x, y = max(x, 0), max(y, 0)
                w, h = min(w, 512 - x), min(h, 512 - y)
                seen = dict(ann=a['id'], split=split, sim=sim_of[split].get(a['id'], np.nan), size=max(a['bbox'][2], a['bbox'][3]), labelled=0, unlabelled=0, best_unlabelled=0.)
                if split == 'val':
                    sel = p_img == image['id']
                    gt = np.array([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]]])
                    seen['detected'] = bool(sel.any() and iou_xyxy(gt, p_box[sel]).max() >= .5)
                if w >= 8 and h >= 8 and float(q[y:y + h, x:x + w].float().std()) > 1:
                    template = q[y:y + h, x:x + w]
                    for k in top:
                        dy, dx = divmod(int(where[k]), 128)
                        dy, dx = (dy - 128 if dy >= 64 else dy) * 4, (dx - 128 if dx >= 64 else dx) * 4
                        hits = [r for r in (ncc_search(template, ref[k], x + s * dx, y + s * dy) for s in (1, -1)) if r is not None]
                        if not hits:
                            continue
                        score, bx, by = max(hits)
                        if score < .9:
                            continue
                        gts = np.array(ref_gt[int(ref_id[k])], np.float64).reshape(-1, 4)
                        labelled = len(gts) and iou_xyxy(np.array([[bx, by, bx + w, by + h]], np.float64), gts).max() >= .5
                        seen['labelled' if labelled else 'unlabelled'] += 1
                        if not labelled:
                            seen['best_unlabelled'] = max(seen['best_unlabelled'], score)
                rows.append(seen)
            if n % 300 == 0:
                print(split, n, len(todo), flush=True)
    json.dump(rows, open(args.out / 'appearances.json', 'w'))

    def summary(sel):
        if not sel:
            return None
        lab, unl = np.array([r['labelled'] for r in sel]), np.array([r['unlabelled'] for r in sel])
        return dict(n=len(sel), in_another_train_tile=r3(((lab + unl) > 0).mean() * 100), labelled_somewhere=r3((lab > 0).mean() * 100),
                    unlabelled_somewhere=r3((unl > 0).mean() * 100), only_unlabelled=r3(((unl > 0) & (lab == 0)).mean() * 100),
                    appearances=int((lab + unl).sum()), unlabelled_share_of_appearances=r3(unl.sum() / max((lab + unl).sum(), 1) * 100))
    report = {}
    for split in ('val', 'train'):
        part = [r for r in rows if r['split'] == split]
        report[split] = dict(all=summary(part), seen=summary([r for r in part if r['sim'] >= .9]), novel=summary([r for r in part if r['sim'] < .9]))
        if split == 'val':
            report[split]['novel_detected'] = summary([r for r in part if r['sim'] < .9 and r['detected']])
            report[split]['novel_missed'] = summary([r for r in part if r['sim'] < .9 and not r['detected']])
            report[split]['seen_missed'] = summary([r for r in part if r['sim'] >= .9 and not r['detected']])
            for name, test in (('only_unlabelled', lambda r: r['unlabelled'] > 0 and r['labelled'] == 0), ('labelled_somewhere', lambda r: r['labelled'] > 0),
                               ('in_no_train_tile', lambda r: r['labelled'] + r['unlabelled'] == 0)):
                sel = [r for r in part if test(r)]
                report[split][f'recall_{name}'] = dict(n=len(sel), recall=r3(np.mean([r['detected'] for r in sel]) * 100) if sel else None)
    (args.out / 'tile_overlap_labels.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1), flush=True)


if __name__ == '__main__':
    main()
