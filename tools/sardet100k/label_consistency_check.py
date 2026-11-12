"""D15: are the detector's background false positives targets that are annotated in another, overlapping training image?

SARDet-100K tiles overlap (D12): the same ground appears in several images. A detection that lies on no GT of its own
image, but whose pixels correlate >= .9 with a GT crop of another TRAIN image of the same size, shows the same object
labelled there and unlabelled here. That is an annotation inconsistency which needs no human judgement. The share found
this way is a lower bound of the missing-label share: an unlabelled object that is unlabelled in every tile cannot be found.

Queries, per split (12,000 sampled train images of D12, all val images):
  fp      detections with score >= .1 and IoU < .1 with every GT of the image (at most 20 per image, by score);
          `inside` marks those lying mostly inside a GT (intersection / detection area >= .5), which are part errors
  random  one GT-sized box per GT with zero overlap with every GT (control: how often does plain background match?)
Reference: the GT crops of all train images. Matching is D12's (24x24 crop, zero mean, unit norm, long side within 10%,
other image); each query is tried at scale .9, 1 and 1.1 because a detection is not placed exactly like the annotation.

  python tools/sardet100k/label_consistency_check.py match OUT --name E000 --train-pred T.npz --val-pred V.npz --d12 D12
  python tools/sardet100k/label_consistency_check.py gallery OUT --name E000 [--split train]
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
from generalisation_gap import encode, load_split  # noqa: E402
from generalisation_probe import cell_name, negative_boxes  # noqa: E402

SCALES = (.9, 1., 1.1)
SCORE_BINS = [(.1, .3), (.3, .5), (.5, 1.01)]


def best_match_index(query, q_size, q_image, reference, r_size, r_image, device, chunk=2048):
    import torch
    reference = torch.as_tensor(reference, device=device)
    r_size = torch.as_tensor(r_size, device=device)
    r_image = torch.as_tensor(r_image, device=device)
    best, index = [], []
    for start in range(0, len(query), chunk):
        part = torch.as_tensor(query[start:start + chunk], device=device)
        scores = part @ reference.T
        ratio = torch.as_tensor(q_size[start:start + chunk], device=device)[:, None] / r_size[None].clamp(min=1e-3)
        scores[(ratio - 1).abs() > .1] = -1
        scores[torch.as_tensor(q_image[start:start + chunk], device=device)[:, None] == r_image[None]] = -1
        value, where = scores.max(1)
        best.append(value.cpu().numpy())
        index.append(where.cpu().numpy())
    return np.concatenate(best), np.concatenate(index)


def queries(root, split, data, pred_path, image_ids):
    with np.load(pred_path) as z:
        keep = z['score'] >= .1
        p_img, p_score, p_box, p_label = z['image_id'][keep], z['score'][keep], z['box'][keep].astype(np.float64), z['label'][keep]
    order = np.argsort(p_img, kind='stable')
    p_img, p_score, p_box, p_label = p_img[order], p_score[order], p_box[order], p_label[order]
    by_image = defaultdict(list)
    for a in data['annotations']:
        by_image[a['image_id']].append(a)
    rows = defaultdict(list)
    tasks = []
    for image in data['images']:
        if image_ids is not None and image['id'] not in image_ids:
            continue
        anns = by_image[image['id']]
        w, h = image['width'], image['height']
        gt = np.array([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]] for a in anns], np.float64).reshape(-1, 4)
        s, e = np.searchsorted(p_img, image['id'], 'left'), np.searchsorted(p_img, image['id'], 'right')
        box, score, label = p_box[s:e], p_score[s:e], p_label[s:e]
        box = np.clip(box, 0, [w, h, w, h])
        ok = (box[:, 2] - box[:, 0] >= 2) & (box[:, 3] - box[:, 1] >= 2)
        inside = np.zeros(len(box))
        if len(gt) and len(box):
            ok &= iou_xyxy(box, gt).max(1) < .1
            lt, rb = np.maximum(box[:, None, :2], gt[None, :, :2]), np.minimum(box[:, None, 2:], gt[None, :, 2:])
            inter = np.clip(rb - lt, 0, None).prod(-1)
            inside = (inter / np.clip((box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]), 1e-6, None)[:, None]).max(1)
        idx = np.flatnonzero(ok)
        idx = idx[np.argsort(-score[idx], kind='stable')][:20]
        rand = np.zeros((0, 4))
        if len(gt):
            rand = negative_boxes(np.clip(gt / np.array([w, h, w, h]), 0, 1), np.random.default_rng(image['id'])) * np.array([w, h, w, h])
        boxes = np.concatenate([box[idx], rand])
        if not len(boxes):
            continue
        rows['box'].append(boxes)
        rows['kind'].append(np.r_[np.zeros(len(idx), np.int8), np.ones(len(rand), np.int8)])
        rows['score'].append(np.r_[score[idx], np.full(len(rand), -1.)])
        rows['label'].append(np.r_[label[idx], np.full(len(rand), -1)])
        rows['inside'].append(np.r_[inside[idx], np.zeros(len(rand))])
        rows['image'].append(np.full(len(boxes), image['id']))
        rows['width'].append(np.full(len(boxes), w))
        rows['n_gt'].append(np.full(len(boxes), len(gt)))
        path = str(root / f'JPEGImages/{split}' / image['file_name'])
        for scale in SCALES:
            cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
            bw, bh = (boxes[:, 2] - boxes[:, 0]) * scale, (boxes[:, 3] - boxes[:, 1]) * scale
            tasks.append((path, np.stack([cx - bw / 2, cy - bh / 2, bw, bh], 1).tolist()))
    with mp.Pool(32) as pool:
        crops = pool.map(encode, tasks, chunksize=64)
    out = {k: np.concatenate(v) for k, v in rows.items()}
    # tasks are image-major, scale-minor: regroup to (scale, query)
    out['crops'] = [np.concatenate(crops[k::len(SCALES)]) for k in range(len(SCALES))]
    return out


def match(args):
    from error_gallery import ROOT
    train, ref, ref_anns = load_split(ROOT, 'train')
    print('reference GT crops', len(ref), flush=True)
    r_size = np.array([max(a['bbox'][2], a['bbox'][3]) for a in ref_anns], np.float32)
    r_image = np.array([a['image_id'] for a in ref_anns], np.int64)
    r_label = np.array([a['category_id'] for a in ref_anns])
    r_id = np.array([a['id'] for a in ref_anns])
    names = {c['id']: c['name'] for c in train['categories']}
    chosen = set(np.load(args.d12 / 'similarity.npz')['train_images'].tolist())
    report = {}
    for split, pred, ids in (('train', args.train_pred, chosen), ('val', args.val_pred, None)):
        data = train if split == 'train' else json.load(open(ROOT / f'Annotations/{split}.json'))
        q = queries(ROOT, split, data, pred, ids)
        size = np.maximum(q['box'][:, 2] - q['box'][:, 0], q['box'][:, 3] - q['box'][:, 1]).astype(np.float32)
        q_image = q['image'] if split == 'train' else -1 - q['image']
        sim, where = np.full(len(size), -1., np.float32), np.zeros(len(size), np.int64)
        for scale, crops in zip(SCALES, q.pop('crops')):
            s, w = best_match_index(crops, size * scale, q_image, ref, r_size, r_image, args.device)
            better = s > sim
            sim[better], where[better] = s[better], w[better]
        q.update(sim=sim, ref_ann=r_id[where], ref_label=r_label[where], ref_image=r_image[where])
        np.savez_compressed(args.out / f'{args.name}_{split}_queries.npz', **q)
        fp, rand = q['kind'] == 0, q['kind'] == 1
        n_images = len(ids) if ids is not None else len(data['images'])
        table = dict(images=n_images, gt=int(sum(1 for a in data['annotations'] if ids is None or a['image_id'] in ids)),
                     random=dict(n=int(rand.sum()), twin=r3((sim[rand] >= .9).mean() * 100)), bins={}, cells={})
        outside = fp & (q['inside'] < .5)
        for lo, hi in SCORE_BINS:
            m = fp & (q['score'] >= lo) & (q['score'] < hi)
            mo = m & outside
            table['bins'][f'{lo}-{hi}'] = dict(
                n=int(m.sum()), per_100_images=r3(m.sum() / n_images * 100), inside_gt=r3((q['inside'][m] >= .5).mean() * 100),
                outside_n=int(mo.sum()), twin=r3((sim[mo] >= .9).mean() * 100), twin_same_class=r3(((sim[mo] >= .9) & (q['ref_label'][mo] == q['label'][mo])).mean() * 100),
                twin_07=r3((sim[mo] >= .7).mean() * 100))
        strong = outside & (q['score'] >= .3)
        cell = np.array([cell_name(w, names[int(c)], s) if c >= 0 else '' for w, c, s in zip(q['width'], q['label'], size)])
        for c in np.unique(cell[strong]):
            m = strong & (cell == c)
            if m.sum() >= 30:
                table['cells'][c] = dict(n=int(m.sum()), twin=r3((sim[m] >= .9).mean() * 100), twin_07=r3((sim[m] >= .7).mean() * 100))
        report[split] = table
        print(split, json.dumps(table, ensure_ascii=False), flush=True)
    (args.out / f'{args.name}_label_consistency.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))


def context_corr(path_a, box_a, path_b, box_b, cache):
    """Correlation of the neighbourhoods of two boxes, over the largest window around both that lies inside both images.

    A small bright blob correlates with any other blob, so the box crop alone does not prove identity for small targets;
    the neighbourhood does. Returns (corr, window / box size); (-1, 0) when the common window is under 1.5 box sizes or 32 px.
    """
    from PIL import Image
    views = []
    for path in (path_a, path_b):
        if path not in cache:
            if len(cache) > 256:
                cache.clear()
            v = np.asarray(Image.open(path), np.float32)
            cache[path] = v.mean(2) if v.ndim == 3 else v
        views.append(cache[path])
    ca, cb = ((box_a[0] + box_a[2]) / 2, (box_a[1] + box_a[3]) / 2), ((box_b[0] + box_b[2]) / 2, (box_b[1] + box_b[3]) / 2)
    side = max(box_a[2] - box_a[0], box_a[3] - box_a[1])
    want = max(side * 2, 48)
    left = min(want, ca[0], cb[0])
    right = min(want, views[0].shape[1] - ca[0], views[1].shape[1] - cb[0])
    up = min(want, ca[1], cb[1])
    down = min(want, views[0].shape[0] - ca[1], views[1].shape[0] - cb[1])
    if min(left + right, up + down) < max(1.5 * side, 32):
        return -1., 0.
    crops = []
    for view, c in zip(views, (ca, cb)):
        x0, x1, y0, y1 = int(round(c[0] - left)), int(round(c[0] + right)), int(round(c[1] - up)), int(round(c[1] + down))
        crops.append(Image.fromarray(view[max(y0, 0):y1, max(x0, 0):x1]).resize((48, 48), Image.BILINEAR))
    u, v = (np.asarray(c, np.float32).ravel() for c in crops)
    u, v = u - u.mean(), v - v.mean()
    return float(u @ v / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-6)), float(min(left + right, up + down) / side)


def confirm(args):
    """Keep only the matches whose neighbourhoods also coincide (same ground seen in two tiles)."""
    from error_gallery import ROOT
    train = json.load(open(ROOT / 'Annotations/train.json'))
    names = {c['id']: c['name'] for c in train['categories']}
    train_file = {i['id']: i['file_name'] for i in train['images']}
    ann_of = {a['id']: a for a in train['annotations']}
    report = {}
    for split in ('train', 'val'):
        q = dict(np.load(args.out / f'{args.name}_{split}_queries.npz'))
        data = train if split == 'train' else json.load(open(ROOT / f'Annotations/{split}.json'))
        file_of = {i['id']: i['file_name'] for i in data['images']}
        ctx = np.full(len(q['sim']), -2., np.float32)
        cache = {}
        for k in np.flatnonzero(q['sim'] >= .9):
            a = ann_of[int(q['ref_ann'][k])]
            b = a['bbox']
            ctx[k], _ = context_corr(str(ROOT / f'JPEGImages/{split}' / file_of[int(q['image'][k])]), q['box'][k],
                                     str(ROOT / 'JPEGImages/train' / train_file[a['image_id']]), [b[0], b[1], b[0] + b[2], b[1] + b[3]], cache)
        q['ctx'] = ctx
        np.savez_compressed(args.out / f'{args.name}_{split}_queries.npz', **q)
        size = np.maximum(q['box'][:, 2] - q['box'][:, 0], q['box'][:, 3] - q['box'][:, 1])
        cell = np.array([cell_name(w, names[int(c)], s) if c >= 0 else '' for w, c, s in zip(q['width'], q['label'], size)])
        fp, rand = (q['kind'] == 0) & (q['inside'] < .5), q['kind'] == 1
        same = (ctx >= .8) & (q['ref_label'] == q['label'])
        table = dict(random=dict(n=int(rand.sum()), same_scene=r3((ctx[rand] >= .8).mean() * 100)), bins={}, cells={})
        for lo, hi in SCORE_BINS:
            m = fp & (q['score'] >= lo) & (q['score'] < hi)
            table['bins'][f'{lo}-{hi}'] = dict(n=int(m.sum()), crop_twin=r3((q['sim'][m] >= .9).mean() * 100), same_scene=r3((ctx[m] >= .8).mean() * 100),
                                               same_scene_and_class=r3(same[m].mean() * 100), untestable=r3(((q['sim'][m] >= .9) & (ctx[m] == -1)).mean() * 100))
        strong = fp & (q['score'] >= .3)
        for c in np.unique(cell[strong]):
            m = strong & (cell == c)
            if m.sum() >= 30:
                table['cells'][c] = dict(n=int(m.sum()), crop_twin=r3((q['sim'][m] >= .9).mean() * 100), same_scene=r3((ctx[m] >= .8).mean() * 100),
                                         same_scene_and_class=r3(same[m].mean() * 100), high_score_n=int((m & (q['score'] >= .5)).sum()),
                                         high_score_same_scene=r3((ctx[m & (q['score'] >= .5)] >= .8).mean() * 100) if (m & (q['score'] >= .5)).sum() else None)
        report[split] = table
        print(split, json.dumps(table, ensure_ascii=False), flush=True)
    (args.out / f'{args.name}_label_consistency_confirmed.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))


def gallery(args):
    """Contact sheet: left, the false positive in its image; right, the matching GT in the other training image."""
    from PIL import Image, ImageDraw
    from error_gallery import ROOT
    q = dict(np.load(args.out / f'{args.name}_{args.split}_queries.npz'))
    train = json.load(open(ROOT / 'Annotations/train.json'))
    data = train if args.split == 'train' else json.load(open(ROOT / f'Annotations/{args.split}.json'))
    file_of = {i['id']: i['file_name'] for i in data['images']}
    train_file = {i['id']: i['file_name'] for i in train['images']}
    ann_of = {a['id']: a for a in train['annotations']}
    gt_of = defaultdict(list)
    for a in data['annotations']:
        gt_of[a['image_id']].append(a['bbox'])
    lo, hi = args.sim
    keep = (q['kind'] == 0) & (q['inside'] < .5) & (q['score'] >= args.score) & (q['sim'] >= lo) & (q['sim'] < hi)
    if args.width:
        keep &= q['width'] == args.width
    pick = np.random.default_rng(0).permutation(np.flatnonzero(keep))[:args.count]
    tile, cols = 160, 4

    def view(path, box, others, colour):
        image = Image.open(path).convert('RGB')
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        half = max(box[2] - box[0], box[3] - box[1], 24) * 1.5
        region = (cx - half, cy - half, cx + half, cy + half)
        k = tile / (2 * half)
        out = image.crop(tuple(int(round(v)) for v in region)).resize((tile, tile), Image.BILINEAR)
        draw = ImageDraw.Draw(out)
        for x, y, w, h in others:
            draw.rectangle([(x - region[0]) * k, (y - region[1]) * k, (x + w - region[0]) * k, (y + h - region[1]) * k], outline=(0, 200, 0))
        draw.rectangle([(box[0] - region[0]) * k, (box[1] - region[1]) * k, (box[2] - region[0]) * k, (box[3] - region[1]) * k], outline=colour)
        return out
    rows = (len(pick) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * (2 * tile + 12), rows * (tile + 16)), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    for n, k in enumerate(pick):
        x0, y0 = (n % cols) * (2 * tile + 12), (n // cols) * (tile + 16)
        sheet.paste(view(ROOT / f'JPEGImages/{args.split}' / file_of[int(q['image'][k])], q['box'][k], gt_of[int(q['image'][k])], (255, 60, 60)), (x0, y0 + 14))
        a = ann_of[int(q['ref_ann'][k])]
        b = a['bbox']
        sheet.paste(view(ROOT / 'JPEGImages/train' / train_file[a['image_id']], [b[0], b[1], b[0] + b[2], b[1] + b[3]], [], (0, 200, 0)), (x0 + tile + 2, y0 + 14))
        draw.text((x0 + 2, y0 + 1), f"score {q['score'][k]:.2f} sim {q['sim'][k]:.2f} pred {int(q['label'][k])} gt {int(q['ref_label'][k])}", fill=(255, 255, 255))
    path = args.out / f'{args.name}_{args.split}_w{args.width}_sim{lo}-{hi}.png'
    sheet.save(path)
    print('written', path, 'candidates', int(keep.sum()))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    m = sub.add_parser('match')
    m.add_argument('out', type=Path)
    m.add_argument('--name', required=True)
    m.add_argument('--train-pred', type=Path, required=True)
    m.add_argument('--val-pred', type=Path, required=True)
    m.add_argument('--d12', type=Path, required=True)
    m.add_argument('--device', default='cuda:0')
    g = sub.add_parser('gallery')
    g.add_argument('out', type=Path)
    g.add_argument('--name', required=True)
    g.add_argument('--split', default='train')
    g.add_argument('--score', type=float, default=.3)
    g.add_argument('--sim', type=float, nargs=2, default=[.9, 1.01])
    g.add_argument('--count', type=int, default=24)
    g.add_argument('--width', type=int, default=0, help='only images of this width (source)')
    c = sub.add_parser('confirm')
    c.add_argument('out', type=Path)
    c.add_argument('--name', required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    {'match': match, 'gallery': gallery, 'confirm': confirm}[args.command](args)


if __name__ == '__main__':
    main()
