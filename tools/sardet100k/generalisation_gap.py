"""D12: is the remaining detection loss a generalisation problem? No training, no model change; val and train only.

Two measurements, the same for every run:
  1. Train versus val. The same checkpoint detects a random subset of TRAIN images and all VAL images. A cell whose
     recall is high on train and lower on val loses by generalisation, not by capacity or by image evidence.
  2. Similarity to the training set. Every GT gets `sim` = the highest correlation of its 24x24 crop with the crop of
     any TRAIN GT of similar size (long-side ratio within 10 %) from a DIFFERENT image; this is the "seen" measure of
     verify_metrics.py (seen = sim >= .9) kept as a continuous value. Recall is then read per similarity bin, inside
     one source x class cell, so that class and source cannot explain the trend.

Cells: val images carry the pixel-matched source group; train images have no source map, so cells that must be
comparable between train and val are named by image size and class (e.g. 512/aircraft), and tanks in 256-px images
are split by size (long side < 16 px = OGSOD-like, otherwise MSAR-like; 139/140 correct on val).

Detection at the operating point: score >= .3, IoU >= .5, greedy by score per image and class. A missed GT is
`no_box` when no box of its class reaches IoU .5 at any score, `low_score` otherwise.

Commands (server; OUT directly under E000's independent_verify_20261106/):
  similarity OUT [--device cuda:0] [--train-images 12000]     OUT/similarity.npz
  predict RUN OUT --name E008 [--device cuda:0]               OUT/train_predictions_<name>.npz
  analyse OUT --runs E000=VAL_NPZ E008=VAL_NPZ ...            OUT/generalisation_gap.json
  ap OUT --runs E000=VAL_NPZ ...                              OUT/ap_by_similarity.json (COCO AP per similarity range)
"""
import argparse
import json
import multiprocessing as mp
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_report import group_of, iou_xyxy, r3  # noqa: E402

CROP = 24
SIM_BINS = [(-1.01, .5), (.5, .7), (.7, .9), (.9, 1.01)]
SIZE_BINS = [(0, 16), (16, 32), (32, 64), (64, 128), (128, 1e9)]


def unit(array):
    vector = np.asarray(array, np.float32).ravel()
    vector = vector - vector.mean()
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def encode(task):
    from PIL import Image
    path, boxes = task
    values = np.asarray(Image.open(path), np.float32)
    if values.ndim == 3:
        values = values.mean(2)
    height, width = values.shape
    crops = []
    for x, y, w, h in boxes:
        x0, y0 = min(max(int(x), 0), width - 1), min(max(int(y), 0), height - 1)
        x1, y1 = min(max(int(np.ceil(x + w)), x0 + 1), width), min(max(int(np.ceil(y + h)), y0 + 1), height)
        crops.append(unit(Image.fromarray(values[y0:y1, x0:x1]).resize((CROP, CROP), Image.BILINEAR)))
    return np.stack(crops) if crops else np.zeros((0, CROP * CROP), np.float32)


def load_split(root, split, image_ids=None):
    data = json.load(open(root / f'Annotations/{split}.json'))
    by_image = defaultdict(list)
    for a in data['annotations']:
        by_image[a['image_id']].append(a)
    images = [i for i in data['images'] if image_ids is None or i['id'] in image_ids]
    tasks = [(str(root / f'JPEGImages/{split}' / i['file_name']), [a['bbox'] for a in by_image[i['id']]]) for i in images]
    with mp.Pool(32) as pool:
        crops = np.concatenate(pool.map(encode, tasks, chunksize=64))
    anns = [a for i in images for a in by_image[i['id']]]
    return data, crops, anns


def best_match(query, q_size, q_image, reference, r_size, r_image, device, chunk=2048):
    """Best correlation with a reference crop of similar size from another image."""
    import torch
    reference = torch.as_tensor(reference, device=device)
    r_size = torch.as_tensor(r_size, device=device)
    r_image = torch.as_tensor(r_image, device=device)
    best = []
    for start in range(0, len(query), chunk):
        part = torch.as_tensor(query[start:start + chunk], device=device)
        scores = part @ reference.T
        ratio = torch.as_tensor(q_size[start:start + chunk], device=device)[:, None] / r_size[None].clamp(min=1e-3)
        scores[(ratio - 1).abs() > .1] = -1
        scores[torch.as_tensor(q_image[start:start + chunk], device=device)[:, None] == r_image[None]] = -1
        best.append(scores.max(1).values.cpu().numpy())
    return np.concatenate(best)


def similarity(args):
    from error_gallery import ROOT
    train, train_crops, train_anns = load_split(ROOT, 'train')
    print('train crops', len(train_crops), flush=True)
    val, val_crops, val_anns = load_split(ROOT, 'val')
    print('val crops', len(val_crops), flush=True)
    size = lambda anns: np.array([max(a['bbox'][2], a['bbox'][3]) for a in anns], np.float32)  # noqa: E731
    # Train image ids are offset so that a val image never equals a train image id.
    t_image = np.array([a['image_id'] for a in train_anns], np.int64)
    v_image = -1 - np.array([a['image_id'] for a in val_anns], np.int64)
    rng = np.random.default_rng(0)
    ids = np.array(sorted({i['id'] for i in train['images']}))
    subset = set(np.sort(rng.choice(ids, min(args.train_images, len(ids)), replace=False)).tolist())
    pick = np.flatnonzero(np.isin(t_image, list(subset)))
    val_sim = best_match(val_crops, size(val_anns), v_image, train_crops, size(train_anns), t_image, args.device)
    train_sim = best_match(train_crops[pick], size(train_anns)[pick], t_image[pick], train_crops, size(train_anns), t_image, args.device)
    np.savez_compressed(args.out / 'similarity.npz',
                        val_ann=np.array([a['id'] for a in val_anns]), val_sim=val_sim,
                        train_ann=np.array([train_anns[k]['id'] for k in pick]), train_sim=train_sim,
                        train_images=np.array(sorted(subset)))
    print('written', args.out / 'similarity.npz', 'val seen share', r3((val_sim >= .9).mean()),
          'train-subset seen share', r3((train_sim >= .9).mean()), flush=True)


class EvalImages:
    def __init__(self, folder, images):
        self.folder, self.images = folder, images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        import torchvision.transforms.functional as F
        from PIL import Image
        image = Image.open(f"{self.folder}/{self.images[index]['file_name']}").convert('RGB')
        size = image.size
        return F.to_tensor(F.resize(image, [640, 640])), index, size[0], size[1]


def predict(args):
    import torch
    from engine.core import YAMLConfig
    from error_gallery import ROOT
    chosen = set(np.load(args.out / 'similarity.npz')['train_images'].tolist())
    data = json.load(open(ROOT / 'Annotations/train.json'))
    images = [i for i in data['images'] if i['id'] in chosen]
    cfg = YAMLConfig(str(args.run / 'resolved_config.yml'))
    model, post = cfg.model, cfg.postprocessor
    state = torch.load(args.run / 'best_stg2.pth', map_location='cpu', weights_only=False)
    model.load_state_dict(state['ema']['module'])
    model, post = model.to(args.device).eval(), post.to(args.device).eval()
    loader = torch.utils.data.DataLoader(EvalImages(ROOT / 'JPEGImages/train', images), batch_size=32, num_workers=8)
    rows = []
    with torch.no_grad():
        for batch, indices, widths, heights in loader:
            outputs = model(batch.to(args.device))
            sizes = torch.stack([widths, heights], 1).float().to(args.device)
            for index, result in zip(indices.tolist(), post(outputs, sizes)):
                n = len(result['scores'])
                rows.append((np.full(n, images[index]['id']), result['labels'].cpu().numpy(),
                             result['scores'].cpu().numpy(), result['boxes'].cpu().numpy()))
    np.savez_compressed(args.out / f'train_predictions_{args.name}.npz', image_id=np.concatenate([r[0] for r in rows]),
                        label=np.concatenate([r[1] for r in rows]), score=np.concatenate([r[2] for r in rows]),
                        box=np.concatenate([r[3] for r in rows]))
    print('written', args.out / f'train_predictions_{args.name}.npz', len(images), flush=True)


def detect_flags(annotations, image_ids, saved, score_thr=.3, iou_thr=.5):
    """Per GT annotation id: (detected at the operating point, best IoU of a same-class box at any score)."""
    by_image = defaultdict(list)
    for a in annotations['annotations']:
        if a['image_id'] in image_ids:
            by_image[a['image_id']].append(a)
    order = np.argsort(saved['image_id'], kind='stable')
    img_sorted = saved['image_id'][order]
    starts = np.searchsorted(img_sorted, sorted(image_ids), 'left')
    ends = np.searchsorted(img_sorted, sorted(image_ids), 'right')
    out = {}
    for image_id, s, e in zip(sorted(image_ids), starts, ends):
        anns = by_image.get(image_id, [])
        if not anns:
            continue
        idx = order[s:e]
        labels, scores, boxes = saved['label'][idx], saved['score'][idx], saved['box'][idx]
        gt = np.array([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]] for a in anns], np.float64)
        gt_label = np.array([a['category_id'] for a in anns])
        for c in np.unique(gt_label):
            g = np.flatnonzero(gt_label == c)
            d = np.flatnonzero(labels == c)
            if not len(d):
                for k in g:
                    out[anns[k]['id']] = (False, 0.)
                continue
            iou = iou_xyxy(boxes[d].astype(np.float64), gt[g])
            best_any = iou.max(0)
            taken = np.zeros(len(g), bool)
            for di in np.argsort(-scores[d], kind='stable'):
                if scores[d][di] < score_thr:
                    break
                cand = np.where(taken, -1, iou[di])
                j = int(cand.argmax())
                if cand[j] >= iou_thr:
                    taken[j] = True
            for k, t, b in zip(g, taken, best_any):
                out[anns[k]['id']] = (bool(t), float(b))
    return out


def cell_of(image, name, long_side, group=None):
    """Cell name that exists for both splits: image size / class, tanks of 256-px images split by size."""
    size = f"{image['width']}" if image['width'] == image['height'] else 'misc'
    if name == 'tank' and size == '256':
        return '256/tank<16' if long_side < 16 else '256/tank>=16'
    return f'{size}/{name}'


def summarise(rows):
    """rows: list of (detected, best_any, sim, long_side)."""
    if not rows:
        return None
    det = np.array([r[0] for r in rows])
    best = np.array([r[1] for r in rows])
    return dict(gt=len(rows), recall=r3(det.mean() * 100), missed=int((~det).sum()),
                no_box=int((~det & (best < .5)).sum()), low_score=int((~det & (best >= .5)).sum()))


def analyse(args):
    from error_gallery import ROOT
    sim = np.load(args.out / 'similarity.npz')
    val = json.load(open(ROOT / 'Annotations/val.json'))
    train = json.load(open(ROOT / 'Annotations/train.json'))
    names = {c['id']: c['name'] for c in val['categories']}
    groups = {int(k): v for k, v in json.load(open(args.out.parent / 'error_gallery_val/source.json')).items()}
    train_ids = set(sim['train_images'].tolist())
    val_ids = {i['id'] for i in val['images']}
    sim_of = {'val': dict(zip(sim['val_ann'].tolist(), sim['val_sim'].tolist())),
              'train': dict(zip(sim['train_ann'].tolist(), sim['train_sim'].tolist()))}
    info = {'val': {i['id']: i for i in val['images']}, 'train': {i['id']: i for i in train['images']}}
    ann = {'val': val, 'train': train}
    report = dict(runs={}, similarity=dict(val_seen_share=r3((sim['val_sim'] >= .9).mean()), train_seen_share=r3((sim['train_sim'] >= .9).mean())))
    flags_all = {}
    for spec in args.runs:
        name, val_npz = spec.split('=')
        saved = {}
        for split, path in (('val', val_npz), ('train', args.out / f'train_predictions_{name}.npz')):
            with np.load(path) as z:
                saved[split] = {k: z[k] for k in z.files}
        run = dict(cells={}, source_cells={})
        for split, ids in (('val', val_ids), ('train', train_ids)):
            flags = detect_flags(ann[split], ids, saved[split])
            flags_all[name, split] = flags
            buckets, source_buckets = defaultdict(list), defaultdict(list)
            for a in ann[split]['annotations']:
                if a['id'] not in flags:
                    continue
                image = info[split][a['image_id']]
                long_side = max(a['bbox'][2], a['bbox'][3])
                row = (*flags[a['id']], sim_of[split].get(a['id'], np.nan), long_side)
                buckets[cell_of(image, names[a['category_id']], long_side)].append(row)
                if split == 'val':
                    g = group_of(groups[a['image_id']], image['width'], image['height'])
                    source_buckets[f"{g}/{names[a['category_id']]}"].append(row)
            for cell, rows in buckets.items():
                entry = run['cells'].setdefault(cell, {})
                entry[split] = summarise(rows)
                entry[f'{split}_by_sim'] = {f'{lo:.1f}-{hi:.1f}': summarise([r for r in rows if lo <= r[2] < hi]) for lo, hi in SIM_BINS}
                entry[f'{split}_by_sim_size'] = {
                    f'{int(a)}-{int(min(b, 9999))}': {f'{lo:.1f}-{hi:.1f}': summarise([r for r in rows if lo <= r[2] < hi and a <= r[3] < b])
                                                       for lo, hi in SIM_BINS}
                    for a, b in SIZE_BINS if sum(a <= r[3] < b for r in rows) >= 100}
            for cell, rows in source_buckets.items():
                run['source_cells'][cell] = dict(all=summarise(rows), by_sim={f'{lo:.1f}-{hi:.1f}': summarise([r for r in rows if lo <= r[2] < hi])
                                                                             for lo, hi in SIM_BINS})
        report['runs'][name] = run
    # Where do the extra misses of one run against the first run fall, by similarity (val)?
    base = args.runs[0].split('=')[0]
    report['extra_misses_vs_' + base] = {}
    for spec in args.runs[1:]:
        name = spec.split('=')[0]
        table = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))  # cell -> sim bin -> [gt, newly missed, newly found]
        for a in val['annotations']:
            if a['id'] not in flags_all[base, 'val']:
                continue
            image = info['val'][a['image_id']]
            cell = cell_of(image, names[a['category_id']], max(a['bbox'][2], a['bbox'][3]))
            s = sim_of['val'].get(a['id'], np.nan)
            key = next((f'{lo:.1f}-{hi:.1f}' for lo, hi in SIM_BINS if lo <= s < hi), 'nan')
            b, n = flags_all[base, 'val'][a['id']][0], flags_all[name, 'val'][a['id']][0]
            t = table[cell][key]
            t[0] += 1
            t[1] += int(b and not n)
            t[2] += int(n and not b)
        report['extra_misses_vs_' + base][name] = {c: {k: dict(gt=v[0], newly_missed=v[1], newly_found=v[2]) for k, v in d.items()} for c, d in table.items()}
    (args.out / 'generalisation_gap.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print('written', args.out / 'generalisation_gap.json', flush=True)


def ap_by_similarity(args):
    """COCO AP restricted to the GTs of one similarity range. GTs outside the range are ignored (a detection matched
    to one of them is ignored too); every unmatched detection still counts as a false positive in every range."""
    from faster_coco_eval import COCO, COCOeval_faster
    from error_gallery import ROOT
    sim = np.load(args.out / 'similarity.npz')
    sim_of = dict(zip(sim['val_ann'].tolist(), sim['val_sim'].tolist()))
    path = ROOT / 'Annotations/val.json'
    val = json.load(open(path))
    names = {c['id']: c['name'] for c in val['categories']}
    groups = {int(k): v for k, v in json.load(open(args.out.parent / 'error_gallery_val/source.json')).items()}
    info = {i['id']: i for i in val['images']}
    members = defaultdict(list)
    for i in val['images']:
        members[group_of(groups[i['id']], i['width'], i['height'])].append(i['id'])
    ranges = {'all': (-2, 2), 'seen (sim>=.9)': (.9, 2), 'novel (sim<.9)': (-2, .9), 'sim .7-.9': (.7, .9), 'sim <.7': (-2, .7)}
    report = {}
    for spec in args.runs:
        name, npz = spec.split('=')
        with np.load(npz) as z:
            results = [dict(image_id=int(i), category_id=int(c), score=float(sc), bbox=[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])], area=1.)
                       for i, c, sc, b in zip(z['image_id'], z['label'], z['score'], z['box'])]
        report[name] = {}
        for label, (lo, hi) in ranges.items():
            gt = COCO(str(path))
            for a in gt.anns.values():
                a['area'] = 1. if lo <= sim_of[a['id']] < hi else 100.
            dt = gt.loadRes(results)
            for d in dt.anns.values():
                d['area'] = 1.
            out = {}
            for scope, ids in [('all', None)] + [(g, v) for g, v in members.items() if g in ('other_512', 'ogsod', 'other_256', 'other_800', 'msar')]:
                e = COCOeval_faster(gt, dt, 'bbox')
                if ids is not None:
                    e.params.imgIds = sorted(ids)
                e.params.areaRng = [[0., 10.]]
                e.params.areaRngLbl = ['all']
                e.evaluate()
                e.accumulate()
                precision, recall = np.asarray(e.eval['precision']), np.asarray(e.eval['recall'])
                last = len(e.params.maxDets) - 1
                per_class = {}
                for k, c in enumerate(e.params.catIds):
                    pr, rc = precision[:, :, k, 0, last], recall[:, k, 0, last]
                    if (rc < 0).all():
                        continue
                    per_class[names[c]] = dict(AP=r3(pr[pr > -1].mean() * 100), AP50=r3(pr[0][pr[0] > -1].mean() * 100),
                                               AP75=r3(pr[5][pr[5] > -1].mean() * 100), AR=r3(rc[rc > -1].mean() * 100))
                out[scope] = dict(mean={m: r3(np.mean([v[m] for v in per_class.values()])) for m in ('AP', 'AP50', 'AP75', 'AR')}, per_class=per_class)
            report[name][label] = out
            print(name, label, out['all']['mean'], flush=True)
    counts = defaultdict(lambda: defaultdict(int))
    for a in val['annotations']:
        for label, (lo, hi) in ranges.items():
            if lo <= sim_of[a['id']] < hi:
                counts[label][names[a['category_id']]] += 1
    report['gt_counts'] = {k: dict(v) for k, v in counts.items()}
    (args.out / 'ap_by_similarity.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print('written', args.out / 'ap_by_similarity.json', flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('similarity')
    s.add_argument('out', type=Path)
    s.add_argument('--device', default='cuda:0')
    s.add_argument('--train-images', type=int, default=12000)
    p = sub.add_parser('predict')
    p.add_argument('run', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--name', required=True)
    p.add_argument('--device', default='cuda:0')
    a = sub.add_parser('analyse')
    a.add_argument('out', type=Path)
    a.add_argument('--runs', nargs='+', required=True)
    b = sub.add_parser('ap')
    b.add_argument('out', type=Path)
    b.add_argument('--runs', nargs='+', required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    {'similarity': similarity, 'predict': predict, 'analyse': analyse, 'ap': ap_by_similarity}[args.command](args)


if __name__ == '__main__':
    main()
