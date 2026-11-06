"""Independent check of the P0 metrics: fresh inference, own COCO-style AP, and seen/unseen split analysis."""
import json
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

RUN = Path(sys.argv[1])
OUTPUT = Path(sys.argv[2])
ROOT = Path('/root/zxy/TCRA-SAR/datasets/SARDet_100K')
OUTPUT.mkdir(parents=True, exist_ok=False)
CROP = 24
THUMB = 32
SPLITS = ['val', 'test']
report = {}


def log(*args):
    print(time.strftime('%H:%M:%S'), *args, flush=True)


def unit(array):
    vector = np.asarray(array, np.float32).ravel()
    vector = vector - vector.mean()
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def encode(task):
    folder, name, boxes = task
    image = Image.open(f'{folder}/{name}')
    values = np.asarray(image, np.float32)
    if values.ndim == 3:
        values = values.mean(2)
    height, width = values.shape
    thumb = unit(Image.fromarray(values).resize((THUMB, THUMB), Image.BILINEAR))
    crops = []
    for x, y, w, h in boxes:
        x0, y0 = min(max(int(x), 0), width - 1), min(max(int(y), 0), height - 1)
        x1, y1 = min(max(int(np.ceil(x + w)), x0 + 1), width), min(max(int(np.ceil(y + h)), y0 + 1), height)
        crops.append(unit(Image.fromarray(values[y0:y1, x0:x1]).resize((CROP, CROP), Image.BILINEAR)))
    return thumb, (np.stack(crops) if crops else np.zeros((0, CROP * CROP), np.float32))


def load_split(split):
    data = json.load(open(ROOT / f'Annotations/{split}.json'))
    by_image = defaultdict(list)
    for a in data['annotations']:
        by_image[a['image_id']].append(a)
    tasks = [(str(ROOT / f'JPEGImages/{split}'), im['file_name'], [a['bbox'] for a in by_image[im['id']]])
             for im in data['images']]
    with mp.Pool(32) as pool:
        encoded = pool.map(encode, tasks, chunksize=64)
    thumbs = np.stack([e[0] for e in encoded])
    crops = np.concatenate([e[1] for e in encoded])
    ids = [a['id'] for im in data['images'] for a in by_image[im['id']]]
    sizes = np.array([max(a['bbox'][2], a['bbox'][3]) for im in data['images'] for a in by_image[im['id']]], np.float32)
    return data, thumbs, crops, ids, sizes


def best_match(query, reference, sizes_q=None, sizes_r=None, chunk=1024):
    import torch
    device = 'cuda'
    reference = torch.as_tensor(reference, device=device)
    if sizes_r is not None:
        sizes_r = torch.as_tensor(sizes_r, device=device)
    best = []
    for start in range(0, len(query), chunk):
        part = torch.as_tensor(query[start:start + chunk], device=device)
        scores = part @ reference.T
        if sizes_q is not None:
            ratio = torch.as_tensor(sizes_q[start:start + chunk], device=device)[:, None] / sizes_r[None, :].clamp(min=1e-3)
            scores[(ratio - 1).abs() > .1] = -1
        best.append(scores.max(1).values.cpu().numpy())
    return np.concatenate(best)


class EvalImages:
    def __init__(self, folder, images):
        self.folder, self.images = folder, images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        import torchvision.transforms.functional as F
        image = Image.open(f"{self.folder}/{self.images[index]['file_name']}").convert('RGB')
        size = image.size
        return F.to_tensor(F.resize(image, [640, 640])), index, size[0], size[1]


def infer(data, split):
    import torch
    from engine.core import YAMLConfig
    cfg = YAMLConfig(str(RUN / 'resolved_config.yml'))
    model, post = cfg.model, cfg.postprocessor
    state = torch.load(RUN / 'best_stg2.pth', map_location='cpu')
    print('load', model.load_state_dict(state['ema']['module'], strict=False))
    model = model.cuda().eval()
    post = post.cuda().eval()
    loader = torch.utils.data.DataLoader(EvalImages(ROOT / f'JPEGImages/{split}', data['images']), batch_size=64, num_workers=8)
    rows = []
    with torch.no_grad():
        for images, indices, widths, heights in loader:
            outputs = model(images.cuda())
            sizes = torch.stack([widths, heights], 1).cuda()
            for index, result in zip(indices.tolist(), post(outputs, sizes)):
                image_id = data['images'][index]['id']
                boxes, scores, labels = result['boxes'].cpu().numpy(), result['scores'].cpu().numpy(), result['labels'].cpu().numpy()
                rows.append((np.full(len(scores), image_id), labels, scores, boxes))
    return tuple(np.concatenate([r[i] for r in rows]) for i in range(4))


def iou_matrix(dets, gts):
    gx1, gy1 = gts[:, 0], gts[:, 1]
    gx2, gy2 = gx1 + gts[:, 2], gy1 + gts[:, 3]
    x1 = np.maximum(dets[:, None, 0], gx1)
    y1 = np.maximum(dets[:, None, 1], gy1)
    x2 = np.minimum(dets[:, None, 2], gx2)
    y2 = np.minimum(dets[:, None, 3], gy2)
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (dets[:, 2] - dets[:, 0])[:, None] * (dets[:, 3] - dets[:, 1])[:, None] + gts[:, 2] * gts[:, 3] - inter
    return inter / np.maximum(area, 1e-9)


THRESHOLDS = np.linspace(.5, .95, 10)
RECALLS = np.linspace(0, 1, 101)


def coco_ap(data, predictions, keep_gt=None, keep_images=None):
    """COCO-style AP (area all, 100 dets per image and class). GT outside keep_gt is ignored."""
    image_ids, labels, scores, boxes = predictions
    gts = defaultdict(list)
    for a in data['annotations']:
        if keep_images is None or a['image_id'] in keep_images:
            ignored = bool(a.get('iscrowd', 0)) or (keep_gt is not None and a['id'] not in keep_gt)
            gts[(a['image_id'], a['category_id'])].append((a['bbox'], ignored))
    index = defaultdict(list)
    for position in np.flatnonzero(np.ones(len(scores), bool)):
        if keep_images is None or image_ids[position] in keep_images:
            index[(int(image_ids[position]), int(labels[position]))].append(position)
    per_class = {}
    for category in sorted(c['id'] for c in data['categories']):
        keys = {k for k in list(gts) + list(index) if k[1] == category}
        positives, det_scores, matched, skipped = 0, [], [], []
        for key in keys:
            rows = gts.get(key, [])
            order = sorted(range(len(rows)), key=lambda i: rows[i][1])
            ignored = np.array([rows[i][1] for i in order], bool)
            boxes_gt = np.array([rows[i][0] for i in order], np.float32).reshape(-1, 4)
            positives += int((~ignored).sum())
            positions = np.array(index.get(key, []), int)
            if len(positions):
                positions = positions[np.argsort(-scores[positions], kind='mergesort')][:100]
            count = len(positions)
            hit = np.zeros((len(THRESHOLDS), count), bool)
            skip = np.zeros((len(THRESHOLDS), count), bool)
            if count and len(boxes_gt):
                ious = iou_matrix(boxes[positions], boxes_gt)
                for t, threshold in enumerate(THRESHOLDS):
                    used = np.zeros(len(boxes_gt), bool)
                    for d in range(count):
                        if ious[d].max() < threshold:
                            continue
                        best, chosen = min(threshold, 1 - 1e-10), -1
                        for g in range(len(boxes_gt)):
                            if used[g]:
                                continue
                            if chosen > -1 and not ignored[chosen] and ignored[g]:
                                break
                            if ious[d, g] < best:
                                continue
                            best, chosen = ious[d, g], g
                        if chosen > -1:
                            used[chosen] = True
                            hit[t, d] = not ignored[chosen]
                            skip[t, d] = ignored[chosen]
            if count:
                det_scores.append(scores[positions])
                matched.append(hit)
                skipped.append(skip)
        if not positives:
            continue
        all_scores = np.concatenate(det_scores) if det_scores else np.zeros(0)
        order = np.argsort(-all_scores, kind='mergesort')
        all_hit = np.concatenate(matched, 1)[:, order] if matched else np.zeros((10, 0), bool)
        all_skip = np.concatenate(skipped, 1)[:, order] if skipped else np.zeros((10, 0), bool)
        values = []
        for t in range(len(THRESHOLDS)):
            valid = ~all_skip[t]
            tp = np.cumsum(all_hit[t][valid])
            fp = np.cumsum(~all_hit[t][valid])
            recall = tp / positives
            precision = tp / np.maximum(tp + fp, 1e-12)
            precision = np.maximum.accumulate(precision[::-1])[::-1] if len(precision) else precision
            where = np.searchsorted(recall, RECALLS, side='left')
            sampled = np.zeros(101)
            inside = where < len(precision)
            sampled[inside] = precision[where[inside]]
            values.append(sampled.mean())
        per_class[category] = values
    matrix = np.array(list(per_class.values()))
    names = {c['id']: c['name'] for c in data['categories']}
    return dict(AP=float(matrix.mean() * 100), AP50=float(matrix[:, 0].mean() * 100), AP75=float(matrix[:, 5].mean() * 100),
                gt=sum(1 for rows in gts.values() for r in rows if not r[1]),
                per_class={names[c]: round(float(np.mean(v)) * 100, 2) for c, v in per_class.items()})


def main():
    log('encode train')
    train = load_split('train')
    for split in SPLITS:
        log('encode', split)
        data, thumbs, crops, ids, sizes = load_split(split)
        log('match', split)
        image_best = best_match(thumbs, train[1])
        crop_best = best_match(crops, train[2], sizes, train[4])
        gt_ids = np.array(ids)
        result = dict(images=len(data['images']), gt=len(ids),
                      crowd=sum(int(a.get('iscrowd', 0)) for a in data['annotations']))
        result['whole_image_match_rate'] = {str(t): round(float((image_best >= t).mean()), 4) for t in (.9, .95, .98, .99)}
        result['crop_match_rate'] = {str(t): round(float((crop_best >= t).mean()), 4) for t in (.8, .9, .95)}
        bins = [(0, 16), (16, 32), (32, 64), (64, 1e9)]
        result['crop_match_rate_0.9_by_size'] = {f'{a}-{b}': [int(((sizes >= a) & (sizes < b)).sum()),
            round(float((crop_best[(sizes >= a) & (sizes < b)] >= .9).mean()), 4)] for a, b in bins}
        log('infer', split)
        predictions = infer(data, split)
        np.savez_compressed(OUTPUT / f'{split}_predictions.npz', image_id=predictions[0], label=predictions[1],
                            score=predictions[2], box=predictions[3])
        log('ap', split)
        result['independent_all'] = coco_ap(data, predictions)
        log(split, result['independent_all'])
        for threshold in (.9, .95):
            seen = set(gt_ids[crop_best >= threshold].tolist())
            unseen = set(gt_ids[crop_best < threshold].tolist())
            result[f'gt_seen_{threshold}'] = coco_ap(data, predictions, keep_gt=seen)
            result[f'gt_unseen_{threshold}'] = coco_ap(data, predictions, keep_gt=unseen)
        duplicate = {im['id'] for im, score in zip(data['images'], image_best) if score >= .95}
        fresh = {im['id'] for im, score in zip(data['images'], image_best) if score < .95}
        result['images_duplicate_0.95'] = coco_ap(data, predictions, keep_images=duplicate)
        result['images_fresh_0.95'] = coco_ap(data, predictions, keep_images=fresh)
        report[split] = result
        (OUTPUT / 'report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
        log(json.dumps(result, ensure_ascii=False))
    log('DONE')


if __name__ == '__main__':
    main()
