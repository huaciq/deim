"""Zero-training structure checks. Train labels and val only; no test. Source is anonymous, so radiometric clusters stand in for it."""
import json
import multiprocessing as mp
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

RUN = Path(sys.argv[1])
OUT = RUN / 'independent_verify_20261106/structure'
ROOT = Path('/root/zxy/TCRA-SAR/datasets/SARDet_100K')
K = 8
THR = np.linspace(.5, .95, 10)
REC = np.linspace(0, 1, 101)
train = json.load(open(ROOT / 'Annotations/train.json'))
val = json.load(open(ROOT / 'Annotations/val.json'))
names = {c['id']: c['name'] for c in val['categories']}
report = {}


def describe(task):
    folder, name = task
    image = Image.open(f'{folder}/{name}').convert('L')
    v = np.asarray(image, np.float32)
    t = np.asarray(image.resize((32, 32), Image.BILINEAR), np.float32).ravel()
    t = t - t.mean()
    n = np.linalg.norm(t)
    p1, p50, p99 = np.percentile(v, [1, 50, 99])
    return (t / n if n > 0 else t), (float(np.log2(v.shape[1])), float(np.log1p(p50)), float(np.log1p(p99)), float(v.std() / max(v.mean(), 1.)), float((v < 16).mean()))


def by_image(data):
    rows = defaultdict(list)
    for a in data['annotations']:
        rows[a['image_id']].append(a)
    return rows


def corners(a):
    x, y, w, h = a['bbox']
    return [x, y, x + w, y + h]


def iou(a, b):
    x1, y1 = np.maximum(a[:, None, 0], b[:, 0]), np.maximum(a[:, None, 1], b[:, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[:, 2]), np.minimum(a[:, None, 3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = (a[:, 2] - a[:, 0])[:, None] * (a[:, 3] - a[:, 1])[:, None] + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return inter / np.maximum(union, 1e-9)


def kmeans(x, k, seed=0):
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(len(x), k, replace=False)]
    for _ in range(50):
        d = ((x[:, None] - centers[None]) ** 2).sum(2)
        a = d.argmin(1)
        centers = np.stack([x[a == i].mean(0) if (a == i).any() else centers[i] for i in range(k)])
    return centers, a


def pixel_same(task):
    a, b = task
    A = np.asarray(Image.open(a).convert('L'), np.float32)
    B = np.asarray(Image.open(b).convert('L'), np.float32)
    return A.shape == B.shape and np.abs(A - B).mean() < 1


def compare_labels(pairs, rows_a, rows_b):
    stats = defaultdict(list)
    for ia, ib in pairs:
        ga, gb = rows_a[ia], rows_b[ib]
        if not ga:
            continue
        if not gb:
            for x in ga:
                stats[x['category_id']].append((0., 0.))
            continue
        boxes = np.array([corners(x) for x in gb], np.float32)
        labels = np.array([x['category_id'] for x in gb])
        for x in ga:
            m = iou(np.array([corners(x)], np.float32), boxes)[0]
            same = m[labels == x['category_id']].max() if (labels == x['category_id']).any() else 0.
            stats[x['category_id']].append((float(same), float(m.max())))
    return {names[c]: dict(n=len(v), unmatched_iou_lt_0_1=round(float(np.mean([s < .1 for s, _ in v])), 4), other_class_only=round(float(np.mean([s < .1 and o >= .5 for s, o in v])), 4), partial_0_1_to_0_75=round(float(np.mean([.1 <= s < .75 for s, _ in v])), 4), median_iou=round(float(np.median([s for s, _ in v])), 3)) for c, v in stats.items()}


class Images:
    def __init__(self, rows, folder):
        self.rows, self.folder = rows, folder

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import torchvision.transforms.functional as F
        image = Image.open(self.folder / self.rows[i]['file_name']).convert('RGB')
        return F.to_tensor(F.resize(image, [640, 640])), i, image.size[0], image.size[1]


def ap_curve(s, h, n):
    r = np.argsort(-s, kind='mergesort')
    h = h[r]
    tp, fp = np.cumsum(h), np.cumsum(~h)
    rc = tp / n
    pr = tp / np.maximum(tp + fp, 1e-12)
    pr = np.maximum.accumulate(pr[::-1])[::-1] if len(pr) else pr
    w = np.searchsorted(rc, REC, side='left')
    o = np.zeros(101)
    k = w < len(pr)
    o[k] = pr[w[k]]
    return o.mean()


def evaluate(pred, rows_val, image_ids):
    out = {}
    for c, name in names.items():
        lines = {t: ([], []) for t in range(10)}
        n = 0
        for image_id in image_ids:
            cur = [a for a in rows_val[image_id] if a['category_id'] == c]
            n += len(cur)
            labels, scores, boxes = pred[image_id]
            keep = np.flatnonzero(labels == c)
            if not len(keep):
                continue
            keep = keep[np.argsort(-scores[keep], kind='mergesort')][:100]
            m = iou(boxes[keep], np.array([corners(a) for a in cur], np.float32)) if cur else None
            for t, th in enumerate(THR):
                used = set()
                hit = np.zeros(len(keep), bool)
                if cur:
                    for d in range(len(keep)):
                        if m[d].max() < th:
                            continue
                        best, chosen = th, -1
                        for g in range(len(cur)):
                            if g not in used and m[d, g] >= best:
                                best, chosen = m[d, g], g
                        if chosen >= 0:
                            used.add(chosen)
                            hit[d] = True
                lines[t][0].append(scores[keep])
                lines[t][1].append(hit)
        if n:
            v = [ap_curve(np.concatenate(lines[t][0]), np.concatenate(lines[t][1]), n) for t in range(10)]
            out[name] = dict(gt=n, AP=round(float(np.mean(v)) * 100, 2), AP50=round(v[0] * 100, 2), AP75=round(v[5] * 100, 2))
    if out:
        out['mean'] = round(float(np.mean([v['AP'] for v in out.values()])), 2)
    return out


def adabn(train_indices_by_group, val_indices_by_group):
    import torch
    from engine.core import YAMLConfig
    cfg = YAMLConfig(str(RUN / 'resolved_config.yml'))
    model, post = cfg.model, cfg.postprocessor
    state = torch.load(RUN / 'best_stg2.pth', map_location='cpu')['ema']['module']
    post = post.cuda().eval()
    results = {}

    def calibrate(indices, limit=3000):
        model.load_state_dict(state)
        model.cuda().eval()
        norms = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        for m in norms:
            m.reset_running_stats()
            m.momentum = None
            m.train()
        rng = np.random.default_rng(0)
        chosen = rng.permutation(indices)[:limit]
        loader = torch.utils.data.DataLoader(Images([train['images'][i] for i in chosen], ROOT / 'JPEGImages/train'), batch_size=64, num_workers=8)
        with torch.no_grad():
            for batch, *_ in loader:
                model(batch.cuda())
        for m in norms:
            m.eval()

    def predict(indices, store):
        rows = [val['images'][i] for i in indices]
        loader = torch.utils.data.DataLoader(Images(rows, ROOT / 'JPEGImages/val'), batch_size=64, num_workers=8)
        with torch.no_grad():
            for batch, order, w, h in loader:
                for i, r in zip(order.tolist(), post(model(batch.cuda()), torch.stack([w, h], 1).cuda())):
                    store[rows[i]['id']] = (r['labels'].cpu().numpy().astype(int), r['scores'].cpu().numpy(), r['boxes'].cpu().numpy())

    all_val = list(range(len(val['images'])))
    baseline = {}
    model.load_state_dict(state)
    model.cuda().eval()
    predict(all_val, baseline)
    pooled = {}
    calibrate(np.arange(len(train['images'])), limit=8000)
    predict(all_val, pooled)
    grouped = {}
    for group, indices in val_indices_by_group.items():
        calibrate(train_indices_by_group[group])
        predict(indices, grouped)
        print('calibrated', group, flush=True)
    return baseline, pooled, grouped


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows_t, rows_v = by_image(train), by_image(val)
    with mp.Pool(32) as pool:
        desc_t = pool.map(describe, [(str(ROOT / 'JPEGImages/train'), i['file_name']) for i in train['images']], chunksize=64)
        desc_v = pool.map(describe, [(str(ROOT / 'JPEGImages/val'), i['file_name']) for i in val['images']], chunksize=64)
    thumb_t, feat_t = np.stack([d[0] for d in desc_t]), np.array([d[1] for d in desc_t], np.float32)
    thumb_v, feat_v = np.stack([d[0] for d in desc_v]), np.array([d[1] for d in desc_v], np.float32)
    mean, std = feat_t.mean(0), feat_t.std(0) + 1e-6
    centers, assign_t = kmeans((feat_t - mean) / std, K)
    assign_v = ((((feat_v - mean) / std)[:, None] - centers[None]) ** 2).sum(2).argmin(1)

    # 1. class/source structure
    classes_per = Counter(len({a['category_id'] for a in rows_t[i['id']]}) for i in train['images'])
    report['classes_per_train_image'] = dict(classes_per)
    clusters = {}
    gt_total, gt_major = 0, 0
    for k in range(K):
        idx = np.flatnonzero(assign_t == k)
        counts = Counter(a['category_id'] for i in idx for a in rows_t[train['images'][i]['id']])
        widths = Counter(train['images'][i]['width'] for i in idx)
        clusters[k] = dict(train_images=int(len(idx)), val_images=int((assign_v == k).sum()), widths=widths.most_common(3), classes={names[c]: n for c, n in counts.most_common()},
                           median_p50=round(float(np.expm1(np.median(feat_t[idx, 1]))), 1), median_p99=round(float(np.expm1(np.median(feat_t[idx, 2]))), 1), median_cv=round(float(np.median(feat_t[idx, 3])), 3), share_lt16=round(float(np.median(feat_t[idx, 4])), 3))
        gt_total += sum(counts.values())
        gt_major += max(counts.values()) if counts else 0
    report['clusters'] = clusters
    report['cluster_majority_class_share_of_gt'] = round(gt_major / gt_total, 4)
    class_img = {}
    for c, name in names.items():
        per = Counter(int(k) for i, k in zip(train['images'], assign_t) if any(a['category_id'] == c for a in rows_t[i['id']]))
        total = sum(per.values())
        class_img[name] = {str(k): round(v / total, 3) for k, v in per.most_common(3)}
    report['class_top_clusters'] = class_img

    # 2. missing or inconsistent labels through pixel-identical pairs
    import torch
    T = torch.as_tensor(thumb_t).cuda()
    V = torch.as_tensor(thumb_v).cuda()
    candidates_tt = []
    for start in range(0, len(T), 1024):
        s, i = (T[start:start + 1024] @ T.T).topk(4, dim=1)
        s, i = s.cpu().numpy(), i.cpu().numpy()
        for r in range(len(s)):
            a = start + r
            for score, b in zip(s[r], i[r]):
                if b > a and score >= .98 and train['images'][a]['width'] == train['images'][b]['width']:
                    candidates_tt.append((a, int(b)))
    candidates_vt = []
    for start in range(0, len(V), 1024):
        s, i = (V[start:start + 1024] @ T.T).topk(3, dim=1)
        s, i = s.cpu().numpy(), i.cpu().numpy()
        for r in range(len(s)):
            for score, b in zip(s[r], i[r]):
                if score >= .98 and val['images'][start + r]['width'] == train['images'][b]['width']:
                    candidates_vt.append((start + r, int(b)))
    del T, V
    torch.cuda.empty_cache()
    print('candidates', len(candidates_tt), len(candidates_vt), flush=True)
    path_t = lambda i: str(ROOT / 'JPEGImages/train' / train['images'][i]['file_name'])
    path_v = lambda i: str(ROOT / 'JPEGImages/val' / val['images'][i]['file_name'])
    with mp.Pool(32) as pool:
        same_tt = pool.map(pixel_same, [(path_t(a), path_t(b)) for a, b in candidates_tt], chunksize=64)
        same_vt = pool.map(pixel_same, [(path_v(a), path_t(b)) for a, b in candidates_vt], chunksize=64)
    pairs_tt = [p for p, ok in zip(candidates_tt, same_tt) if ok]
    pairs_vt = [p for p, ok in zip(candidates_vt, same_vt) if ok]
    report['pixel_identical'] = dict(train_train_pairs=len(pairs_tt), val_train_pairs=len(pairs_vt),
                                     train_images_in_a_pair=len({x for p in pairs_tt for x in p}),
                                     train_train=compare_labels([(train['images'][a]['id'], train['images'][b]['id']) for a, b in pairs_tt], rows_t, rows_t),
                                     val_train=compare_labels([(val['images'][a]['id'], train['images'][b]['id']) for a, b in pairs_vt], rows_v, rows_t))
    # High-score background false positives on val images with a labelled twin in train.
    saved = np.load(RUN / 'independent_verify_20261106/val_predictions.npz')
    PI, PC, PS, PB = saved['image_id'], saved['label'].astype(int), saved['score'], saved['box']
    twin = defaultdict(list)
    for a, b in pairs_vt:
        twin[val['images'][a]['id']].append(train['images'][b]['id'])
    keep = np.isin(PI, list(twin)) & (PS >= .3)
    fp_total, fp_supported = Counter(), Counter()
    for p in np.flatnonzero(keep):
        image_id = int(PI[p])
        gt = rows_v[image_id]
        box = PB[p][None]
        if gt and iou(box, np.array([corners(a) for a in gt], np.float32)).max() >= .1:
            continue
        fp_total[names[PC[p]]] += 1
        for other in twin[image_id]:
            same = [a for a in rows_t[other] if a['category_id'] == PC[p]]
            if same and iou(box, np.array([corners(a) for a in same], np.float32)).max() >= .5:
                fp_supported[names[PC[p]]] += 1
                break
    report['val_fp_labelled_in_identical_train_twin'] = {k: dict(fp=v, labelled_in_twin=fp_supported[k]) for k, v in fp_total.items()}

    # 3. BN statistics per radiometric cluster
    train_groups = {k: np.flatnonzero(assign_t == k) for k in range(K)}
    val_groups = {k: np.flatnonzero(assign_v == k).tolist() for k in range(K) if (assign_v == k).any() and len(train_groups[k]) >= 200}
    baseline, pooled, grouped = adabn(train_groups, val_groups)
    ids_all = [i['id'] for i in val['images']]
    ids_group = [val['images'][i]['id'] for v in val_groups.values() for i in v]
    report['adabn'] = dict(images_in_groups=len(ids_group), base=evaluate(baseline, rows_v, ids_group), pooled=evaluate(pooled, rows_v, ids_group), per_cluster=evaluate(grouped, rows_v, ids_group), per_group={})
    for k, v in val_groups.items():
        ids = [val['images'][i]['id'] for i in v]
        report['adabn']['per_group'][k] = dict(base=evaluate(baseline, rows_v, ids), per_cluster=evaluate(grouped, rows_v, ids))
    (OUT / 'report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print('RESULT', json.dumps(report, ensure_ascii=False), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
