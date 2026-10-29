#!/usr/bin/env python3
"""Frozen SAR-AIRcraft local-evidence ranking probe. See docs/local_evidence_probe.md."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODES = ('score', 'cls', 'cls_fdr', 'cls_fdr_roi', 'cls_fdr_border', 'cls_fdr_local')
TARGETS = ('same_class', 'geometry')


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def write_npz(path, **arrays):
    path = Path(path)
    tmp = path.with_suffix('.partial')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    tmp.replace(path)


def save_checkpoint(path, value):
    tmp = path.with_suffix('.partial')
    torch.save(value, tmp)
    tmp.replace(path)


def sampling_points(boxes):
    """Normalized predicted cxcywh -> body [B,Q,9,2], border [B,Q,4,3,8,2].

    Side order left, top, right, bottom. Offset order inside, on, outside.
    The normal offsets are fractions of the predicted width/height, not pixels.
    """
    boxes = boxes.detach().float()
    centers, extent = boxes[..., :2], boxes[..., 2:]
    t = (torch.arange(3, device=boxes.device) + .5) / 3 - .5
    yy, xx = torch.meshgrid(t, t, indexing='ij')
    unit = torch.stack((xx, yy), -1).reshape(9, 2)
    body = centers[..., None, :] + unit * extent[..., None, :]
    along = (torch.arange(8, device=boxes.device) + .5) / 8 - .5
    normal = torch.tensor([.45, .50, .55], device=boxes.device)
    a = along.expand(3, 8)
    n = normal[:, None].expand(3, 8)
    sides = torch.stack((torch.stack((-n, a), -1), torch.stack((a, -n), -1),
                         torch.stack((n, a), -1), torch.stack((a, n), -1)))
    border = centers[..., None, None, None, :] + sides * extent[..., None, None, None, :]
    return body, border


def local_descriptors(feature, boxes, seed, channels=8):
    """Fixed random orthogonal channel projection; no learned/GT-dependent pooling."""
    generator = torch.Generator().manual_seed(seed)
    projection, _ = torch.linalg.qr(torch.randn(feature.shape[1], channels, generator=generator))
    projected = F.conv2d(feature.float(), projection.T.to(feature.device)[:, :, None, None])
    body, border = sampling_points(boxes)
    points = torch.cat((body, border.flatten(2, 4)), dim=2)
    valid = ((points >= 0) & (points <= 1)).all(-1)
    b, q, n, _ = points.shape
    samples = F.grid_sample(projected, (points * 2 - 1).reshape(b, q, n, 2),
                            mode='bilinear', padding_mode='border', align_corners=False)
    samples = samples.permute(0, 2, 3, 1)  # B,Q,points,channels
    roi = torch.cat((samples[:, :, :9].flatten(2), valid[:, :, :9].float()), -1)
    sides = samples[:, :, 9:].reshape(b, q, 4, 3, 8, channels)
    # Keep normal position and side identity; aggregate only along the side.
    edge = torch.cat((sides.mean(-2).flatten(2), sides.std(-2, unbiased=False).flatten(2),
                      valid[:, :, 9:].float().reshape(b, q, 4, 3, 8).mean(-1).flatten(2)), -1)
    return roi, edge


class FeatureTap:
    """Read-only hooks. HGNetv2 S2 is already computed by the unmodified baseline."""
    def __init__(self, model):
        self.values = {}
        self.handles = [
            model.backbone.stages[0].register_forward_hook(self._save('s2')),
            model.decoder.register_forward_pre_hook(self._save_input),
        ]

    def _save(self, key):
        def hook(module, inputs, output):
            self.values[key] = output.detach()
        return hook

    def _save_input(self, module, inputs):
        self.values['p3'] = inputs[0][0].detach()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.values.clear()


def detection_arrays(outputs, sizes, lqe, project, maps, topk=300):
    logits, boxes, corners = (outputs[k].float() for k in ('pred_logits', 'pred_boxes', 'pred_corners'))
    b, q, c = logits.shape
    probabilities = logits.sigmoid()
    score, index = probabilities.flatten(1).topk(topk, dim=-1)
    query, label = index // c, index % c
    take = lambda x: x.gather(1, query[..., None].expand(-1, -1, x.shape[-1]))
    boxes = take(boxes)
    xyxy = torch.cat((boxes[..., :2] - boxes[..., 2:] / 2, boxes[..., :2] + boxes[..., 2:] / 2), -1)
    xyxy = xyxy * sizes.repeat(1, 2)[:, None, :]  # original sizes are (width, height)
    distribution = corners.reshape(b, q, 4, -1).softmax(-1)
    mean = (distribution * project).sum(-1)
    std = (distribution * (project - mean[..., None]).square()).sum(-1).clamp_min(0).sqrt()
    entropy = -(distribution * distribution.clamp_min(1e-12).log()).sum(-1)
    peak = distribution.amax(-1)
    quality = lqe(torch.zeros_like(logits[..., :1]), corners)
    roi, border = [], []
    for seed, key in enumerate(('s2', 'p3')):
        r, e = local_descriptors(maps[key], boxes, seed=173 + seed)
        roi.append(r)
        border.append(e)
    return dict(score=score, label=label, query_id=query, box=xyxy, box_normalized=boxes,
                probs=take(probabilities), quality=take(quality),
                fdr=take(torch.cat((mean, std, peak, entropy), -1)),
                roi=torch.cat(roi, -1), border=torch.cat(border, -1),
                orig_size=sizes[:, None, :].expand(-1, topk, -1))


def export_features(args, rank, world, device):
    from engine.core import YAMLConfig
    from engine.deim.dfine_utils import weighting_function
    from torch.utils.data import DataLoader

    root = Path(args.output)
    cfg = YAMLConfig(str(root / 'resolved_config.yml'))
    model = cfg.model.to(device).eval().requires_grad_(False)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    use_ema = bool(state.get('ema'))
    model.load_state_dict(state['ema']['module'] if use_ema else state['model'], strict=True)
    del state
    post = cfg.postprocessor
    if not post.use_focal_loss or post.remap_mscoco_category:
        raise ValueError('This probe requires sigmoid scores and zero-based original category IDs.')
    loader = cfg.val_dataloader
    dataset = loader.dataset
    subset = [i for i in range(rank, len(dataset), world)
              if not (root / 'features' / f'{dataset.ids[i]}.npz').exists()]
    loader = DataLoader(dataset, batch_size=args.export_batch_size // world, sampler=subset,
                        num_workers=args.workers, collate_fn=loader.collate_fn,
                        pin_memory=device.type == 'cuda', drop_last=False)
    dec = model.decoder
    lqe = dec.decoder.lqe_layers[dec.decoder.eval_idx]
    project = weighting_function(dec.reg_max, dec.up, dec.reg_scale).float().to(device)
    tap = FeatureTap(model)
    print(f'EXPORT rank={rank} remaining_images={len(subset)} weights={"EMA" if use_ema else "model"}', flush=True)
    with torch.inference_mode():
        for step, (images, targets) in enumerate(loader):
            images = images.to(device)
            outputs = model(images)
            sizes = torch.stack([t['orig_size'] for t in targets]).to(device)
            data = detection_arrays(outputs, sizes, lqe, project, tap.values, post.num_top_queries)
            thumbnails = F.interpolate(images.mean(1, keepdim=True), size=(32, 32), mode='area').flatten(1)
            for i, target in enumerate(targets):
                image_id = int(target['image_id'])
                arrays = {k: value[i].cpu().numpy() for k, value in data.items()}
                for key in ('roi', 'border'):
                    arrays[key] = arrays[key].astype(np.float16)
                arrays.update(image_id=np.full(len(arrays['score']), image_id, dtype=np.int64),
                              thumbnail=thumbnails[i].cpu().numpy())
                write_npz(root / 'features' / f'{image_id}.npz', **arrays)
            print(f'EXPORT rank={rank} batch={step + 1}/{len(loader)}', flush=True)
    tap.close()
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def load_features(root, ann):
    ids = sorted(int(im['id']) for im in ann['images'])
    records = []
    for image_id in ids:
        with np.load(Path(root) / 'features' / f'{image_id}.npz', allow_pickle=False) as record:
            records.append(dict(record))
    return ({k: np.concatenate([r[k] for r in records]) for k in records[0] if k != 'thumbnail'},
            np.array(ids), np.stack([r['thumbnail'] for r in records]))


def thumbnail_groups(thumbnails, threshold=.95):
    """Correlation connected components, no labels, fingerprints or file hashes."""
    x = thumbnails.astype(np.float64)
    x -= x.mean(1, keepdims=True)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    x /= np.maximum(norm, 1e-12)
    parent = np.arange(len(x))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(*np.where(np.triu(x @ x.T >= threshold, 1))):
        parent[find(j)] = find(i)
    return np.array([find(i) for i in range(len(x))])


def grouped_folds(image_ids, groups, seed, folds=2):
    """Balance image counts without splitting a group or using any GT labels."""
    unique, counts = np.unique(np.asarray(groups).astype(str), return_counts=True)
    if len(unique) < folds:
        raise ValueError(f'Need at least {folds} distinct groups; got {len(unique)}.')
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    order = order[np.argsort(-counts[order], kind='stable')]
    loads = np.zeros(folds, int)
    assignment = {}
    for i in order:
        fold = int(np.argmin(loads))
        assignment[unique[i]] = fold
        loads[fold] += counts[i]
    return {int(i): assignment[str(g)] for i, g in zip(image_ids, groups)}


def overlap(boxes, gt, ioa=False):
    if len(gt) == 0:
        return np.zeros((len(boxes), 0), np.float32)
    intersection = np.maximum(0, np.minimum(boxes[:, None, 2:], gt[None, :, 2:]) -
                              np.maximum(boxes[:, None, :2], gt[None, :, :2])).prod(-1)
    area = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(-1)
    denominator = area[:, None] if ioa else (area[:, None] + (gt[:, 2:] - gt[:, :2]).prod(-1) - intersection)
    return intersection / np.maximum(denominator, 1e-12)


def targets_for(d, ann):
    """Same-class and class-agnostic max IoU. Separate nearest-class correctness."""
    annotations = {}
    for item in ann['annotations']:
        annotations.setdefault(int(item['image_id']), []).append(item)
    y = np.zeros((len(d['score']), 2), np.float32)
    correct = np.zeros(len(y), bool)
    eligible = np.ones(len(y), bool)
    for image_id in np.unique(d['image_id']):
        rows = np.flatnonzero(d['image_id'] == image_id)
        items = annotations.get(int(image_id), [])
        normal = [g for g in items if not g.get('iscrowd', 0) and not g.get('ignore', 0)]
        ignored = [g for g in items if g.get('iscrowd', 0) or g.get('ignore', 0)]
        def boxes(items):
            a = np.array([g['bbox'] for g in items], np.float32).reshape(-1, 4)
            a[:, 2:] += a[:, :2]
            return a
        iou = overlap(d['box'][rows], boxes(normal))
        if normal:
            classes = np.array([g['category_id'] for g in normal])
            nearest = iou.argmax(1)
            y[rows, 1] = iou[np.arange(len(rows)), nearest]
            y[rows, 0] = (iou * (d['label'][rows, None] == classes[None, :])).max(1)
            correct[rows] = (y[rows, 1] >= .5) & (classes[nearest] == d['label'][rows])
        if ignored:
            ignored_overlap = overlap(d['box'][rows], boxes(ignored), ioa=True).max(1)
            eligible[rows] = (ignored_overlap < .5) | (y[rows, 1] >= .5)
    return y, correct, eligible


def feature_matrix(d, mode):
    s = np.clip(d['score'][:, None], 1e-6, 1 - 1e-6)
    parts = [s, np.log(s / (1 - s))]
    if mode != 'score':
        parts += [d['probs'], d['quality'], np.eye(d['probs'].shape[1])[d['label']]]
    if mode.startswith('cls_fdr'):
        # Size is a shared control: the local probe must improve beyond size cues.
        parts += [d['fdr'], d['box_normalized'], np.log1p(d['orig_size'] / 640)]
    if mode in ('cls_fdr_roi', 'cls_fdr_local'):
        parts.append(d['roi'])
    if mode in ('cls_fdr_border', 'cls_fdr_local'):
        parts.append(d['border'])
    return np.concatenate(parts, axis=1).astype(np.float32)


def make_net(dim):
    return nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Dropout(.2),
                         nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))


def fit_fold(x, y, score, eligible, train, test, args, job, device, writer=None):
    root = Path(args.output)
    checkpoint = root / 'models' / f'{job}.pth'
    train = train & eligible
    if not train.any():
        raise ValueError('No eligible training candidates in fold.')
    torch.manual_seed(args.fit_seed)
    mu, sd = x[train].mean(0), np.maximum(x[train].std(0), 1e-3)
    net = make_net(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    epoch = 0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        net.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        mu, sd, epoch = state['mean'], state['std'], state['epoch'] + 1
        torch.set_rng_state(state['rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state(state['cuda_rng'], device)
    X = torch.as_tensor((x[train] - mu) / sd, device=device)
    Y = torch.as_tensor(y[train], device=device)
    weights = torch.as_tensor(score[train] / max(score[train].mean(), 1e-8), device=device)
    for ep in range(epoch, args.epochs):
        net.train()
        loss_sum, weight_sum = 0., 0.
        for idx in torch.randperm(len(X), device=device).split(args.fit_batch_size):
            loss_per_row = F.binary_cross_entropy_with_logits(net(X[idx]), Y[idx], reduction='none').mean(-1)
            loss = (loss_per_row * weights[idx]).sum() / weights[idx].sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float((loss_per_row.detach() * weights[idx]).sum())
            weight_sum += float(weights[idx].sum())
        value = loss_sum / weight_sum
        save_checkpoint(checkpoint, dict(model=net.state_dict(), optimizer=optimizer.state_dict(),
            mean=mu, std=sd, epoch=ep, rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
            job=job, targets=TARGETS, input_dim=x.shape[1]))
        row = dict(job=job, epoch=ep, train_loss=value)
        with (root / 'models' / f'{job}.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row) + '\n')
        if writer is not None:
            writer.add_scalar(f'loss/{job}', value, ep)
        if ep == 0 or (ep + 1) % 10 == 0 or ep + 1 == args.epochs:
            print(f'FIT {job} epoch={ep + 1}/{args.epochs} loss={value:.6f}', flush=True)
    net.eval()
    validation = torch.as_tensor((x[test] - mu) / sd, device=device)
    with torch.inference_mode():
        q = torch.cat([net(part).sigmoid().cpu() for part in validation.split(args.fit_batch_size)]).numpy()
    write_npz(root / 'oof' / f'{job}.npz', rows=np.flatnonzero(test), quality=q)
    return q


def competition_score(d):
    return d['score'] * np.sqrt(d['score'] / np.maximum(d['probs'].sum(1), 1e-8))


def coco_metrics(ann_path, d, scores):
    from faster_coco_eval import COCO, COCOeval_faster
    b = d['box']
    detections = [dict(image_id=int(i), category_id=int(c), score=float(s),
                      bbox=[float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])])
                  for i, c, s, box in zip(d['image_id'], d['label'], scores, b)]
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ann_path))
        evaluation = COCOeval_faster(gt, gt.loadRes(detections), 'bbox')
        evaluation.params.imgIds = sorted(int(v) for v in np.unique(d['image_id']))
        evaluation.params.maxDets = [1, 10, 100]
        evaluation.evaluate()
        evaluation.accumulate()
        evaluation.summarize()
    return dict(zip(('AP', 'AP50', 'AP75'), (evaluation.stats[:3] * 100).tolist()))


def conditional_geometry(d, y, correct, prediction):
    from scipy.stats import rankdata
    values, counts = [], []
    # Fixed score bins and classes, all chosen without looking at model improvements.
    bins = np.minimum((d['score'] * 10).astype(int), 9)
    for label in np.unique(d['label']):
        for bucket in range(10):
            rows = correct & (d['label'] == label) & (bins == bucket)
            if rows.sum() < 15:
                continue
            a, b = rankdata(prediction[rows]), rankdata(y[rows, 1])
            if a.std() == 0 or b.std() == 0:
                continue
            values.append(float(np.corrcoef(a, b)[0, 1]))
            counts.append(int(rows.sum()))
    return dict(spearman=float(np.average(values, weights=counts)) if counts else None,
                candidates=sum(counts), strata=len(counts))


def create_folds(args, image_ids, thumbnails):
    if args.groups:
        mapping = json.loads(Path(args.groups).read_text(encoding='utf-8'))
        groups = [str(mapping[str(int(i))]) for i in image_ids]
        source = 'provided_scene_groups'
    else:
        groups = thumbnail_groups(thumbnails, args.group_threshold).astype(str).tolist()
        source = 'thumbnail_correlation_components_approximate'
    return dict(source=source, group_threshold=args.group_threshold, n_groups=len(set(groups)),
                groups={str(int(i)): g for i, g in zip(image_ids, groups)},
                repetitions=[{str(i): f for i, f in grouped_folds(image_ids, groups, args.seed + r).items()}
                             for r in range(args.repeats)])


def fit_probes(args, d, ann, rank, world, device):
    from torch.utils.tensorboard import SummaryWriter
    root = Path(args.output)
    folds = json.loads((root / 'folds.json').read_text(encoding='utf-8'))
    y, correct, eligible = targets_for(d, ann)
    del correct
    writer = SummaryWriter(str(root / 'summary' / f'rank{rank}'))
    job_index = 0
    for mode in MODES:
        x = feature_matrix(d, mode)
        for rep in range(args.repeats):
            assignment = folds['repetitions'][rep]
            fold_ids = np.array([assignment[str(int(i))] for i in d['image_id']])
            for fold in (0, 1):
                job = f'{mode}_rep{rep}_fold{fold}'
                assigned = job_index % world == rank
                job_index += 1
                if not assigned or (root / 'oof' / f'{job}.npz').exists():
                    continue
                args.fit_seed = args.seed + rep * 100 + fold
                fit_fold(x, y, d['score'], eligible, fold_ids != fold, fold_ids == fold,
                         args, job, device, writer)
    writer.close()


def reference_metrics(args, d, ann):
    path = Path(args.output) / 'reference_metrics.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    y, correct, eligible = targets_for(d, ann)
    score = d['score']
    result = dict(baseline=coco_metrics(args.ann, d, score),
                  competition=coco_metrics(args.ann, d, competition_score(d)),
                  oracle={target: coco_metrics(args.ann, d, score * y[:, j])
                          for j, target in enumerate(TARGETS)},
                  geometry_baseline=conditional_geometry(d, y, correct & eligible, score))
    write_json(path, result)
    print('REFERENCE', json.dumps(result), flush=True)
    return result


def report(args):
    from torch.utils.tensorboard import SummaryWriter
    root = Path(args.output)
    ann = json.loads(Path(args.ann).read_text(encoding='utf-8'))
    d, ids, thumbs = load_features(root, ann)
    del ids, thumbs
    y, correct, eligible = targets_for(d, ann)
    correct &= eligible
    score = d['score']
    reference = reference_metrics(args, d, ann)
    baseline, competition = reference['baseline'], reference['competition']
    result = dict(**reference, probes={},
                  units='AP points (COCO stats multiplied by 100)',
                  note='Frozen detector; grouped OOF calibration on val; repeats are not independent detector seeds.')
    folds = json.loads((root / 'folds.json').read_text(encoding='utf-8'))
    result['grouping'] = {k: v for k, v in folds.items() if k not in ('groups', 'repetitions')}
    for mode in MODES:
        rows = []
        for rep in range(args.repeats):
            q = np.empty((len(score), 2), np.float32)
            count = np.zeros(len(score), int)
            assignment = folds['repetitions'][rep]
            expected = np.array([assignment[str(int(i))] for i in d['image_id']])
            for fold in (0, 1):
                with np.load(root / 'oof' / f'{mode}_rep{rep}_fold{fold}.npz') as saved:
                    index = saved['rows']
                    if not np.array_equal(index, np.flatnonzero(expected == fold)):
                        raise ValueError('OOF prediction rows do not match the saved fold.')
                    q[index] = saved['quality']
                    count[index] += 1
            if not np.all(count == 1):
                raise ValueError('Every candidate must have exactly one held-out prediction.')
            entry = {target: coco_metrics(args.ann, d, score * q[:, j]) for j, target in enumerate(TARGETS)}
            entry['geometry_quality_correlation'] = conditional_geometry(d, y, correct, q[:, 1])
            entry['geometry_rank_correlation'] = conditional_geometry(d, y, correct, score * q[:, 1])
            rows.append(entry)
            write_npz(root / 'ranks' / f'{mode}_rep{rep}.npz', image_id=d['image_id'],
                      query_id=d['query_id'], label=d['label'], box=d['box'],
                      original_score=score, quality=q, scores=score[:, None] * q)
            print('RESULT', mode, rep, json.dumps(entry), flush=True)
        dim = feature_matrix(d, mode).shape[1]
        means, deviations = {}, {}
        for target in TARGETS:
            values = np.array([[row[target][m] for m in ('AP', 'AP50', 'AP75')] for row in rows])
            means[target] = dict(zip(('AP', 'AP50', 'AP75'), values.mean(0).tolist()))
            deviations[target] = dict(zip(('AP', 'AP50', 'AP75'), values.std(0).tolist()))
        result['probes'][mode] = dict(input_dim=dim, parameters=sum(p.numel() for p in make_net(dim).parameters()),
                                     mean=means, std=deviations, repetitions=rows)
    with SummaryWriter(str(root / 'summary' / 'evaluation')) as writer:
        for mode, result_row in result['probes'].items():
            for rep, entry in enumerate(result_row['repetitions']):
                for target in TARGETS:
                    for metric, value in entry[target].items():
                        writer.add_scalar(f'{mode}/{target}/{metric}', value, rep)
    local = result['probes']['cls_fdr_local']
    control = result['probes']['cls_fdr']
    gaps = [a['same_class']['AP'] - b['same_class']['AP']
            for a, b in zip(local['repetitions'], control['repetitions'])]
    existing = dict(original=baseline['AP'], competition=competition['AP'],
                    **{name: result['probes'][name]['mean']['same_class']['AP']
                       for name in ('score', 'cls', 'cls_fdr')})
    best_control = max(existing, key=existing.get)
    result['local_comparison'] = dict(
        delta_vs_cls_fdr=float(np.mean(gaps)), paired_deltas=gaps,
        delta_vs_competition=local['mean']['same_class']['AP'] - competition['AP'],
        best_existing_control=best_control,
        delta_vs_best_existing=local['mean']['same_class']['AP'] - existing[best_control],
        geometry_delta_vs_cls_fdr=local['mean']['geometry']['AP'] - control['mean']['geometry']['AP'],
        preliminary_candidate=bool(min(gaps) > 0 and
                                   local['mean']['same_class']['AP'] - existing[best_control] >= 1))
    write_json(root / 'results.json', result)
    lines = ['# Local evidence ranking probe', '', result['note'], '',
             '| Variant | AP | AP50 | AP75 | geometric-target AP |', '|---|---:|---:|---:|---:|']
    for name, value in (('original', baseline), ('competition rule', competition)):
        lines.append(f"| {name} | {value['AP']:.3f} | {value['AP50']:.3f} | {value['AP75']:.3f} | — |")
    for mode, row in result['probes'].items():
        mean, std = row['mean']['same_class'], row['std']['same_class']
        lines.append(f"| {mode} | {mean['AP']:.3f} ± {std['AP']:.3f} | {mean['AP50']:.3f} | "
                     f"{mean['AP75']:.3f} | {row['mean']['geometry']['AP']:.3f} |")
    lines += ['', 'Local vs controls:', '```json', json.dumps(result['local_comparison'], indent=2), '```', '',
              'Check conditional geometry correlations in results.json before attributing gains to localization.',
              'A negative result applies to this frozen representation and fixed descriptor/probe only.']
    (root / 'results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


def prepare(args, world):
    import yaml
    from engine.core import YAMLConfig
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'manifest.json').exists():
        raise FileExistsError('Existing run: use resume --output to continue it.')
    args.output = str(root)
    for name in ('config', 'checkpoint', 'ann', 'img_folder', 'groups'):
        if getattr(args, name, None):
            setattr(args, name, str(Path(getattr(args, name)).resolve()))
    cfg = YAMLConfig(args.config)
    cfg.yaml_cfg.pop('__include__', None)
    cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    cfg.yaml_cfg['val_dataloader']['dataset'].update(img_folder=args.img_folder, ann_file=args.ann)
    cfg.yaml_cfg['val_dataloader'].pop('batch_size', None)
    cfg.yaml_cfg['val_dataloader'].update(total_batch_size=args.export_batch_size,
                                         num_workers=args.workers, shuffle=False, drop_last=False)
    ops = cfg.yaml_cfg['val_dataloader']['dataset']['transforms']['ops']
    if [o['type'] for o in ops] != ['Resize', 'ConvertPILImage']:
        raise ValueError('Expected the baseline deterministic Resize + ConvertPILImage val transforms.')
    for folder in ('features', 'models', 'oof', 'ranks', 'summary'):
        (root / folder).mkdir(exist_ok=True)
    (root / 'resolved_config.yml').write_text(yaml.safe_dump(cfg.yaml_cfg, sort_keys=False), encoding='utf-8')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    status = subprocess.check_output(['git', 'status', '--short'], text=True)
    write_json(root / 'manifest.json', dict(args=vars(args), git_revision=revision, worktree_status=status,
        torch_version=torch.__version__, numpy_version=np.__version__, world_size=world,
        detector=dict(mode='frozen eval', precision='FP32', initialization=args.checkpoint,
                      weights='EMA if present, otherwise model', augmentation=ops,
                      original_training_protocol=cfg.yaml_cfg, export_global_batch=args.export_batch_size),
        probe=dict(targets=TARGETS, modes=MODES, folds=2, head=[64, 32, 2], dropout=.2,
                   optimizer='AdamW', minibatch_per_independent_fit=args.fit_batch_size,
                   projection_seeds=[173, 174], projection_channels=8,
                   local_cache_dtype='float16; converted to float32 for fitting',
                   roi_grid=3, side_points=8, normal_offsets=[-.05, 0, .05],
                   ema=False, amp=False, ranking='original score * estimated quality',
                   topk=cfg.postprocessor.num_top_queries,
                   evaluation='COCO bbox maxDets 100, same frozen candidates')))


def read_args(output):
    data = json.loads((Path(output) / 'manifest.json').read_text(encoding='utf-8'))
    return argparse.Namespace(**data['args'])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    run.add_argument('-c', '--config', required=True)
    run.add_argument('-r', '--checkpoint', required=True)
    run.add_argument('--ann', required=True)
    run.add_argument('--img-folder', required=True)
    run.add_argument('--output', required=True)
    run.add_argument('--groups', help='Optional JSON object mapping every image_id to a scene_id.')
    run.add_argument('--group-threshold', type=float, default=.95)
    run.add_argument('--export-batch-size', type=int, default=16, help='GLOBAL across export GPUs.')
    run.add_argument('--workers', type=int, default=2)
    run.add_argument('--epochs', type=int, default=60)
    run.add_argument('--fit-batch-size', type=int, default=4096)
    run.add_argument('--lr', type=float, default=1e-3)
    run.add_argument('--weight-decay', type=float, default=1e-3)
    run.add_argument('--repeats', type=int, default=3)
    run.add_argument('--seed', type=int, default=0)
    run.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    for command in ('resume', 'report'):
        sub.add_parser(command).add_argument('--output', required=True)
    return parser.parse_args()


def main():
    from datetime import timedelta
    import torch.distributed as dist
    args = parse_args()
    command, output = args.command, args.output
    if command != 'run':
        args = read_args(output)
    if command == 'report':
        report(args)
        return
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device('cuda', local_rank) if args.device == 'cuda' else torch.device('cpu')
    torch.set_num_threads(2)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo', timeout=timedelta(hours=6))
    def barrier():
        if world > 1:
            dist.barrier(device_ids=[local_rank] if device.type == 'cuda' else None)
    try:
        if args.export_batch_size < world or args.export_batch_size % world:
            raise ValueError('Global export batch must be positive and divisible by world size.')
        if command == 'run' and rank == 0:
            prepare(args, world)
        barrier()
        args = read_args(output)
        export_features(args, rank, world, device)
        barrier()
        ann = json.loads(Path(args.ann).read_text(encoding='utf-8'))
        d, ids, thumbs = load_features(args.output, ann)
        if rank == 0 and not (Path(args.output) / 'folds.json').exists():
            write_json(Path(args.output) / 'folds.json', create_folds(args, ids, thumbs))
        if rank == 0:
            reference_metrics(args, d, ann)
        barrier()
        fit_probes(args, d, ann, rank, world, device)
        barrier()
        if rank == 0:
            del d
            report(args)
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
