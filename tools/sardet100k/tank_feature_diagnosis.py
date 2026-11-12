"""Where does OGSOD tank evidence disappear between the input and the final score? Val only, no training, weights unchanged.

Commands (run from the repository root, RUN = E000 run directory, OUT = fresh output directory):
  collect RUN OUT   One forward per image; per-GT funnel rows, per-stage response statistics, frozen-feature samples.
  select  RUN OUT   Choose the 30 visualised samples from the collected rows (seeded).
  render  RUN OUT   Raw feature dumps and figures for the 30 samples.
  sweep   RUN OUT   Same weights at other input sizes (eval_spatial_size is cleared at runtime; nothing on disk changes).
  sweep_report RUN OUT   Tank AP and error counts from the sweep predictions.
  extras  RUN OUT   Spatial profile of the encoder tank map around GT centres, native saturation, train duplicates, train size mix.
  summary RUN OUT   Tables from the files above.

Definitions
  Response tiers: `raw` is the channel tensor as hooked; `aggregate` is its per-cell L2 norm (also mean absolute value);
  `display` is the aggregate mapped to colour. Statistics use raw and aggregate only, never display values.
  GT and ring means are weighted by each cell's overlap area with the region, because at stride 32 a 20 px box is less
  than one cell. Ring: guard = max(side/4, stride/2), width = max(side, 2*stride), other GT boxes excluded.
  Status follows error_gallery.match_image at score 0.3, IoU 0.5.
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from error_gallery import ROOT, iou_matrix, match_image  # noqa: E402

TANK = 3
SOURCES = ('ogsod', 'msar')
INPUT = 640
SCORE, IOU = .3, .5
STRIDES = {'input': 1, 'S2': 4, 'P3': 8, 'P4': 16, 'P5': 32, 'F3': 8, 'F4': 16, 'F5': 32, 'M3': 8, 'M4': 16, 'M5': 32, 'T3': 8, 'T4': 16, 'T5': 32}
FEATURE_STAGES = ('S2', 'P3', 'P4', 'P5', 'F3', 'F4', 'F5', 'M3', 'M4', 'M5')
PROB_STAGES = ('T3', 'T4', 'T5')
SWEEP_SIZES = {224: 64, 320: 64, 480: 48, 640: 32, 800: 24, 960: 16, 1280: 8}
NEG_MARGIN = 48.  # Keeps a negative point out of the GT's own stride-32 cell.


# ---- data ----
def load_val(out):
    annotations = json.load(open(ROOT / 'Annotations/val.json'))
    source = {int(k): v for k, v in json.load(open(out.parent / 'error_gallery_val/source.json')).items()}
    grouped = defaultdict(list)
    for a in annotations['annotations']:
        grouped[a['image_id']].append(a)
    return annotations, source, grouped


def image_targets(rows):
    cls = np.array([a['category_id'] for a in rows], int)
    box = np.array([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]] for a in rows], np.float32).reshape(-1, 4)
    return cls, box


def to_input(path, size=INPUT):
    import torchvision.transforms.functional as F
    image = Image.open(path).convert('RGB')
    return F.to_tensor(F.resize(image, [size, size])), image.size


def load_model(run, device):
    import torch
    from engine.core import YAMLConfig
    cfg = YAMLConfig(str(run / 'resolved_config.yml'))
    model, post = cfg.model, cfg.postprocessor
    model.load_state_dict(torch.load(run / 'best_stg2.pth', map_location='cpu', weights_only=False)['ema']['module'])
    return model.to(device).eval(), post.to(device).eval()


class Hooks:
    """Read-only forward hooks. Decoder layers are reached by setting eval_idx at runtime in the loaded object."""

    def __init__(self, model):
        self.data, self.model = {}, model
        stages = model.backbone.stages
        for name, index in (('S2', 0), ('P3', 1), ('P4', 2), ('P5', 3)):
            stages[index].register_forward_hook(self.save(name))
        model.decoder.enc_output.register_forward_hook(self.save('memory'))
        model.decoder.enc_score_head.register_forward_hook(self.save('enc_logits'))

    def save(self, name):
        def hook(_, __, output):
            self.data[name] = output.detach()
        return hook

    def run(self, x):
        """Returns stage tensors plus per-decoder-layer logits and boxes, in the order of the baseline forward."""
        import torch
        model, decoder = self.model, self.model.decoder
        neck = model.encoder(model.backbone(x))
        layers = []
        for k in range(decoder.decoder.num_layers):
            decoder.decoder.eval_idx = k
            out = decoder(neck)
            layers.append((torch.sigmoid(out['pred_logits'])[0].cpu().numpy(), out['pred_boxes'][0].cpu().numpy()))
        last = out
        memory, logits = self.data['memory'], self.data['enc_logits']
        anchors, valid = decoder.anchors, decoder.valid_mask
        boxes = torch.sigmoid(decoder.enc_bbox_head(memory) + anchors)
        boxes = torch.where(valid, boxes, torch.zeros_like(boxes))
        shapes = [f.shape[-2:] for f in neck]
        counts = [h * w for h, w in shapes]
        prob = torch.sigmoid(logits)
        stages = {k: self.data[k][0].cpu().numpy() for k in ('S2', 'P3', 'P4', 'P5')}
        for i, (name, n, (h, w)) in enumerate(zip(('3', '4', '5'), counts, shapes)):
            stages[f'F{name}'] = neck[i][0].cpu().numpy()
            start = sum(counts[:i])
            stages[f'M{name}'] = memory[0, start:start + n].T.reshape(-1, h, w).cpu().numpy()
            stages[f'T{name}'] = prob[0, start:start + n, TANK].reshape(1, h, w).cpu().numpy()
        best = logits.max(-1).values[0]
        order = torch.argsort(best, descending=True)
        rank = torch.zeros_like(order)
        rank[order] = torch.arange(len(order), device=order.device)
        enc = dict(prob=prob[0].cpu().numpy(), boxes=boxes[0].cpu().numpy(), rank=rank.cpu().numpy(), selected=order[:decoder.num_queries].cpu().numpy(),
                   counts=counts, shapes=[tuple(int(v) for v in s) for s in shapes])
        return stages, enc, layers, last


def xyxy(cxcywh, size=INPUT):
    out = np.empty_like(cxcywh)
    out[..., 0], out[..., 1] = cxcywh[..., 0] - cxcywh[..., 2] / 2, cxcywh[..., 1] - cxcywh[..., 3] / 2
    out[..., 2], out[..., 3] = cxcywh[..., 0] + cxcywh[..., 2] / 2, cxcywh[..., 1] + cxcywh[..., 3] / 2
    return out * size


# ---- region statistics ----
def cell_weights(box, stride, h, w):
    ex, ey = np.arange(w + 1) * stride, np.arange(h + 1) * stride
    ox = np.clip(np.minimum(ex[1:], box[2]) - np.maximum(ex[:-1], box[0]), 0, None)
    oy = np.clip(np.minimum(ey[1:], box[3]) - np.maximum(ey[:-1], box[1]), 0, None)
    return np.outer(oy, ox)


def grow(box, margin):
    return np.array([box[0] - margin, box[1] - margin, box[2] + margin, box[3] + margin], np.float32)


def region_masks(box, others, stride, h, w):
    """Cell weights of the box and of its ring; other GT boxes (with guard) are removed from the ring."""
    side = float(max(box[2] - box[0], box[3] - box[1]))
    guard, width = max(side / 4, stride / 2), max(side, 2 * stride)
    inner = cell_weights(box, stride, h, w)
    ring = cell_weights(grow(box, guard + width), stride, h, w) - cell_weights(grow(box, guard), stride, h, w)
    for other in others:
        ring -= np.minimum(cell_weights(grow(other, guard), stride, h, w), np.clip(ring, 0, None))
    return inner, np.clip(ring, 0, None)


def weighted(values, weights):
    total = weights.sum()
    if total <= 0:
        return float('nan'), float('nan')
    mean = float((values * weights).sum() / total)
    return mean, float(np.sqrt(max(((values - mean) ** 2 * weights).sum() / total, 0)))


def region_stats(tensor, stride, box, others):
    """tensor: C x H x W raw values. Returns aggregate statistics plus the raw-vector cosine distance between box and ring."""
    _, h, w = tensor.shape
    l2 = np.sqrt((tensor ** 2).sum(0)) if tensor.shape[0] > 1 else np.abs(tensor[0])
    absolute = np.abs(tensor).mean(0)
    inner, ring = region_masks(box, others, stride, h, w)
    if inner.sum() <= 0:
        return None
    gt_mean, _ = weighted(l2, inner)
    ring_mean, ring_std = weighted(l2, ring)
    abs_gt, _ = weighted(absolute, inner)
    abs_ring, _ = weighted(absolute, ring)
    row = dict(gt_mean=gt_mean, gt_peak=float(l2[inner > 0].max()), ring_mean=ring_mean, ring_std=ring_std, ratio=gt_mean / ring_mean if ring_mean > 0 else float('nan'),
               z=(gt_mean - ring_mean) / ring_std if ring_std > 0 else float('nan'), abs_ratio=abs_gt / abs_ring if abs_ring > 0 else float('nan'),
               image_pct=float((l2 < gt_mean).mean()), peak_pct=float((l2 < l2[inner > 0].max()).mean()), ring_area=float(ring.sum()))
    if tensor.shape[0] > 1 and ring.sum() > 0:
        a = (tensor * inner).sum((1, 2))
        b = (tensor * ring).sum((1, 2))
        row['cos_dist'] = float(1 - (a @ b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
    return row


def center_cell(tensor, stride, x, y):
    _, h, w = tensor.shape
    return tensor[:, min(max(int(y // stride), 0), h - 1), min(max(int(x // stride), 0), w - 1)]


# ---- collect ----
def funnel_row(g, box, enc, layers, enc_all_boxes, centers_all):
    """One GT's trail: encoder cell scores and selection, then decoder layers. Boxes are in input pixels."""
    inside = (centers_all[:, 0] >= box[0]) & (centers_all[:, 0] < box[2]) & (centers_all[:, 1] >= box[1]) & (centers_all[:, 1] < box[3])
    row = {}
    start = 0
    levels = np.zeros(len(centers_all), int)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    for i, (n, stride) in enumerate(zip(enc['counts'], (8, 16, 32))):
        levels[start:start + n] = i
        h, w = enc['shapes'][i]
        cell = min(int(cy // stride), h - 1) * w + min(int(cx // stride), w - 1)
        row[f'enc_prob_center_L{stride}'] = float(enc['prob'][start + cell, TANK])
        row[f'enc_rank_center_L{stride}'] = int(enc['rank'][start + cell])
        start += n
    if not inside.any():
        inside = np.zeros(len(centers_all), bool)
        start = 0
        for i, (n, stride) in enumerate(zip(enc['counts'], (8, 16, 32))):
            h, w = enc['shapes'][i]
            inside[start + min(int(cy // stride), h - 1) * w + min(int(cx // stride), w - 1)] = True
            start += n
    probs = enc['prob'][:, TANK]
    picked = np.flatnonzero(inside)[np.argmax(probs[inside])]
    row.update(enc_prob_max_in_gt=float(probs[picked]), enc_prob_level=int((8, 16, 32)[levels[picked]]), enc_rank_best_in_gt=int(enc['rank'][inside].min()),
               enc_selected_in_gt=int((enc['rank'][inside] < len(enc['selected'])).sum()), enc_cutoff_prob=float(enc['prob'].max(1)[enc['selected'][-1]]))
    all_iou = iou_matrix(enc_all_boxes, box[None])[:, 0]
    best_cell = int(all_iou.argmax())
    selected_iou = all_iou[enc['selected']]
    row.update(enc_best_iou_all=float(all_iou[best_cell]), enc_best_iou_all_rank=int(enc['rank'][best_cell]), enc_best_iou_selected=float(selected_iou.max()),
               enc_selected_iou_ge_01=int((selected_iou >= .1).sum()))
    final_iou = None
    for k, (scores, boxes) in enumerate(layers):
        iou = iou_matrix(xyxy(boxes), box[None])[:, 0]
        best = int(iou.argmax())
        good = iou >= IOU
        tank = scores[:, TANK]
        row.update({f'dec{k}_best_iou': float(iou[best]), f'dec{k}_score_at_best_iou': float(tank[best]), f'dec{k}_best_tp_score': float(tank[good].max()) if good.any() else 0.,
                    f'dec{k}_count_iou50': int(good.sum())})
        final_iou, final_scores, final_boxes = iou, scores, boxes
    good = final_iou >= IOU
    query = int(np.flatnonzero(good)[np.argmax(final_scores[good, TANK])]) if good.any() else int(final_iou.argmax())
    row['q_level'] = int((8, 16, 32)[levels[enc['selected'][query]]])
    row['q_init_iou'] = float(selected_iou[query])
    row['q_enc_rank'] = int(query)
    row['q_enc_tank_prob'] = float(probs[enc['selected'][query]])
    for k, (scores, boxes) in enumerate(layers):
        row[f'q_iou_{k}'] = float(iou_matrix(xyxy(boxes)[query][None], box[None])[0, 0])
        row[f'q_score_{k}'] = float(scores[query, TANK])
    row['q_top_class'] = int(final_scores[query].argmax())
    row['q_top_class_score'] = float(final_scores[query].max())
    wrong = (final_iou >= .1) & (final_iou < IOU)
    row['mislocalised_top_score'] = float(final_scores[wrong, TANK].max()) if wrong.any() else 0.
    return row, xyxy(final_boxes)[query], float(final_scores[query, TANK]), float(final_iou[query])


def negatives(rng, boxes, side_of, count, size=INPUT):
    """Points at least NEG_MARGIN from every GT edge, near one reference GT, so they lie in another stride-32 cell."""
    points = []
    for g in side_of:
        cx, cy = (boxes[g][0] + boxes[g][2]) / 2, (boxes[g][1] + boxes[g][3]) / 2
        half = max(boxes[g][2] - boxes[g][0], boxes[g][3] - boxes[g][1]) / 2
        for _ in range(count):
            for _ in range(20):
                angle, radius = rng.uniform(0, 2 * np.pi), half + NEG_MARGIN * rng.uniform(1, 2)
                x, y = cx + radius * np.cos(angle), cy + radius * np.sin(angle)
                if 0 <= x < size and 0 <= y < size and ((x < boxes[:, 0] - NEG_MARGIN) | (x > boxes[:, 2] + NEG_MARGIN) | (y < boxes[:, 1] - NEG_MARGIN) | (y > boxes[:, 3] + NEG_MARGIN)).all():
                    points.append((x, y, g))
                    break
    return points


def collect(args):
    import torch
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    annotations, source, grouped = load_val(out)
    images = [i for i in annotations['images'] if source[i['id']] in SOURCES and any(a['category_id'] == TANK for a in grouped[i['id']])]
    model, post = load_model(args.run, args.device)
    hooks = Hooks(model)
    saved = np.load(args.run / 'independent_verify_20261106/val_predictions.npz')
    gt_rows, stage_rows, probe_meta, probe_vectors = [], [], [], defaultdict(list)
    rng = np.random.default_rng(args.seed)
    differences = []
    with torch.no_grad():
        for count, image in enumerate(images):
            image_id = image['id']
            x, (w, h) = to_input(ROOT / 'JPEGImages/val' / image['file_name'])
            x = x[None].to(args.device)
            stages, enc, layers, last = hooks.run(x)
            result = post(last, torch.tensor([[w, h]], device=args.device, dtype=torch.float32))[0]
            p_cls, p_score, p_box = result['labels'].cpu().numpy().astype(int), result['scores'].cpu().numpy(), result['boxes'].cpu().numpy()
            mask = saved['image_id'] == image_id
            if mask.any():  # The saved predictions were made by the independent evaluation; this forward must match them.
                saved_top = np.sort(saved['score'][mask])[::-1][:20]
                differences.append(float(np.abs(np.sort(p_score)[::-1][:len(saved_top)] - saved_top).max()))
            rows = grouped[image_id]
            g_cls, g_native = image_targets(rows)
            sx, sy = INPUT / w, INPUT / h
            g_input = g_native * np.array([sx, sy, sx, sy], np.float32)
            keep, kinds, targets, fn_kind, low_box, taken = match_image(p_cls, p_score, p_box, g_cls, g_native, SCORE, IOU)
            all_iou = iou_matrix(p_box, g_native)
            stages['input'] = x[0, :1].cpu().numpy()
            centers = []
            for i, stride in enumerate((8, 16, 32)):
                H, W = enc['shapes'][i]
                yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
                centers.append(np.stack([(xx.ravel() + .5) * stride, (yy.ravel() + .5) * stride], 1))
            centers = np.concatenate(centers).astype(np.float32)
            enc_all_boxes = xyxy(enc['boxes'])
            tank_gt = np.flatnonzero(g_cls == TANK)
            all_centers = np.stack([(g_input[:, 0] + g_input[:, 2]) / 2, (g_input[:, 1] + g_input[:, 3]) / 2], 1)
            for g in tank_gt:
                box = g_input[g]
                annotation_id = rows[g]['id']
                others = np.delete(g_input, g, axis=0)
                funnel, q_box, q_score, q_iou = funnel_row(g, box, enc, layers, enc_all_boxes, centers)
                same = p_cls == TANK
                ious = np.where(same, all_iou[:, g], 0)
                best_box = int(ious.argmax())
                distance = np.linalg.norm(np.delete(all_centers, g, axis=0) - all_centers[g], axis=1) if len(others) else np.array([1e4])
                native_side = float(max(g_native[g][2] - g_native[g][0], g_native[g][3] - g_native[g][1]))
                input_side = float(max(box[2] - box[0], box[3] - box[1]))
                tp_score = tp_iou = float('nan')
                if taken[g] >= 0:
                    d = keep[taken[g]]
                    tp_score, tp_iou = float(p_score[d]), float(all_iou[d, g])
                gt_rows.append(dict(
                    gt_id=annotation_id, image_id=image_id, file_name=image['file_name'], source=source[image_id], native_w=w, native_h=h,
                    gt_x1=float(g_native[g][0]), gt_y1=float(g_native[g][1]), gt_x2=float(g_native[g][2]), gt_y2=float(g_native[g][3]),
                    native_gw=float(g_native[g][2] - g_native[g][0]), native_gh=float(g_native[g][3] - g_native[g][1]), native_side=native_side,
                    input_gw=float(box[2] - box[0]), input_gh=float(box[3] - box[1]), input_side=input_side, n_gt=len(g_cls), nn_input=float(distance.min()),
                    status='TP' if fn_kind[g] is None else fn_kind[g], tp_score=tp_score, tp_iou=tp_iou,
                    best_box_iou=float(ious[best_box]), best_box_score=float(p_score[best_box]),
                    res_dx=float((p_box[best_box][0] + p_box[best_box][2] - g_native[g][0] - g_native[g][2]) / 2), res_dy=float((p_box[best_box][1] + p_box[best_box][3] - g_native[g][1] - g_native[g][3]) / 2),
                    res_dw=float(p_box[best_box][2] - p_box[best_box][0] - g_native[g][2] + g_native[g][0]), res_dh=float(p_box[best_box][3] - p_box[best_box][1] - g_native[g][3] + g_native[g][1]),
                    q_box=[round(float(v), 2) for v in q_box], q_final_score=q_score, q_final_iou=q_iou, **funnel))
                for name, tensor in stages.items():
                    row = region_stats(tensor, STRIDES[name], box, others)
                    if row:
                        stage_rows.append(dict(kind='gt', owner=annotation_id, stage=name, image_id=image_id, **row))
                cx, cy = all_centers[g]
                probe_meta.append(dict(kind='gt', owner=annotation_id, image_id=image_id, source=source[image_id]))
                for name in (*FEATURE_STAGES, *PROB_STAGES):
                    probe_vectors[name].append(center_cell(stages[name], STRIDES[name], cx, cy))
            # Tank background false positives: same statistics, same probe points.
            for position, (d, kind) in enumerate(zip(keep, kinds)):
                if kind != 'Bkg' or p_cls[d] != TANK:
                    continue
                box = p_box[d] * np.array([sx, sy, sx, sy], np.float32)
                tag = f'fp_{image_id}_{position}'
                for name, tensor in stages.items():
                    row = region_stats(tensor, STRIDES[name], box, g_input)
                    if row:
                        stage_rows.append(dict(kind='fp', owner=tag, stage=name, image_id=image_id, score=float(p_score[d]), **row))
                probe_meta.append(dict(kind='fp', owner=tag, image_id=image_id, source=source[image_id], score=float(p_score[d])))
                for name in (*FEATURE_STAGES, *PROB_STAGES):
                    probe_vectors[name].append(center_cell(stages[name], STRIDES[name], (box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
            for x_, y_, g in negatives(rng, g_input, tank_gt, 2):
                probe_meta.append(dict(kind='neg', owner=rows[g]['id'], image_id=image_id, source=source[image_id]))
                for name in (*FEATURE_STAGES, *PROB_STAGES):
                    probe_vectors[name].append(center_cell(stages[name], STRIDES[name], x_, y_))
            print('collect', count + 1, len(images), flush=True)
    json.dump(gt_rows, open(out / 'gt_rows.json', 'w'))
    json.dump(stage_rows, open(out / 'stage_rows.json', 'w'))
    json.dump(probe_meta, open(out / 'probe_meta.json', 'w'))
    np.savez(out / 'probe_vectors.npz', **{k: np.stack(v).astype(np.float32) for k, v in probe_vectors.items()})
    counts = Counter((r['source'], r['status']) for r in gt_rows)
    fp = Counter(r['owner'] for r in stage_rows if r['kind'] == 'fp' and r['stage'] == 'input')
    report = dict(images=len(images), status={f'{a}|{b}': c for (a, b), c in sorted(counts.items())}, tank_fp_bkg_in_tank_images=len(fp),
                  max_top20_score_difference_vs_saved=max(differences) if differences else None, seed=args.seed, input=INPUT, score=SCORE, iou=IOU)
    (out / 'collect_report.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report), flush=True)


# ---- select ----
def select(args):
    rng = random.Random(args.seed)
    rows = json.load(open(args.out / 'gt_rows.json'))
    ogsod = [r for r in rows if r['source'] == 'ogsod']
    miss, hit = [r for r in ogsod if r['status'] != 'TP'], [r for r in ogsod if r['status'] == 'TP']
    msar = [r for r in rows if r['source'] == 'msar' and r['status'] == 'TP']

    def pick_distinct(pool, n, used):
        rng.shuffle(pool)
        chosen = []
        for limit in (1, 2, 99):
            for r in pool:
                if len(chosen) >= n:
                    return chosen
                if r not in chosen and used[r['image_id']] < limit:
                    chosen.append(r)
                    used[r['image_id']] += 1
        return chosen

    # Misses keep the status mix of the population, by largest remainder.
    kinds = Counter(r['status'] for r in miss)
    exact = {k: 10 * v / len(miss) for k, v in kinds.items()}
    quota = {k: int(v) for k, v in exact.items()}
    for k in sorted(exact, key=lambda k: exact[k] - quota[k], reverse=True)[:10 - sum(quota.values())]:
        quota[k] += 1
    used_miss = Counter()
    chosen_miss = []
    for k, n in sorted(quota.items()):
        chosen_miss += pick_distinct([r for r in miss if r['status'] == k], n, used_miss)
    # Hits matched to each miss on input long side, from other images, so size is not the visible difference.
    used_hit = Counter(r['image_id'] for r in chosen_miss)
    chosen_hit = []
    for m in sorted(chosen_miss, key=lambda r: r['input_side']):
        pool = sorted((r for r in hit if r not in chosen_hit), key=lambda r: (used_hit[r['image_id']] > 0, abs(r['input_side'] - m['input_side']), r['gt_id']))
        chosen_hit.append(pool[0])
        used_hit[pool[0]['image_id']] += 1
    # MSAR: one image each, spread over the size quantiles of its own population.
    msar_sorted = sorted(msar, key=lambda r: r['input_side'])
    used_msar, chosen_msar = Counter(), []
    for i in range(10):
        a, b = len(msar_sorted) * i // 10, len(msar_sorted) * (i + 1) // 10
        chosen_msar += pick_distinct(msar_sorted[a:b], 1, used_msar)
    table = [dict(group=g, sample=f'{g}_{i:02d}', **r) for g, part in (('ogsod_hit', chosen_hit), ('ogsod_miss', chosen_miss), ('msar_hit', chosen_msar)) for i, r in enumerate(sorted(part, key=lambda r: r['input_side']))]
    json.dump(table, open(args.out / 'samples.json', 'w'), indent=1)
    fields = ['sample', 'group', 'gt_id', 'file_name', 'native_w', 'native_h', 'native_gw', 'native_gh', 'native_side', 'input_gw', 'input_gh', 'input_side', 'status', 'tp_score', 'tp_iou', 'best_box_score', 'best_box_iou', 'q_final_score', 'q_final_iou', 'n_gt', 'nn_input']
    with open(args.out / 'samples.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fields, extrasaction='ignore')
        writer.writeheader()
        for r in table:
            writer.writerow({k: round(r[k], 3) if isinstance(r[k], float) else r[k] for k in fields})
    print(Counter(r['group'] for r in table), quota, 'images', {g: len({r['image_id'] for r in table if r['group'] == g}) for g in ('ogsod_hit', 'ogsod_miss', 'msar_hit')}, flush=True)


# ---- render ----
ANCHORS = np.array([(0, 0, 4), (40, 11, 84), (101, 21, 110), (159, 42, 99), (212, 72, 66), (245, 125, 21), (250, 193, 39), (252, 255, 164)], np.float32)
LUT = np.stack([np.interp(np.linspace(0, 7, 256), np.arange(8), ANCHORS[:, c]) for c in range(3)], 1).astype(np.uint8)
GT_C, RING_C, PRED_C, BEST_C = (60, 230, 90), (255, 255, 255), (70, 190, 255), (255, 80, 200)
FONT, SMALL = ImageFont.load_default(size=13), ImageFont.load_default(size=11)
CELL = 240


def sampled(values, stride, x0, y0, span, display):
    """Nearest-neighbour map to display pixels: every feature cell stays visible, nothing is smoothed."""
    h, w = values.shape
    position = x0 + (np.arange(display) + .5) * span / display, y0 + (np.arange(display) + .5) * span / display
    jx, iy = np.clip((position[0] // stride).astype(int), 0, w - 1), np.clip((position[1] // stride).astype(int), 0, h - 1)
    return values[np.ix_(iy, jx)]


def colour(values, low, high):
    scaled = np.clip((values - low) / max(high - low, 1e-9), 0, 1)
    return LUT[(scaled * 255).astype(np.uint8)]


def scale_box(box, x0, y0, factor):
    return [(box[0] - x0) * factor, (box[1] - y0) * factor, (box[2] - x0) * factor, (box[3] - y0) * factor]


def panel(rgb, title, boxes=(), grid=None, x0=0, y0=0, factor=1.):
    image = Image.fromarray(rgb).convert('RGB')
    draw = ImageDraw.Draw(image)
    if grid:
        step = grid * factor
        if step >= 6:
            for v in np.arange(-(x0 % grid) * factor, image.width, step):
                draw.line([v, 0, v, image.height], fill=(120, 120, 120), width=1)
            for v in np.arange(-(y0 % grid) * factor, image.height, step):
                draw.line([0, v, image.width, v], fill=(120, 120, 120), width=1)
    for box, color, width in boxes:
        draw.rectangle(scale_box(box, x0, y0, factor), outline=color, width=width)
    left, top, right, bottom = draw.textbbox((0, 0), title, font=SMALL)
    draw.rectangle([0, 0, right - left + 6, bottom - top + 5], fill=(0, 0, 0))
    draw.text((3, 2 - top), title, fill=(240, 240, 240), font=SMALL)
    return image


def l2_map(tensor):
    return np.sqrt((tensor ** 2).sum(0)) if tensor.shape[0] > 1 else tensor[0]


def render(args):
    import torch
    out = args.out / 'samples'
    (out / 'raw').mkdir(parents=True, exist_ok=True)
    table = json.load(open(args.out / 'samples.json'))
    annotations, source, grouped = load_val(args.out)
    images = {i['id']: i for i in annotations['images']}
    model, post = load_model(args.run, args.device)
    hooks = Hooks(model)
    store = []
    with torch.no_grad():
        for row in table:
            image = images[row['image_id']]
            x, (w, h) = to_input(ROOT / 'JPEGImages/val' / image['file_name'])
            stages, enc, layers, last = hooks.run(x[None].to(args.device))
            result = post(last, torch.tensor([[w, h]], device=args.device, dtype=torch.float32))[0]
            stages['input'] = x[:1].numpy()
            np.savez_compressed(out / 'raw' / f"{row['sample']}.npz", **{k: v.astype(np.float16) for k, v in stages.items()})
            maps = {k: l2_map(v) for k, v in stages.items()}
            np.savez_compressed(out / 'raw' / f"{row['sample']}_l2.npz", **{k: v.astype(np.float32) for k, v in maps.items()})
            store.append((row, x.numpy().transpose(1, 2, 0), maps, result['labels'].cpu().numpy().astype(int), result['scores'].cpu().numpy(), result['boxes'].cpu().numpy(), w, h, grouped[row['image_id']]))
            print('forward', row['sample'], flush=True)
    bounds = {}
    for name in store[0][2]:
        values = np.concatenate([s[2][name].ravel() for s in store])
        bounds[name] = (0., 1.) if name in PROB_STAGES or name == 'input' else (float(np.percentile(values, 1)), float(np.percentile(values, 99.5)))
    (args.out / 'display_scales.json').write_text(json.dumps(bounds, indent=1))

    sheet_cells = defaultdict(dict)
    for row, rgb, maps, labels, scores, boxes, w, h, targets in store:
        sx, sy = INPUT / w, INPUT / h
        box = [row['gt_x1'] * sx, row['gt_y1'] * sy, row['gt_x2'] * sx, row['gt_y2'] * sy]
        side = row['input_side']
        span = int(min(max(120, side * 3), INPUT))
        x0 = int(np.clip((box[0] + box[2]) / 2 - span / 2, 0, INPUT - span))
        y0 = int(np.clip((box[1] + box[3]) / 2 - span / 2, 0, INPUT - span))
        factor = CELL / span
        other_boxes = [[a['bbox'][0] * sx, a['bbox'][1] * sy, (a['bbox'][0] + a['bbox'][2]) * sx, (a['bbox'][1] + a['bbox'][3]) * sy] for a in targets if a['id'] != row['gt_id']]
        marks = [(box, GT_C, 1)] + [(b, (0, 150, 60), 1) for b in other_boxes]

        def zoom(name, low=None, high=None, grid=True, extra=()):
            low_, high_ = bounds[name] if low is None else (low, high)
            stride = STRIDES[name]
            values = sampled(maps[name], stride, x0, y0, span, CELL)
            return panel(colour(values, low_, high_), f'{name} L2 [{low_:.2f},{high_:.2f}]' if name not in PROB_STAGES else f'{name} P(tank) [0,1]', (*marks, *extra), stride if grid else None, x0, y0, factor)

        def whole(name):
            low_, high_ = bounds[name]
            values = sampled(maps[name], STRIDES[name], 0, 0, INPUT, CELL)
            return panel(colour(values, low_, high_), f'{name} whole, shared scale', [(box, GT_C, 1)], None, 0, 0, CELL / INPUT)

        whole_input = np.asarray(Image.fromarray((rgb * 255).astype(np.uint8)).resize((CELL, CELL), Image.BILINEAR))
        zoom_input = (sampled(rgb[..., 0], 1, x0, y0, span, CELL) * 255).astype(np.uint8)
        zoom_rgb = np.stack([zoom_input] * 3, -1)
        low_ring = max(side / 4, 4)
        ring_outer = [box[0] - low_ring - max(side, 8), box[1] - low_ring - max(side, 8), box[2] + low_ring + max(side, 8), box[3] + low_ring + max(side, 8)]
        keep = (labels == TANK) & (scores >= .05)
        pred_marks = [([b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy], PRED_C if s >= SCORE else (150, 110, 255), 1) for b, s in zip(boxes[keep], scores[keep])]
        q_box = row['q_box']
        q_mark = [(q_box, BEST_C, 1)]
        pred_panel = panel(zoom_rgb, f"preds>=.05, pink query {row['q_final_score']:.2f}/{row['q_final_iou']:.2f}", [*marks, *pred_marks, *q_mark], None, x0, y0, factor)
        grid_rows = [
            [panel(whole_input, 'input whole (no stretch)', [(box, GT_C, 1)], None, 0, 0, CELL / INPUT), whole('P3'), whole('P4'), whole('P5')],
            [panel(zoom_rgb, f'input zoom x{factor:.1f} (white=ring S3)', [*marks, (ring_outer, RING_C, 1)], None, x0, y0, factor), zoom('P3'), zoom('P4'), zoom('P5')],
            [zoom('S2'), zoom('F3'), zoom('F4'), zoom('F5')],
            [pred_panel, zoom('T3'), zoom('T4'), zoom('T5')],
        ]
        per_image = []
        for name in ('input', 'P3', 'P4', 'P5'):
            whole_map = maps[name]
            low_, high_ = float(whole_map.min()), float(whole_map.max())
            per_image.append(panel(colour(sampled(whole_map, STRIDES[name], x0, y0, span, CELL), low_, high_), f'{name} PER-IMAGE min-max [{low_:.2f},{high_:.2f}]', marks, None, x0, y0, factor))
        grid_rows.append(per_image)
        header = 54
        canvas = Image.new('RGB', (CELL * 4 + 15, header + (CELL + 5) * len(grid_rows)), (24, 24, 24))
        draw = ImageDraw.Draw(canvas)
        confidence = f"TP score {row['tp_score']:.3f} IoU {row['tp_iou']:.3f}" if row['status'] == 'TP' else f"{row['status']}: best same-class box score {row['best_box_score']:.3f} IoU {row['best_box_iou']:.3f}"
        draw.text((5, 3), f"{row['sample']}  {row['file_name']}  image {w}x{h}  GT native {row['native_gw']:.1f}x{row['native_gh']:.1f}  input {row['input_gw']:.1f}x{row['input_gh']:.1f}", fill=(240, 240, 240), font=FONT)
        draw.text((5, 20), f"{confidence}  |  GT in image {row['n_gt']}, nearest GT {row['nn_input']:.0f} input px", fill=(240, 240, 240), font=FONT)
        draw.text((5, 37), 'Rows 1-4 use one fixed scale per stage across all 30 samples; row 5 is per-image min-max and is not comparable.', fill=(190, 190, 190), font=SMALL)
        for r, cells in enumerate(grid_rows):
            for c, cell in enumerate(cells):
                canvas.paste(cell, (c * (CELL + 5), header + r * (CELL + 5)))
        canvas.save(out / f"{row['sample']}.png")
        small = {name: zoom(name, grid=False).resize((128, 128), Image.NEAREST) for name in ('P3', 'P4', 'P5', 'F3', 'F4', 'T3', 'T4')}
        small['input'] = panel(zoom_rgb, '', marks, None, x0, y0, factor).resize((128, 128), Image.BILINEAR)
        small['P3_norm'] = per_image[1].resize((128, 128), Image.NEAREST)
        small['input_norm'] = per_image[0].resize((128, 128), Image.BILINEAR)
        for name, cell in small.items():
            draw = ImageDraw.Draw(cell)
            draw.rectangle([0, 0, 128, 12], fill=(0, 0, 0))
            draw.text((2, 1), f"{row['status']} {row['input_side']:.0f}px", fill=(255, 255, 255), font=SMALL)
            sheet_cells[row['group'], name][row['sample']] = cell
        print('figure', row['sample'], flush=True)

    def sheet(name, stages, note):
        left = 96
        groups = ('ogsod_hit', 'ogsod_miss', 'msar_hit')
        canvas = Image.new('RGB', (left + 131 * 10, 22 + 131 * len(stages) * len(groups)), (24, 24, 24))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 4), note, fill=(220, 220, 220), font=FONT)
        y = 22
        for group in groups:
            for stage in stages:
                draw.text((4, y + 50), f'{group}\n{stage}', fill=(240, 240, 240), font=SMALL)
                for i, key in enumerate(sorted(sheet_cells[group, stage])):
                    canvas.paste(sheet_cells[group, stage][key], (left + i * 131, y))
                y += 131
        canvas.save(args.out / f'{name}.png')

    sheet('sheet_backbone', ['input', 'P3', 'P4', 'P5'], 'Backbone aggregate L2 on a fixed scale per stage across all samples. Each column is sorted by input long side.')
    sheet('sheet_neck_head', ['input', 'F3', 'F4', 'T3', 'T4'], 'Neck aggregate L2 on a fixed scale, encoder tank probability on [0,1].')
    sheet('sheet_normalised_warning', ['input', 'input_norm', 'P3', 'P3_norm'], 'Fixed scale vs per-image min-max (_norm): the same data, showing what single-image normalisation hides.')


# ---- sweep ----
class Sweep:
    def __init__(self, images, size):
        self.images, self.size = images, size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        x, (w, h) = to_input(ROOT / 'JPEGImages/val' / self.images[index]['file_name'], self.size)
        return x, index, w, h


def sweep(args):
    import torch
    annotations, source, _ = load_val(args.out)
    images = [i for i in annotations['images'] if source[i['id']] in SOURCES]
    model, post = load_model(args.run, args.device)
    model.encoder.eval_spatial_size = None
    model.decoder.eval_spatial_size = None
    folder = args.out / 'sweep'
    folder.mkdir(parents=True, exist_ok=True)
    for size, batch in SWEEP_SIZES.items():
        path = folder / f'pred_{size}.npz'
        if path.exists():
            continue
        loader = torch.utils.data.DataLoader(Sweep(images, size), batch_size=batch, num_workers=8)
        keep = defaultdict(list)
        with torch.no_grad():
            for x, index, w, h in loader:
                results = post(model(x.to(args.device)), torch.stack([w, h], 1).to(args.device).float())
                for i, r in zip(index.tolist(), results):
                    s = r['scores'].cpu().numpy()
                    use = s >= .001
                    keep['image_id'].append(np.full(use.sum(), images[i]['id']))
                    keep['label'].append(r['labels'].cpu().numpy()[use])
                    keep['score'].append(s[use])
                    keep['box'].append(r['boxes'].cpu().numpy()[use])
        np.savez_compressed(path, **{k: np.concatenate(v) for k, v in keep.items()})
        print('sweep', size, flush=True)


def sweep_report(args):
    from faster_coco_eval import COCO
    from source_class_eval import SIZE_BINS, long_side_gt, run_eval
    annotations, source, grouped = load_val(args.out)
    path = ROOT / 'Annotations/val.json'
    names = {c['id']: c['name'] for c in annotations['categories']}
    members = {s: [i['id'] for i in annotations['images'] if source[i['id']] == s] for s in SOURCES}
    report = {}
    for size in SWEEP_SIZES:
        saved = np.load(args.out / 'sweep' / f'pred_{size}.npz')
        results = [dict(image_id=int(i), category_id=int(c), score=float(s), bbox=[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])])
                   for i, c, s, b in zip(saved['image_id'], saved['label'], saved['score'], saved['box'])]
        scales = {i['id']: size / max(i['width'], i['height']) for i in annotations['images']}
        entry = {}
        for system in ('orig', 'input'):
            gt = COCO(str(path))
            long_side_gt(gt, scales, system == 'input')
            dt = gt.loadRes(results)
            long_side_gt(dt, scales, system == 'input')
            for s in SOURCES:
                for (c, name), metrics in run_eval(gt, dt, members[s], sorted(names)).items():
                    entry.setdefault(s, {}).setdefault(system, {}).setdefault(names[c], {})[name] = {k: round(v, 2) for k, v in metrics.items()}
        order = np.argsort(saved['image_id'], kind='stable')
        values, first = np.unique(saved['image_id'][order], return_index=True)
        slices = dict(zip(values.tolist(), np.split(order, first[1:])))
        for s in SOURCES:
            counts, ious = Counter(), []
            by_bin = defaultdict(Counter)
            for image_id in members[s]:
                g_cls, g_box = image_targets(grouped[image_id]) if grouped[image_id] else (np.zeros(0, int), np.zeros((0, 4), np.float32))
                pos = slices.get(image_id, np.zeros(0, int))
                keep, kinds, targets, fn_kind, _, taken = match_image(saved['label'][pos].astype(int), saved['score'][pos], saved['box'][pos], g_cls, g_box, SCORE, IOU)
                for d, kind in zip(keep, kinds):
                    if kind == 'Bkg' and saved['label'][pos[d]] == TANK:
                        counts['FP_bkg'] += 1
                for g in np.flatnonzero(g_cls == TANK):
                    side = float(max(g_box[g][2] - g_box[g][0], g_box[g][3] - g_box[g][1])) * scales[image_id]
                    label = next(n for n, a, b in SIZE_BINS[1:] if a <= side < b)
                    status = fn_kind[g] or 'TP'
                    counts[status] += 1
                    counts['GT'] += 1
                    by_bin[label]['GT'] += 1
                    by_bin[label][status] += 1
                    if status == 'TP':
                        d = keep[taken[g]]
                        ious.append(float(iou_matrix(saved['box'][pos[d]][None], g_box[g][None])[0, 0]))
            entry.setdefault(s, {})['tank_counts'] = dict(counts)
            entry[s]['tank_by_input_bin'] = {k: dict(v) for k, v in by_bin.items()}
            entry[s]['tank_tp_iou_median'] = round(float(np.median(ious)), 4) if ious else None
        report[str(size)] = entry
        print('sweep report', size, flush=True)
    (args.out / 'sweep_report.json').write_text(json.dumps(report, indent=1))


# ---- extras ----
PROFILE_REACH = 6  # Cells on each side of the GT centre.


def native_saturation(task):
    """Saturated share and contrast in native pixels, per GT. No resizing, so nothing here comes from the network input."""
    path, boxes = task
    values = np.asarray(Image.open(path).convert('L'), np.float32)
    h, w = values.shape
    rows = []
    for x1, y1, x2, y2 in boxes:
        x0, y0, x3, y3 = max(int(x1), 0), max(int(y1), 0), min(int(np.ceil(x2)), w), min(int(np.ceil(y2)), h)
        side = max(x3 - x0, y3 - y0)
        a0, b0, a1, b1 = max(x0 - side, 0), max(y0 - side, 0), min(x3 + side, w), min(y3 + side, h)
        mask = np.ones((b1 - b0, a1 - a0), bool)
        mask[y0 - b0:y3 - b0, x0 - a0:x3 - a0] = False
        inner, ring = values[y0:y3, x0:x3], values[b0:b1, a0:a1][mask]
        if not inner.size:  # Zero-area annotation.
            rows.append(dict(gt_sat=float('nan'), ring_sat=float('nan'), gt_mean=float('nan'), ring_mean=float('nan'), ring_std=float('nan'), gt_peak=float('nan')))
            continue
        rows.append(dict(gt_sat=float((inner >= 250).mean()), ring_sat=float((ring >= 250).mean()) if ring.size else float('nan'), gt_mean=float(inner.mean()), ring_mean=float(ring.mean()) if ring.size else float('nan'),
                         ring_std=float(ring.std()) if ring.size else float('nan'), gt_peak=float(inner.max())))
    return rows


def thumb_only(path):
    image = Image.open(path).convert('L')
    t = np.asarray(image.resize((32, 32), Image.BILINEAR), np.float32).ravel()
    t = t - t.mean()
    n = np.linalg.norm(t)
    return t / n if n > 0 else t


def image_texture(path):
    v = np.asarray(Image.open(path).convert('L'), np.float32)
    lap = v[1:-1, 1:-1] * 4 - v[:-2, 1:-1] - v[2:, 1:-1] - v[1:-1, :-2] - v[1:-1, 2:]
    return float(lap.std()), float(v.mean()), float((v >= 250).mean())


def layout_task(task):
    """Largest set of this image's tank boxes reproduced by one train image under one integer shift. Per-GT flags for that image."""
    boxes, train_boxes = task
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    sizes = boxes[:, 2:] - boxes[:, :2]
    best, flags = 0., np.zeros(len(boxes), bool)
    for other in train_boxes:
        other_centers = (other[:, :2] + other[:, 2:]) / 2
        other_sizes = other[:, 2:] - other[:, :2]
        shifts = np.unique(np.round((centers[:12, None] - other_centers[None]).reshape(-1, 2)), axis=0)
        for shift in shifts:
            distance = np.abs(centers[:, None] - shift - other_centers[None]).max(-1)
            near = distance.argmin(1)
            hit = (distance.min(1) <= 1.5) & (np.abs(sizes - other_sizes[near]).max(1) <= 2)
            if hit.mean() > best:
                best, flags = float(hit.mean()), hit
    return best, flags.tolist()


def extras(args):
    """Spatial shape of the encoder tank map around GT centres, native saturation, image-level train duplicates, train size mix, train layout overlap."""
    import multiprocessing as mp
    import torch
    out = args.out
    annotations, source, grouped = load_val(out)
    rows = json.load(open(out / 'gt_rows.json'))
    status = {r['gt_id']: r['status'] for r in rows}
    images = {i['id']: i for i in annotations['images']}
    tank_images = sorted({r['image_id'] for r in rows})
    report = {}
    model, _ = load_model(args.run, args.device)
    hooks = Hooks(model)
    size = 2 * PROFILE_REACH + 1
    stacks = defaultdict(lambda: np.zeros((size, size), np.float64))
    counts = Counter()
    with torch.no_grad():
        for count, image_id in enumerate(tank_images):
            x, _ = to_input(ROOT / 'JPEGImages/val' / images[image_id]['file_name'])
            stages, enc, _, _ = hooks.run(x[None].to(args.device))
            s = INPUT / images[image_id]['width']
            image_rows = [r for r in rows if r['image_id'] == image_id]
            for r in image_rows:
                cx, cy = (r['gt_x1'] + r['gt_x2']) / 2 * s, (r['gt_y1'] + r['gt_y2']) / 2 * s
                for name in ('T3', 'T4', 'T5', 'F3', 'P3'):
                    stride = STRIDES[name]
                    values = stages[name][0] if name.startswith('T') else np.sqrt((stages[name] ** 2).sum(0))
                    H, W = values.shape
                    i, j = int(cy // stride), int(cx // stride)
                    if PROFILE_REACH <= i < H - PROFILE_REACH and PROFILE_REACH <= j < W - PROFILE_REACH:
                        patch = values[i - PROFILE_REACH:i + PROFILE_REACH + 1, j - PROFILE_REACH:j + PROFILE_REACH + 1]
                        state = 'hit' if r['status'] == 'TP' else 'miss'
                        isolated = 'isolated' if r['nn_input'] >= 2 * PROFILE_REACH * stride else 'all'
                        for key in {(r['source'], state, name, 'all'), (r['source'], state, name, isolated)}:
                            stacks[key] += patch
                            counts[key] += 1
            print('profile', count + 1, len(tank_images), flush=True)
    profiles = {}
    for key, stack in stacks.items():
        mean = stack / counts[key]
        base = np.percentile(mean, 10)
        along_x, along_y = mean[PROFILE_REACH] - base, mean[:, PROFILE_REACH] - base
        peak = max(along_x[PROFILE_REACH], along_y[PROFILE_REACH], 1e-12)

        def width(profile):
            return float((profile >= peak / 2).sum())

        def centroid(profile):
            weights = np.clip(profile, 0, None)
            return float((weights * (np.arange(size) - PROFILE_REACH)).sum() / max(weights.sum(), 1e-12))
        profiles['|'.join(key)] = dict(n=counts[key], center=round(float(mean[PROFILE_REACH, PROFILE_REACH]), 4), width_x_cells=width(along_x), width_y_cells=width(along_y), centroid_x_cells=round(centroid(along_x), 3), centroid_y_cells=round(centroid(along_y), 3),
                                       profile_x=np.round(mean[PROFILE_REACH], 4).tolist(), profile_y=np.round(mean[:, PROFILE_REACH], 4).tolist())
    report['profiles'] = profiles

    tasks = []
    keys = []
    for image_id in tank_images:
        image_rows = [r for r in rows if r['image_id'] == image_id]
        tasks.append((str(ROOT / 'JPEGImages/val' / images[image_id]['file_name']), [(r['gt_x1'], r['gt_y1'], r['gt_x2'], r['gt_y2']) for r in image_rows]))
        keys += [r['gt_id'] for r in image_rows]
    with mp.Pool(16) as pool:
        stats = [row for part in pool.map(native_saturation, tasks) for row in part]
        native = dict(zip(keys, stats))
        json.dump({str(k): v for k, v in native.items()}, open(out / 'native_stats.json', 'w'))
        report['native'] = {}
        for label, test in (('ogsod_hit', lambda r: r['source'] == 'ogsod' and r['status'] == 'TP'), ('ogsod_miss', lambda r: r['source'] == 'ogsod' and r['status'] != 'TP'), ('msar_hit', lambda r: r['source'] == 'msar' and r['status'] == 'TP')):
            part = [native[r['gt_id']] for r in rows if test(r)]
            report['native'][label] = {f: quartiles([p[f] for p in part]) for f in part[0]}
            report['native'][label]['scr_db_median'] = round(float(np.nanmedian([10 * np.log10(max(p['gt_mean'], 1) / max(p['ring_mean'], 1)) for p in part])), 3)
            report['native'][label]['ring_saturated_gt50pct_share'] = round(float(np.nanmean([p['ring_sat'] > .5 for p in part])), 3)

        train = json.load(open(ROOT / 'Annotations/train.json'))
        names = [i['file_name'] for i in train['images'] if i['width'] == 256 and i['height'] == 256]
        references = np.stack(pool.map(thumb_only, [str(ROOT / 'JPEGImages/train' / n) for n in names], chunksize=128)).astype(np.float32)
        queries = np.stack(pool.map(thumb_only, [str(ROOT / 'JPEGImages/val' / images[i]['file_name']) for i in tank_images], chunksize=8)).astype(np.float32)
        texture = dict(zip(tank_images, pool.map(image_texture, [str(ROOT / 'JPEGImages/val' / images[i]['file_name']) for i in tank_images], chunksize=8)))
        # Tank layouts in train: pixel thumbnails miss overlapping tiles, box positions do not.
        train_by_image = defaultdict(list)
        for a in train['annotations']:
            if a['category_id'] == TANK and a['bbox'][2] > 0 and a['bbox'][3] > 0:
                train_by_image[a['image_id']].append([a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]])
        train_layouts = [np.array(v, np.float32) for v in train_by_image.values() if len(v) >= 3]
        layout_tasks, layout_keys = [], []
        for image_id in tank_images:
            image_rows = [r for r in rows if r['image_id'] == image_id and r['native_gw'] > 0 and r['native_gh'] > 0]
            if len(image_rows) >= 3:
                layout_tasks.append((np.array([[r['gt_x1'], r['gt_y1'], r['gt_x2'], r['gt_y2']] for r in image_rows], np.float32), train_layouts))
                layout_keys.append([r['gt_id'] for r in image_rows])
        layouts = pool.map(layout_task, layout_tasks, chunksize=1)
    seen = {}
    image_share = {}
    for ids_, (share, flags) in zip(layout_keys, layouts):
        for g, flag in zip(ids_, flags):
            seen[g] = flag
            image_share[g] = share
    json.dump({str(k): dict(seen=v, image_share=image_share[k]) for k, v in seen.items()}, open(out / 'train_layout.json', 'w'))
    report['train_layout'] = {}
    for s in SOURCES:
        part = [r for r in rows if r['source'] == s and r['gt_id'] in seen]
        entry = dict(gt=len(part), layout_seen_share=round(float(np.mean([seen[r['gt_id']] for r in part])), 3))
        for label, test in (('gt_seen', lambda r: seen[r['gt_id']]), ('gt_unseen', lambda r: not seen[r['gt_id']]), ('image_share_ge_.8', lambda r: image_share[r['gt_id']] >= .8), ('image_share_.3_.8', lambda r: .3 <= image_share[r['gt_id']] < .8), ('image_share_lt_.3', lambda r: image_share[r['gt_id']] < .3)):
            subset = [r for r in part if test(r)]
            tps = [r['tp_iou'] for r in subset if r['status'] == 'TP']
            entry[label] = dict(gt=len(subset), images=len({r['image_id'] for r in subset}), recall=round(float(np.mean([r['status'] == 'TP' for r in subset])), 3) if subset else None, tp_iou=round(float(np.median(tps)), 3) if tps else None,
                                side=float(np.median([r['input_side'] for r in subset])) if subset else None, enc_prob=round(float(np.median([r['enc_prob_max_in_gt'] for r in subset])), 3) if subset else None)
        report['train_layout'][s] = entry
    # Same question inside one size bin, so size cannot explain it.
    report['train_layout_ogsod_by_size'] = {}
    for low, high in ((0, 16), (16, 24), (24, 1e5)):
        for label, value in (('seen', True), ('unseen', False)):
            subset = [r for r in rows if r['source'] == 'ogsod' and r['gt_id'] in seen and seen[r['gt_id']] == value and low <= r['input_side'] < high]
            report['train_layout_ogsod_by_size'][f'{low}-{high}|{label}'] = dict(gt=len(subset), images=len({r['image_id'] for r in subset}), recall=round(float(np.mean([r['status'] == 'TP' for r in subset])), 3) if subset else None)
    report['texture'] = {}
    for s in SOURCES:
        ids = [i for i in tank_images if source[i] == s]
        lap = np.array([texture[i][0] for i in ids])
        report['texture'][s] = dict(lap_histogram=np.histogram(lap, bins=[0, 20, 40, 60, 80, 100, 120, 160, 1000])[0].tolist(), mean_quantiles=np.round(np.percentile([texture[i][1] for i in ids], [10, 50, 90]), 1).tolist(), sat_quantiles=np.round(np.percentile([texture[i][2] for i in ids], [10, 50, 90]), 3).tolist())
        for label, test in (('smooth_lap_lt_70', lambda v: v < 70), ('speckled_lap_ge_70', lambda v: v >= 70)):
            subset = [r for r in rows if r['source'] == s and test(texture[r['image_id']][0])]
            tps = [r['tp_iou'] for r in subset if r['status'] == 'TP']
            report['texture'][s][label] = dict(images=len({r['image_id'] for r in subset}), gt=len(subset), recall=round(float(np.mean([r['status'] == 'TP' for r in subset])), 3) if subset else None, tp_iou=round(float(np.median(tps)), 3) if tps else None, side=float(np.median([r['input_side'] for r in subset])) if subset else None)
    best = np.array([(references @ q).max() for q in queries])
    dup = dict(zip(tank_images, best))
    report['train_duplicates'] = {}
    for s in SOURCES:
        ids = [i for i in tank_images if source[i] == s]
        values = np.array([dup[i] for i in ids])
        part = [r for r in rows if r['source'] == s]
        entry = dict(images=len(ids), corr_quantiles=np.round(np.percentile(values, [10, 50, 90]), 3).tolist())
        for threshold in (.9, .95, .98):
            near = [r for r in part if dup[r['image_id']] >= threshold]
            far = [r for r in part if dup[r['image_id']] < threshold]
            entry[f'ge_{threshold}'] = dict(images=int((values >= threshold).sum()), gt=len(near), recall=round(float(np.mean([r['status'] == 'TP' for r in near])), 3) if near else None,
                                            recall_rest=round(float(np.mean([r['status'] == 'TP' for r in far])), 3) if far else None,
                                            tp_iou_near=round(float(np.nanmedian([r['tp_iou'] for r in near])), 3) if near else None, tp_iou_rest=round(float(np.nanmedian([r['tp_iou'] for r in far])), 3) if far else None)
        report['train_duplicates'][s] = entry
    edges = [(0, 8), (8, 16), (16, 32), (32, 64), (64, 1e5)]
    train_sizes = Counter()
    for a in train['annotations']:
        if a['category_id'] == TANK:
            side = max(a['bbox'][2], a['bbox'][3])
            train_sizes[next(f'{low}-{high}' for low, high in edges if low <= side < high)] += 1
    report['train_tank_native_side'] = dict(train_sizes)
    report['train_tank_total'] = sum(train_sizes.values())
    (out / 'extras.json').write_text(json.dumps(report, indent=1))
    print('extras done', flush=True)


# ---- summary ----
def auc(positive, negative):
    positive, negative = np.asarray(positive, float), np.asarray(negative, float)
    positive, negative = positive[np.isfinite(positive)], negative[np.isfinite(negative)]
    if not len(positive) or not len(negative):
        return float('nan')
    joined = np.concatenate([positive, negative])
    order = joined.argsort(kind='mergesort')
    rank = np.empty(len(joined))
    sorted_values = joined[order]
    unique, start, counts = np.unique(sorted_values, return_index=True, return_counts=True)
    rank[order] = np.repeat(start + (counts + 1) / 2, counts)
    return float((rank[:len(positive)].sum() - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative)))


def cluster_ci(values, flags, images, boot=400, seed=0):
    """AUC of flagged vs unflagged, resampling whole images; 82 images carry 1216 GTs, so GT-level intervals would be too narrow."""
    rng = np.random.default_rng(seed)
    values, flags, images = np.asarray(values, float), np.asarray(flags, bool), np.asarray(images)
    unique = np.unique(images)
    index = {u: np.flatnonzero(images == u) for u in unique}
    samples = []
    for _ in range(boot):
        chosen = np.concatenate([index[u] for u in rng.choice(unique, len(unique))])
        samples.append(auc(values[chosen][flags[chosen]], values[chosen][~flags[chosen]]))
    return [round(float(v), 3) for v in np.nanpercentile(samples, [2.5, 97.5])]


def quartiles(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    a, b, c = np.percentile(values, [25, 50, 75])
    return dict(n=len(values), median=round(float(b), 3), q1=round(float(a), 3), q3=round(float(c), 3))


def funnel_stage(r):
    if r['status'] == 'TP':
        return '5 detected'
    if r['enc_selected_in_gt'] == 0 and r['enc_best_iou_selected'] < .1:
        return '1 no selected query'
    if r['dec3_best_iou'] < IOU:
        return '2 selected, no box IoU>=.5'
    if r['dec3_best_tp_score'] < .05:
        return '3 box IoU>=.5, score<.05'
    if r['dec3_best_tp_score'] < SCORE:
        return '4 box IoU>=.5, score .05-.3'
    return '4b box IoU>=.5, score>=.3, unmatched'


def fit_lda(train, labels, components=64, shrink=.1):
    mean, std = train.mean(0), train.std(0) + 1e-6
    z = (train - mean) / std
    _, _, vt = np.linalg.svd(z - z.mean(0), full_matrices=False)
    basis = vt[:components]
    y = z @ basis.T
    a, b = y[labels].mean(0), y[~labels].mean(0)
    within = np.cov(np.vstack([y[labels] - a, y[~labels] - b]).T)
    d = within.shape[0]
    within = (1 - shrink) * within + shrink * np.trace(within) / d * np.eye(d)
    weights = np.linalg.solve(within, a - b)
    return lambda x: ((x - mean) / std) @ basis.T @ weights


def probes(out):
    meta = json.load(open(out / 'probe_meta.json'))
    data = np.load(out / 'probe_vectors.npz')
    gt_status = {r['gt_id']: r['status'] for r in json.load(open(out / 'gt_rows.json'))}
    kind, source, images = (np.array([m[k] for m in meta]) for k in ('kind', 'source', 'image_id'))
    hit = np.array([m['kind'] == 'gt' and gt_status[m['owner']] == 'TP' for m in meta])
    miss = np.array([m['kind'] == 'gt' and gt_status[m['owner']] != 'TP' for m in meta])
    report = {}
    oof = {}
    for s in SOURCES:
        mask = source == s
        ids = np.unique(images[mask])
        fold = {u: i % 5 for i, u in enumerate(np.random.default_rng(0).permutation(ids))}
        fold_of = np.array([fold.get(u, -1) for u in images])
        report[s] = {}
        for stage in FEATURE_STAGES:
            scores = np.full(len(meta), np.nan)
            x = data[stage]
            for f in range(5):
                train, test = mask & (fold_of != f), mask & (fold_of == f)
                scores[test] = fit_lda(x[train], kind[train] == 'gt')(x[test])
            oof[s, stage] = scores
            row = {}
            for label, positive in (('all_gt', hit | miss), ('hit', hit), ('miss', miss)):
                for neg_label, negative in (('local', kind == 'neg'), ('fp', kind == 'fp')):
                    p, n = scores[mask & positive], scores[mask & negative]
                    row[f'{label}_vs_{neg_label}'] = round(auc(p, n), 3) if len(p) >= 5 and len(n) >= 5 else None
            report[s][stage] = row
        for stage in PROB_STAGES:
            scores = data[stage][:, 0]
            row = {}
            for label, positive in (('all_gt', hit | miss), ('hit', hit), ('miss', miss)):
                for neg_label, negative in (('local', kind == 'neg'), ('fp', kind == 'fp')):
                    p, n = scores[mask & positive], scores[mask & negative]
                    row[f'{label}_vs_{neg_label}'] = round(auc(p, n), 3) if len(p) >= 5 and len(n) >= 5 else None
            report[s][stage + '_model_head'] = row
    # Do the two sources' tanks share a representation? Fit on one, score the other.
    report['cross'] = {}
    for fit_on, test_on in (('msar', 'ogsod'), ('ogsod', 'msar')):
        report['cross'][f'{fit_on}_to_{test_on}'] = {}
        for stage in FEATURE_STAGES:
            train, test = (source == fit_on) & (kind != 'fp'), source == test_on
            scores = fit_lda(data[stage][train], kind[train] == 'gt')(data[stage][test])
            pos, neg = (hit | miss)[test], (kind == 'neg')[test]
            report['cross'][f'{fit_on}_to_{test_on}'][stage] = dict(all=round(auc(scores[pos], scores[neg]), 3), hit=round(auc(scores[hit[test]], scores[neg]), 3), miss=round(auc(scores[miss[test]], scores[neg]), 3))
    return report


def summary(args):
    out = args.out
    gts = json.load(open(out / 'gt_rows.json'))
    stages = json.load(open(out / 'stage_rows.json'))
    by_id = {r['gt_id']: r for r in gts}
    for r in gts:
        r['funnel'] = funnel_stage(r)
    groups = {
        'ogsod_hit': lambda r: r['source'] == 'ogsod' and r['status'] == 'TP',
        'ogsod_miss': lambda r: r['source'] == 'ogsod' and r['status'] != 'TP',
        'ogsod_NoBox': lambda r: r['source'] == 'ogsod' and r['status'] == 'NoBox',
        'ogsod_LowScore': lambda r: r['source'] == 'ogsod' and r['status'] == 'LowScore',
        'ogsod_Loc': lambda r: r['source'] == 'ogsod' and r['status'] == 'Loc',
        'msar_hit': lambda r: r['source'] == 'msar' and r['status'] == 'TP',
        'msar_miss': lambda r: r['source'] == 'msar' and r['status'] != 'TP',
    }
    scalar_fields = ['native_side', 'input_side', 'n_gt', 'nn_input', 'enc_prob_max_in_gt', 'enc_rank_best_in_gt', 'enc_selected_in_gt', 'enc_best_iou_all', 'enc_best_iou_selected', 'enc_cutoff_prob',
                     'dec0_best_iou', 'dec1_best_iou', 'dec2_best_iou', 'dec3_best_iou', 'dec0_best_tp_score', 'dec1_best_tp_score', 'dec2_best_tp_score', 'dec3_best_tp_score', 'dec3_count_iou50',
                     'q_init_iou', 'q_iou_0', 'q_iou_3', 'q_score_0', 'q_score_1', 'q_score_2', 'q_score_3', 'q_enc_tank_prob', 'q_enc_rank', 'q_top_class_score', 'mislocalised_top_score', 'best_box_iou', 'best_box_score', 'tp_iou', 'tp_score', 'res_dx', 'res_dy', 'res_dw', 'res_dh']
    report = dict(status=dict(Counter(f"{r['source']}|{r['status']}" for r in gts)), funnel=dict(Counter(f"{r['source']}|{r['funnel']}" for r in gts)))
    report['q_level'] = dict(Counter(f"{r['source']}|{'hit' if r['status'] == 'TP' else 'miss'}|L{r['q_level']}" for r in gts))
    report['scalars'] = {name: {f: quartiles([r[f] for r in gts if test(r)]) for f in scalar_fields} for name, test in groups.items()}
    # Size bins inside OGSOD: same-size comparison of hits and misses.
    bins = [(0, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 1e5)]
    report['ogsod_recall_by_input_side'] = {}
    for a, b in bins:
        part = [r for r in gts if r['source'] == 'ogsod' and a <= r['input_side'] < b]
        report['ogsod_recall_by_input_side'][f'{a}-{b}'] = dict(gt=len(part), hit=sum(r['status'] == 'TP' for r in part), images=len({r['image_id'] for r in part}))
    table = defaultdict(lambda: defaultdict(list))
    for s in stages:
        if s['kind'] == 'gt':
            r = by_id[s['owner']]
            for name, test in groups.items():
                if test(r):
                    table[s['stage']][name].append(s)
        elif s['kind'] == 'fp':
            table[s['stage']]['ogsod_fp' if s['image_id'] in {r['image_id'] for r in gts if r['source'] == 'ogsod'} else 'msar_fp'].append(s)
    metrics = ('gt_mean', 'ring_mean', 'ratio', 'z', 'abs_ratio', 'image_pct', 'peak_pct', 'cos_dist')
    report['stages'] = {stage: {group: {m: quartiles([s.get(m, float('nan')) for s in rows]) for m in metrics} for group, rows in entries.items()} for stage, entries in table.items()}
    # Hit vs miss within OGSOD, overall and inside fixed size bins, with image-cluster bootstrap.
    report['hit_vs_miss'] = {}
    ogsod_rows = defaultdict(list)
    for s in stages:
        if s['kind'] == 'gt' and by_id[s['owner']]['source'] == 'ogsod':
            ogsod_rows[s['stage']].append(s)
    for stage, rows in ogsod_rows.items():
        report['hit_vs_miss'][stage] = {}
        flags = [by_id[s['owner']]['status'] == 'TP' for s in rows]
        images = [s['image_id'] for s in rows]
        sides = np.array([by_id[s['owner']]['input_side'] for s in rows])
        for m in ('ratio', 'z', 'peak_pct', 'cos_dist'):
            values = np.array([s.get(m, np.nan) for s in rows], float)
            f = np.array(flags)
            pairs = []
            for a, b in bins[1:5]:  # Stratified AUC: hit and miss compared only at similar size.
                inside = (sides >= a) & (sides < b)
                if inside.sum():
                    pairs.append((auc(values[inside & f], values[inside & ~f]), min((inside & f).sum(), (inside & ~f).sum())))
            usable = [(v, w) for v, w in pairs if np.isfinite(v)]
            stratified = float(np.average([v for v, _ in usable], weights=[w for _, w in usable])) if usable else float('nan')
            report['hit_vs_miss'][stage][m] = dict(auc=round(auc(values[f], values[~f]), 3), ci=cluster_ci(values, f, images), size_stratified_auc=round(stratified, 3))
    # Scalars carrying the same question through the head.
    report['hit_vs_miss_head'] = {}
    rows = [r for r in gts if r['source'] == 'ogsod']
    flags = np.array([r['status'] == 'TP' for r in rows])
    images = [r['image_id'] for r in rows]
    for field in ('input_side', 'n_gt', 'nn_input', 'enc_prob_max_in_gt', 'q_enc_tank_prob', 'q_score_0', 'q_score_3', 'enc_best_iou_selected', 'dec3_best_iou'):
        values = np.array([r[field] for r in rows], float)
        report['hit_vs_miss_head'][field] = dict(auc=round(auc(values[flags], values[~flags]), 3), ci=cluster_ci(values, flags, images))
    # Per-image structure: is failure a property of whole images?
    per_image = defaultdict(lambda: [0, 0])
    for r in rows:
        per_image[r['image_id']][0] += 1
        per_image[r['image_id']][1] += r['status'] == 'TP'
    recalls = np.array([b / a for a, b in per_image.values() if a >= 5])
    report['ogsod_per_image_recall'] = dict(images=len(per_image), images_ge5=len(recalls), quantiles=np.round(np.percentile(recalls, [0, 10, 25, 50, 75, 90, 100]), 3).tolist(),
                                            miss_share_of_worst_10pct=round(float(sum(sorted(((a - b) for a, b in per_image.values()), reverse=True)[:max(len(per_image) // 10, 1)]) / max(sum(a - b for a, b in per_image.values()), 1)), 3))
    report['probes'] = probes(out)
    (out / 'summary.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=0)[:200], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('collect', 'select', 'render', 'sweep', 'sweep_report', 'extras', 'summary'))
    parser.add_argument('run', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == '__main__':
    main()
