"""Baseline vs Q-MAL on SARDet-100K val: where does Q-MAL lose AP?

Usage (one GPU, or --device cpu):
    python tools/sardet100k/qmal_ranking_analysis.py --run base=RUN_DIR --run qmal=RUN_DIR --output OUT_DIR

Each run is evaluated from its own resolved_config.yml and EMA weights of --checkpoint (default best_stg2);
val predictions are cached in OUT_DIR/<name>_val_predictions.npz and reused if present.
Evaluation follows COCO bbox rules (per image and class top 100, greedy matching with ignored GT last,
101 recall points); "small" means GT long side < 16 original-image pixels, and an unmatched detection
counts in a size subset only if its own long side is in the range (COCO uses area instead of long side).

Questions answered (report.md):
  Q1  At fixed class and actual IoU, do small targets still get extra-low scores?
  Q2  At IoU 0.5 / 0.75 / 0.9, are more unqualified boxes ranked above qualified ones under Q-MAL?
  Q3  For < 16 px targets, is the AP loss from having no qualified box or from ranking it too low?
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

INF = float('inf')
SIZE_BINS = [(0, 8), (8, 16), (16, 32), (32, 64), (64, INF)]
IOU_EDGES = [.5, .6, .7, .8, .9, 1.01]
THRESHOLDS = (.5, .75, .9)
COCO_THRESHOLDS = np.round(np.linspace(.5, .95, 10), 2)
RECALLS = np.linspace(0, 1, 101)
SMALL = (0, 16)
ALL = (0, INF)
MAX_DETS = 100


def log(*args):
    print(time.strftime('%H:%M:%S'), *args, flush=True)


# ----------------------------------------------------------------------------- inference
class EvalImages:
    def __init__(self, folder, images):
        self.folder, self.images = folder, images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        import torchvision.transforms.functional as F
        image = Image.open(f"{self.folder}/{self.images[index]['file_name']}").convert('RGB')
        width, height = image.size
        return F.to_tensor(F.resize(image, [640, 640])), index, width, height


def infer(run, checkpoint, images, folder, device, batch_size, workers):
    import torch
    from engine.core import YAMLConfig
    cfg = YAMLConfig(str(run / 'resolved_config.yml'))
    model, post = cfg.model, cfg.postprocessor
    state = torch.load(run / f'{checkpoint}.pth', map_location='cpu')
    weights = state['ema']['module'] if state.get('ema') else state['model']
    result = model.load_state_dict(weights, strict=False)
    if result.missing_keys:
        raise RuntimeError(f'{run}: missing weights {result.missing_keys[:5]}')
    model, post = model.to(device).eval(), post.to(device).eval()
    loader = torch.utils.data.DataLoader(EvalImages(folder, images), batch_size=batch_size, num_workers=workers)
    rows = []
    with torch.no_grad():
        for step, (batch, indices, widths, heights) in enumerate(loader):
            outputs = model(batch.to(device))
            sizes = torch.stack([widths, heights], 1).to(device)
            for index, out in zip(indices.tolist(), post(outputs, sizes)):
                scores = out['scores'].float().cpu().numpy()
                rows.append((np.full(len(scores), images[index]['id']), out['labels'].cpu().numpy(), scores,
                             out['boxes'].float().cpu().numpy()))
            if step % 20 == 0:
                log(f'  {run.name}: {step * batch_size}/{len(images)}')
    return tuple(np.concatenate([r[i] for r in rows]) for i in range(4))


# ----------------------------------------------------------------------------- matching
def pair_iou(boxes, gts):
    x1 = np.maximum(boxes[:, None, 0], gts[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], gts[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], gts[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], gts[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area_g = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])
    return inter / np.maximum(area_b[:, None] + area_g[None] - inter, 1e-12)


def long_side(boxes):
    return np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]) if len(boxes) else np.zeros(0)


def build_cells(gt, predictions):
    """One cell per (image, class) holding scores sorted descending (top 100), IoU matrix and sizes."""
    image_id, label, score, box = predictions
    by_key = defaultdict(list)
    for i, key in enumerate(zip(image_id.tolist(), label.tolist())):
        by_key[key].append(i)
    cells = []
    for key in set(by_key) | set(gt):
        dets = np.array(by_key.get(key, []), int)
        if len(dets):
            dets = dets[np.argsort(-score[dets], kind='mergesort')][:MAX_DETS]
        boxes = box[dets] if len(dets) else np.zeros((0, 4))
        gboxes, crowd = gt.get(key, (np.zeros((0, 4)), np.zeros(0, bool)))
        cells.append(dict(key=key, score=score[dets] if len(dets) else np.zeros(0), box=boxes,
                          dlong=long_side(boxes), glong=long_side(gboxes), crowd=crowd, gbox=gboxes,
                          iou=pair_iou(boxes, gboxes) if len(dets) and len(gboxes) else np.zeros((len(dets), len(gboxes)))))
    return cells


def greedy(iou, threshold, gt_ignore, crowd):
    """COCO matching: detections in score order take the best unmatched GT, non-ignored GT first."""
    match = -np.ones(iou.shape[0], int)
    if iou.shape[1] == 0:
        return match
    taken = np.zeros(iou.shape[1], bool)
    limit = min(threshold, 1 - 1e-10)
    for d in np.flatnonzero((iou >= limit).any(1)):
        for ignored in (False, True):
            ok = ~taken & (gt_ignore == ignored) & (iou[d] >= limit)
            if ok.any():
                g = int(np.argmax(np.where(ok, iou[d], -1)))
                match[d] = g
                taken[g] = not crowd[g]
                break
    return match


def evaluate(cells, threshold, size):
    """Per class: kept detections (score, tp, fp type) and GT counts for one IoU threshold and GT size range."""
    lo, hi = size
    out = defaultdict(lambda: dict(score=[], tp=[], fptype=[], npos=0, gt_hit=[], gt_any=[]))
    for cell in cells:
        c = cell['key'][1]
        gt_ignore = cell['crowd'] | (cell['glong'] < lo) | (cell['glong'] >= hi)
        match = greedy(cell['iou'], threshold, gt_ignore, cell['crowd'])
        matched = match >= 0
        det_ignore = np.where(matched, gt_ignore[np.maximum(match, 0)] if len(gt_ignore) else False,
                              (cell['dlong'] < lo) | (cell['dlong'] >= hi))
        keep = ~det_ignore
        best = cell['iou'].max(1) if cell['iou'].shape[1] else np.zeros(len(match))
        # 0 TP, 1 duplicate (IoU >= t with a taken GT), 2 poor localization (0.1 <= IoU < t), 3 background
        fptype = np.where(matched, 0, np.where(best >= threshold, 1, np.where(best >= .1, 2, 3)))
        entry = out[c]
        entry['score'].append(cell['score'][keep])
        entry['tp'].append(matched[keep])
        entry['fptype'].append(fptype[keep])
        valid = ~gt_ignore
        entry['npos'] += int(valid.sum())
        hit = np.zeros(len(gt_ignore), bool)
        hit[match[matched]] = True
        entry['gt_hit'].append(hit[valid])
        any_ok = (cell['iou'] >= threshold).any(0) if cell['iou'].shape[0] else np.zeros(len(gt_ignore), bool)
        entry['gt_any'].append(any_ok[valid])
    for entry in out.values():
        for k in ('score', 'tp', 'fptype', 'gt_hit', 'gt_any'):
            entry[k] = np.concatenate(entry[k]) if entry[k] else np.zeros(0)
    return dict(out)


def coco_ap(score, tp, npos):
    if npos == 0:
        return float('nan')
    order = np.argsort(-score, kind='mergesort')
    tp = tp[order].astype(bool)
    tpc, fpc = np.cumsum(tp), np.cumsum(~tp)
    recall = tpc / npos
    precision = tpc / np.maximum(tpc + fpc, np.spacing(1))
    precision = np.maximum.accumulate(precision[::-1])[::-1] if len(precision) else precision
    index = np.searchsorted(recall, RECALLS, side='left')
    q = np.zeros(len(RECALLS))
    valid = index < len(precision)
    q[valid] = precision[index[valid]]
    return float(q.mean())


def nanmean(values):
    values = [v for v in values if v == v]
    return float(np.mean(values)) if values else float('nan')


# ----------------------------------------------------------------------------- questions
def q1_scores_at_fixed_iou(gt, predictions, classes):
    """For each GT, the highest-scoring same-class detection with IoU >= .5: its score by class x IoU bin x size bin."""
    cells = build_cells(gt, predictions)
    rows = []
    for cell in cells:
        if not cell['iou'].size:
            continue
        for g in np.flatnonzero(~cell['crowd']):
            ok = cell['iou'][:, g] >= .5
            if ok.any():
                d = int(np.argmax(np.where(ok, cell['score'], -1)))
                rows.append((cell['key'][1], cell['glong'][g], cell['iou'][d, g], cell['score'][d]))
    rows = np.array(rows) if rows else np.zeros((0, 4))
    table, gaps = {}, {}
    for c in classes + ['all']:
        sel = rows if c == 'all' else rows[rows[:, 0] == classes.index(c)]
        table[c] = {}
        for i in range(len(IOU_EDGES) - 1):
            in_bin = sel[(sel[:, 2] >= IOU_EDGES[i]) & (sel[:, 2] < IOU_EDGES[i + 1])]
            table[c][f'{IOU_EDGES[i]:.1f}'] = {
                f'{a}-{b}': [round(float(np.median(s[:, 3])), 3) if len(s) else None, int(len(s))]
                for a, b in SIZE_BINS for s in [in_bin[(in_bin[:, 1] >= a) & (in_bin[:, 1] < b)]]}
    # size gap within class x IoU bin: median(<16) - median(>=32), weighted by the smaller group
    for c_index, c in enumerate(classes):
        sel = rows[rows[:, 0] == c_index]
        num = den = 0.
        for i in range(len(IOU_EDGES) - 1):
            in_bin = sel[(sel[:, 2] >= IOU_EDGES[i]) & (sel[:, 2] < IOU_EDGES[i + 1])]
            small, large = in_bin[in_bin[:, 1] < 16, 3], in_bin[in_bin[:, 1] >= 32, 3]
            if len(small) >= 20 and len(large) >= 20:
                w = min(len(small), len(large))
                num += w * (np.median(small) - np.median(large))
                den += w
        gaps[c] = round(num / den, 3) if den else None
    return dict(table=table, gap_small_minus_large=gaps), rows


def q2_ranking(cells, classes, size):
    result = {}
    for t in THRESHOLDS:
        per_class = evaluate(cells, t, size)
        rows = {}
        for c_index, c in enumerate(classes):
            e = per_class.get(c_index)
            if e is None or e['npos'] == 0:
                continue
            tp = e['tp'].astype(bool)
            tp_scores = e['score'][tp]
            row = dict(AP=round(100 * coco_ap(e['score'], tp, e['npos']), 2),
                       recall=round(100 * tp.sum() / e['npos'], 2), npos=e['npos'], tp=int(tp.sum()))
            # mean number of unqualified boxes of each kind ranked above a qualified box
            for name, kind in (('dup', 1), ('poor_loc', 2), ('background', 3)):
                fp = np.sort(e['score'][(~tp) & (e['fptype'] == kind)])
                above = len(fp) - np.searchsorted(fp, tp_scores, side='right') if len(tp_scores) else np.zeros(0)
                row[f'fp_above_tp_{name}'] = round(float(above.mean()), 3) if len(above) else None
                row[f'fp_count_{name}'] = int(len(fp))
            fp_all = np.sort(e['score'][~tp])
            above = len(fp_all) - np.searchsorted(fp_all, tp_scores, side='right')
            row['fp_above_tp_all'] = round(float(above.mean()), 3) if len(above) else None
            # share of qualified boxes beaten by at least one poorly localized box (pairwise view)
            row['tp_median_score'] = round(float(np.median(tp_scores)), 3) if len(tp_scores) else None
            rows[c] = row
        rows['mean'] = dict(AP=round(nanmean([r['AP'] for r in rows.values()]), 2))
        result[str(t)] = rows
    return result


def q3_small_loss(cells, classes):
    """AP of < 16 px GT split into: no qualified box (1 - oracle AP) and ranking (oracle AP - AP)."""
    result = {}
    per_t = {t: evaluate(cells, t, SMALL) for t in COCO_THRESHOLDS}
    for c_index, c in enumerate(classes + ['all']):
        rows = {}
        for t in list(THRESHOLDS) + ['0.5:0.95']:
            ts = COCO_THRESHOLDS if t == '0.5:0.95' else [t]
            ap, oracle, recall, anyq = [], [], [], []
            for tt in ts:
                entries = per_t[round(float(tt), 2)]
                chosen = list(entries.values()) if c == 'all' else [entries.get(c_index)]
                aps, ors, rcs, anys = [], [], [], []
                for e in chosen:
                    if e is None or e['npos'] == 0:
                        continue
                    tp = e['tp'].astype(bool)
                    aps.append(coco_ap(e['score'], tp, e['npos']))
                    ors.append(coco_ap(np.where(tp, INF, e['score']), tp, e['npos']))
                    rcs.append(tp.sum() / e['npos'])
                    anys.append(e['gt_any'].mean())
                ap.append(nanmean(aps)); oracle.append(nanmean(ors)); recall.append(nanmean(rcs)); anyq.append(nanmean(anys))
            a, o = np.mean(ap), np.mean(oracle)
            rows[str(t)] = dict(AP=round(100 * a, 2), oracle_rank_AP=round(100 * o, 2),
                                loss_no_qualified_box=round(100 * (1 - o), 2), loss_ranking=round(100 * (o - a), 2),
                                recall=round(100 * np.mean(recall), 2), gt_with_any_qualified_box=round(100 * np.mean(anyq), 2))
        # why small GT lack a qualified box (IoU 0.5): imprecise same-class box, other class on it, or nothing near
        rows['missing_breakdown_0.5'] = missing_breakdown(cells, None if c == 'all' else c_index)
        result[c] = rows
    return result


def missing_breakdown(cells, class_index):
    by_image = defaultdict(list)
    for cell in cells:
        by_image[cell['key'][0]].append(cell)
    counts = defaultdict(int)
    for image_cells in by_image.values():
        for cell in image_cells:
            if class_index is not None and cell['key'][1] != class_index:
                continue
            small = np.flatnonzero(~cell['crowd'] & (cell['glong'] < 16))
            for g in small:
                same = cell['iou'][:, g] if cell['iou'].shape[0] else np.zeros(0)
                if (same >= .5).any():
                    counts['has_qualified'] += 1
                    continue
                other = any(len(o['box']) and (pair_iou(o['box'], cell['gbox'][g:g + 1])[:, 0] >= .5).any()
                            for o in image_cells if o is not cell)
                if (same >= .1).any():
                    counts['same_class_imprecise'] += 1
                elif other:
                    counts['other_class_box'] += 1
                else:
                    counts['nothing_near'] += 1
    total = sum(counts.values())
    return {k: [counts[k], round(100 * counts[k] / total, 2) if total else None]
            for k in ('has_qualified', 'same_class_imprecise', 'other_class_box', 'nothing_near')}


# ----------------------------------------------------------------------------- report
def markdown(report, names, classes):
    lines = ['# Q-MAL vs baseline: val ranking analysis', '']
    lines.append('Self-computed AP@[.5:.95] (all sizes), to compare with the training log val AP:')
    for n in names:
        lines.append(f'- {n}: {report[n]["ap_all"]}')
    lines += ['', '## Q1 score at fixed class and actual IoU', '',
              'Size gap = median score(<16 px) - median score(>=32 px) inside the same class and IoU bin '
              '(bins with >= 20 GT on each side, weighted by the smaller group). Negative = small targets scored lower.', '',
              '| class | ' + ' | '.join(names) + ' |', '| --- |' + ' ---: |' * len(names)]
    for c in classes:
        lines.append(f'| {c} | ' + ' | '.join(str(report[n]['q1']['gap_small_minus_large'][c]) for n in names) + ' |')
    for c in classes + ['all']:
        lines += ['', f'### {c}: median score (n) by IoU bin x GT long side', '',
                  '| run | IoU | ' + ' | '.join(f'{a}-{b}' for a, b in SIZE_BINS) + ' |',
                  '| --- | --- |' + ' ---: |' * len(SIZE_BINS)]
        for n in names:
            for iou_bin, sizes in report[n]['q1']['table'][c].items():
                lines.append(f'| {n} | {iou_bin} | ' + ' | '.join(
                    f'{v[0]} ({v[1]})' if v[0] is not None else f'- ({v[1]})' for v in sizes.values()) + ' |')
    for scope in ('q2_all', 'q2_small'):
        title = 'all sizes' if scope == 'q2_all' else 'GT long side < 16 px'
        lines += ['', f'## Q2 unqualified boxes ranked above qualified boxes ({title})', '',
                  'fp_above = mean number of unqualified boxes with a higher score than each qualified box '
                  '(dup: IoU >= t on an already matched GT; poor_loc: 0.1 <= IoU < t; background: IoU < 0.1).', '',
                  '| t | class | run | AP | recall | fp_above all | dup | poor_loc | background |',
                  '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for t in THRESHOLDS:
            for c in classes:
                for n in names:
                    r = report[n][scope][str(t)].get(c)
                    if r:
                        lines.append(f'| {t} | {c} | {n} | {r["AP"]} | {r["recall"]} | {r["fp_above_tp_all"]} | '
                                     f'{r["fp_above_tp_dup"]} | {r["fp_above_tp_poor_loc"]} | {r["fp_above_tp_background"]} |')
            for n in names:
                lines.append(f'| {t} | mean | {n} | {report[n][scope][str(t)]["mean"]["AP"]} | | | | | |')
    lines += ['', '## Q3 < 16 px targets: no qualified box vs ranking', '',
              'oracle_rank_AP puts every qualified (matched) box above all others; '
              'loss_no_qualified_box = 100 - oracle; loss_ranking = oracle - AP.', '',
              '| class | t | run | AP | oracle | loss: no qualified box | loss: ranking | recall |',
              '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for c in classes + ['all']:
        for t in [str(x) for x in THRESHOLDS] + ['0.5:0.95']:
            for n in names:
                r = report[n]['q3'][c][t]
                lines.append(f'| {c} | {t} | {n} | {r["AP"]} | {r["oracle_rank_AP"]} | {r["loss_no_qualified_box"]} | '
                             f'{r["loss_ranking"]} | {r["recall"]} |')
    lines += ['', 'Small GT without a same-class box at IoU >= 0.5 (count, %):', '',
              '| class | run | has qualified | same class, imprecise | other class box | nothing near |',
              '| --- | --- | ---: | ---: | ---: | ---: |']
    for c in classes + ['all']:
        for n in names:
            b = report[n]['q3'][c]['missing_breakdown_0.5']
            lines.append(f'| {c} | {n} | ' + ' | '.join(f'{v[0]} ({v[1]})' for v in b.values()) + ' |')
    return '\n'.join(lines) + '\n'


def load_gt(path):
    data = json.load(open(path))
    classes = [c['name'] for c in sorted(data['categories'], key=lambda c: c['id'])]
    ids = [c['id'] for c in sorted(data['categories'], key=lambda c: c['id'])]
    if ids != list(range(len(ids))):
        raise ValueError(f'expected zero-based contiguous category ids, got {ids}')
    gt = defaultdict(lambda: ([], []))
    for a in data['annotations']:
        x, y, w, h = a['bbox']
        boxes, crowd = gt[(a['image_id'], a['category_id'])]
        boxes.append([x, y, x + w, y + h])
        crowd.append(bool(a.get('iscrowd', 0)))
    gt = {k: (np.array(b, np.float64), np.array(c, bool)) for k, (b, c) in gt.items()}
    return data, classes, gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='append', required=True, help='name=RUN_DIR; give the baseline first')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('/data2/zxy/sar/datasets/SARDet_100K'))
    parser.add_argument('--checkpoint', default='best_stg2')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    runs = [(s.split('=', 1)[0], Path(s.split('=', 1)[1])) for s in args.run]
    args.output.mkdir(parents=True, exist_ok=True)
    data, classes, gt = load_gt(args.data_root / 'Annotations/val.json')
    report = dict(checkpoint=args.checkpoint, runs={n: str(p) for n, p in runs})
    for name, run in runs:
        cache = args.output / f'{name}_val_predictions.npz'
        if cache.exists():
            log(name, 'reuse', cache)
            saved = np.load(cache)
            predictions = tuple(saved[k] for k in ('image_id', 'label', 'score', 'box'))
        else:
            log(name, 'infer', run)
            predictions = infer(run, args.checkpoint, data['images'], args.data_root / 'JPEGImages/val',
                                args.device, args.batch_size, args.workers)
            np.savez_compressed(cache, image_id=predictions[0], label=predictions[1], score=predictions[2], box=predictions[3])
        predictions = (predictions[0].astype(np.int64), predictions[1].astype(np.int64),
                       predictions[2].astype(np.float64), predictions[3].astype(np.float64))
        cells = build_cells(gt, predictions)
        log(name, 'evaluate')
        ap_all = nanmean([nanmean([coco_ap(e['score'], e['tp'].astype(bool), e['npos'])
                                   for e in evaluate(cells, t, ALL).values()]) for t in COCO_THRESHOLDS])
        q1, _ = q1_scores_at_fixed_iou(gt, predictions, classes)
        report[name] = dict(ap_all=round(100 * ap_all, 2), q1=q1, q2_all=q2_ranking(cells, classes, ALL),
                            q2_small=q2_ranking(cells, classes, SMALL), q3=q3_small_loss(cells, classes))
        log(name, 'AP@[.5:.95]', report[name]['ap_all'])
    names = [n for n, _ in runs]
    (args.output / 'report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    (args.output / 'report.md').write_text(markdown(report, names, classes), encoding='utf-8')
    log('wrote', args.output / 'report.md')


if __name__ == '__main__':
    main()
