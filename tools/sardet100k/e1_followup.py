"""E1 follow-up diagnosis: where does AP50 go, how are final queries ranked, and what do box-positive /
class-background queries become. No training, no model change. Val only; test is never read.

Both runs are measured with their own EMA weights (the checkpoint used in the E1 report), their own matcher, the
same 10,492 val images, the same 640x640 input and the configured postprocessor rule (sigmoid, top 300 over
300 queries x 6 classes). Every final detection is rebuilt from the last decoder layer's 300 queries, so each
detection keeps its query index and can be linked to the training assignment of that query.

Commands (repository root on the server; OUT directly under E000's independent_verify_20261106/):
  collect RUN OUT --checkpoint PATH [--device cuda:0] [--batch 8]
      queries.npz   last decoder layer: logits (N,300,6) float32, boxes (N,300,4) normalised xyxy
      assign.pt     per image with GTs: matcher indices of enc / pre / dec1-4 and the union set (training rule)
  analyse BASE_OUT RUN_OUT --out DIR [--boot 500]
      DIR/followup.json  tasks 1-3 tables; DIR/cases/*.png  20 cases (E000 | E1); contact sheet
  plot DIR  (local) DIR/figures/fig1-fig5

Definitions
  Evaluation       COCO bbox rules reimplemented (area all, maxDets 100 per image and class, 101 recall points,
                   greedy matching by score); checked against COCOeval of the E1 report.
  Detection type   at IoU .5, in this order: TP (matched); duplicate (unmatched, IoU >= .5 with a same-class GT);
                   class confusion (IoU >= .5 with a GT of another class); localisation (best same-class IoU in
                   [.1, .5)); background (otherwise). Only the detections COCO evaluates (top 100 per image/class).
  GT candidates    queries whose highest-IoU GT among the class-c GTs is this GT, with IoU >= .1 (survey rule);
                   top = highest class-c score, best = highest IoU.
  BP-CB query      box-positive / class-background at the last layer: in the union set (box, FGL and DDF losses
                   supervise it towards its union GT) but not matched by the last-layer matcher (MAL target 0 for
                   every class at that layer). This is DEIMCriterion's boxpos_clsbg_dec4.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_report import GROUP_ORDER, group_of, iou_xyxy, r3, sigmoid  # noqa: E402

STAGES = ['enc', 'pre', 'dec1', 'dec2', 'dec3', 'dec4']
THRESH = np.round(np.arange(.5, .951, .05), 2)
RECALL = np.linspace(0, 1, 101)
TYPES = ['tp', 'duplicate', 'class', 'localisation', 'background']
SCORE_CUTS = [.05, .1, .2, .3, .5, .7]
BINS = np.round(np.arange(0, 1.0001, .02), 2)


# ---------------------------------------------------------------- collect
def collect(args):
    import torch
    from error_gallery import ROOT
    from tank_feature_diagnosis import load_val, to_input, xyxy
    from tank_matching_counterfactual import Runner
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    annotations, source, grouped = load_val(out)
    runner = Runner(args.run, args.device, args.checkpoint)
    model, inner, crit, saved = runner.model, runner.inner, runner.criterion, runner.saved
    images = annotations['images']
    ids, logits_all, boxes_all, assign = [], [], [], {}
    batch = []
    with torch.no_grad():
        for n, image in enumerate(images):
            x, _ = to_input(ROOT / 'JPEGImages/val' / image['file_name'])
            batch.append((image, x))
            if len(batch) < args.batch and n < len(images) - 1:
                continue
            xb = torch.stack([b[1] for b in batch]).to(args.device)
            outs = {}
            for k in range(inner.num_layers):
                inner.eval_idx = k
                result = model(xb)
                outs[f'dec{k + 1}'] = (result['pred_logits'].float(), result['pred_boxes'].float())
                if k == 0:
                    ind = saved['topk_ind']
                    enc_logits = torch.gather(saved['enc_logits'], 1, ind[..., None].expand(-1, -1, saved['enc_logits'].shape[-1]))
                    outs['enc'] = (enc_logits.float(), torch.sigmoid(saved['enc_box_delta'] + saved['topk_anchors']).float())
                    pre_boxes, pre_logits = saved['pre']
                    outs['pre'] = (pre_logits.float(), pre_boxes.float())
            inner.eval_idx = runner.default_idx
            has = [j for j, (img, _) in enumerate(batch) if grouped[img['id']]]
            if has:
                targets = [runner.targets(batch[j][0], grouped[batch[j][0]['id']])[0][0] for j in has]
                sel = torch.tensor(has, device=args.device)
                ind = {s: runner.matcher({'pred_logits': outs[s][0][sel], 'pred_boxes': outs[s][1][sel]}, targets)['indices'] for s in STAGES}
                go = crit._get_go_indices(ind['dec4'], [ind[s] for s in ('dec1', 'dec2', 'dec3', 'pre', 'enc')])
                for t, j in enumerate(has):
                    rec = {s: (ind[s][t][0].cpu().numpy(), ind[s][t][1].cpu().numpy()) for s in STAGES}
                    rec['go'] = (go[t][0].cpu().numpy(), go[t][1].cpu().numpy())
                    assign[batch[j][0]['id']] = rec
            for j, (img, _) in enumerate(batch):
                ids.append(img['id'])
                logits_all.append(outs['dec4'][0][j].cpu().numpy())
                boxes_all.append(xyxy(outs['dec4'][1][j].cpu().numpy(), 1))
            batch = []
            if (n + 1) % 1000 < args.batch:
                print('images', n + 1, len(images), flush=True)
    np.savez_compressed(out / 'queries.npz', image_id=np.array(ids), logits=np.stack(logits_all).astype(np.float32),
                        boxes=np.stack(boxes_all).astype(np.float32))
    torch.save(assign, out / 'assign.pt')
    (out / 'collect_meta.json').write_text(json.dumps(dict(run=str(args.run), checkpoint=str(args.checkpoint), epoch=runner.epoch,
                                                           matcher=dict(cost_class=runner.matcher.cost_class, cost_bbox=runner.matcher.cost_bbox,
                                                                        cost_giou=runner.matcher.cost_giou),
                                                           images=len(ids), images_with_gt=len(assign), batch=args.batch), indent=1))
    print('collected', out, flush=True)


# ---------------------------------------------------------------- per-run tables
class Val:
    def __init__(self, out):
        from tank_feature_diagnosis import load_val
        annotations, source, grouped = load_val(out)
        self.names = {c['id']: c['name'] for c in annotations['categories']}
        self.classes = sorted(self.names)
        self.info = {i['id']: i for i in annotations['images']}
        self.group = {i['id']: group_of(source[i['id']], i['width'], i['height']) for i in annotations['images']}
        self.gt = {}
        for i in annotations['images']:
            anns = grouped[i['id']]
            w, h = i['width'], i['height']
            box = np.array([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]] for a in anns], np.float32).reshape(-1, 4)
            self.gt[i['id']] = dict(label=np.array([a['category_id'] for a in anns], int), box=box,
                                    crowd=np.array([a.get('iscrowd', 0) for a in anns], bool), norm=box / np.array([w, h, w, h], np.float32))


def load_run(out):
    import torch
    with np.load(out / 'queries.npz') as z:
        data = {k: z[k] for k in z.files}
    return data, torch.load(out / 'assign.pt', weights_only=False), json.loads((out / 'collect_meta.json').read_text())


def detections(val, data):
    """Postprocessor rule per image: sigmoid, top 300 over (query, class). Returns per-image dicts."""
    dets = {}
    for n, image_id in enumerate(data['image_id'].tolist()):
        score = sigmoid(data['logits'][n].astype(np.float64))
        flat = np.argsort(-score.ravel(), kind='stable')[:300]
        q, c = np.divmod(flat, score.shape[1])
        w, h = val.info[image_id]['width'], val.info[image_id]['height']
        dets[image_id] = dict(query=q, label=c, score=score.ravel()[flat], box=data['boxes'][n][q] * np.array([w, h, w, h], np.float32))
    return dets


def evaluate(val, dets):
    """COCO matching per image and class. Returns detection records and per-class arrays for AP."""
    recs = defaultdict(list)
    for image_id, d in dets.items():
        gt = val.gt[image_id]
        ious_all = iou_xyxy(d['box'], gt['box']) if len(gt['label']) else np.zeros((len(d['box']), 0))
        g = val.group[image_id]
        for c in val.classes:
            sel = np.flatnonzero(d['label'] == c)[:100]
            gidx = np.flatnonzero(gt['label'] == c)
            crowd = gt['crowd'][gidx]
            order = np.argsort(crowd, kind='stable')  # COCO puts ignored GTs last
            gidx, crowd = gidx[order], crowd[order]
            npos = int((~crowd).sum())
            recs['npos', c].append((image_id, npos))
            if not len(sel):
                continue
            m = ious_all[sel][:, gidx] if len(gidx) else np.zeros((len(sel), 0))
            tp = np.zeros((len(sel), len(THRESH)), bool)
            ignored = np.zeros((len(sel), len(THRESH)), bool)
            cand = np.flatnonzero(m.max(1) >= .5) if len(gidx) else np.empty(0, int)
            for t, thr in enumerate(THRESH):
                taken = np.zeros(len(gidx), bool)
                for di in cand:
                    best, best_iou = -1, min(thr, 1 - 1e-10)
                    for gj in range(len(gidx)):
                        if taken[gj] and not crowd[gj]:
                            continue
                        if best > -1 and not crowd[best] and crowd[gj]:
                            break
                        if m[di, gj] < best_iou:
                            continue
                        best, best_iou = gj, m[di, gj]
                    if best > -1:
                        taken[best] = True
                        if crowd[best]:
                            ignored[di, t] = True
                        else:
                            tp[di, t] = True
            same = m.max(1) if len(gidx) else np.zeros(len(sel))
            other = ious_all[sel][:, gt['label'] != c].max(1) if (gt['label'] != c).any() else np.zeros(len(sel))
            kind = np.where(tp[:, 0], 0, np.where(same >= .5, 1, np.where(other >= .5, 2, np.where(same >= .1, 3, 4))))
            recs['det'].append(dict(image=np.full(len(sel), image_id), group=np.full(len(sel), g), label=np.full(len(sel), c),
                                    score=d['score'][sel], query=d['query'][sel], tp=tp, ignored=ignored, kind=kind, same_iou=same))
    det = {k: np.concatenate([r[k] for r in recs['det']]) for k in recs['det'][0]}
    npos = {c: dict(recs['npos', c]) for c in val.classes}
    return det, npos


def ap_arrays(det, npos, val, images):
    """Per class: score-sorted detections of the chosen images, their image index and TP flags; GT count per image."""
    pos = {int(i): k for k, i in enumerate(images)}
    keep = np.array([int(i) in pos for i in det['image']])
    out = {}
    for c in val.classes:
        s = np.flatnonzero(keep & (det['label'] == c))
        s = s[np.argsort(-det['score'][s], kind='stable')]
        gt_count = np.array([npos[c].get(int(i), 0) for i in images], float)
        if gt_count.sum() == 0:
            continue
        out[c] = dict(img=np.array([pos[int(i)] for i in det['image'][s]], int), tp=det['tp'][s], ig=det['ignored'][s],
                      score=det['score'][s], gt=gt_count, kind=det['kind'][s])
    return out


def ap_from(arr, weight=None, drop=None):
    """AP per threshold for one class (COCO 101-point). weight: per-image bootstrap weights. drop: mask of detections removed."""
    w = np.ones(len(arr['gt'])) if weight is None else weight
    dw = w[arr['img']].astype(float)
    if drop is not None:
        dw = dw * (~drop)
    npos = (w * arr['gt']).sum()
    if npos == 0:
        return np.full(len(THRESH), np.nan), None, None
    valid = ~arr['ig']
    tpw = arr['tp'] * dw[:, None]
    fpw = (~arr['tp'] & valid) * dw[:, None]
    tpc, fpc = np.cumsum(tpw, 0), np.cumsum(fpw, 0)
    recall = tpc / npos
    precision = tpc / np.maximum(tpc + fpc, np.finfo(float).eps)
    precision = np.maximum.accumulate(precision[::-1], 0)[::-1]
    ap = np.empty(len(THRESH))
    curve = np.zeros((len(THRESH), len(RECALL)))
    for t in range(len(THRESH)):
        idx = np.searchsorted(recall[:, t], RECALL, side='left')
        ok = idx < len(recall)
        curve[t, ok] = precision[idx[ok], t]
        ap[t] = curve[t].mean()
    return ap, curve, recall[-1] if len(recall) else np.zeros(len(THRESH))


def group_ap(arrs, weight=None, drops=None):
    """Mean over classes (each class with GT in the subset) of per-threshold AP."""
    aps = {c: ap_from(a, weight, None if drops is None else drops[c])[0] for c, a in arrs.items()}
    return np.nanmean(np.stack(list(aps.values())), 0), aps


# ---------------------------------------------------------------- task 2 and 3 per run
def per_gt(val, data, assign, dets, det):
    """Final-layer ranking of each GT's candidates and the fate of the best box; BP-CB queries."""
    kind_of = {}
    for i, q, c, k, s in zip(det['image'], det['query'], det['label'], det['kind'], det['score']):
        kind_of[int(i), int(q), int(c)] = (int(k), float(s))
    rows, bp = [], []
    for n, image_id in enumerate(data['image_id'].tolist()):
        gt = val.gt[image_id]
        if not len(gt['label']):
            continue
        score = sigmoid(data['logits'][n].astype(np.float64))
        order = np.argsort(-score.ravel(), kind='stable')
        rank_flat = np.empty(order.size, int)
        rank_flat[order] = np.arange(order.size)
        boxes = data['boxes'][n]
        a = assign[image_id]
        m4 = dict(zip(a['dec4'][1].tolist(), a['dec4'][0].tolist()))  # gt -> matched query at the last layer
        matched4 = set(a['dec4'][0].tolist())
        go_q, go_g = a['go']
        stage_sets = {s: set(zip(a[s][0].tolist(), a[s][1].tolist())) for s in STAGES}
        ious = iou_xyxy(boxes, gt['norm'])
        g = val.group[image_id]
        for c in np.unique(gt['label']):
            idx = np.flatnonzero(gt['label'] == c)
            m = ious[:, idx]
            owner, best_iou = m.argmax(1), m.max(1)
            # rank of each query's class-c score among the 300 queries
            rank_c = np.empty(300, int)
            rank_c[np.argsort(-score[:, c], kind='stable')] = np.arange(300)
            for j, t in enumerate(idx):
                cand = np.flatnonzero((owner == j) & (best_iou >= .1))
                row = dict(image=image_id, group=g, label=int(c), gt=int(t), n_cand=len(cand))
                if len(cand):
                    s, iou = score[cand, c], m[cand, j]
                    top, best = int(cand[np.argmax(s)]), int(cand[np.argmax(iou)])
                    rank_in_cand = int((s > score[best, c]).sum()) + 1
                    kb = kind_of.get((image_id, best, int(c)))
                    kt = kind_of.get((image_id, top, int(c)))
                    row.update(top=top, best=best, top_iou=float(m[top, j]), best_iou=float(m[best, j]),
                               top_score=float(score[top, c]), best_score=float(score[best, c]),
                               best_rank_cand=rank_in_cand, best_rank_class=int(rank_c[best]) + 1,
                               best_rank_flat=int(rank_flat[best * score.shape[1] + c]) + 1,
                               best_kind=-1 if kb is None else kb[0], top_kind=-1 if kt is None else kt[0],
                               hi_cand_03=int(((s >= .3) & (iou >= .5)).sum()),
                               matched=m4.get(int(t), -1), best_is_matched=int(m4.get(int(t), -1) == best),
                               top_is_matched=int(m4.get(int(t), -1) == top))
                rows.append(row)
        # BP-CB: union-set queries not matched at the last layer.
        for q, t in zip(go_q.tolist(), go_g.tolist()):
            c = int(gt['label'][t])
            mq = m4.get(t, -1)
            k = kind_of.get((image_id, q, c))
            rec = dict(image=image_id, group=g, label=c, gt=t, query=q, matched4=int(q in matched4),
                       iou=float(ious[q, t]), score=float(score[q, c]), kind=-1 if k is None else k[0],
                       stages=''.join('1' if (q, t) in stage_sets[s] else '0' for s in STAGES),
                       mq=mq, mq_iou=float(ious[mq, t]) if mq >= 0 else np.nan, mq_score=float(score[mq, c]) if mq >= 0 else np.nan)
            bp.append(rec)
    return rows, bp


def table(rows, keys):
    return {k: np.array([r.get(k, np.nan) for r in rows], dtype=object if k == 'group' else float) for k in keys}


def summarise_gt(rows, names):
    keys = ['label', 'n_cand', 'top_iou', 'best_iou', 'top_score', 'best_score', 'best_rank_cand', 'best_rank_class',
            'best_rank_flat', 'best_kind', 'top_kind', 'hi_cand_03', 'best_is_matched', 'top_is_matched']
    t = table(rows, keys)
    t['group'] = np.array([r['group'] for r in rows])
    out = {}

    def stats(mask):
        has = mask & (t['n_cand'] > 0)
        n = int(has.sum())
        if not n:
            return dict(gt=int(mask.sum()))
        beaten = (t['best_iou'] - t['top_iou']) >= .05
        return dict(
            gt=int(mask.sum()), no_candidate=r3(1 - n / mask.sum()),
            top_is_best=r3(np.mean(t['best_rank_cand'][has] == 1)), top_beaten_005=r3(np.mean(beaten[has])),
            best_rank_cand_median=r3(np.median(t['best_rank_cand'][has])), best_rank_cand_mean=r3(np.mean(t['best_rank_cand'][has])),
            best_rank_le2=r3(np.mean(t['best_rank_cand'][has] <= 2)), best_rank_class_median=r3(np.median(t['best_rank_class'][has])),
            best_kept_top300=r3(np.mean(t['best_rank_flat'][has] <= 300)), best_evaluated=r3(np.mean(t['best_kind'][has] >= 0)),
            best_is_tp50=r3(np.mean(t['best_kind'][has] == 0)), best_is_duplicate=r3(np.mean(t['best_kind'][has] == 1)),
            top_is_tp50=r3(np.mean(t['top_kind'][has] == 0)),
            best_score_median=r3(np.median(t['best_score'][has])), top_score_median=r3(np.median(t['top_score'][has])),
            score_gap_when_beaten=r3(np.median((t['top_score'] - t['best_score'])[has & beaten])) if (has & beaten).any() else None,
            best_iou_median=r3(np.median(t['best_iou'][has])), top_iou_median=r3(np.median(t['top_iou'][has])),
            hi_cand_03_mean=r3(np.mean(t['hi_cand_03'][has])), gt_with_2plus_hi_cand=r3(np.mean(t['hi_cand_03'][has] >= 2)),
            best_is_matched=r3(np.mean(t['best_is_matched'][has])), top_is_matched=r3(np.mean(t['top_is_matched'][has])))

    out['all'] = stats(np.ones(len(rows), bool))
    for g in GROUP_ORDER:
        if (t['group'] == g).any():
            out[g] = stats(t['group'] == g)
            for c in sorted(set(t['label'][t['group'] == g].astype(int))):
                out[f'{g}/{names[c]}'] = stats((t['group'] == g) & (t['label'] == c))
    return out


def summarise_bp(bp, det, names):
    keys = ['label', 'matched4', 'iou', 'score', 'kind', 'mq', 'mq_iou', 'mq_score']
    t = table(bp, keys)
    t['group'] = np.array([r['group'] for r in bp])
    stages = np.array([r['stages'] for r in bp])
    gts = Counter((r['image'], r['gt']) for r in bp)
    pos_dup = (det['kind'] == 1)
    out = {}

    def stats(mask_g, gt_count, dup_mask):
        b = mask_g & (t['matched4'] == 0)
        p = mask_g & (t['matched4'] == 1)
        r = dict(gt=gt_count, union_queries=int(mask_g.sum()), bpcb_queries=int(b.sum()), bpcb_per_gt=r3(b.sum() / max(gt_count, 1)))
        for name, mm in (('bpcb', b), ('matched', p)):
            if not mm.any():
                continue
            r[name] = dict(iou_median=r3(np.median(t['iou'][mm])), iou_ge_075=r3(np.mean(t['iou'][mm] >= .75)),
                           score_median=r3(np.median(t['score'][mm])), score_ge_03=r3(np.mean(t['score'][mm] >= .3)),
                           score_ge_05=r3(np.mean(t['score'][mm] >= .5)), evaluated=r3(np.mean(t['kind'][mm] >= 0)),
                           **{f'{k}_of_all': r3(np.mean(t['kind'][mm] == i)) for i, k in enumerate(TYPES)},
                           **{f'{k}_at03': r3(np.mean((t['kind'][mm] == i) & (t['score'][mm] >= .3))) for i, k in enumerate(TYPES[:2])})
        if b.any():
            has_mq = b & (t['mq'] >= 0)
            r['bpcb']['better_box_than_matched'] = r3(np.mean(t['iou'][has_mq] > t['mq_iou'][has_mq] + .05)) if has_mq.any() else None
            r['bpcb']['higher_score_than_matched'] = r3(np.mean(t['score'][has_mq] > t['mq_score'][has_mq])) if has_mq.any() else None
            sc = Counter(stages[b])
            r['bpcb']['stage_patterns_top'] = {k: r3(v / b.sum()) for k, v in sc.most_common(6)}
            r['bpcb']['matched_at_stage'] = {s: r3(np.mean([x[i] == '1' for x in stages[b]])) for i, s in enumerate(STAGES)}
        # Share of duplicates (score >= .3) that are BP-CB queries.
        dup = dup_mask & (det['score'] >= .3)
        bp_keys = {(x['image'], x['query'], x['label']) for x, mk in zip(bp, b) if mk}
        dk = [(int(i), int(q), int(c)) in bp_keys for i, q, c in zip(det['image'][dup], det['query'][dup], det['label'][dup])]
        r['duplicates_at03'] = int(dup.sum())
        r['duplicates_at03_from_bpcb'] = r3(np.mean(dk)) if len(dk) else None
        hist_b = np.histogram(t['score'][b], BINS)[0].tolist() if b.any() else None
        hist_p = np.histogram(t['score'][p], BINS)[0].tolist() if p.any() else None
        r['score_hist'] = dict(bpcb=hist_b, matched=hist_p)
        return r

    allgt = len(gts)
    out['all'] = stats(np.ones(len(bp), bool), allgt, pos_dup)
    for g in GROUP_ORDER:
        mg = t['group'] == g
        if mg.any():
            out[g] = stats(mg, len({(x['image'], x['gt']) for x, k in zip(bp, mg) if k}), pos_dup & (det['group'] == g))
            for c in sorted(set(t['label'][mg].astype(int))):
                mc = mg & (t['label'] == c)
                out[f'{g}/{names[c]}'] = stats(mc, len({(x['image'], x['gt']) for x, k in zip(bp, mc) if k}),
                                               pos_dup & (det['group'] == g) & (det['label'] == c))
    return out


# ---------------------------------------------------------------- task 1 per run
def task1_run(val, det, npos, dets):
    images = sorted(val.info)
    by_group = defaultdict(list)
    for i in images:
        by_group[val.group[i]].append(i)
    res = {}
    arr_all = ap_arrays(det, npos, val, images)
    ap_all, ap_cls = group_ap(arr_all)
    res['ap_by_threshold'] = {'all': [r3(x * 100) for x in ap_all]}
    res['ap_by_threshold_class'] = {val.names[c]: [r3(x * 100) for x in v] for c, v in ap_cls.items()}
    curves = {}
    for c, a in arr_all.items():
        _, curve, _ = ap_from(a)
        curves[val.names[c]] = {'.50': [r3(x) for x in curve[0]], '.75': [r3(x) for x in curve[5]]}
    curves['mean'] = {k: [r3(x) for x in np.mean([np.array(v[k], float) for v in curves.values()], 0)] for k in ('.50', '.75')}
    res['pr_all'] = curves
    # Oracle removals at IoU .5 (not deployable): which detection type explains the AP50 level?
    res['ap50_without'] = {}
    for k, name in enumerate(TYPES[1:], start=1):
        drops = {c: a['kind'] == k for c, a in arr_all.items()}
        res['ap50_without'][name] = r3(group_ap(arr_all, drops=drops)[0][0] * 100)
    res['ap50_without']['none'] = r3(ap_all[0] * 100)
    # Class-wise NMS (IoU .7 / .5 / .3) as a diagnostic of how near the duplicates are.
    res['source_class'] = {}
    res['pr_cells'] = {}
    for g in GROUP_ORDER:
        if g not in by_group:
            continue
        arr = ap_arrays(det, npos, val, by_group[g])
        gm, gc = group_ap(arr)
        res['ap_by_threshold'][g] = [r3(x * 100) for x in gm]
        for c, v in gc.items():
            key = f'{g}/{val.names[c]}'
            drops = arr[c]['kind'] == 1
            res['source_class'][key] = dict(AP=r3(np.mean(v) * 100), AP50=r3(v[0] * 100), AP75=r3(v[5] * 100),
                                            AP50_without_duplicates=r3(ap_from(arr[c], drop=drops)[0][0] * 100),
                                            gt=int(arr[c]['gt'].sum()))
            if key in ('ogsod/tank', 'other_512/aircraft', 'other_256/ship', 'msar/ship', 'other_1000/ship', 'ogsod/bridge'):
                _, curve, _ = ap_from(arr[c])
                res['pr_cells'][key] = {'.50': [r3(x) for x in curve[0]], '.75': [r3(x) for x in curve[5]]}
    # Score distributions per detection type (evaluated detections).
    res['type_counts'] = {}
    res['type_hist'] = {}
    for k, name in enumerate(TYPES):
        mk = det['kind'] == k
        res['type_counts'][name] = {f'>={s}': int((mk & (det['score'] >= s)).sum()) for s in SCORE_CUTS}
        res['type_counts'][name]['all'] = int(mk.sum())
        res['type_hist'][name] = np.histogram(det['score'][mk], BINS)[0].tolist()
        if mk.any():
            res['type_counts'][name]['score_q50_q90_q99'] = [r3(np.quantile(det['score'][mk], q)) for q in (.5, .9, .99)]
    res['type_counts_by_group_at03'] = {}
    for g in GROUP_ORDER:
        mg = det['group'] == g
        if mg.any():
            res['type_counts_by_group_at03'][g] = {name: int((mg & (det['kind'] == k) & (det['score'] >= .3)).sum()) for k, name in enumerate(TYPES)}
    # Detections per image.
    per_image = {}
    for g in GROUP_ORDER + ['all']:
        ids = images if g == 'all' else by_group.get(g, [])
        if not ids:
            continue
        gt_n = np.array([len(val.gt[i]['label']) for i in ids])
        row = dict(images=len(ids), gt_per_image=r3(gt_n.mean()))
        for s in (.1, .3, .5):
            cnt = np.array([(dets[i]['score'] >= s).sum() for i in ids])
            row[f'dets_ge{s}'] = r3(cnt.mean())
            row[f'dets_ge{s}_over_gt'] = r3(cnt.sum() / max(gt_n.sum(), 1))
            row[f'images_more_than_gt_ge{s}'] = r3(np.mean(cnt > gt_n))
        per_image[g] = row
    res['per_image'] = per_image
    return res, arr_all, by_group


def nms_keep(det, dets, thr):
    """Class-wise greedy NMS over the evaluated detections (diagnostic only). det rows are contiguous per image/class."""
    keep_mask = np.ones(len(det['image']), bool)
    key = det['image'].astype(np.int64) * 10 + det['label']
    bounds = np.flatnonzero(np.diff(key)) + 1
    blocks = np.split(np.arange(len(key)), bounds)
    for b in blocks:
        image_id = int(det['image'][b[0]])
        d = dets[image_id]
        boxes = d['box'][_rows_of(d, det['query'][b], det['label'][b])]
        order = np.argsort(-det['score'][b], kind='stable')
        ious = iou_xyxy(boxes[order], boxes[order])
        removed = np.zeros(len(b), bool)
        for x in range(len(b)):
            if removed[x]:
                continue
            removed[x + 1:] |= ious[x, x + 1:] > thr
        keep_mask[b[order[removed]]] = False
    return keep_mask


def _rows_of(d, queries, labels):
    lookup = {(int(q), int(c)): k for k, (q, c) in enumerate(zip(d['query'], d['label']))}
    return np.array([lookup[int(q), int(c)] for q, c in zip(queries, labels)])


def paired_bootstrap(arr_a, arr_b, n_images, reps, seed=0):
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(reps):
        w = rng.multinomial(n_images, np.full(n_images, 1 / n_images)).astype(float)
        a = group_ap(arr_a, w)[0]
        b = group_ap(arr_b, w)[0]
        diffs.append([np.mean(b) - np.mean(a), b[0] - a[0], b[5] - a[5]])
    d = np.array(diffs) * 100
    return {k: dict(ci95=[r3(np.percentile(d[:, i], 2.5)), r3(np.percentile(d[:, i], 97.5))],
                    p_le_0=r3(np.mean(d[:, i] <= 0)), p_ge_0=r3(np.mean(d[:, i] >= 0))) for i, k in enumerate(('AP', 'AP50', 'AP75'))}


# ---------------------------------------------------------------- cases
def draw_cases(val, runs_assign, labels, bps, out, n_cases=20):
    from PIL import Image, ImageDraw
    from error_gallery import ROOT
    b = [x for x in bps[1] if not x['matched4'] and x['kind'] == 1 and x['score'] >= .3]
    priority = ['ogsod', 'other_512', 'other_256', 'msar', 'other_800', 'other_1000', 'other_224', 'other_misc']
    quota = {'ogsod': 6, 'other_512': 4, 'other_256': 3, 'msar': 3, 'other_800': 1, 'other_1000': 1, 'other_224': 1, 'other_misc': 1}
    chosen, seen = [], set()
    for g in priority:
        pool = sorted([x for x in b if x['group'] == g], key=lambda x: -x['score'])
        k = 0
        for x in pool:
            if x['image'] in seen or k >= quota[g]:
                continue
            chosen.append(x)
            seen.add(x['image'])
            k += 1
    for x in sorted(b, key=lambda x: -x['score']):
        if len(chosen) >= n_cases:
            break
        if x['image'] not in seen:
            chosen.append(x)
            seen.add(x['image'])
    chosen = chosen[:n_cases]
    (out / 'cases').mkdir(parents=True, exist_ok=True)
    meta, tiles = [], []
    for n, case in enumerate(chosen):
        image_id, t = case['image'], case['gt']
        info = val.info[image_id]
        img = Image.open(ROOT / 'JPEGImages/val' / info['file_name']).convert('RGB')
        w, h = img.size
        gt = val.gt[image_id]
        gx = gt['box'][t]
        side = max(gx[2] - gx[0], gx[3] - gx[1]) * 3 + 40
        cx, cy = (gx[0] + gx[2]) / 2, (gx[1] + gx[3]) / 2
        crop = [max(0, cx - side / 2), max(0, cy - side / 2), min(w, cx + side / 2), min(h, cy + side / 2)]
        panels, info_rows = [], []
        for r in range(2):
            data, assign = runs_assign[r]
            row = int(np.flatnonzero(data['image_id'] == image_id)[0])
            score = sigmoid(data['logits'][row].astype(np.float64))
            boxes = data['boxes'][row] * np.array([w, h, w, h])
            c = int(gt['label'][t])
            a = assign[image_id]
            m4 = dict(zip(a['dec4'][1].tolist(), a['dec4'][0].tolist()))
            mq = m4.get(t, -1)
            matched4 = set(a['dec4'][0].tolist())
            bpq = [int(q) for q, g in zip(*a['go']) if g == t and int(q) not in matched4]
            panel = img.crop(tuple(int(v) for v in crop))
            scale = 320 / max(panel.size)
            panel = panel.resize((max(1, int(panel.size[0] * scale)), max(1, int(panel.size[1] * scale))))
            dr = ImageDraw.Draw(panel)

            def rect(bx, colour, width=2):
                bb = [(bx[0] - crop[0]) * scale, (bx[1] - crop[1]) * scale, (bx[2] - crop[0]) * scale, (bx[3] - crop[1]) * scale]
                dr.rectangle(bb, outline=colour, width=width)
                return bb

            for j in range(len(gt['label'])):
                rect(gt['box'][j], (255, 255, 255), 1)
            rect(gx, (255, 255, 0), 2)
            lines = [f'{labels[r]}  GT={val.names[c]}']
            iou_t = iou_xyxy(boxes, gx[None])[:, 0]
            hi = [q for q in np.flatnonzero((score[:, c] >= .3) & (iou_t >= .5)) if q != mq and q not in bpq]
            for q in hi:
                rect(boxes[q], (90, 160, 255), 1)
            for k, q in enumerate(sorted(bpq, key=lambda q: -score[q, c])):
                rect(boxes[q], (255, 60, 60), 2)
                if k < 4:
                    lines.append(f'BP-CB q{q}: s={score[q, c]:.2f} IoU={iou_t[q]:.2f}')
            if mq >= 0:
                rect(boxes[mq], (60, 230, 60), 2)
                lines.insert(1, f'matched q{mq}: s={score[mq, c]:.2f} IoU={iou_t[mq]:.2f}')
            lines.append(f'other s>=.3 IoU>=.5: {len(hi)}')
            canvas = Image.new('RGB', (330, 330 + 14 * len(lines)), (20, 20, 20))
            canvas.paste(panel, (5, 5))
            d2 = ImageDraw.Draw(canvas)
            for k, s in enumerate(lines):
                d2.text((6, 332 + 14 * k), s, fill=(235, 235, 235))
            panels.append(canvas)
            info_rows.append(dict(run=labels[r], matched=int(mq), matched_score=r3(score[mq, c]) if mq >= 0 else None,
                                  matched_iou=r3(iou_t[mq]) if mq >= 0 else None,
                                  bpcb=[dict(query=int(q), score=r3(score[q, c]), iou=r3(iou_t[q])) for q in bpq], other_hi=len(hi)))
        H = max(p.size[1] for p in panels)
        both = Image.new('RGB', (sum(p.size[0] for p in panels) + 10, H + 20), (0, 0, 0))
        ImageDraw.Draw(both).text((6, 4), f'case {n + 1:02d}  image {image_id}  {case["group"]}', fill=(255, 255, 255))
        x0 = 0
        for p in panels:
            both.paste(p, (x0, 20))
            x0 += p.size[0] + 10
        path = out / 'cases' / f'case_{n + 1:02d}_{case["group"]}_{val.names[int(gt["label"][t])]}_{image_id}.png'
        both.save(path)
        tiles.append(both)
        meta.append(dict(case=n + 1, image=image_id, file=info['file_name'], group=case['group'], cls=val.names[int(gt['label'][t])], gt=t, runs=info_rows))
    if tiles:
        cols = 4
        tw, th = max(t.size[0] for t in tiles), max(t.size[1] for t in tiles)
        sheet = Image.new('RGB', (cols * tw, ((len(tiles) + cols - 1) // cols) * th), (0, 0, 0))
        for k, t in enumerate(tiles):
            sheet.paste(t, ((k % cols) * tw, (k // cols) * th))
        sheet.save(out / 'cases' / 'contact_sheet.png')
    return meta


def analyse(args):
    val = Val(args.base)
    labels = args.labels
    result = dict(meta={}, task1={}, task2={}, task3={})
    per, runs_assign = [], []
    for lab, out in zip(labels, (args.base, args.run)):
        data, assign, meta = load_run(out)
        runs_assign.append((data, assign))
        dets = detections(val, data)
        det, npos = evaluate(val, dets)
        t1, arr_all, by_group = task1_run(val, det, npos, dets)
        # NMS diagnostic
        t1['ap_after_classwise_nms'] = {}
        for thr in (.7, .5):
            keep = nms_keep(det, dets, thr)
            # Matching is redone on the kept list (TP flags change when a higher-scored duplicate is removed).
            sub = {k: v[keep] for k, v in det.items()}
            re_dets = {}
            order = np.argsort(sub['image'], kind='stable')
            ids_sorted = sub['image'][order]
            for i in val.info:
                d = dets[i]
                lo, hi = np.searchsorted(ids_sorted, i, 'left'), np.searchsorted(ids_sorted, i, 'right')
                mk = order[lo:hi]
                rows = np.sort(_rows_of(d, sub['query'][mk], sub['label'][mk])) if len(mk) else np.empty(0, int)
                re_dets[i] = dict(query=d['query'][rows], label=d['label'][rows], score=d['score'][rows], box=d['box'][rows])
            det2, npos2 = evaluate(val, re_dets)
            arr2 = ap_arrays(det2, npos2, val, sorted(val.info))
            ap2 = group_ap(arr2)[0]
            t1['ap_after_classwise_nms'][str(thr)] = dict(AP=r3(np.mean(ap2) * 100), AP50=r3(ap2[0] * 100), AP75=r3(ap2[5] * 100),
                                                          kept=int(keep.sum()), removed=int((~keep).sum()))
            print(lab, 'nms', thr, t1['ap_after_classwise_nms'][str(thr)], flush=True)
        rows, bp = per_gt(val, data, assign, dets, det)
        result['meta'][lab] = meta
        result['task1'][lab] = t1
        result['task2'][lab] = summarise_gt(rows, val.names)
        result['task3'][lab] = summarise_bp(bp, det, val.names)
        per.append(dict(arr=arr_all, by_group=by_group, det=det, npos=npos, bp=bp))
        print(lab, 'AP by threshold', t1['ap_by_threshold']['all'], flush=True)
    # Paired bootstrap over images: all val and each group.
    images = sorted(val.info)
    result['task1']['paired_bootstrap'] = {'all': paired_bootstrap(per[0]['arr'], per[1]['arr'], len(images), args.boot)}
    for g in GROUP_ORDER:
        ids = per[0]['by_group'].get(g)
        if not ids:
            continue
        a = ap_arrays(per[0]['det'], per[0]['npos'], val, ids)
        b = ap_arrays(per[1]['det'], per[1]['npos'], val, ids)
        result['task1']['paired_bootstrap'][g] = paired_bootstrap(a, b, len(ids), args.boot)
        print('bootstrap', g, result['task1']['paired_bootstrap'][g], flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    result['task3']['cases'] = draw_cases(val, runs_assign, labels, [p['bp'] for p in per], args.out)
    (args.out / 'followup.json').write_text(json.dumps(result, indent=1, ensure_ascii=False, default=float))
    print('written', args.out / 'followup.json', flush=True)


# ---------------------------------------------------------------- plot (local)
COLOURS = {'E000': '#2a78d6', 'E1': '#eb6834'}
TYPE_NAMES = {'tp': 'TP', 'duplicate': '重复 FP', 'class': '类别混淆 FP', 'localisation': '定位 FP', 'background': '背景 FP'}


def _style(plt):
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'], 'axes.unicode_minus': False,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e5e0',
                         'grid.linewidth': .6, 'axes.edgecolor': '#8a8984', 'axes.labelcolor': '#52514e',
                         'xtick.color': '#52514e', 'ytick.color': '#52514e', 'font.size': 9, 'figure.facecolor': '#fcfcfb',
                         'axes.facecolor': '#fcfcfb', 'legend.frameon': False})


def plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _style(plt)
    d = args.dir
    r = json.loads((d / 'followup.json').read_text(encoding='utf-8'))
    labs = [k for k in r['task1'] if k != 'paired_bootstrap'][:2]
    t1 = {k: r['task1'][k] for k in labs}
    fig_dir = d / 'figures'
    fig_dir.mkdir(exist_ok=True)
    centres = (BINS[:-1] + BINS[1:]) / 2

    # 1 AP by IoU threshold
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={'width_ratios': [1, 1.5]})
    for k in labs:
        ax[0].plot(THRESH, t1[k]['ap_by_threshold']['all'], marker='o', ms=4, lw=2, color=COLOURS[k], label=k)
    ax[0].set_xlabel('IoU 阈值')
    ax[0].set_ylabel('AP（类别平均）')
    ax[0].legend()
    ax[0].set_title('逐阈值 AP，全部 val')
    groups = [g for g in GROUP_ORDER if g in t1[labs[0]]['ap_by_threshold']]
    rows = ['all'] + groups
    delta = np.array([np.array(t1[labs[1]]['ap_by_threshold'][g], float) - np.array(t1[labs[0]]['ap_by_threshold'][g], float) for g in rows])
    lim = np.nanmax(np.abs(delta))
    im = ax[1].imshow(delta, cmap='RdBu', vmin=-lim, vmax=lim, aspect='auto')
    ax[1].set_xticks(range(len(THRESH)), [f'{v:.2f}' for v in THRESH])
    ax[1].set_yticks(range(len(rows)), rows)
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            ax[1].text(j, i, f'{delta[i, j]:+.1f}', ha='center', va='center', fontsize=6.5, color='#0b0b0b')
    ax[1].grid(False)
    ax[1].set_title(f'ΔAP（{labs[1]} − {labs[0]}），按来源与 IoU 阈值')
    fig.colorbar(im, ax=ax[1], shrink=.8)
    fig.tight_layout()
    fig.savefig(fig_dir / 'fig1_ap_by_threshold.png', dpi=160)
    plt.close(fig)

    # 2 score distribution by detection type
    fig, ax = plt.subplots(1, 4, figsize=(12.5, 3.1), sharex=True)
    for a, name in zip(ax, ['tp', 'duplicate', 'background', 'localisation']):
        for k in labs:
            h = np.array(t1[k]['type_hist'][name], float)
            a.step(centres, np.where(h > 0, h, np.nan), where='mid', lw=1.8, color=COLOURS[k], label=k)
        a.set_yscale('log')
        a.set_title(TYPE_NAMES[name])
        a.set_xlabel('分数')
    ax[0].set_ylabel('检测数（对数）')
    ax[0].legend()
    fig.suptitle('被评测检测（每图每类前 100）按 IoU .5 类型的分数分布', fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / 'fig2_score_by_type.png', dpi=160)
    plt.close(fig)

    # 3 PR curves
    cells = [('mean', '.50'), ('mean', '.75'), ('ogsod/tank', '.50'), ('other_512/aircraft', '.50'), ('other_256/ship', '.50'), ('other_1000/ship', '.50')]
    fig, ax = plt.subplots(2, 3, figsize=(11, 6.2), sharex=True, sharey=True)
    for a, (cell, thr) in zip(ax.ravel(), cells):
        for k in labs:
            curve = t1[k]['pr_all']['mean'][thr] if cell == 'mean' else t1[k]['pr_cells'].get(cell, {}).get(thr)
            if curve:
                a.plot(RECALL, curve, lw=2, color=COLOURS[k], label=k)
        a.set_title(f'{"全部 val，类别平均" if cell == "mean" else cell}  IoU {thr}')
        a.set_xlim(0, 1)
        a.set_ylim(0, 1.02)
    for a in ax[1]:
        a.set_xlabel('召回')
    for a in ax[:, 0]:
        a.set_ylabel('精度')
    ax[0, 0].legend(loc='lower left')
    fig.tight_layout()
    fig.savefig(fig_dir / 'fig3_pr_curves.png', dpi=160)
    plt.close(fig)

    # 4 final-query ranking by group
    t2 = {k: r['task2'][k] for k in labs}
    groups2 = ['all'] + [g for g in GROUP_ORDER if g in t2[labs[0]]] + ['ogsod/tank', 'other_512/aircraft', 'other_256/ship']
    metrics = [('top_is_best', '最高分 query = 最佳框'), ('best_is_tp50', '最佳框成为 TP@.5'), ('gt_with_2plus_hi_cand', '≥2 个候选 s≥.3 且 IoU≥.5')]
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)
    y = np.arange(len(groups2))
    for a, (m, title) in zip(ax, metrics):
        for j, k in enumerate(labs):
            vals = [t2[k].get(g, {}).get(m) or 0 for g in groups2]
            a.barh(y + (j - .5) * .4, vals, height=.38, color=COLOURS[k], label=k)
        a.set_title(title)
        a.set_xlim(0, 1)
    ax[0].set_yticks(y, groups2)
    ax[0].invert_yaxis()
    ax[0].legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(fig_dir / 'fig4_final_ranking.png', dpi=160)
    plt.close(fig)

    # 5 BP-CB queries
    t3 = {k: r['task3'][k] for k in labs}
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for k in labs:
        hb = np.array(t3[k]['all']['score_hist']['bpcb'], float)
        hm = np.array(t3[k]['all']['score_hist']['matched'], float)
        ax[0].step(centres, np.where(hb > 0, hb, np.nan), where='mid', lw=2, color=COLOURS[k], label=f'{k} BP-CB')
        ax[0].step(centres, np.where(hm > 0, hm, np.nan), where='mid', lw=1.2, ls='--', color=COLOURS[k], label=f'{k} 末层匹配')
    ax[0].set_yscale('log')
    ax[0].set_xlabel('GT 类别分数（末层）')
    ax[0].set_ylabel('query 数（对数）')
    ax[0].legend(fontsize=7)
    ax[0].set_title('框正向 / 分类背景（BP-CB）query 与末层匹配 query 的分数')
    groups3 = ['all'] + [g for g in GROUP_ORDER if g in t3[labs[0]]]
    y = np.arange(len(groups3))
    for j, k in enumerate(labs):
        vals = [(t3[k].get(g, {}).get('duplicates_at03_from_bpcb') or 0) for g in groups3]
        ax[1].barh(y + (j - .5) * .4, vals, height=.38, color=COLOURS[k], label=k)
    ax[1].set_yticks(y, groups3)
    ax[1].invert_yaxis()
    ax[1].set_xlim(0, 1)
    ax[1].legend(loc='lower right')
    ax[1].set_title('分数 ≥ .3 的重复 FP 中 BP-CB query 的占比')
    fig.tight_layout()
    fig.savefig(fig_dir / 'fig5_bpcb.png', dpi=160)
    plt.close(fig)
    print('figures', fig_dir)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('collect')
    c.add_argument('run', type=Path)
    c.add_argument('out', type=Path)
    c.add_argument('--checkpoint', type=Path, required=True)
    c.add_argument('--device', default='cuda:0')
    c.add_argument('--batch', type=int, default=8)
    a = sub.add_parser('analyse')
    a.add_argument('base', type=Path)
    a.add_argument('run', type=Path)
    a.add_argument('--out', type=Path, required=True)
    a.add_argument('--labels', nargs=2, default=['E000', 'E1'])
    a.add_argument('--boot', type=int, default=500)
    p = sub.add_parser('plot')
    p.add_argument('dir', type=Path)
    args = parser.parse_args()
    {'collect': collect, 'analyse': analyse, 'plot': plot}[args.command](args)


if __name__ == '__main__':
    main()
