"""Matcher audit for the OGSOD tank pairs of section 11 (top-ranked query vs a better-localised query of the same GT).

Replays the training assignment on val with the objects built from E000's resolved config:
  cfg.criterion.matcher (HungarianMatcher: 2 x focal class cost (alpha .25, gamma 2) + 5 x L1 (normalised cxcywh) + 2 x GIoU),
  one independent match per stage: encoder Top-K (enc_aux_outputs), decoder layer 1 pre-head (pre_outputs), decoder layers 1..4
  (aux_outputs + main), exactly as DEIMCriterion.forward; and the union set of DEIMCriterion._get_go_indices, which the
  box and local losses use (use_uni_set). MAL uses each stage's own one-to-one indices: target IoU**gamma for matched, 0 otherwise.
Targets are all GT boxes of the image (every class), as in training. The cost matrix is recomputed and its assignment
is checked against the matcher's.

Difference from training (inference has no matching): EMA weights instead of the live weights of each step, no
augmentation (the last four epochs use only flips at 640), no denoising queries (they are masked from the matching
queries in self-attention and assigned separately), val images.

Commands:
  replay RUN OUT --tracking DIR --device cuda:0   DIR = tank_layer_tracking output (layers.npz)
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from error_gallery import ROOT, iou_matrix  # noqa: E402
from tank_feature_diagnosis import INPUT, TANK, load_model, load_val, to_input, xyxy  # noqa: E402
from tank_lqe_ranking import sigmoid  # noqa: E402

STAGES = ['enc', 'pre', 'dec1', 'dec2', 'dec3', 'dec4']


def pairs_from_tracking(tracking, source):
    """Same definitions as tank_layer_tracking.analyse: GT fixed once, candidate set fixed once, final rank by dec4 total."""
    d = np.load(tracking / 'layers.npz')
    image, gt, gt_image = d['image_id'], d['gt'], d['gt_image']
    boxes = np.concatenate([d['enc_box'][:, None], d['box']], 1)
    total = d['raw'][:, -1] + d['lqe'][:, -1]
    N, S = boxes.shape[:2]
    query = np.zeros(N, int)
    iou = np.zeros((N, S))
    owner = np.full(N, -1)
    out = []
    for i in np.unique(image):
        rows, g = np.flatnonzero(image == i), np.flatnonzero(gt_image == i)
        query[rows] = np.arange(len(rows))
        if not len(g):
            continue
        m = np.stack([iou_matrix(boxes[rows, s], gt[g]) for s in range(S)], 1)
        k = m.max(1).argmax(1)
        iou[rows] = m[np.arange(len(rows)), :, k]
        owner[rows] = k  # index among this image's tank GTs (non-degenerate, annotation order)
    rel = iou.max(1) >= .1
    for i in np.unique(image):
        if source[int(i)] != 'ogsod':
            continue
        rows = np.flatnonzero((image == i) & rel)
        rank = np.argsort(np.argsort(-total[rows]))
        for k in np.unique(owner[rows]):
            sub = rows[owner[rows] == k]
            t = sub[np.argmin(rank[np.isin(rows, sub)])]
            b = sub[np.argmax(iou[sub, -1])]
            if b != t and iou[b, -1] >= iou[t, -1] + .05:
                out.append(dict(image_id=int(i), tank_gt=int(k), top=int(query[t]), better=int(query[b])))
    return out


def costs(matcher, logits, boxes, labels, gt_boxes):
    """Recompute the matcher's cost terms (weighted), for every query against every target."""
    import torch
    from engine.deim.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
    prob = torch.sigmoid(logits)[:, labels]
    neg = (1 - matcher.alpha) * (prob ** matcher.gamma) * (-(1 - prob + 1e-8).log())
    pos = matcher.alpha * ((1 - prob) ** matcher.gamma) * (-(prob + 1e-8).log())
    c_class = matcher.cost_class * (pos - neg)
    c_l1 = matcher.cost_bbox * torch.cdist(boxes, gt_boxes, p=1)
    c_giou = matcher.cost_giou * -generalized_box_iou(box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(gt_boxes))
    return c_class, c_l1, c_giou


def replay(args):
    import torch
    from engine.core import YAMLConfig
    from scipy.optimize import linear_sum_assignment
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    annotations, source, grouped = load_val(out)
    images = {i['id']: i for i in annotations['images']}
    pairs = pairs_from_tracking(args.tracking, source)
    by_image = defaultdict(list)
    for p in pairs:
        by_image[p['image_id']].append(p)
    cfg = YAMLConfig(str(args.run / 'resolved_config.yml'))
    criterion = cfg.criterion
    matcher = criterion.matcher
    model, _ = load_model(args.run, args.device)
    decoder = model.decoder
    inner = decoder.decoder
    default_idx = inner.eval_idx
    saved = {}
    decoder.enc_score_head.register_forward_hook(lambda _, __, o: saved.__setitem__('enc_logits', o.detach()))
    decoder.enc_bbox_head.register_forward_hook(lambda _, __, o: saved.__setitem__('enc_box_delta', o.detach()))
    inner.register_forward_hook(lambda _, __, o: saved.__setitem__('pre', (o[4].detach(), o[5].detach())))
    original = decoder._select_topk

    def select(memory, logits, anchors, topk):
        result = original(memory, logits, anchors, topk)
        saved['topk_ind'] = torch.topk(logits.max(-1).values, topk, dim=-1)[1].detach()
        saved['topk_anchors'] = result[2].detach()
        return result

    decoder._select_topk = select
    gamma = criterion.gamma
    rows, checks = [], Counter()
    with torch.no_grad():
        for image_id, plist in by_image.items():
            image = images[image_id]
            x, (w, h) = to_input(ROOT / 'JPEGImages/val' / image['file_name'])
            x = x[None].to(args.device)
            anns = grouped[image_id]
            labels = torch.tensor([a['category_id'] for a in anns], device=args.device)
            xy = np.array([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]] for a in anns], np.float32) / np.array([w, h, w, h], np.float32)
            cxcywh = np.stack([(xy[:, 0] + xy[:, 2]) / 2, (xy[:, 1] + xy[:, 3]) / 2, xy[:, 2] - xy[:, 0], xy[:, 3] - xy[:, 1]], 1)
            targets = [dict(labels=labels, boxes=torch.tensor(cxcywh, device=args.device))]
            tank_rows = [n for n, a in enumerate(anns) if a['category_id'] == TANK and a['bbox'][2] > 0 and a['bbox'][3] > 0]
            stage_out = {}
            for k in range(inner.num_layers):
                inner.eval_idx = k
                result = model(x)
                stage_out[f'dec{k + 1}'] = (result['pred_logits'][0], result['pred_boxes'][0])
                if k == 0:
                    ind = saved['topk_ind'][0]
                    stage_out['enc'] = (saved['enc_logits'][0][ind], torch.sigmoid(saved['enc_box_delta'] + saved['topk_anchors'])[0])
                    pre_boxes, pre_logits = saved['pre']
                    stage_out['pre'] = (pre_logits[0], pre_boxes[0])
            inner.eval_idx = default_idx
            indices, cost = {}, {}
            for s in STAGES:
                logits, boxes = stage_out[s]
                idx = matcher({'pred_logits': logits[None], 'pred_boxes': boxes[None]}, targets)['indices'][0]
                indices[s] = idx
                cc, cl, cg = costs(matcher, logits.float(), boxes.float(), labels, targets[0]['boxes'].float())
                C = (cc + cl + cg).cpu().numpy()
                r_, c_ = linear_sum_assignment(np.nan_to_num(C, nan=1.0))
                checks['assignments'] += 1
                checks['identical_to_matcher'] += int(sorted(zip(r_.tolist(), c_.tolist())) == sorted(zip(idx[0].tolist(), idx[1].tolist())))
                cost[s] = (cc.cpu().numpy(), cl.cpu().numpy(), cg.cpu().numpy())
            # Union set, in the order DEIMCriterion.forward builds it: main = dec4; aux = dec1..dec3, then pre; then encoder.
            go = criterion._get_go_indices([indices['dec4']], [[indices[s]] for s in ('dec1', 'dec2', 'dec3', 'pre', 'enc')])[0]
            go_pairs = set(zip(go[0].tolist(), go[1].tolist()))
            for p in plist:
                t = tank_rows[p['tank_gt']]
                row = dict(p, gt_annotation=anns[t]['id'])
                for s in STAGES:
                    q2t = dict(zip(indices[s][0].tolist(), indices[s][1].tolist()))
                    t2q = {v: k for k, v in q2t.items()}
                    matched = t2q.get(t)
                    logits, boxes = stage_out[s]
                    bx = xyxy(boxes.cpu().numpy(), 1)  # normalised, like gt_box
                    gt_box = np.array(xy[t]) * 1.0
                    iou = iou_matrix(bx[[p['top'], p['better']]], gt_box[None])[:, 0]
                    cc, cl, cg = cost[s]
                    row[s] = dict(
                        matched='top' if matched == p['top'] else 'better' if matched == p['better'] else ('none' if matched is None else 'other'),
                        matched_query=matched,
                        iou_top=float(iou[0]), iou_better=float(iou[1]),
                        tank_prob_top=float(sigmoid(logits[p['top'], TANK].item())), tank_prob_better=float(sigmoid(logits[p['better'], TANK].item())),
                        cost_class_top=float(cc[p['top'], t]), cost_class_better=float(cc[p['better'], t]),
                        cost_l1_top=float(cl[p['top'], t]), cost_l1_better=float(cl[p['better'], t]),
                        cost_giou_top=float(cg[p['top'], t]), cost_giou_better=float(cg[p['better'], t]),
                        top_matched_elsewhere=q2t.get(p['top']) is not None and q2t.get(p['top']) != t,
                        better_matched_elsewhere=q2t.get(p['better']) is not None and q2t.get(p['better']) != t,
                        # MAL target for the tank channel at this stage (decoder stages and encoder; pre also uses MAL).
                        mal_target_top=float(iou[0] ** gamma) if matched == p['top'] else 0.,
                        mal_target_better=float(iou[1] ** gamma) if matched == p['better'] else 0.)
                row['union_set'] = dict(top_in_union=(p['top'], t) in go_pairs, better_in_union=(p['better'], t) in go_pairs,
                                        queries_for_this_gt=sorted(q for q, g in go_pairs if g == t))
                rows.append(row)
    json.dump(rows, open(out / 'pairs.json', 'w'))
    report = summarise(rows, checks, len(pairs))
    (out / 'report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False))


def summarise(rows, checks, n_pairs):
    med = lambda v: round(float(np.median(v)), 3)
    report = dict(pairs=n_pairs, replayed=len(rows), cost_replication=dict(checks))
    per = {}
    for s in STAGES:
        R = [r[s] for r in rows]
        who = Counter(x['matched'] for x in R)
        diff = lambda k: [x[f'{k}_better'] - x[f'{k}_top'] for x in R]  # positive: the better query costs more
        per[s] = dict(matched_share={k: round(who[k] / len(R), 3) for k in ('top', 'better', 'other', 'none')},
                      median_cost_better_minus_top=dict(cls=med(diff('cost_class')), l1=med(diff('cost_l1')), giou=med(diff('cost_giou')),
                                                        total=med([a + b + c for a, b, c in zip(diff('cost_class'), diff('cost_l1'), diff('cost_giou'))])),
                      better_has_lower_box_cost=round(float(np.mean([(x['cost_l1_better'] + x['cost_giou_better']) < (x['cost_l1_top'] + x['cost_giou_top']) for x in R])), 3),
                      better_has_higher_class_cost=round(float(np.mean([x['cost_class_better'] > x['cost_class_top'] for x in R])), 3),
                      better_box_wins_but_class_loses=round(float(np.mean([((x['cost_l1_better'] + x['cost_giou_better']) < (x['cost_l1_top'] + x['cost_giou_top'])) and x['matched'] != 'better' for x in R])), 3),
                      median_iou_top=med([x['iou_top'] for x in R]), median_iou_better=med([x['iou_better'] for x in R]),
                      median_prob_top=med([x['tank_prob_top'] for x in R]), median_prob_better=med([x['tank_prob_better'] for x in R]),
                      better_mal_target_zero=round(float(np.mean([x['mal_target_better'] == 0 for x in R])), 3),
                      better_matched_to_another_gt=round(float(np.mean([x['better_matched_elsewhere'] for x in R])), 3))
    report['per_stage'] = per
    seq = Counter(tuple(r[s]['matched'] for s in STAGES) for r in rows)
    report['matched_sequence_top10 (enc, pre, dec1..dec4)'] = [[list(k), v] for k, v in seq.most_common(10)]
    report['same_query_matched_all_decoder_layers'] = round(float(np.mean([len({r[s]['matched_query'] for s in ('dec1', 'dec2', 'dec3', 'dec4')}) == 1 for r in rows])), 3)
    report['union_set'] = dict(top_in_union=round(float(np.mean([r['union_set']['top_in_union'] for r in rows])), 3),
                               better_in_union=round(float(np.mean([r['union_set']['better_in_union'] for r in rows])), 3),
                               median_queries_per_gt_in_union=med([len(r['union_set']['queries_for_this_gt']) for r in rows]))
    report['better_never_matched_any_stage'] = round(float(np.mean([all(r[s]['matched'] != 'better' for s in STAGES) for r in rows])), 3)
    report['better_mal_zero_at_all_decoder_layers'] = round(float(np.mean([all(r[s]['mal_target_better'] == 0 for s in ('dec1', 'dec2', 'dec3', 'dec4')) for r in rows])), 3)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('replay',))
    parser.add_argument('run', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--tracking', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    replay(args)


if __name__ == '__main__':
    main()
