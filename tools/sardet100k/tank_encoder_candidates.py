"""Encoder candidates for every val tank GT (OGSOD and MSAR): box quality against classification rank. Val only, weights unchanged.

For each GT, over all encoder cells (anchors), without looking at scores:
  best-IoU candidate: IoU, tank probability, max-class probability, global rank (the order Top-300 uses), tank-only rank, selected or not;
  among candidates with IoU >= 0.5: how many, the best rank, whether any was selected;
  final status from the existing funnel rows (score 0.3, IoU 0.5).
Also: high-score candidates that sit on background, and rank-quality correlations.

Usage (repository root on the server):
  python tools/sardet100k/tank_encoder_candidates.py RUN OUT --device cuda:0
OUT must sit under RUN/independent_verify_20261106/ next to tank_feature_diagnosis/ (gt_rows.json) and error_gallery_val/.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from error_gallery import ROOT, iou_matrix  # noqa: E402
from tank_feature_diagnosis import INPUT, SOURCES, TANK, image_targets, load_model, load_val, to_input, xyxy  # noqa: E402

GOOD = .5
NAMES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor')


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float('nan')
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def native_stats(image, box):
    """8-bit grey statistics of the GT and a ring of width max(side, 8) after a guard of side/4, original pixels."""
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1, 1.)
    guard, width = side / 4, max(side, 8.)
    h, w = image.shape
    yy, xx = np.mgrid[0:h, 0:w] + .5
    inside = (xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2)
    outer = (xx >= x1 - guard - width) & (xx < x2 + guard + width) & (yy >= y1 - guard - width) & (yy < y2 + guard + width)
    near = (xx >= x1 - guard) & (xx < x2 + guard) & (yy >= y1 - guard) & (yy < y2 + guard)
    ring = outer & ~near
    if not inside.any() or not ring.any():
        return None
    g, r = image[inside], image[ring]
    return dict(gt_mean=float(g.mean()), ring_mean=float(r.mean()), ring_std=float(r.std()),
                scr_db=float(10 * np.log10(max(g.mean(), 1e-3) / max(r.mean(), 1e-3))), contrast_z=float((g.mean() - r.mean()) / max(r.std(), 1e-3)),
                gt_sat=float((g >= 250).mean()), ring_sat=float((r >= 250).mean()), image_mean=float(image.mean()))


def run(args):
    import torch
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    annotations, source, grouped = load_val(out)
    funnel = {r['gt_id']: r for r in json.load(open(out.parent / 'tank_feature_diagnosis/gt_rows.json'))}
    images = [i for i in annotations['images'] if source[i['id']] in SOURCES and any(a['category_id'] == TANK for a in grouped[i['id']])]
    model, _ = load_model(args.run, args.device)
    decoder = model.decoder
    saved = {}
    decoder.enc_score_head.register_forward_hook(lambda _, __, o: saved.__setitem__('logits', o.detach()))
    decoder.enc_output.register_forward_hook(lambda _, __, o: saved.__setitem__('memory', o.detach()))
    rows, cand = [], []
    with torch.no_grad():
        for count, image in enumerate(images):
            x, (w, h) = to_input(ROOT / 'JPEGImages/val' / image['file_name'])
            model(x[None].to(args.device))
            logits, memory = saved['logits'][0], saved['memory']
            boxes = torch.sigmoid(decoder.enc_bbox_head(memory) + decoder.anchors)[0]
            valid = decoder.valid_mask[0, :, 0]
            prob = torch.sigmoid(logits)
            # Same order as the Top-K selection: max over classes of the encoder logits.
            best = logits.max(-1).values.masked_fill(~valid, -1e9)
            order = torch.argsort(best, descending=True)
            rank = torch.empty_like(order)
            rank[order] = torch.arange(len(order), device=order.device)
            tank_order = torch.argsort(logits[:, TANK].masked_fill(~valid, -1e9), descending=True)
            tank_rank = torch.empty_like(tank_order)
            tank_rank[tank_order] = torch.arange(len(tank_order), device=order.device)
            order = order.cpu().numpy()
            prob, boxes, rank, tank_rank, valid = prob.cpu().numpy(), xyxy(boxes.cpu().numpy()), rank.cpu().numpy(), tank_rank.cpu().numpy(), valid.cpu().numpy()
            k = decoder.num_queries
            counts = [(INPUT // s) ** 2 for s in (8, 16, 32)]
            assert sum(counts) == len(valid), (counts, len(valid))
            level = np.repeat([8, 16, 32], counts)
            g_cls, g_native = image_targets(grouped[image['id']])
            sx, sy = INPUT / w, INPUT / h
            g_input = g_native * np.array([sx, sy, sx, sy], np.float32)
            iou_all = iou_matrix(boxes, g_input)
            iou_all[~valid] = 0
            grey = np.asarray(Image.open(ROOT / 'JPEGImages/val' / image['file_name']).convert('L'), np.float32)
            tank_gt = np.flatnonzero(g_cls == TANK)
            # Candidates near tank GTs (for correlations) and high-score candidates (for background).
            best_tank_iou = iou_all[:, tank_gt].max(1) if len(tank_gt) else np.zeros(len(valid))
            best_any_iou = iou_all.max(1) if len(g_cls) else np.zeros(len(valid))
            top = np.flatnonzero((rank < k) & valid)
            for c in top:
                cand.append(dict(image_id=image['id'], source=source[image['id']], rank=int(rank[c]), level=int(level[c]), top_class=int(prob[c].argmax()),
                                 tank_prob=float(prob[c, TANK]), max_prob=float(prob[c].max()), iou_tank=float(best_tank_iou[c]), iou_any=float(best_any_iou[c])))
            for g in tank_gt:
                ious = iou_all[:, g]
                b = int(ious.argmax())
                good = np.flatnonzero(ious >= GOOD)
                near = np.flatnonzero(ious >= .1)
                f = funnel.get(grouped[image['id']][g]['id'], {})
                stats = native_stats(grey, g_native[g]) or {}
                row = dict(gt_id=grouped[image['id']][g]['id'], image_id=image['id'], file_name=image['file_name'], source=source[image['id']],
                           native_side=float(max(g_native[g][2] - g_native[g][0], g_native[g][3] - g_native[g][1])), native_area=float((g_native[g][2] - g_native[g][0]) * (g_native[g][3] - g_native[g][1])),
                           nn_input=f.get('nn_input'), n_gt=int(len(g_cls)), status=f.get('status'),
                           best_iou=float(ious[b]), best_tank_prob=float(prob[b, TANK]), best_max_prob=float(prob[b].max()), best_top_class=NAMES[int(prob[b].argmax())],
                           best_rank=int(rank[b]), best_tank_rank=int(tank_rank[b]), best_selected=bool(rank[b] < k), best_level=int(level[b]),
                           n_good=int(len(good)), good_best_rank=int(rank[good].min()) if len(good) else None, good_selected=bool((rank[good] < k).any()) if len(good) else False,
                           good_best_tank_prob=float(prob[good, TANK].max()) if len(good) else None,
                           near_spearman=spearman(prob[near, TANK], ious[near]) if len(near) >= 5 else None,
                           near_top_score_is_top_iou=bool(near[np.argmax(prob[near, TANK])] == near[np.argmax(ious[near])]) if len(near) else None,
                           near_iou_of_top_score=float(ious[near[np.argmax(prob[near, TANK])]]) if len(near) else None,
                           cutoff_max_prob=float(prob[order[k - 1].item()].max()), **stats)
                rows.append(row)
            if (count + 1) % 20 == 0:
                print('images', count + 1, len(images), flush=True)
    json.dump(rows, open(out / 'gt_candidates.json', 'w'))
    json.dump(cand, open(out / 'top300_candidates.json', 'w'))
    print('saved', len(rows), len(cand), flush=True)


def med(values):
    v = np.array([x for x in values if x is not None and np.isfinite(x)], float)
    return round(float(np.median(v)), 3) if len(v) else None


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return None
    allv = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(allv)) + 1
    return round(float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))), 3)


def report(args):
    out = args.out
    rows = [r for r in json.load(open(out / 'gt_candidates.json')) if r['native_side'] > 0]
    cand = json.load(open(out / 'top300_candidates.json'))
    result = {}
    for src in SOURCES:
        R = [r for r in rows if r['source'] == src]
        groups = defaultdict(list)
        for r in R:
            hi_iou, hi = r['best_iou'] >= GOOD, r['best_selected']
            groups[{(True, True): 'A', (False, True): 'B', (True, False): 'C', (False, False): 'D'}[(hi_iou, hi)]].append(r)
        table = {}
        for name in 'ABCD':
            G = groups[name]
            table[name] = dict(gt=len(G), images=len({r['image_id'] for r in G}), best_iou=med([r['best_iou'] for r in G]), tank_prob=med([r['best_tank_prob'] for r in G]),
                               rank=med([r['best_rank'] for r in G]), detected=round(float(np.mean([r['status'] == 'TP' for r in G])), 3) if G else None)
        missed_good = [r for r in R if r['n_good'] and not r['good_selected']]
        has_good = [r for r in R if r['n_good']]
        q1 = dict(gt=len(R), has_iou50_candidate=len(has_good), has_but_none_selected=len(missed_good), images=len({r['image_id'] for r in missed_good}),
                  their_status=dict(Counter(r['status'] for r in missed_good)), best_rank_median=med([r['good_best_rank'] for r in missed_good]),
                  best_rank_quartiles=np.percentile([r['good_best_rank'] for r in missed_good], [25, 75]).round().tolist() if missed_good else None,
                  within_rank_600=sum(r['good_best_rank'] < 600 for r in missed_good), within_rank_1000=sum(r['good_best_rank'] < 1000 for r in missed_good),
                  best_tank_prob_median=med([r['good_best_tank_prob'] for r in missed_good]), no_iou50_candidate=len(R) - len(has_good),
                  no_iou50_status=dict(Counter(r['status'] for r in R if not r['n_good'])))
        # Q2: missed good candidates against selected good candidates (A-like), same source.
        sel_good = [r for r in R if r['n_good'] and r['good_selected']]
        keys = ('native_side', 'native_area', 'nn_input', 'n_gt', 'gt_mean', 'ring_mean', 'ring_std', 'scr_db', 'contrast_z', 'gt_sat', 'ring_sat', 'image_mean')
        q2 = {k: dict(not_selected=med([r.get(k) for r in missed_good]), selected=med([r.get(k) for r in sel_good]),
                      auc_not_selected_higher=auc([r[k] for r in missed_good if r.get(k) is not None], [r[k] for r in sel_good if r.get(k) is not None])) for k in keys}
        q2['top_class_of_best_candidate_not_selected'] = dict(Counter(r['best_top_class'] for r in missed_good))
        image_share = Counter(r['image_id'] for r in missed_good).most_common(8)
        q2['concentration_top8_images'] = dict(images=[i for i, _ in image_share], share=round(sum(c for _, c in image_share) / max(len(missed_good), 1), 3))
        # Q3: Top-300 candidates predicting tank, by IoU to tank GTs.
        C = [c for c in cand if c['source'] == src and c['top_class'] == TANK]
        q3 = {}
        for label, thr in (('tank_top300_all', 0.), ('tank_prob_ge_0.3', .3), ('tank_prob_ge_0.5', .5)):
            S = [c for c in C if c['tank_prob'] >= thr]
            q3[label] = dict(n=len(S), iou_lt_0_1=round(float(np.mean([c['iou_tank'] < .1 for c in S])), 3) if S else None,
                             iou_lt_0_1_any_gt=round(float(np.mean([c['iou_any'] < .1 for c in S])), 3) if S else None,
                             iou_0_1_to_0_5=round(float(np.mean([.1 <= c['iou_tank'] < .5 for c in S])), 3) if S else None,
                             iou_ge_0_5=round(float(np.mean([c['iou_tank'] >= .5 for c in S])), 3) if S else None,
                             iou_ge_0_75=round(float(np.mean([c['iou_tank'] >= .75 for c in S])), 3) if S else None)
        for lvl in (8, 16, 32):
            S = [c for c in C if c['tank_prob'] >= .3 and c['level'] == lvl]
            q3[f'tank_prob_ge_0.3_L{lvl}'] = dict(n=len(S), iou_lt_0_1=round(float(np.mean([c['iou_tank'] < .1 for c in S])), 3) if S else None)
        bins = [0, .05, .1, .2, .3, .5, 1.01]
        q3['by_tank_prob'] = [dict(range=f'{a}-{b}', n=sum(a <= c['tank_prob'] < b for c in C),
                                   iou_lt_0_1=round(float(np.mean([c['iou_tank'] < .1 for c in C if a <= c['tank_prob'] < b])), 3) if any(a <= c['tank_prob'] < b for c in C) else None,
                                   iou_ge_0_5=round(float(np.mean([c['iou_tank'] >= .5 for c in C if a <= c['tank_prob'] < b])), 3) if any(a <= c['tank_prob'] < b for c in C) else None)
                              for a, b in zip(bins[:-1], bins[1:])]
        # Q4: correlations.
        near = [c for c in C if c['iou_tank'] >= .1]
        q4 = dict(per_gt_best_iou_vs_tank_prob=round(spearman([r['best_iou'] for r in R], [r['best_tank_prob'] for r in R]), 3),
                  per_gt_best_iou_vs_rank=round(spearman([r['best_iou'] for r in R], [-r['best_rank'] for r in R]), 3),
                  within_gt_median_spearman=med([r['near_spearman'] for r in R]),
                  within_gt_top_score_is_top_iou=round(float(np.mean([r['near_top_score_is_top_iou'] for r in R if r['near_top_score_is_top_iou'] is not None])), 3),
                  within_gt_iou_of_top_score_median=med([r['near_iou_of_top_score'] for r in R]),
                  within_gt_best_iou_median=med([r['best_iou'] for r in R]),
                  top300_near_gt_tank_prob_vs_iou=round(spearman([c['tank_prob'] for c in near], [c['iou_tank'] for c in near]), 3), top300_near_gt_n=len(near))
        result[src] = dict(groups=table, q1=q1, q2=q2, q3=q3, q4=q4)
    (out / 'report.json').write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps(result, indent=1, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('run', 'report'))
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.run = args.run_dir
    {'run': run, 'report': report}[args.command](args)


if __name__ == '__main__':
    main()
