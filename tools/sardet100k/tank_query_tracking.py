"""Track each of the 300 queries from the encoder Top-K to the last decoder layer, and measure duplicates. Val only, E000 unchanged.

Identity: at inference there are no denoising queries, so decoder query j is built from encoder Top-K position j
(content = enc_topk_memory[j], reference = enc_bbox_head(enc_topk_memory[j]) + anchor). Indices are recomputed with the
same call as DFINETransformer._select_topk on the same tensor, and the encoder box is read from the enc_bbox_head hook.

Commands:
  collect RUN OUT --device cuda:0   tank images of OGSOD and MSAR val; per-query arrays
  analyse RUN OUT --lqe-dir DIR     tracking statistics, plus duplicate analysis on DIR/candidates.npz (tank_lqe_ranking collect)
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from error_gallery import ROOT, iou_matrix  # noqa: E402
from tank_feature_diagnosis import INPUT, SOURCES, TANK, load_model, load_val, to_input, xyxy  # noqa: E402
from tank_lqe_ranking import MAX_DETS, evaluate, sigmoid, spearman  # noqa: E402

BOOT = 1000


def collect(args):
    import torch
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    annotations, source, grouped = load_val(out)
    images = [i for i in annotations['images'] if source[i['id']] in SOURCES and any(a['category_id'] == TANK for a in grouped[i['id']])]
    model, _ = load_model(args.run, args.device)
    decoder = model.decoder
    saved = {}
    decoder.enc_score_head.register_forward_hook(lambda _, __, o: saved.__setitem__('enc_logits', o.detach()))
    decoder.enc_bbox_head.register_forward_hook(lambda _, __, o: saved.__setitem__('enc_box_delta', o.detach()))
    lqe = decoder.decoder.lqe_layers[decoder.decoder.eval_idx]
    lqe.reg_conf.register_forward_hook(lambda _, __, o: saved.__setitem__('quality', o.detach()))
    original = decoder._select_topk

    def select(memory, logits, anchors, topk):
        result = original(memory, logits, anchors, topk)
        saved['topk_ind'] = torch.topk(logits.max(-1).values, topk, dim=-1)[1].detach()  # same call as the 'default' branch
        saved['topk_anchors'] = result[2].detach()
        return result

    assert decoder.query_select_method == 'default'
    decoder._select_topk = select
    store = defaultdict(list)
    with torch.no_grad():
        for image in images:
            x, (w, h) = to_input(ROOT / 'JPEGImages/val' / image['file_name'])
            result = model(x[None].to(args.device))
            ind = saved['topk_ind'][0]
            enc_logits = saved['enc_logits'][0][ind]
            enc_box = torch.sigmoid(saved['enc_box_delta'] + saved['topk_anchors'])[0]
            quality = saved['quality'][0, :, 0]
            logits = result['pred_logits'][0]
            n = len(ind)
            store['image_id'].append(np.full(n, image['id']))
            store['query'].append(np.arange(n))
            store['enc_tank_logit'].append(enc_logits[:, TANK].cpu().numpy())
            store['enc_max_logit'].append(enc_logits.max(-1).values.cpu().numpy())
            store['enc_box'].append(xyxy(enc_box.cpu().numpy()))
            store['dec_raw_tank'].append((logits[:, TANK] - quality).cpu().numpy())
            store['lqe'].append(quality.cpu().numpy())
            store['dec_final_tank'].append(logits[:, TANK].cpu().numpy())
            store['dec_box'].append(xyxy(result['pred_boxes'][0].cpu().numpy()))
            tanks = [a for a in grouped[image['id']] if a['category_id'] == TANK]
            gt = np.array([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]] for a in tanks], np.float32) * np.array([INPUT / w, INPUT / h] * 2, np.float32)
            store['gt'].append(gt)
            store['gt_image'].append(np.full(len(gt), image['id']))
    np.savez_compressed(out / 'tracking.npz', **{k: np.concatenate(v) for k, v in store.items()})
    print('saved', len(images), 'images', flush=True)


def boot_ci(stat, groups, seed=0):
    """stat(index array) -> float; images resampled with replacement."""
    rng = np.random.default_rng(seed)
    keys = np.unique(groups)
    members = {k: np.flatnonzero(groups == k) for k in keys}
    values = []
    for _ in range(BOOT):
        take = np.concatenate([members[k] for k in rng.choice(keys, len(keys))])
        values.append(stat(take))
    return [round(float(np.nanpercentile(values, 2.5)), 3), round(float(np.nanpercentile(values, 97.5)), 3)]


def tracking(out, source):
    d = np.load(out / 'tracking.npz')
    image, gt, gt_image = d['image_id'], d['gt'], d['gt_image']
    enc_iou, dec_iou, owner = np.zeros(len(image)), np.zeros(len(image)), np.full(len(image), -1)
    gt_index = {}
    for i in np.unique(image):
        rows, g = np.flatnonzero(image == i), np.flatnonzero(gt_image == i)
        me, md = iou_matrix(d['enc_box'][rows], gt[g]), iou_matrix(d['dec_box'][rows], gt[g])
        # One GT per query for both stages: the GT it overlaps most at either stage.
        k = np.maximum(me, md).argmax(1)
        enc_iou[rows], dec_iou[rows], owner[rows] = me[np.arange(len(rows)), k], md[np.arange(len(rows)), k], g[k]
        gt_index[int(i)] = g
    enc_s, raw, lqe, final = sigmoid(d['enc_tank_logit']), d['dec_raw_tank'], d['lqe'], d['dec_final_tank']
    dec_s = sigmoid(final)
    # Decoder rank: final tank score within the image; encoder rank: Top-K position (max-class logit).
    dec_rank = np.zeros(len(image), int)
    for i in np.unique(image):
        rows = np.flatnonzero(image == i)
        dec_rank[rows[np.argsort(-final[rows])]] = np.arange(len(rows))
    report = {}
    for src in SOURCES:
        sel = np.array([source[int(i)] == src for i in image])
        rel = sel & (np.maximum(enc_iou, dec_iou) >= .1)
        r = dict(images=len(np.unique(image[sel])), queries=int(sel.sum()), queries_overlapping_tank_gt=int(rel.sum()))
        d_iou, d_logit, d_prob = dec_iou - enc_iou, final - d['enc_tank_logit'], dec_s - enc_s

        def classes(mask, dead_iou=0., dead_logit=0.):
            up_i, down_i = d_iou > dead_iou, d_iou < -dead_iou
            up_s, down_s = d_logit > dead_logit, d_logit < -dead_logit
            groups = dict(A=up_i & up_s, B=up_i & down_s, C=down_i & up_s, D=down_i & down_s)
            total = mask.sum()
            res = {k: dict(n=int((v & mask).sum()), share=round(float((v & mask).sum() / max(total, 1)), 3),
                           enc_iou=round(float(np.median(enc_iou[v & mask])), 3) if (v & mask).any() else None,
                           dec_iou=round(float(np.median(dec_iou[v & mask])), 3) if (v & mask).any() else None,
                           enc_score=round(float(np.median(enc_s[v & mask])), 3) if (v & mask).any() else None,
                           dec_score=round(float(np.median(dec_s[v & mask])), 3) if (v & mask).any() else None) for k, v in groups.items()}
            res['unchanged_or_mixed'] = int(total - sum(x['n'] for x in res.values()))
            return res

        r['classes_all_overlapping'] = classes(rel)
        r['classes_dead_zone_iou_0.02_logit_0.1'] = classes(rel, .02, .1)
        # Decisive queries: the top-ranked decoder query of each GT (the one that becomes the detection).
        top = np.zeros(len(image), bool)
        for g in np.unique(owner[rel]):
            rows = np.flatnonzero(rel & (owner == g))
            top[rows[np.argmin(dec_rank[rows])]] = True
        r['classes_top_query_per_gt'] = classes(top)
        strong_c = rel & (d_iou <= -.1) & (d_prob >= .1)
        strong_b = rel & (d_iou >= .1) & (d_prob <= -.1)
        r['strong_cases'] = {'C: IoU down >= 0.1 and score up >= 0.1': int(strong_c.sum()), 'B: IoU up >= 0.1 and score down >= 0.1': int(strong_b.sum()),
                             'C that are the top query of their GT': int((strong_c & top).sum()), 'B that are the top query of their GT': int((strong_b & top).sum())}
        # Harm: the GT's top-ranked decoder query is not its best decoder box.
        harm = []
        for g in np.unique(owner[rel]):
            rows = np.flatnonzero(rel & (owner == g))
            t = rows[np.argmin(dec_rank[rows])]
            harm.append((dec_iou[rows].max() - dec_iou[t], dec_iou[t] < .75 <= dec_iou[rows].max()))
        harm = np.array(harm)
        r['gt_top_query_vs_best'] = dict(gts=len(harm), median_iou_lost=round(float(np.median(harm[:, 0])), 3),
                                         share_top_below_075_while_best_above=round(float(harm[:, 1].mean()), 3))
        # Correlations on the same query sets, image bootstrap.
        corr = {}
        for label, mask in (('all_queries', sel), ('overlapping_tank_gt', rel)):
            idx = np.flatnonzero(mask)
            groups = image[idx]
            pairs = {'enc logit vs enc IoU': (d['enc_tank_logit'], enc_iou), 'dec cls logit vs dec IoU': (raw, dec_iou), 'LQE vs dec IoU': (lqe, dec_iou),
                     'dec cls + LQE vs dec IoU': (final, dec_iou), 'delta logit vs delta IoU': (d_logit, d_iou), 'enc rank vs dec rank': (-d['query'].astype(float), -dec_rank.astype(float))}
            corr[label] = {k: dict(spearman=round(spearman(a[idx], b[idx]), 3), ci95=boot_ci(lambda t, a=a, b=b: spearman(a[idx][t], b[idx][t]), groups)) for k, (a, b) in pairs.items()}
        r['spearman'] = corr
        r['medians_overlapping'] = dict(enc_iou=round(float(np.median(enc_iou[rel])), 3), dec_iou=round(float(np.median(dec_iou[rel])), 3),
                                        delta_iou=round(float(np.median(d_iou[rel])), 3), enc_score=round(float(np.median(enc_s[rel])), 3), dec_score=round(float(np.median(dec_s[rel])), 3),
                                        raw_logit=round(float(np.median(raw[rel])), 3), lqe=round(float(np.median(lqe[rel])), 3), final_logit=round(float(np.median(final[rel])), 3))
        report[src] = r
    # Per-query table for inspection.
    rows = [dict(image_id=int(image[q]), query=int(d['query'][q]), enc_tank_logit=round(float(d['enc_tank_logit'][q]), 3), enc_iou=round(float(enc_iou[q]), 3),
                 dec_raw_logit=round(float(raw[q]), 3), lqe=round(float(lqe[q]), 3), dec_logit=round(float(final[q]), 3), dec_iou=round(float(dec_iou[q]), 3),
                 enc_rank=int(d['query'][q]), dec_rank=int(dec_rank[q]), gt=int(owner[q])) for q in np.flatnonzero(np.maximum(enc_iou, dec_iou) >= .1)]
    json.dump(rows, open(out / 'tracked_queries.json', 'w'))
    return report


def duplicates(out, lqe_dir, annotations, source, grouped):
    data = np.load(lqe_dir / 'candidates.npz')
    image_id, logits, box = data['image_id'], data['final_logits'], data['box']
    final = sigmoid(logits[:, TANK])
    gts = {}
    for image in annotations['images']:
        if source[image['id']] in SOURCES:
            gts[image['id']] = np.array([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]]
                                         for a in grouped[image['id']] if a['category_id'] == TANK and a['bbox'][2] > 0 and a['bbox'][3] > 0], np.float32).reshape(-1, 4)
    by_image = defaultdict(list)
    for n, i in enumerate(image_id):
        by_image[int(i)].append(n)
    from scipy.optimize import linear_sum_assignment
    report = {}
    for src in SOURCES:
        imgs = [i for i in by_image if source[i] == src]
        stats = defaultdict(list)
        variants = defaultdict(dict)
        breakdown = {name: Counter() for name in ('current', 'current_score_ge_0.3')}
        for i in imgs:
            rows = np.array(by_image[i])
            g = gts[i]
            m = iou_matrix(box[rows], g) if len(g) else np.zeros((len(rows), 0))
            best = m.max(1) if len(g) else np.zeros(len(rows))
            owner = m.argmax(1) if len(g) else np.full(len(rows), -1)
            order = np.argsort(-final[rows], kind='mergesort')
            top = order[:MAX_DETS]
            if len(g):
                stats['candidates_iou75_per_gt'] += (m >= .75).sum(0).tolist()
                stats['candidates_iou50_per_gt'] += (m >= .5).sum(0).tolist()
                stats['top100_cover_iou50_per_gt'] += (m[top] >= .5).sum(0).tolist()
                stats['image_unique_cover_iou50'].append(float((m[top] >= .5).any(0).mean()))
                stats['image_unique_cover_iou75'].append(float((m[top] >= .75).any(0).mean()))
            # COCO-style greedy at IoU .5 over the current top 100: TP, duplicate, poor localisation, background.
            for name, keep in (('current', top), ('current_score_ge_0.3', top[final[rows][top] >= .3])):
                used = np.zeros(len(g), bool)
                for q in keep:
                    if not len(g) or best[q] < .1:
                        breakdown[name]['background'] += 1
                        continue
                    cand = np.where(used, -1, m[q])
                    k = int(cand.argmax())
                    if cand[k] >= .5:
                        used[k] = True
                        breakdown[name]['TP'] += 1
                    elif (m[q] >= .5).any():
                        breakdown[name]['duplicate'] += 1
                    else:
                        breakdown[name]['localisation'] += 1
            # Ranking variants.
            s_cur = final[rows]
            s_iou = best + 1e-3 * s_cur
            overlap = best >= .1
            # One-to-one GT matching on IoU (Hungarian), pairs with IoU >= .1.
            matched = np.zeros(len(rows), bool)
            if len(g):
                r_, c_ = linear_sum_assignment(-m)
                ok = m[r_, c_] >= .1
                matched[r_[ok]] = True
            # Dedup with the current choice: per GT keep the current top-scoring overlapping query.
            cur_pick = np.zeros(len(rows), bool)
            for k in range(len(g)):
                cand = np.flatnonzero(overlap & (owner == k))
                if len(cand):
                    cur_pick[cand[np.argmax(s_cur[cand])]] = True
            options = {
                'current': (s_cur, np.ones(len(rows), bool)),
                'IoU oracle (previous)': (s_iou, np.ones(len(rows), bool)),
                'dedup only: per GT keep current top-scoring query, drop other overlapping': (s_cur, cur_pick | ~overlap),
                'dedup + best box: per GT keep best-IoU query (one-to-one), current scores, background kept': (s_cur, matched | ~overlap),
                'GT-aware duplicate-suppressed oracle: one-to-one best boxes ranked by IoU, everything else dropped': (best, matched),
                'same oracle, background kept below at current scores': (np.where(matched, 1 + best, s_cur * 1e-3), matched | ~overlap),
            }
            for name, (score, keep) in options.items():
                rr = np.flatnonzero(keep)
                o = rr[np.argsort(-score[rr], kind='mergesort')][:MAX_DETS]
                variants[name][i] = (score[o], box[rows][o])
        r = {}
        for k, v in stats.items():
            v = np.array(v, float)
            r[k] = dict(mean=round(float(v.mean()), 3), median=round(float(np.median(v)), 3), share_ge_2=round(float((v >= 2).mean()), 3), share_0=round(float((v == 0).mean()), 3))
        r['top100_breakdown_iou50'] = {k: dict(v, tp_to_duplicate=round(v['TP'] / max(v['duplicate'], 1), 2)) for k, v in breakdown.items()}
        gt_src = {i: gts[i] for i in imgs}
        r['ap'] = {name: evaluate(dets, gt_src) for name, dets in variants.items()}
        report[src] = r
    return report


def analyse(args):
    annotations, source, grouped = load_val(args.out)
    report = dict(tracking=tracking(args.out, source), duplicates=duplicates(args.out, args.lqe_dir, annotations, source, grouped))
    (args.out / 'report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('collect', 'analyse'))
    parser.add_argument('run', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--lqe-dir', type=Path)
    args = parser.parse_args()
    {'collect': collect, 'analyse': analyse}[args.command](args)


if __name__ == '__main__':
    main()
