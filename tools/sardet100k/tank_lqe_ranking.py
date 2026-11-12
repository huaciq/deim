"""LQE and candidate ranking for tanks on OGSOD and MSAR val (E000, weights unchanged, no training).

Code path (engine/deim/dfine_decoder.py, deim_criterion.py, postprocessor.py):
  encoder Top-300: max over classes of enc_score_head logits; LQE is not involved;
  last decoder layer: logits = score_head(query) + LQE(top-4 probabilities and mean of each edge distribution), one scalar added to all classes;
  LQE has no loss of its own; it is trained only through MAL on the summed logits (target IoU**1.5 for matched queries, 0 otherwise);
  inference uses only the last layer; final score = sigmoid(logits), top 300 over 300 queries x 6 classes.

Commands:
  collect RUN OUT --device cuda:0   one forward per OGSOD/MSAR val image; saves arrays
  analyse RUN OUT                   experiment 1 statistics, experiment 3 re-ranking, experiment 2 figure
OUT must sit under RUN/independent_verify_20261106/ (error_gallery_val/source.json is read from there).
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from error_gallery import ROOT, iou_matrix  # noqa: E402
from tank_feature_diagnosis import INPUT, SOURCES, TANK, load_model, load_val, to_input, xyxy  # noqa: E402

THRESHOLDS = np.linspace(.5, .95, 10)
RECALL = np.linspace(0, 1, 101)
MAX_DETS = 100


def collect(args):
    import torch
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    annotations, source, grouped = load_val(out)
    images = [i for i in annotations['images'] if source[i['id']] in SOURCES]
    model, post = load_model(args.run, args.device)
    decoder = model.decoder
    saved = {}
    decoder.enc_score_head.register_forward_hook(lambda _, __, o: saved.__setitem__('enc_logits', o.detach()))
    decoder.enc_output.register_forward_hook(lambda _, __, o: saved.__setitem__('memory', o.detach()))
    lqe = decoder.decoder.lqe_layers[decoder.decoder.eval_idx]
    lqe.reg_conf.register_forward_hook(lambda _, __, o: saved.__setitem__('quality', o.detach()))
    store = defaultdict(list)
    enc_rows = []
    with torch.no_grad():
        for count, image in enumerate(images):
            x, (w, h) = to_input(ROOT / 'JPEGImages/val' / image['file_name'])
            result = model(x[None].to(args.device))
            logits, boxes = result['pred_logits'][0], result['pred_boxes'][0]
            quality = saved['quality'][0, :, 0]
            # Official post-processing: top 300 over (query, class) pairs.
            official = post(result, torch.tensor([[w, h]], device=args.device, dtype=torch.float32))[0]
            scale = np.array([w, h, w, h], np.float32)
            store['image_id'].append(np.full(len(quality), image['id']))
            store['raw_tank'].append((logits[:, TANK] - quality).cpu().numpy())
            store['lqe'].append(quality.cpu().numpy())
            store['final_logits'].append(logits.cpu().numpy())
            store['box'].append(xyxy(boxes.cpu().numpy(), 1) * scale)
            keep = official['labels'] == TANK
            store['official_image_id'].append(np.full(int(keep.sum()), image['id']))
            store['official_score'].append(official['scores'][keep].cpu().numpy())
            store['official_box'].append(official['boxes'][keep].cpu().numpy())
            tanks = [a for a in grouped[image['id']] if a['category_id'] == TANK]
            if tanks:
                gt = np.array([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]] for a in tanks], np.float32) * np.array([INPUT / w, INPUT / h] * 2, np.float32)
                enc_logits = saved['enc_logits'][0]
                valid = decoder.valid_mask[0, :, 0]
                enc_box = torch.sigmoid(decoder.enc_bbox_head(saved['memory']) + decoder.anchors)[0]
                iou = iou_matrix(xyxy(enc_box.cpu().numpy()), gt)
                iou[~valid.cpu().numpy()] = 0
                order_max = torch.argsort(enc_logits.max(-1).values.masked_fill(~valid, -1e9), descending=True)[:decoder.num_queries].cpu().numpy()
                order_tank = torch.argsort(enc_logits[:, TANK].masked_fill(~valid, -1e9), descending=True)[:decoder.num_queries].cpu().numpy()
                enc_prob = torch.sigmoid(enc_logits[:, TANK]).cpu().numpy()
                for g in range(len(gt)):
                    oracle = np.sort(iou[:, g])[::-1][:decoder.num_queries]
                    enc_rows.append(dict(image_id=image['id'], source=source[image['id']], width=float(tanks[g]['bbox'][2]), height=float(tanks[g]['bbox'][3]),
                                         current=float(iou[order_max, g].max()), tank_only=float(iou[order_tank, g].max()), oracle=float(oracle.max())))
                best = iou.max(1)
                store['enc_scatter'].append(np.stack([np.full(len(order_max), image['id']), enc_prob[order_max], best[order_max]], 1))
            if (count + 1) % 500 == 0:
                print('images', count + 1, len(images), flush=True)
    np.savez_compressed(out / 'candidates.npz', **{k: np.concatenate(v) for k, v in store.items()})
    json.dump(enc_rows, open(out / 'encoder_gt_rows.json', 'w'))
    print('saved', len(images), 'images', len(enc_rows), 'tank GT', flush=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float('nan')
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float('nan')
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def evaluate(dets, gts):
    """COCO-style tank AP. dets: image -> (scores, boxes) already limited to MAX_DETS; gts: image -> boxes."""
    total = sum(len(v) for v in gts.values())
    lines = [([], []) for _ in THRESHOLDS]
    for image, (scores, boxes) in dets.items():
        gt = gts.get(image, np.zeros((0, 4)))
        order = np.argsort(-scores, kind='mergesort')
        scores, boxes = scores[order], boxes[order]
        m = iou_matrix(boxes, gt) if len(gt) and len(boxes) else np.zeros((len(boxes), len(gt)))
        for t, thr in enumerate(THRESHOLDS):
            used = np.zeros(len(gt), bool)
            hit = np.zeros(len(boxes), bool)
            for d in range(len(boxes)):
                if not len(gt):
                    break
                cand = np.where(used, -1, m[d])
                g = int(cand.argmax())
                if cand[g] >= thr:
                    used[g] = True
                    hit[d] = True
            lines[t][0].append(scores)
            lines[t][1].append(hit)
    ap = []
    for s, h in lines:
        s, h = np.concatenate(s), np.concatenate(h)
        order = np.argsort(-s, kind='mergesort')
        h = h[order]
        tp, fp = np.cumsum(h), np.cumsum(~h)
        rc, pr = tp / max(total, 1), tp / np.maximum(tp + fp, 1e-12)
        pr = np.maximum.accumulate(pr[::-1])[::-1] if len(pr) else pr
        idx = np.searchsorted(rc, RECALL, side='left')
        q = np.zeros(len(RECALL))
        ok = idx < len(pr)
        q[ok] = pr[idx[ok]]
        ap.append(q.mean())
    return dict(AP=round(100 * float(np.mean(ap)), 2), AP50=round(100 * ap[0], 2), AP75=round(100 * ap[5], 2))


def analyse(args):
    out = args.out
    annotations, source, grouped = load_val(out)
    data = np.load(out / 'candidates.npz')
    image_id, raw, lqe, logits, box = data['image_id'], data['raw_tank'], data['lqe'], data['final_logits'], data['box']
    final = sigmoid(logits[:, TANK])
    gts = {}
    for image in annotations['images']:
        if source[image['id']] in SOURCES:
            gts[image['id']] = np.array([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]]
                                         for a in grouped[image['id']] if a['category_id'] == TANK], np.float32).reshape(-1, 4)
    by_image = defaultdict(list)
    for n, i in enumerate(image_id):
        by_image[int(i)].append(n)
    by_image = {k: np.array(v) for k, v in by_image.items()}
    # Best tank-GT IoU and which GT, per query.
    q_iou, q_gt = np.zeros(len(image_id)), np.full(len(image_id), -1)
    for i, rows in by_image.items():
        if len(gts[i]):
            m = iou_matrix(box[rows], gts[i])
            q_iou[rows], q_gt[rows] = m.max(1), m.argmax(1)
    report = {'code_path': __doc__.split('Commands:')[0].strip()}
    variants = {
        'classification only: sigmoid(cls)': sigmoid(raw),
        'current: sigmoid(cls + LQE)': final,
        'LQE only': lqe,
        'sigmoid(cls + 0.5 LQE)': sigmoid(raw + .5 * lqe),
        'sigmoid(cls + 2 LQE)': sigmoid(raw + 2 * lqe),
        'sigmoid(cls + 4 LQE)': sigmoid(raw + 4 * lqe),
        'IoU oracle (not deployable)': q_iou + 1e-3 * final,
    }
    for src in SOURCES:
        imgs = [i for i in by_image if source[i] == src]
        tank_imgs = [i for i in imgs if len(gts[i])]
        rows_all = np.concatenate([by_image[i] for i in tank_imgs])
        near = rows_all[q_iou[rows_all] >= .1]
        r = {}
        # ---- experiment 1 ----
        hi, lo = near[q_iou[near] >= .85], near[(q_iou[near] >= .35) & (q_iou[near] < .45)]
        bins = [.1, .3, .5, .6, .7, .8, .9, 1.01]
        r['lqe'] = dict(
            queries_in_tank_images=len(rows_all), queries_near_tank_gt=len(near),
            lqe_mean_all=round(float(lqe[rows_all].mean()), 3), lqe_std_all=round(float(lqe[rows_all].std()), 3),
            lqe_mean_near=round(float(lqe[near].mean()), 3), lqe_std_near=round(float(lqe[near].std()), 3),
            cls_logit_std_near=round(float(raw[near].std()), 3),
            spearman_lqe_iou=round(spearman(lqe[near], q_iou[near]), 3), spearman_cls_iou=round(spearman(raw[near], q_iou[near]), 3),
            spearman_final_iou=round(spearman(final[near], q_iou[near]), 3), spearman_lqe_cls=round(spearman(lqe[near], raw[near]), 3),
            iou_ge_085=dict(n=len(hi), lqe_mean=round(float(lqe[hi].mean()), 3) if len(hi) else None, cls_mean=round(float(raw[hi].mean()), 3) if len(hi) else None),
            iou_035_045=dict(n=len(lo), lqe_mean=round(float(lqe[lo].mean()), 3) if len(lo) else None, cls_mean=round(float(raw[lo].mean()), 3) if len(lo) else None),
            auc_09_vs_04=dict(lqe=round(auc(lqe[hi], lqe[lo]), 3), cls=round(auc(raw[hi], raw[lo]), 3), final=round(auc(final[hi], final[lo]), 3)),
            by_iou=[dict(range=f'{a:.1f}-{min(b, 1):.1f}', n=int(((q_iou[near] >= a) & (q_iou[near] < b)).sum()),
                         lqe=round(float(lqe[near][(q_iou[near] >= a) & (q_iou[near] < b)].mean()), 3) if ((q_iou[near] >= a) & (q_iou[near] < b)).any() else None,
                         cls=round(float(raw[near][(q_iou[near] >= a) & (q_iou[near] < b)].mean()), 3) if ((q_iou[near] >= a) & (q_iou[near] < b)).any() else None)
                    for a, b in zip(bins[:-1], bins[1:])])
        # LQE effect on ordering: within each image, near-GT queries ranked by cls vs cls + LQE; and per-GT pick.
        rank_corr, changed, picks = [], 0, 0
        for i in tank_imgs:
            rows = by_image[i]
            rows = rows[q_iou[rows] >= .1]
            if len(rows) >= 3:
                rank_corr.append(spearman(raw[rows], raw[rows] + lqe[rows]))
            for g in np.unique(q_gt[rows]):
                cand = rows[q_gt[rows] == g]
                picks += 1
                changed += int(cand[np.argmax(raw[cand])] != cand[np.argmax(raw[cand] + lqe[cand])])
        r['lqe']['ordering'] = dict(within_image_spearman_cls_vs_cls_plus_lqe=round(float(np.median(rank_corr)), 4) if rank_corr else None,
                                    gt_whose_top_query_changes_with_lqe=round(changed / max(picks, 1), 3), gts=picks)
        # ---- experiment 3 ----
        gt_src = {i: gts[i] for i in imgs}
        official = {}
        oid = data['official_image_id']
        for i in imgs:
            k = oid == i
            s, b = data['official_score'][k], data['official_box'][k]
            o = np.argsort(-s)[:MAX_DETS]
            official[i] = (s[o], b[o])
        table = {'official post-processing (reference)': evaluate(official, gt_src)}
        n_gt = sum(len(v) for v in gt_src.values())
        for name, score in variants.items():
            dets, top_iou, best_pick = {}, [], []
            rec50 = rec75 = 0
            for i in imgs:
                rows = by_image[i]
                o = rows[np.argsort(-score[rows], kind='mergesort')][:MAX_DETS]
                dets[i] = (score[o], box[o])
                if len(gts[i]):
                    m = iou_matrix(box[o], gts[i])
                    rec50 += int((m.max(0) >= .5).sum())
                    rec75 += int((m.max(0) >= .75).sum())
                    for g in range(len(gts[i])):
                        hit = np.flatnonzero(m[:, g] >= .1)
                        best_pick.append(m[hit[0], g] if len(hit) else 0.)
            res = evaluate(dets, gt_src)
            res.update(recall100_iou50=round(rec50 / max(n_gt, 1), 4), recall100_iou75=round(rec75 / max(n_gt, 1), 4),
                       top_ranked_overlap_iou_median=round(float(np.median(best_pick)), 3), top_ranked_overlap_iou_ge_075=round(float(np.mean(np.array(best_pick) >= .75)), 3))
            table[name] = res
        # Ceiling: best IoU among all 300 queries per GT.
        ceil = []
        for i in tank_imgs:
            ceil += iou_matrix(box[by_image[i]], gts[i]).max(0).tolist()
        r['reranking'] = table
        r['ceiling_best_of_300_iou_median'] = round(float(np.median(ceil)), 3)
        enc = [e for e in json.load(open(out / 'encoder_gt_rows.json')) if e['source'] == src and e['width'] > 0 and e['height'] > 0]
        r['encoder_top300_recall'] = {name: dict(iou50=round(float(np.mean([e[k] >= .5 for e in enc])), 4), iou75=round(float(np.mean([e[k] >= .75 for e in enc])), 4))
                                      for name, k in (('current: max-class logit', 'current'), ('tank logit only', 'tank_only'), ('IoU oracle', 'oracle'))}
        r['encoder_top300_recall']['note'] = 'LQE does not exist at the encoder stage, so it cannot be applied there.'
        r['images'] = len(imgs)
        r['tank_gt'] = n_gt
        report[src] = r
        print(src, json.dumps(r, ensure_ascii=False), flush=True)
    (out / 'report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    # Plot data only: tank images, score >= 0.01. Plotting needs matplotlib, which the server env may lack.
    enc = data['enc_scatter']
    has_tank = np.isin(image_id, np.unique(enc[:, 0]))
    np.savez_compressed(out / 'scatter.npz', enc_score=enc[:, 1], enc_iou=enc[:, 2], enc_source=np.array([source[int(i)] for i in enc[:, 0]]),
                        dec_score=final[has_tank], dec_iou=q_iou[has_tank], dec_source=np.array([source[int(i)] for i in image_id[has_tank]]))
    print('saved scatter.npz; plot with: tank_lqe_ranking.py plot RUN OUT', flush=True)


def plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = args.out
    data = np.load(out / 'scatter.npz')
    surface, ink, muted, series = '#fcfcfb', '#0b0b0b', '#52514e', '#2a78d6'
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.6), facecolor=surface, sharex=True, sharey=True)
    panels = [('Encoder Top-300 candidates', data['enc_score'], data['enc_iou'], data['enc_source']),
              ('Decoder final queries', data['dec_score'], data['dec_iou'], data['dec_source'])]
    for row, (title, score, iou, src) in enumerate(panels):
        for col, name in enumerate(SOURCES):
            ax = axes[row, col]
            keep = (src == name) & (score >= .01)
            ax.set_facecolor(surface)
            ax.scatter(score[keep], iou[keep], s=4, alpha=.25, color=series, linewidths=0, rasterized=True)
            edges = np.linspace(.01, 1, 21)
            mids, med = [], []
            for a, b in zip(edges[:-1], edges[1:]):
                k = keep & (score >= a) & (score < b)
                mids.append((a + b) / 2)
                # Bins with too few points stay empty, so the line never bridges a gap in the data.
                med.append(np.median(iou[k]) if k.sum() >= 15 else np.nan)
            ax.plot(mids, med, color=ink, linewidth=2, marker='o', markersize=3.5, label='median IoU per score bin (>= 15 points)')
            for y in (.5, .75):
                ax.axhline(y, color=muted, linewidth=.8, linestyle=(0, (3, 3)))
            ax.axvline(.3, color=muted, linewidth=.8, linestyle=(0, (3, 3)))
            rho = spearman(score[keep & (iou >= .1)], iou[keep & (iou >= .1)])
            ax.set_title(f'{name.upper()} - {title}\n{int(keep.sum())} candidates (score >= 0.01), Spearman on IoU >= 0.1: {rho:.2f}', fontsize=9.5, color=ink, loc='left')
            ax.grid(color='#e6e5e1', linewidth=.6)
            ax.set_axisbelow(True)
            for s in ax.spines.values():
                s.set_color('#c9c8c2')
            ax.tick_params(colors=muted, labelsize=8.5)
            ax.set_xlim(0, 1)
            ax.set_ylim(-.02, 1.02)
            if row == 1:
                ax.set_xlabel('Tank classification score', color=ink, fontsize=9.5)
            if col == 0:
                ax.set_ylabel('IoU with the best-matching tank GT', color=ink, fontsize=9.5)
    axes[0, 0].legend(loc='upper left', fontsize=8.5, frameon=False, labelcolor=ink)
    fig.suptitle('E000 val: tank score against box quality (dashed: score 0.3, IoU 0.5 and 0.75; IoU 0 = background)', fontsize=10.5, color=ink, x=.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(out / 'score_iou_scatter.png', dpi=150, facecolor=surface)
    print('figure', out / 'score_iou_scatter.png', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('collect', 'analyse', 'plot'))
    parser.add_argument('run', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    {'collect': collect, 'analyse': analyse, 'plot': plot}[args.command](args)


if __name__ == '__main__':
    main()
