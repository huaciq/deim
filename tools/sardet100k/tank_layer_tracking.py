"""Layer-by-layer tracking of the same 300 queries: encoder Top-K, then every decoder layer. Val tank images, E000 unchanged.

Each decoder layer has its own score head and LQE (all kept, the model is not converted to deploy). At inference the
decoder computes scores only at eval_idx and stops, so the loaded object's decoder.decoder.eval_idx is set to 0..L-1 in
turn and the image is forwarded again; nothing on disk changes. The encoder selection does not depend on eval_idx, so
query j is the same query in every pass; this is asserted on the encoder indices.

Fixed for all stages: query identity, its GT (the tank GT it overlaps most at any stage) and the candidate set
(queries with IoU >= 0.1 to that GT at any stage). Stage 'enc' is the encoder Top-K box and logit.

Commands:
  collect RUN OUT --device cuda:0
  analyse RUN OUT
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
from tank_lqe_ranking import sigmoid, spearman  # noqa: E402
from tank_query_tracking import boot_ci  # noqa: E402


def collect(args):
    import torch
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    annotations, source, grouped = load_val(out)
    images = [i for i in annotations['images'] if source[i['id']] in SOURCES and any(a['category_id'] == TANK for a in grouped[i['id']])]
    model, _ = load_model(args.run, args.device)
    decoder = model.decoder
    inner = decoder.decoder
    layers = inner.num_layers
    default_idx = inner.eval_idx
    assert default_idx == layers - 1 and decoder.query_select_method == 'default'
    saved = {}
    decoder.enc_score_head.register_forward_hook(lambda _, __, o: saved.__setitem__('enc_logits', o.detach()))
    decoder.enc_bbox_head.register_forward_hook(lambda _, __, o: saved.__setitem__('enc_box_delta', o.detach()))
    for k in range(layers):
        inner.lqe_layers[k].reg_conf.register_forward_hook(lambda _, __, o, k=k: saved.__setitem__(f'quality{k}', o.detach()))
    original = decoder._select_topk

    def select(memory, logits, anchors, topk):
        result = original(memory, logits, anchors, topk)
        saved['topk_ind'] = torch.topk(logits.max(-1).values, topk, dim=-1)[1].detach()
        saved['topk_anchors'] = result[2].detach()
        return result

    decoder._select_topk = select
    store = defaultdict(list)
    worst = 0.
    with torch.no_grad():
        for image in images:
            x, (w, h) = to_input(ROOT / 'JPEGImages/val' / image['file_name'])
            x = x[None].to(args.device)
            inner.eval_idx = default_idx
            reference = model(x)
            ind0 = saved['topk_ind'][0].clone()
            enc_logits = saved['enc_logits'][0][ind0]
            enc_box = torch.sigmoid(saved['enc_box_delta'] + saved['topk_anchors'])[0]
            raw, lqe, box = [], [], []
            for k in range(layers):
                inner.eval_idx = k
                result = model(x)
                assert torch.equal(saved['topk_ind'][0], ind0), 'encoder selection changed between passes'
                q = saved[f'quality{k}'][0, :, 0]
                raw.append((result['pred_logits'][0, :, TANK] - q).cpu().numpy())
                lqe.append(q.cpu().numpy())
                box.append(xyxy(result['pred_boxes'][0].cpu().numpy()))
                if k == default_idx:
                    worst = max(worst, float((result['pred_logits'] - reference['pred_logits']).abs().max()), float((result['pred_boxes'] - reference['pred_boxes']).abs().max()))
            inner.eval_idx = default_idx
            n = len(ind0)
            store['image_id'].append(np.full(n, image['id']))
            store['enc_logit'].append(enc_logits[:, TANK].cpu().numpy())
            store['enc_box'].append(xyxy(enc_box.cpu().numpy()))
            store['raw'].append(np.stack(raw, 1))
            store['lqe'].append(np.stack(lqe, 1))
            store['box'].append(np.stack(box, 1))
            tanks = [a for a in grouped[image['id']] if a['category_id'] == TANK and a['bbox'][2] > 0 and a['bbox'][3] > 0]
            gt = np.array([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]] for a in tanks], np.float32) * np.array([INPUT / w, INPUT / h] * 2, np.float32)
            store['gt'].append(gt.reshape(-1, 4))
            store['gt_image'].append(np.full(len(gt), image['id']))
    np.savez_compressed(out / 'layers.npz', **{k: np.concatenate(v) for k, v in store.items()})
    report = dict(images=len(images), decoder_layers=layers, max_abs_difference_last_pass_vs_default=worst)
    (out / 'collect_report.json').write_text(json.dumps(report, indent=1))
    print(report, flush=True)


def med(v):
    return round(float(np.median(v)), 3) if len(v) else None


def analyse(args):
    out = args.out
    _, source, _ = load_val(out)
    d = np.load(out / 'layers.npz')
    image, gt, gt_image = d['image_id'], d['gt'], d['gt_image']
    L = d['raw'].shape[1]
    stages = ['enc'] + [f'dec{k + 1}' for k in range(L)]
    boxes = np.concatenate([d['enc_box'][:, None], d['box']], 1)          # N x S x 4
    total = np.concatenate([d['enc_logit'][:, None], d['raw'] + d['lqe']], 1)  # encoder has no LQE
    cls = np.concatenate([d['enc_logit'][:, None], d['raw']], 1)
    lqe = d['lqe']
    N, S = boxes.shape[:2]
    iou = np.zeros((N, S))
    owner = np.full(N, -1)
    for i in np.unique(image):
        rows, g = np.flatnonzero(image == i), np.flatnonzero(gt_image == i)
        if not len(g):
            continue
        m = np.stack([iou_matrix(boxes[rows, s], gt[g]) for s in range(S)], 1)  # rows x S x G
        k = m.max(1).argmax(1)                       # GT fixed once, over all stages
        iou[rows] = m[np.arange(len(rows)), :, k]
        owner[rows] = g[k]
    relevant = iou.max(1) >= .1                      # candidate set fixed once
    score = sigmoid(total)
    final = total[:, -1]
    rank = np.zeros(N, int)
    for i in np.unique(image):
        rows = np.flatnonzero(image == i)
        rank[rows[np.argsort(-final[rows])]] = np.arange(len(rows))
    report = dict(stages=stages, note='S = tank logit; decoder S = score-head logit + LQE of that layer; encoder has no LQE.')
    for src in SOURCES:
        sel = np.array([source[int(i)] == src for i in image])
        rel = sel & relevant
        idx = np.flatnonzero(rel)
        r = dict(images=len(np.unique(image[sel])), queries=int(rel.sum()))
        per = {}
        for s, name in enumerate(stages):
            per[name] = dict(
                iou_mean=round(float(iou[rel, s].mean()), 3), iou_median=med(iou[rel, s]),
                score_mean=round(float(score[rel, s].mean()), 3), score_median=med(score[rel, s]),
                spearman_cls_iou=round(spearman(cls[idx, s], iou[idx, s]), 3),
                spearman_cls_iou_ci=boot_ci(lambda t, s=s: spearman(cls[idx[t], s], iou[idx[t], s]), image[idx]),
                spearman_total_iou=round(spearman(total[idx, s], iou[idx, s]), 3))
            if s:
                per[name].update(lqe_mean=round(float(lqe[rel, s - 1].mean()), 3), lqe_std=round(float(lqe[rel, s - 1].std()), 3),
                                 spearman_lqe_iou=round(spearman(lqe[idx, s - 1], iou[idx, s]), 3))
        r['per_stage'] = per
        # LQE share at the last layer.
        a, b = np.abs(d['raw'][rel, -1]), np.abs(lqe[rel, -1])
        r['last_layer_lqe_share'] = dict(median_abs_lqe_over_abs_cls_plus_abs_lqe=round(float(np.median(b / (a + b))), 4),
                                         variance_share_lqe=round(float(lqe[rel, -1].var() / total[rel, -1].var()), 4),
                                         median_abs_cls=round(float(np.median(a)), 3), median_abs_lqe=round(float(np.median(b)), 3))
        # Trajectories. Classes by encoder -> final change of the decisive (top-ranked) query of each GT, as in section 10.
        top = np.zeros(N, bool)
        better = np.zeros(N, bool)
        pairs = 0
        for g in np.unique(owner[rel]):
            rows = np.flatnonzero(rel & (owner == g))
            t = rows[np.argmin(rank[rows])]
            top[t] = True
            b_ = rows[np.argmax(iou[rows, -1])]
            if b_ != t and iou[b_, -1] >= iou[t, -1] + .05:
                better[b_] = True
                pairs += 1
        d_iou, d_s = iou[:, -1] - iou[:, 0], total[:, -1] - total[:, 0]
        groups = {
            'A (top query of its GT: IoU up, score up)': top & rel & (d_iou > 0) & (d_s > 0),
            'C (top query of its GT: IoU down, score up)': top & rel & (d_iou < 0) & (d_s > 0),
            'final top-ranked query of each GT': top & rel,
            'a better-localised query of the same GT (final IoU >= top + 0.05)': better & rel,
            'judged background: overlapping, final score < 0.05': rel & (score[:, -1] < .05),
            'judged background but IoU >= 0.5 at some stage': rel & (score[:, -1] < .05) & (iou.max(1) >= .5),
        }
        traj = {}
        for name, mask in groups.items():
            traj[name] = dict(n=int(mask.sum()), iou=[med(iou[mask, s]) for s in range(S)], score=[med(score[mask, s]) for s in range(S)],
                              cls_logit=[med(cls[mask, s]) for s in range(S)])
        r['trajectories'] = traj
        r['top_vs_better_pairs'] = pairs
        # Paired view: for GTs with a better query, the two queries side by side, stage by stage.
        tops, bets = [], []
        for g in np.unique(owner[rel]):
            rows = np.flatnonzero(rel & (owner == g))
            t = rows[np.argmin(rank[rows])]
            b_ = rows[np.argmax(iou[rows, -1])]
            if b_ != t and iou[b_, -1] >= iou[t, -1] + .05:
                tops.append(t)
                bets.append(b_)
        tops, bets = np.array(tops, int), np.array(bets, int)
        if len(tops):
            r['paired_top_minus_better'] = dict(pairs=len(tops),
                                                iou=[med(iou[tops, s] - iou[bets, s]) for s in range(S)],
                                                score=[med(score[tops, s] - score[bets, s]) for s in range(S)],
                                                share_top_scores_higher=[round(float((total[tops, s] > total[bets, s]).mean()), 3) for s in range(S)],
                                                share_top_iou_higher=[round(float((iou[tops, s] > iou[bets, s]).mean()), 3) for s in range(S)])
        # Where does the per-query score-IoU coupling break: layer-to-layer change correlations.
        steps = {}
        for s in range(1, S):
            di, ds = iou[idx, s] - iou[idx, s - 1], cls[idx, s] - cls[idx, s - 1]
            steps[f'{stages[s - 1]}->{stages[s]}'] = dict(spearman_dcls_diou=round(spearman(ds, di), 3),
                                                          ci=boot_ci(lambda t, di=di, ds=ds: spearman(ds[t], di[t]), image[idx]),
                                                          median_diou=med(di), median_dcls=med(ds),
                                                          share_iou_down_cls_up=round(float(((di < 0) & (ds > 0)).mean()), 3))
        r['layer_steps'] = steps
        report[src] = r
        print(src, json.dumps(r, ensure_ascii=False), flush=True)
    (out / 'report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('collect', 'analyse'))
    parser.add_argument('run', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    {'collect': collect, 'analyse': analyse}[args.command](args)


if __name__ == '__main__':
    main()
