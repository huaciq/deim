"""Automatic diagnostic report for one run, and an E000-vs-E1 comparison. Val only; test is never read.

Every run is measured with its own EMA weights and its own matcher (from its resolved config), on the same images,
with the same GT association rule, the same candidate count (the decoder's 300 queries, the postprocessor's top 300)
and the same code. Nothing here reuses another run's offline results.

Commands (repository root on the server; OUT must sit directly under E000's independent_verify_20261106/, where
error_gallery_val/source.json is):
  collect RUN OUT [--device cuda:0] [--checkpoint PATH]   full-val predictions + per-stage outputs on the tank images
  analyse OUT [--dynamics DIR] [--train-log]              sections 1-6 -> OUT/report.json
  compare BASE_OUT RUN_OUT --out DIR [--labels E000 E1]   (local) change tables and figures -> DIR/comparison.md

Checkpoint: by default the file among best_stg2/best_stg1/final whose saved epoch is the epoch with the highest val AP
in log.txt (the AGENTS rule: select by evaluated AP, not by file name).

Sections
  1 overall           AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100 from log.txt at that epoch, and a recomputation
  2 source x class    COCO AP / AP50 / AP75 / AR per group (ogsod, msar, other_256, other_512, other_800, ...) and class
  3 matching quality  per stage (enc, pre, dec1-4) on tank GTs: matched-query IoU, matched query = best-IoU query,
                      query changes between decoder layers, unweighted matcher cost terms of the matched pairs
  4 query competition top-ranked query of each tank GT: A/B/C/D by encoder -> final change of IoU and tank logit;
                      GTs whose top query is beaten by another query by >= .05 IoU; score and IoU gaps
  5 cross-layer       DEIMCriterion._record_matching_metrics on the replayed assignment (the training-log definitions)
  6 decoder layers    per stage Spearman of classification logit / logit + LQE with IoU, IoU and score levels,
                      layer-to-layer change correlations, LQE share at the last layer
GT association for sections 4 and 6: each query belongs to the tank GT it overlaps most at any stage (enc, dec1-4);
candidates are the queries with IoU >= .1 to that GT at some stage. Fixed once per run, never re-chosen per layer.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

STAGES = ['enc', 'pre', 'dec1', 'dec2', 'dec3', 'dec4']
TRACK = ['enc', 'dec1', 'dec2', 'dec3', 'dec4']
GROUP_ORDER = ['ogsod', 'msar', 'other_256', 'other_512', 'other_800', 'other_224', 'other_1000', 'other_misc']
TANK_SOURCES = ('ogsod', 'msar')
TANK = 3
BOOT = 500
COCO_NAMES = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100']


def group_of(source, width, height):
    """Same rule as dataset_profile.group_of."""
    if source != 'other':
        return source
    return f'other_{width}' if width == height and width in (224, 256, 512, 800, 1000) else 'other_misc'


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float('nan')
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def iou_xyxy(a, b):
    x1, y1 = np.maximum(a[:, None, 0], b[:, 0]), np.maximum(a[:, None, 1], b[:, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[:, 2]), np.minimum(a[:, None, 3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = (a[:, 2] - a[:, 0])[:, None] * (a[:, 3] - a[:, 1])[:, None] + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return inter / np.maximum(union, 1e-12)


def boot_ci(stat, groups, seed=0):
    rng = np.random.default_rng(seed)
    keys = np.unique(groups)
    members = {k: np.flatnonzero(groups == k) for k in keys}
    values = [stat(np.concatenate([members[k] for k in rng.choice(keys, len(keys))])) for _ in range(BOOT)]
    return [round(float(np.nanpercentile(values, 2.5)), 3), round(float(np.nanpercentile(values, 97.5)), 3)]


def r3(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 3)


# ---------------------------------------------------------------- collect
def best_epoch(run):
    rows = [json.loads(s) for s in open(run / 'log.txt') if s.strip()]
    best = max(rows, key=lambda r: r['test_coco_eval_bbox'][0])
    return best


def choose_checkpoint(run, epoch):
    import torch
    found = {}
    for name in ('best_stg2.pth', 'best_stg1.pth', 'final.pth', 'last.pth'):
        path = run / name
        if path.is_file():
            found[name] = torch.load(path, map_location='cpu', weights_only=False).get('last_epoch')
            if found[name] == epoch:
                return path, found
    raise SystemExit(f'No checkpoint saved at the best epoch {epoch}: {found}. Pass --checkpoint explicitly.')


def collect(args):
    import torch
    from engine.core import YAMLConfig
    from error_gallery import ROOT
    from tank_feature_diagnosis import load_val, to_input, xyxy
    from tank_matching_counterfactual import Runner
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    annotations, source, grouped = load_val(out)
    best = best_epoch(args.run)
    if args.checkpoint:
        checkpoint, found = args.checkpoint, {}
    else:
        checkpoint, found = choose_checkpoint(args.run, best['epoch'])
    runner = Runner(args.run, args.device, checkpoint)
    model, inner, crit = runner.model, runner.inner, runner.criterion
    post = YAMLConfig(str(args.run / 'resolved_config.yml')).postprocessor.to(args.device).eval()
    for k in range(inner.num_layers):
        inner.lqe_layers[k].reg_conf.register_forward_hook(lambda _, __, o, k=k: runner.saved.__setitem__(f'quality{k}', o.detach()))
    groups = {i['id']: group_of(source[i['id']], i['width'], i['height']) for i in annotations['images']}

    # Full-val predictions with the default (last) decoder layer and the configured postprocessor.
    ids, labels, scores, boxes = [], [], [], []
    batch = []
    images = annotations['images']
    with torch.no_grad():
        for n, image in enumerate(images):
            x, (w, h) = to_input(ROOT / 'JPEGImages/val' / image['file_name'])
            batch.append((image['id'], x, (w, h)))
            if len(batch) == args.batch or n == len(images) - 1:
                inner.eval_idx = runner.default_idx
                result = post(model(torch.stack([b[1] for b in batch]).to(args.device)),
                              torch.tensor([b[2] for b in batch], device=args.device, dtype=torch.float32))
                for (image_id, _, _), r in zip(batch, result):
                    ids.append(np.full(len(r['scores']), image_id))
                    labels.append(r['labels'].cpu().numpy())
                    scores.append(r['scores'].cpu().numpy())
                    boxes.append(r['boxes'].cpu().numpy())
                batch = []
            if (n + 1) % 1000 == 0:
                print('predicted', n + 1, len(images), flush=True)
    np.savez_compressed(out / 'val_predictions.npz', image_id=np.concatenate(ids), label=np.concatenate(labels),
                        score=np.concatenate(scores), box=np.concatenate(boxes))

    # Per-stage outputs, matching and the union set on the tank images of OGSOD and MSAR.
    crit.matching_stats_class = TANK
    records, cross = [], {s: defaultdict(float) for s in TANK_SOURCES}
    tank_images = [i for i in images if source[i['id']] in TANK_SOURCES and any(a['category_id'] == TANK for a in grouped[i['id']])]
    for image in tank_images:
        anns = grouped[image['id']]
        x, _ = to_input(ROOT / 'JPEGImages/val' / image['file_name'])
        targets, xy = runner.targets(image, anns)
        outs = runner.stages(x[None].to(args.device))
        quality = np.stack([runner.saved[f'quality{k}'][0, :, 0].cpu().numpy() for k in range(inner.num_layers)])
        indices = {s: runner.matcher({'pred_logits': outs[s][0][None], 'pred_boxes': outs[s][1][None]}, targets)['indices'] for s in STAGES}
        go = crit._get_go_indices(indices['dec4'], [indices[s] for s in ('dec1', 'dec2', 'dec3', 'pre', 'enc')])
        outputs = dict(pred_boxes=outs['dec4'][1][None], aux_outputs=[{'pred_boxes': outs[f'dec{k}'][1][None]} for k in (1, 2, 3)],
                       pre_outputs={'pred_boxes': outs['pre'][1][None]}, enc_aux_outputs=[{'pred_boxes': outs['enc'][1][None]}])
        crit._record_matching_metrics(outputs, targets, indices['dec4'], [indices[s] for s in ('dec1', 'dec2', 'dec3', 'pre')], [indices['enc']], go)
        for name, value in crit.matching_metrics.items():
            cross[source[image['id']]][name] += value
        records.append(dict(
            image_id=image['id'], source=source[image['id']], gt_xyxy=np.asarray(xy, np.float32),
            gt_labels=np.array([a['category_id'] for a in anns]),
            logits=np.stack([outs[s][0].float().cpu().numpy() for s in STAGES]),
            boxes=np.stack([xyxy(outs[s][1].float().cpu().numpy(), 1) for s in STAGES]),
            lqe=quality,
            indices={s: (indices[s][0][0].numpy(), indices[s][0][1].numpy()) for s in STAGES},
            go=(go[0][0].numpy(), go[0][1].numpy())))
    torch.save(records, out / 'tank_stages.pt')
    matcher = runner.matcher
    meta = dict(run=str(args.run), checkpoint=str(checkpoint), checkpoint_epochs=found, best_log_row=best, epoch=best['epoch'],
                matcher=dict(cost_class=matcher.cost_class, cost_bbox=matcher.cost_bbox, cost_giou=matcher.cost_giou,
                             alpha=matcher.alpha, gamma=matcher.gamma),
                images=len(images), tank_images=len(tank_images), groups=dict(Counter(groups.values())),
                cross_layer_sums={s: dict(v) for s, v in cross.items()}, device=args.device, batch=args.batch)
    (out / 'collect_meta.json').write_text(json.dumps(meta, indent=1, default=float))
    print('collected', out, flush=True)


# ---------------------------------------------------------------- analyse
def coco_tables(out):
    from faster_coco_eval import COCO, COCOeval_faster
    from error_gallery import ROOT
    from tank_feature_diagnosis import load_val
    annotations, source, _ = load_val(out)
    groups = {i['id']: group_of(source[i['id']], i['width'], i['height']) for i in annotations['images']}
    members = defaultdict(list)
    for image_id, g in groups.items():
        members[g].append(image_id)
    saved = np.load(out / 'val_predictions.npz')
    results = [dict(image_id=int(i), category_id=int(c), score=float(s), bbox=[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])])
               for i, c, s, b in zip(saved['image_id'], saved['label'], saved['score'], saved['box'])]
    gt = COCO(str(ROOT / 'Annotations/val.json'))
    dt = gt.loadRes(results)
    overall = COCOeval_faster(gt, dt, 'bbox')
    overall.evaluate()
    overall.accumulate()
    overall.summarize()
    names = {c['id']: c['name'] for c in annotations['categories']}
    table = {}
    for g in [x for x in GROUP_ORDER if x in members] + ['all']:
        e = COCOeval_faster(gt, dt, 'bbox')
        e.params.imgIds = sorted(members[g] if g != 'all' else groups)
        e.evaluate()
        e.accumulate()
        p, r = np.asarray(e.eval['precision']), np.asarray(e.eval['recall'])  # T,R,K,A,M / T,K,A,M
        row = {}
        for k, c in enumerate(e.params.catIds):
            pk, rk = p[:, :, k, 0, -1], r[:, k, 0, -1]
            if (rk < 0).all():
                continue
            row[names[c]] = dict(AP=r3(pk[pk > -1].mean() * 100), AP50=r3(pk[0][pk[0] > -1].mean() * 100),
                                 AP75=r3(pk[5][pk[5] > -1].mean() * 100), AR=r3(rk[rk > -1].mean() * 100))
        row['mean'] = dict(AP=r3(np.mean([v['AP'] for v in row.values()])), images=len(e.params.imgIds))
        table[g] = row
    return dict(zip(COCO_NAMES, [r3(v * 100) for v in overall.stats[:9]])), table


def per_query_tables(records, matcher):
    """Flatten tank-image records into per-query arrays with the fixed GT association."""
    rows = defaultdict(list)
    for rec in records:
        tank = np.flatnonzero(rec['gt_labels'] == TANK)
        if not len(tank):
            continue
        boxes = rec['boxes']  # S x Q x 4
        track = [STAGES.index(s) for s in TRACK]
        m = np.stack([iou_xyxy(boxes[s], rec['gt_xyxy'][tank]) for s in track], 1)  # Q x T x G
        owner = m.max(1).argmax(1)
        iou = m[np.arange(len(owner)), :, owner]  # Q x T
        cls = np.stack([rec['logits'][s, :, TANK] for s in track], 1)
        lqe = np.concatenate([np.zeros((len(owner), 1)), rec['lqe'].T], 1)  # encoder has no LQE
        rows['image'].append(np.full(len(owner), rec['image_id']))
        rows['source'].append(np.full(len(owner), rec['source']))
        rows['owner'].append(np.array([f"{rec['image_id']}_{g}" for g in owner]))
        rows['iou'].append(iou)
        rows['total'].append(cls)          # decoder logits already include LQE
        rows['raw'].append(cls - lqe)
        rows['lqe'].append(lqe)
    return {k: np.concatenate(v) for k, v in rows.items()}


def analyse(args):
    import torch
    out = args.out
    meta = json.loads((out / 'collect_meta.json').read_text())
    report = dict(meta={k: meta[k] for k in ('run', 'checkpoint', 'epoch', 'matcher', 'images', 'tank_images', 'groups')})
    best = meta['best_log_row']
    overall, table = coco_tables(out)
    report['1_overall'] = dict(log=dict(zip(COCO_NAMES, [r3(v * 100) for v in best['test_coco_eval_bbox'][:9]])), recomputed=overall)
    report['2_source_class'] = table
    records = torch.load(out / 'tank_stages.pt', weights_only=False)
    m = meta['matcher']

    # 3 matching quality.
    sec3 = {}
    for src in TANK_SOURCES:
        per = defaultdict(list)
        for rec in [r for r in records if r['source'] == src]:
            tank = np.flatnonzero(rec['gt_labels'] == TANK)
            matched = {}
            for si, s in enumerate(STAGES):
                q, g = rec['indices'][s]
                q_of = dict(zip(g.tolist(), q.tolist()))
                matched[s] = q_of
                boxes, logits = rec['boxes'][si], rec['logits'][si]
                prob = sigmoid(logits[:, rec['gt_labels']])
                neg = (1 - m['alpha']) * prob ** m['gamma'] * -np.log(1 - prob + 1e-8)
                pos = m['alpha'] * (1 - prob) ** m['gamma'] * -np.log(prob + 1e-8)
                cls_cost = pos - neg
                ious = iou_xyxy(boxes, rec['gt_xyxy'])
                gt_c = np.stack([(rec['gt_xyxy'][:, 0] + rec['gt_xyxy'][:, 2]) / 2, (rec['gt_xyxy'][:, 1] + rec['gt_xyxy'][:, 3]) / 2,
                                 rec['gt_xyxy'][:, 2] - rec['gt_xyxy'][:, 0], rec['gt_xyxy'][:, 3] - rec['gt_xyxy'][:, 1]], 1)
                pr_c = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2, boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], 1)
                for t in tank:
                    qq = q_of[t]
                    per[f'{s}_iou'].append(ious[qq, t])
                    per[f'{s}_is_best'].append(float(qq == ious[:, t].argmax()))
                    per[f'{s}_cls_cost'].append(cls_cost[qq, t])
                    per[f'{s}_l1_cost'].append(np.abs(pr_c[qq] - gt_c[t]).sum())
                    per[f'{s}_giou_cost'].append(-giou(boxes[qq], rec['gt_xyxy'][t]))
            for t in tank:
                layers = [matched[f'dec{k}'][t] for k in (1, 2, 3, 4)]
                per['same_all_dec'].append(float(len(set(layers)) == 1))
                for k in range(3):
                    per[f'change_dec{k + 1}_dec{k + 2}'].append(float(layers[k] != layers[k + 1]))
                per['change_enc_dec4'].append(float(matched['enc'][t] != matched['dec4'][t]))
        res = dict(tank_gt=len(per['dec4_iou']), stages={})
        for s in STAGES:
            v = np.array(per[f'{s}_iou'])
            res['stages'][s] = dict(iou_mean=r3(v.mean()), iou_median=r3(np.median(v)), iou_ge50=r3((v >= .5).mean()), iou_ge75=r3((v >= .75).mean()),
                                    matched_is_best_iou_query=r3(np.mean(per[f'{s}_is_best'])),
                                    cost_median_unweighted=dict(cls=r3(np.median(per[f'{s}_cls_cost'])), l1=r3(np.median(per[f'{s}_l1_cost'])),
                                                                giou=r3(np.median(per[f'{s}_giou_cost']))))
        res['query_change_rate'] = {k: r3(np.mean(per[k])) for k in ('change_dec1_dec2', 'change_dec2_dec3', 'change_dec3_dec4', 'change_enc_dec4')}
        res['same_query_all_decoder_layers'] = r3(np.mean(per['same_all_dec']))
        sec3[src] = res
    report['3_matching_quality'] = sec3

    # 4 query competition and 6 decoder layers, fixed association.
    q = per_query_tables(records, None)
    sec4, sec6 = {}, {}
    for src in TANK_SOURCES:
        sel = (q['source'] == src) & (q['iou'].max(1) >= .1)
        idx = np.flatnonzero(sel)
        iou, total, raw, lqe = q['iou'][idx], q['total'][idx], q['raw'][idx], q['lqe'][idx]
        owners, images = q['owner'][idx], q['image'][idx]
        top, best = {}, {}
        for o in np.unique(owners):
            rows = np.flatnonzero(owners == o)
            top[o] = rows[np.argmax(total[rows, -1])]
            best[o] = rows[np.argmax(iou[rows, -1])]
        t = np.array(list(top.values()))
        d_iou, d_s = iou[t, -1] - iou[t, 0], total[t, -1] - total[t, 0]
        classes = {'A': (d_iou > 0) & (d_s > 0), 'B': (d_iou > 0) & (d_s <= 0), 'C': (d_iou <= 0) & (d_s > 0), 'D': (d_iou <= 0) & (d_s <= 0)}
        gaps = [(o, top[o], best[o]) for o in top if best[o] != top[o] and iou[best[o], -1] >= iou[top[o], -1] + .05]
        sec4[src] = dict(
            gts=len(top), classes_top_query={k: r3(v.mean()) for k, v in classes.items()},
            class_trajectories={k: dict(n=int(v.sum()), enc_iou=r3(np.median(iou[t[v], 0])) if v.any() else None, dec4_iou=r3(np.median(iou[t[v], -1])) if v.any() else None,
                                        enc_score=r3(np.median(sigmoid(total[t[v], 0]))) if v.any() else None, dec4_score=r3(np.median(sigmoid(total[t[v], -1]))) if v.any() else None)
                                for k, v in classes.items()},
            top_beaten_by_005=r3(len(gaps) / max(len(top), 1)), pairs=len(gaps),
            top_below_075_best_above=r3(np.mean([iou[top[o], -1] < .75 <= iou[best[o], -1] for o in top])),
            pair_gap_median=dict(score_top_minus_best=r3(np.median([sigmoid(total[a, -1]) - sigmoid(total[b, -1]) for _, a, b in gaps])) if gaps else None,
                                 iou_top_minus_best=r3(np.median([iou[a, -1] - iou[b, -1] for _, a, b in gaps])) if gaps else None),
            top_iou_median=r3(np.median(iou[t, -1])), best_available_iou_median=r3(np.median(iou[np.array(list(best.values())), -1])))
        layers = {}
        for k, s in enumerate(TRACK):
            layers[s] = dict(spearman_cls_iou=r3(spearman(raw[:, k], iou[:, k])),
                             spearman_cls_iou_ci=boot_ci(lambda tt, k=k: spearman(raw[tt, k], iou[tt, k]), images),
                             spearman_total_iou=r3(spearman(total[:, k], iou[:, k])),
                             iou_mean=r3(iou[:, k].mean()), iou_median=r3(np.median(iou[:, k])),
                             score_mean=r3(sigmoid(total[:, k]).mean()), score_median=r3(np.median(sigmoid(total[:, k]))))
        steps = {}
        for k in range(1, len(TRACK)):
            di, dc = iou[:, k] - iou[:, k - 1], raw[:, k] - raw[:, k - 1]
            steps[f'{TRACK[k - 1]}->{TRACK[k]}'] = dict(spearman_dcls_diou=r3(spearman(dc, di)), median_diou=r3(np.median(di)), median_dcls=r3(np.median(dc)),
                                                         share_iou_down_cls_up=r3(((di < 0) & (dc > 0)).mean()))
        a, b = np.abs(raw[:, -1]), np.abs(lqe[:, -1])
        sec6[src] = dict(queries=len(idx), layers=layers, steps=steps,
                         lqe_last_layer=dict(median_share=r3(np.median(b / np.maximum(a + b, 1e-12))), variance_share=r3(lqe[:, -1].var() / max(total[:, -1].var(), 1e-12))))
    report['4_query_competition'] = sec4
    report['6_decoder_layers'] = sec6

    # 5 cross-layer supervision.
    sec5 = {}
    for src, s in meta['cross_layer_sums'].items():
        gt, union = max(s['match_gt_tank'], 1), max(s['match_union_tank'], 1)
        sec5[src] = dict(tank_gt=s['match_gt_tank'], union_pairs=s['match_union_tank'],
                         box_positive_class_background_per_gt=r3((s['match_union_tank'] - s['match_gt_tank']) / gt),
                         background_run2_share_of_union=r3(s['match_clsbg_run2_tank'] / union),
                         background_all4_share_of_union=r3(s['match_clsbg_all_tank'] / union),
                         same_query_all_decoder_layers=r3(s['match_same_dec_all_tank'] / gt),
                         same_query_consecutive={f'dec{k}-dec{k + 1}': r3(s[f'match_same_dec_{k}_{k + 1}_tank'] / gt) for k in (1, 2, 3)},
                         matched_iou={st: r3(s[f'match_iou_{st}_tank'] / gt) for st in STAGES})
    if args.dynamics and (args.dynamics / 'report.json').is_file():
        dyn = json.loads((args.dynamics / 'report.json').read_text())
        sec5['ema_replay_over_epochs'] = sorted(({'epoch': v['epoch'], **{src: dict(
            matched_iou_dec4=v[src]['matched_iou']['dec4']['mean'], iou75_dec4=v[src]['matched_iou']['dec4']['ge75'],
            same_query_all_decoder_layers=v[src]['same_query_all_decoder_layers'],
            box_positive_class_background_per_gt=v[src]['boxpos_clsbg_per_gt']['dec4'],
            background_run2_share_of_union=v[src]['clsbg_run2_per_union']) for src in TANK_SOURCES}} for v in dyn.values()), key=lambda r: r['epoch'])
    if args.train_log:
        rows = [json.loads(s) for s in open(Path(meta['run']) / 'log.txt') if s.strip()]
        rows = [r for r in rows if 'train_match_gt_tank' in r]
        sec5['training_batches'] = [dict(epoch=r['epoch'],
                                         matched_iou_dec4=r3(r['train_match_iou_dec4_tank'] / r['train_match_gt_tank']),
                                         iou75_dec4=r3(r['train_match_iou75_dec4_tank'] / r['train_match_gt_tank']),
                                         same_query_all_decoder_layers=r3(r['train_match_same_dec_all_tank'] / r['train_match_gt_tank']),
                                         box_positive_class_background_per_gt=r3((r['train_match_union_tank'] - r['train_match_gt_tank']) / r['train_match_gt_tank']),
                                         background_run2_share_of_union=r3(r['train_match_clsbg_run2_tank'] / r['train_match_union_tank'])) for r in rows]
    report['5_cross_layer'] = sec5
    (out / 'report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print('report', out / 'report.json', flush=True)


def giou(a, b):
    """GIoU of two xyxy boxes."""
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(x2 - x1, 0) * max(y2 - y1, 0)
    area_a, area_b = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    hull = (max(a[2], b[2]) - min(a[0], b[0])) * (max(a[3], b[3]) - min(a[1], b[1]))
    return inter / max(union, 1e-12) - (hull - union) / max(hull, 1e-12)


# ---------------------------------------------------------------- compare
def compare(args):
    base, run = (json.loads((p / 'report.json').read_text()) for p in (args.base, args.run_out))
    A, B = args.labels
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    lines = [f'# {B} 与 {A} 对照', '',
             f'- {A}：`{base["meta"]["checkpoint"]}`（第 {base["meta"]["epoch"]} 轮，matcher {base["meta"]["matcher"]}）',
             f'- {B}：`{run["meta"]["checkpoint"]}`（第 {run["meta"]["epoch"]} 轮，matcher {run["meta"]["matcher"]}）',
             f'- 两者都用自己的 EMA 权重与匹配器，val 图（{base["meta"]["images"]} / {run["meta"]["images"]} 张，tank 图 {base["meta"]["tank_images"]} / {run["meta"]["tank_images"]} 张）、GT 关联规则和候选数相同。', '']

    def delta(a, b):
        return '' if a is None or b is None else f'{b - a:+.3f}' if abs(b - a) < 10 and isinstance(a, float) and max(abs(a), abs(b)) <= 1.5 else f'{b - a:+.2f}'

    def table(title, header, rows):
        lines.extend([f'## {title}', '', '| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(['---'] + ['---:'] * (len(header) - 1)) + ' |'])
        lines.extend('| ' + ' | '.join(str(c) if c is not None else '—' for c in r) + ' |' for r in rows)
        lines.append('')

    o1, o2 = base['1_overall'], run['1_overall']
    table('1. 总体（训练日志，最佳轮）', ['指标', A, B, 'Δ'], [[k, o1['log'][k], o2['log'][k], delta(o1['log'][k], o2['log'][k])] for k in COCO_NAMES])
    lines.append(f'重算（本脚本 COCOeval）：{A} AP {o1["recomputed"]["AP"]}，{B} AP {o2["recomputed"]["AP"]}；与日志的差异来自评测实现细节。\n')
    rows = []
    for g in [x for x in GROUP_ORDER + ['all'] if x in base['2_source_class']]:
        for c, v in base['2_source_class'][g].items():
            if c == 'mean':
                continue
            w = run['2_source_class'].get(g, {}).get(c)
            if w:
                rows.append([g, c, v['AP'], w['AP'], delta(v['AP'], w['AP']), v['AP75'], w['AP75'], delta(v['AP75'], w['AP75'])])
    table('2. 来源 × 类别', ['分组', '类别', f'{A} AP', f'{B} AP', 'ΔAP', f'{A} AP75', f'{B} AP75', 'ΔAP75'], rows)
    for src in TANK_SOURCES:
        s1, s2 = base['3_matching_quality'][src], run['3_matching_quality'][src]
        table(f'3. 匹配质量（{src} tank，{s1["tank_gt"]} 个 GT）', ['阶段', f'{A} IoU', f'{B} IoU', f'{A} ≥.75', f'{B} ≥.75', f'{A} 选中最佳框', f'{B} 选中最佳框', f'{A} 分类代价', f'{B} 分类代价', f'{A} GIoU 代价', f'{B} GIoU 代价'],
              [[s, s1['stages'][s]['iou_mean'], s2['stages'][s]['iou_mean'], s1['stages'][s]['iou_ge75'], s2['stages'][s]['iou_ge75'],
                s1['stages'][s]['matched_is_best_iou_query'], s2['stages'][s]['matched_is_best_iou_query'],
                s1['stages'][s]['cost_median_unweighted']['cls'], s2['stages'][s]['cost_median_unweighted']['cls'],
                s1['stages'][s]['cost_median_unweighted']['giou'], s2['stages'][s]['cost_median_unweighted']['giou']] for s in STAGES])
        table(f'3. 匹配变化率（{src} tank）', ['指标', A, B, 'Δ'],
              [[k, s1['query_change_rate'][k], s2['query_change_rate'][k], delta(s1['query_change_rate'][k], s2['query_change_rate'][k])] for k in s1['query_change_rate']]
              + [['4 层同一 query', s1['same_query_all_decoder_layers'], s2['same_query_all_decoder_layers'], delta(s1['same_query_all_decoder_layers'], s2['same_query_all_decoder_layers'])]])
    for src in TANK_SOURCES:
        c1, c2 = base['4_query_competition'][src], run['4_query_competition'][src]
        keys = [('A 比例', lambda c: c['classes_top_query']['A']), ('B 比例', lambda c: c['classes_top_query']['B']),
                ('C 比例', lambda c: c['classes_top_query']['C']), ('D 比例', lambda c: c['classes_top_query']['D']),
                ('第一名被另一 query 超过 ≥ .05 IoU', lambda c: c['top_beaten_by_005']), ('第一名 < .75 而最佳 ≥ .75', lambda c: c['top_below_075_best_above']),
                ('分数差（第一名 − 最佳）', lambda c: c['pair_gap_median']['score_top_minus_best']), ('IoU 差（第一名 − 最佳）', lambda c: c['pair_gap_median']['iou_top_minus_best']),
                ('第一名 IoU 中位数', lambda c: c['top_iou_median']), ('可得最佳 IoU 中位数', lambda c: c['best_available_iou_median'])]
        table(f'4. Query 竞争（{src} tank，{c1["gts"]} / {c2["gts"]} 个 GT）', ['指标', A, B, 'Δ'], [[n, f(c1), f(c2), delta(f(c1), f(c2))] for n, f in keys])
    for src in TANK_SOURCES:
        x1, x2 = base['5_cross_layer'][src], run['5_cross_layer'][src]
        keys = ['box_positive_class_background_per_gt', 'background_run2_share_of_union', 'background_all4_share_of_union', 'same_query_all_decoder_layers']
        table(f'5. 跨层监督（{src} tank，EMA 重放）', ['指标', A, B, 'Δ'], [[k, x1[k], x2[k], delta(x1[k], x2[k])] for k in keys])
    for src in TANK_SOURCES:
        l1, l2 = base['6_decoder_layers'][src], run['6_decoder_layers'][src]
        table(f'6. Decoder 逐层（{src} tank，{l1["queries"]} / {l2["queries"]} 个候选）', ['阶段', f'{A} 分类–IoU ρ', f'{B} 分类–IoU ρ', f'{A} 区间', f'{B} 区间', f'{A} IoU 均值', f'{B} IoU 均值', f'{A} 分数中位数', f'{B} 分数中位数'],
              [[s, l1['layers'][s]['spearman_cls_iou'], l2['layers'][s]['spearman_cls_iou'], l1['layers'][s]['spearman_cls_iou_ci'], l2['layers'][s]['spearman_cls_iou_ci'],
                l1['layers'][s]['iou_mean'], l2['layers'][s]['iou_mean'], l1['layers'][s]['score_median'], l2['layers'][s]['score_median']] for s in TRACK])
        table(f'6. 层间变化（{src} tank）', ['步', f'{A} ρ(Δ分类, ΔIoU)', f'{B} ρ(Δ分类, ΔIoU)', f'{A} ΔIoU 中位数', f'{B} ΔIoU 中位数'],
              [[k, l1['steps'][k]['spearman_dcls_diou'], l2['steps'][k]['spearman_dcls_diou'], l1['steps'][k]['median_diou'], l2['steps'][k]['median_diou']] for k in l1['steps']])
    figures = plot(base, run, A, B, out)
    lines += ['## 图', ''] + [f'![{f.stem}]({f.name})' for f in figures]
    (out / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('written', out / 'comparison.md', flush=True)


def plot(base, run, A, B, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    surface, ink, muted, grid = '#fcfcfb', '#0b0b0b', '#52514e', '#e6e5e1'
    colors = {A: '#2a78d6', B: '#eb6834'}  # categorical slots 1 and 2 of the reference palette, fixed order
    files = []

    def style(ax, title, ylabel):
        ax.set_facecolor(surface)
        ax.set_title(title, fontsize=10, color=ink, loc='left')
        ax.set_ylabel(ylabel, color=ink, fontsize=9)
        ax.grid(color=grid, linewidth=.6)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color('#c9c8c2')
        ax.tick_params(colors=muted, labelsize=8.5)

    # Per-layer score-IoU correlation and IoU, both sources.
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), facecolor=surface)
    for col, src in enumerate(TANK_SOURCES):
        for label, rep in ((A, base), (B, run)):
            layers = rep['6_decoder_layers'][src]['layers']
            rho = [layers[s]['spearman_cls_iou'] for s in TRACK]
            lo = [layers[s]['spearman_cls_iou_ci'][0] for s in TRACK]
            hi = [layers[s]['spearman_cls_iou_ci'][1] for s in TRACK]
            axes[0, col].plot(TRACK, rho, marker='o', markersize=5, linewidth=2, color=colors[label], label=label)
            axes[0, col].fill_between(TRACK, lo, hi, color=colors[label], alpha=.15, linewidth=0)
            axes[1, col].plot(STAGES, [rep['3_matching_quality'][src]['stages'][s]['iou_mean'] for s in STAGES], marker='o', markersize=5, linewidth=2, color=colors[label], label=label)
        style(axes[0, col], f'{src.upper()} tank: class logit vs IoU (Spearman, 95% CI)', 'Spearman')
        style(axes[1, col], f'{src.upper()} tank: matched-query IoU per stage', 'mean IoU')
    axes[0, 0].legend(frameon=False, fontsize=9, labelcolor=ink)
    fig.tight_layout()
    path = out / 'layers_and_matching.png'
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)
    files.append(path)

    # Source x class AP change.
    cells = []
    for g in GROUP_ORDER:
        for c, v in base['2_source_class'].get(g, {}).items():
            w = run['2_source_class'].get(g, {}).get(c)
            if c != 'mean' and w and v['AP'] is not None and w['AP'] is not None:
                cells.append((f'{g} {c}', w['AP'] - v['AP']))
    if cells:
        fig, ax = plt.subplots(figsize=(8, .32 * len(cells) + 1), facecolor=surface)
        names, values = zip(*cells)
        ax.barh(names, values, color=[colors[B] if v >= 0 else colors[A] for v in values], height=.6)
        ax.axvline(0, color=muted, linewidth=.8)
        for y, v in enumerate(values):
            ax.text(v, y, f' {v:+.2f}', va='center', ha='left' if v >= 0 else 'right', fontsize=8, color=ink)
        ax.invert_yaxis()
        style(ax, f'AP change by source and class ({B} − {A}); blue = lower, orange = higher', '')
        fig.tight_layout()
        path = out / 'source_class_delta.png'
        fig.savefig(path, dpi=150, facecolor=surface)
        plt.close(fig)
        files.append(path)

    # Training dynamics, when both replays exist.
    if 'ema_replay_over_epochs' in base['5_cross_layer'] and 'ema_replay_over_epochs' in run['5_cross_layer']:
        keys = [('matched_iou_dec4', 'matched IoU (dec4)'), ('same_query_all_decoder_layers', 'same query, 4 decoder layers'),
                ('box_positive_class_background_per_gt', 'box+ / class-bg queries per GT')]
        fig, axes = plt.subplots(len(TANK_SOURCES), len(keys), figsize=(13, 6.5), facecolor=surface)
        for r, src in enumerate(TANK_SOURCES):
            for c, (k, title) in enumerate(keys):
                for label, rep in ((A, base), (B, run)):
                    rows = rep['5_cross_layer']['ema_replay_over_epochs']
                    axes[r, c].plot([x['epoch'] for x in rows], [x[src][k] for x in rows], marker='o', markersize=4, linewidth=2, color=colors[label], label=label)
                style(axes[r, c], f'{src.upper()}: {title}', '')
                axes[r, c].set_xlabel('epoch (EMA replay on fixed val tank images)', color=ink, fontsize=8.5)
        axes[0, 0].legend(frameon=False, fontsize=9, labelcolor=ink)
        fig.tight_layout()
        path = out / 'training_dynamics.png'
        fig.savefig(path, dpi=150, facecolor=surface)
        plt.close(fig)
        files.append(path)
    return files


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('collect')
    p.add_argument('run', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--batch', type=int, default=16)
    p = sub.add_parser('analyse')
    p.add_argument('out', type=Path)
    p.add_argument('--dynamics', type=Path)
    p.add_argument('--train-log', action='store_true')
    p = sub.add_parser('compare')
    p.add_argument('base', type=Path)
    p.add_argument('run_out', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--labels', nargs=2, default=['E000', 'E1'])
    args = parser.parse_args()
    {'collect': collect, 'analyse': analyse, 'compare': compare}[args.command](args)


if __name__ == '__main__':
    main()
