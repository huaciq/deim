"""Experiments A and B on E000 (val tank images of OGSOD and MSAR; weights unchanged, no training).

A  matching  Counterfactual matching cost. Model outputs, GTs and candidates fixed; only the class-cost weight of the
             full Hungarian match (all GTs of every class, as in training) changes: 2 (E000), 1, 0.5, 0; L1 5, GIoU 2.
B  gradients Gradients of the training losses with respect to each decoder layer's query features q (the layer output
             that feeds score_head[k] and bbox_head[k]; the classification path also passes through LQE):
               L_cls = MAL of layer k with layer k's own one-to-one indices (DEIMCriterion.loss_labels_mal),
               L_box = 5 L1 + 2 GIoU with the union indices of all stages (DEIMCriterion.loss_boxes, use_uni_set).
             FGL/DDF are not included. Losses are un-normalised (num_boxes = 1) so per-query magnitudes compare across images.
             Decoder dropout is 0 in E000, so eval mode gives the same decoder computation as training.

Stage outputs, the matcher and the union set are built exactly as in tank_matcher_audit.py. Differences from training:
EMA weights, no augmentation, no denoising queries, val images.

Commands:
  matching  RUN OUT --tracking DIR --device cuda:0
  gradients RUN OUT --tracking DIR --device cuda:0
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from error_gallery import ROOT, iou_matrix  # noqa: E402
from tank_feature_diagnosis import SOURCES, TANK, load_model, load_val, to_input, xyxy  # noqa: E402
from tank_matcher_audit import STAGES, pairs_from_tracking  # noqa: E402

WEIGHTS = (2., 1., .5, 0.)
BOOT = 1000


class Runner:
    """Per-stage outputs of one image, same hooks as tank_matcher_audit.replay."""

    def __init__(self, run, device, checkpoint=None):
        """checkpoint: optional .pth of this run; its EMA weights replace best_stg2."""
        import torch
        from engine.core import YAMLConfig
        self.torch = torch
        cfg = YAMLConfig(str(run / 'resolved_config.yml'))
        self.criterion = cfg.criterion
        self.matcher = self.criterion.matcher
        if checkpoint is None:
            self.model, _ = load_model(run, device)
        else:
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            self.model = cfg.model
            self.model.load_state_dict(state['ema']['module'])
            self.model = self.model.to(device).eval()
            self.epoch = state.get('last_epoch')
        self.device = device
        decoder = self.model.decoder
        self.inner = decoder.decoder
        self.default_idx = self.inner.eval_idx
        self.saved = {}
        decoder.enc_score_head.register_forward_hook(lambda _, __, o: self.saved.__setitem__('enc_logits', o))
        decoder.enc_bbox_head.register_forward_hook(lambda _, __, o: self.saved.__setitem__('enc_box_delta', o))
        self.inner.register_forward_hook(lambda _, __, o: self.saved.__setitem__('pre', (o[4], o[5])))
        for k, layer in enumerate(self.inner.layers):
            layer.register_forward_hook(lambda _, __, o, k=k: self.saved.__setitem__(f'q{k}', o))
        original = decoder._select_topk

        def select(memory, logits, anchors, topk):
            result = original(memory, logits, anchors, topk)
            self.saved['topk_ind'] = torch.topk(logits.max(-1).values, topk, dim=-1)[1].detach()
            self.saved['topk_anchors'] = result[2].detach()
            return result

        decoder._select_topk = select

    def targets(self, image, anns):
        torch = self.torch
        w, h = image['width'], image['height']
        xy = np.array([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]] for a in anns], np.float32) / np.array([w, h, w, h], np.float32)
        cxcywh = np.stack([(xy[:, 0] + xy[:, 2]) / 2, (xy[:, 1] + xy[:, 3]) / 2, xy[:, 2] - xy[:, 0], xy[:, 3] - xy[:, 1]], 1)
        labels = torch.tensor([a['category_id'] for a in anns], device=self.device)
        return [dict(labels=labels, boxes=torch.tensor(cxcywh, device=self.device))], xy

    def stages(self, x):
        """Detached outputs of every stage and the per-stage indices and union set with the original matcher."""
        torch = self.torch
        out = {}
        with torch.no_grad():
            for k in range(self.inner.num_layers):
                self.inner.eval_idx = k
                result = self.model(x)
                out[f'dec{k + 1}'] = (result['pred_logits'][0].detach(), result['pred_boxes'][0].detach())
                if k == 0:
                    ind = self.saved['topk_ind'][0]
                    out['enc'] = (self.saved['enc_logits'][0][ind].detach(), torch.sigmoid(self.saved['enc_box_delta'] + self.saved['topk_anchors'])[0].detach())
                    pre_boxes, pre_logits = self.saved['pre']
                    out['pre'] = (pre_logits[0].detach(), pre_boxes[0].detach())
        self.inner.eval_idx = self.default_idx
        return out


def cost_terms(logits, boxes, labels, gt_boxes, matcher):
    """Unweighted cost terms of HungarianMatcher (class term before cost_class)."""
    import torch
    from engine.deim.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
    prob = torch.sigmoid(logits.float())[:, labels]
    neg = (1 - matcher.alpha) * (prob ** matcher.gamma) * (-(1 - prob + 1e-8).log())
    pos = matcher.alpha * ((1 - prob) ** matcher.gamma) * (-(prob + 1e-8).log())
    l1 = torch.cdist(boxes.float(), gt_boxes.float(), p=1)
    giou = -generalized_box_iou(box_cxcywh_to_xyxy(boxes.float()), box_cxcywh_to_xyxy(gt_boxes.float()))
    return (pos - neg).cpu().numpy(), l1.cpu().numpy(), giou.cpu().numpy()


def images_of(annotations, source, grouped):
    return [i for i in annotations['images'] if source[i['id']] in SOURCES and any(a['category_id'] == TANK for a in grouped[i['id']])]


def matching(args):
    from scipy.optimize import linear_sum_assignment
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    annotations, source, grouped = load_val(out)
    pairs = defaultdict(list)
    for p in pairs_from_tracking(args.tracking, source):
        pairs[p['image_id']].append(p)
    runner = Runner(args.run, args.device)
    rows, gt_rows, checks = [], [], Counter()
    for image in images_of(annotations, source, grouped):
        anns = grouped[image['id']]
        x, _ = to_input(ROOT / 'JPEGImages/val' / image['file_name'])
        targets, xy = runner.targets(image, anns)
        outs = runner.stages(x[None].to(args.device))
        tank = [n for n, a in enumerate(anns) if a['category_id'] == TANK and a['bbox'][2] > 0 and a['bbox'][3] > 0]
        for s in STAGES:
            logits, boxes = outs[s]
            cls, l1, giou = cost_terms(logits, boxes, targets[0]['labels'], targets[0]['boxes'], runner.matcher)
            iou = iou_matrix(xyxy(boxes.cpu().numpy(), 1), xy)
            prob = 1 / (1 + np.exp(-logits[:, TANK].float().cpu().numpy()))
            assign = {}
            for w in WEIGHTS:
                C = np.nan_to_num(w * cls + runner.matcher.cost_bbox * l1 + runner.matcher.cost_giou * giou, nan=1.0)
                r_, c_ = linear_sum_assignment(C)
                assign[w] = dict(zip(c_.tolist(), r_.tolist()))  # GT -> query
            ref = runner.matcher({'pred_logits': logits[None], 'pred_boxes': boxes[None]}, targets)['indices'][0]
            checks['stages'] += 1
            checks['baseline_identical_to_matcher'] += int(dict(zip(ref[1].tolist(), ref[0].tolist())) == assign[2.])
            non_tank = [n for n in range(len(anns)) if n not in tank]
            for w in WEIGHTS[1:]:
                checks[f'non_tank_gt_rematched_w{w}'] += sum(assign[w][n] != assign[2.][n] for n in non_tank)
                checks['non_tank_gt'] += len(non_tank) if w == WEIGHTS[1] else 0
            for t in tank:
                best = int(iou[:, t].argmax())
                gt_rows.append(dict(source=source[image['id']], stage=s, image_id=image['id'], best_iou=float(iou[best, t]),
                                    **{f'w{w}': dict(query=assign[w][t], iou=float(iou[assign[w][t], t]), is_best=assign[w][t] == best, prob=float(prob[assign[w][t]]),
                                                     cls=float(cls[assign[w][t], t]), l1=float(l1[assign[w][t], t]), giou=float(giou[assign[w][t], t])) for w in WEIGHTS}))
            for p in pairs.get(image['id'], []):
                t = tank[p['tank_gt']]
                rows.append(dict(stage=s, image_id=image['id'], **{f'w{w}': 'top' if assign[w][t] == p['top'] else 'better' if assign[w][t] == p['better'] else 'other' for w in WEIGHTS}))
    json.dump(gt_rows, open(out / 'matching_gt_rows.json', 'w'))
    med = lambda v: round(float(np.median(v)), 3) if len(v) else None
    report = dict(checks=dict(checks), weights=list(WEIGHTS))
    for src in SOURCES:
        rs = {}
        for s in STAGES:
            G = [g for g in gt_rows if g['source'] == src and g['stage'] == s]
            P = [r for r in rows if r['stage'] == s] if src == 'ogsod' else []
            stage = dict(gt=len(G), pairs=len(P), best_available_iou_median=med([g['best_iou'] for g in G]))
            for w in WEIGHTS:
                k = f'w{w}'
                base = [g['w2.0'] for g in G]
                cur = [g[k] for g in G]
                stage[k] = dict(
                    pairs_matched_better=round(float(np.mean([r[k] == 'better' for r in P])), 3) if P else None,
                    pairs_matched_top=round(float(np.mean([r[k] == 'top' for r in P])), 3) if P else None,
                    matched_iou_median=med([c['iou'] for c in cur]), matched_iou_ge_075=round(float(np.mean([c['iou'] >= .75 for c in cur])), 3),
                    matched_iou_ge_05=round(float(np.mean([c['iou'] >= .5 for c in cur])), 3),
                    matched_is_best_box=round(float(np.mean([c['is_best'] for c in cur])), 3),
                    matched_tank_prob_median=med([c['prob'] for c in cur]),
                    matched_cost_median=dict(cls=med([c['cls'] for c in cur]), l1=med([c['l1'] for c in cur]), giou=med([c['giou'] for c in cur])),
                    changed_from_baseline=round(float(np.mean([c['query'] != b['query'] for c, b in zip(cur, base)])), 3),
                    worse_than_baseline_by_005=round(float(np.mean([c['iou'] < b['iou'] - .05 for c, b in zip(cur, base)])), 3),
                    baseline_best_box_lost=int(sum(b['is_best'] and not c['is_best'] for c, b in zip(cur, base))),
                    baseline_iou50_lost=int(sum(b['iou'] >= .5 > c['iou'] for c, b in zip(cur, base))),
                    baseline_iou50_gained=int(sum(c['iou'] >= .5 > b['iou'] for c, b in zip(cur, base))))
            rs[s] = stage
        report[src] = rs
    (out / 'matching_report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({src: {s: report[src][s] for s in ('enc', 'dec2', 'dec4')} for src in SOURCES}, indent=1, ensure_ascii=False))
    print('checks', report['checks'])


def gradients(args):
    import torch
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    annotations, source, grouped = load_val(out)
    better = defaultdict(set)
    for p in pairs_from_tracking(args.tracking, source):
        better[p['image_id']].add(p['better'])
    runner = Runner(args.run, args.device)
    crit = runner.criterion
    rows = []
    for image in images_of(annotations, source, grouped):
        anns = grouped[image['id']]
        x, _ = to_input(ROOT / 'JPEGImages/val' / image['file_name'])
        x = x[None].to(args.device)
        targets, _ = runner.targets(image, anns)
        outs = runner.stages(x)
        indices = {s: runner.matcher({'pred_logits': outs[s][0][None], 'pred_boxes': outs[s][1][None]}, targets)['indices'] for s in STAGES}
        go = crit._get_go_indices(indices['dec4'], [indices[s] for s in ('dec1', 'dec2', 'dec3', 'pre', 'enc')])
        go_q, go_t = go[0][0].tolist(), go[0][1].tolist()
        labels = targets[0]['labels'].tolist()
        for k in range(runner.inner.num_layers):
            runner.inner.eval_idx = k
            with torch.enable_grad():
                result = runner.model(x)
                q = runner.saved[f'q{k}']
                outputs = {'pred_logits': result['pred_logits'], 'pred_boxes': result['pred_boxes']}
                l_cls = crit.loss_labels_mal(outputs, targets, indices[f'dec{k + 1}'], 1.)['loss_mal'] * crit.weight_dict['loss_mal']
                boxes = crit.loss_boxes(outputs, targets, go, 1.)
                l_box = boxes['loss_bbox'] * crit.weight_dict['loss_bbox'] + boxes['loss_giou'] * crit.weight_dict['loss_giou']
                g_cls = torch.autograd.grad(l_cls, q, retain_graph=True)[0][0]
                g_box = torch.autograd.grad(l_box, q)[0][0]
            runner.inner.eval_idx = runner.default_idx
            matched = set(indices[f'dec{k + 1}'][0][0].tolist())
            for query, gt in zip(go_q, go_t):
                if labels[gt] != TANK:
                    continue
                a, b = g_cls[query].float(), g_box[query].float()
                na, nb = float(a.norm()), float(b.norm())
                rows.append(dict(source=source[image['id']], image_id=image['id'], layer=k + 1, query=query,
                                 group='cls positive + box positive' if query in matched else 'cls background + box positive',
                                 better_query=query in better[image['id']],
                                 cos=float((a @ b) / max(na * nb, 1e-12)), norm_cls=na, norm_box=nb))
    json.dump(rows, open(out / 'gradient_rows.json', 'w'))
    report = {}
    rng = np.random.default_rng(0)
    for src in SOURCES:
        rs = {}
        for layer in range(1, runner.inner.num_layers + 1):
            for group in ('cls positive + box positive', 'cls background + box positive'):
                for only_better in (False, True):
                    R = [r for r in rows if r['source'] == src and r['layer'] == layer and r['group'] == group and (r['better_query'] or not only_better)]
                    if not R:
                        continue
                    cos = np.array([r['cos'] for r in R])
                    na, nb = np.array([r['norm_cls'] for r in R]), np.array([r['norm_box'] for r in R])
                    img = np.array([r['image_id'] for r in R])
                    keys = np.unique(img)
                    members = {k: np.flatnonzero(img == k) for k in keys}
                    boots = []
                    for _ in range(BOOT):
                        take = np.concatenate([members[k] for k in rng.choice(keys, len(keys))])
                        boots.append(((cos[take] < 0).mean(), np.median(cos[take])))
                    boots = np.array(boots)
                    rs[f'layer{layer} | {group}' + (' | 480-pair better query' if only_better else '')] = dict(
                        n=len(R), images=len(keys), cos_median=round(float(np.median(cos)), 3),
                        cos_median_ci=np.percentile(boots[:, 1], [2.5, 97.5]).round(3).tolist(),
                        share_cos_negative=round(float((cos < 0).mean()), 3), share_cos_negative_ci=np.percentile(boots[:, 0], [2.5, 97.5]).round(3).tolist(),
                        share_cos_below_minus_03=round(float((cos < -.3).mean()), 3),
                        norm_cls_median=round(float(np.median(na)), 4), norm_box_median=round(float(np.median(nb)), 4),
                        cls_over_box_median=round(float(np.median(na / np.maximum(nb, 1e-12))), 3),
                        strong_conflict=round(float(((cos < -.3) & (na >= .5 * nb)).mean()), 3))
        report[src] = rs
    (out / 'gradient_report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('matching', 'gradients'))
    parser.add_argument('run', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--tracking', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    {'matching': matching, 'gradients': gradients}[args.command](args)


if __name__ == '__main__':
    main()
