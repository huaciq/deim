"""D10.2: can a quality head fed with frozen features predict the absolute localisation quality of a detection beyond
what the classification score already tells? Gate before implementing and training BCLQ. No training of the detector,
no change to it.

Every query whose highest class score is >= .02 (true positives, duplicates, badly localised and background boxes)
gets, from one CPU forward pass of the EMA checkpoint:
  z      6 final class logits, their maximum and its sigmoid                               (8)
  geo    box geometry cx, cy, w, h, log w, log h and the degree-2 monomials of (cx, cy, log w, log h)   (16)
  S5     decoder query feature of the last layer (input of the score head)                 (256)
  S4_h1  first hidden layer of the last box head                                           (256)
  roi    box-conditioned features of the encoder outputs at stride 8 and 16; per level: mean and max of 3x3 bilinear
         samples inside the box, mean of 8 samples on the 1.5x box (corners and edge midpoints), and the 3x3 inside
         samples kept in place after a fixed random projection 256 -> 24                   (2 x 984)
  target t = max IoU of the box with any GT of the image (class-agnostic quality, 0 without overlap)

A small MLP (2 x 256, BCE with the soft target t) is fitted on TRAIN images and applied to VAL images; a head fitted
on val would count val labelling habits as gain (see the BAFR lesson). Secondary: 5-fold by image inside val.
Fusion at inference (BCLQ-v2): z_final[i, c] = z[i, c] + beta * logit(clip(Q[i], .02, .98)) for every class c; queries
below the .02 score threshold get Q = .02. Evaluation: COCO rule of e1_followup on all val.

Feature sets: C0 = z + geo (control: what re-calibrating the score with geometry gives); Q = C0 + S5;
QH = C0 + S5 + S4_h1; ROI = C0 + roi; BOX = C0 + S5 + roi (the BCLQ input); ALL = everything.

Decision fixed before the run (docs/sardet100k_bclq_design_review.md): primary = train -> val, beta = 1. If the best
of Q / QH / ROI / BOX / ALL beats the control C0 by >= 0.5 AP with AP50 not more than 0.5 below the unfused baseline,
the branch is implemented and trained once, with the inputs that carry the gain; otherwise it is not trained.

Usage (server, CPU; OUT directly under E000's independent_verify_20261106/):
  extract RUN OUT --checkpoint PATH --split val|train [--max-images 12000] [--threads 16]
  fit OUT --train TRAIN_DIR        heads fitted on train (and 5-fold inside val), val predictions saved
  fuse OUT [--workers 13] [--boot 200]   fused AP on all val (no torch in this step)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from box_evidence_probe import box_points, mesh, sample  # noqa: E402
from e1_report import iou_xyxy, r3, sigmoid  # noqa: E402

THRESHOLD = .02
SETS = {'C0': ['z', 'geo'], 'Q': ['z', 'geo', 'S5'], 'QH': ['z', 'geo', 'S5', 'S4_h1'], 'ROI': ['z', 'geo', 'roi'],
        'BOX': ['z', 'geo', 'S5', 'roi'], 'ALL': ['z', 'geo', 'S5', 'S4_h1', 'roi']}
BETAS = [.5, 1., 2.]
CELLS = ['ogsod/tank', 'ogsod/bridge', 'ogsod/harbor', 'other_512/aircraft', 'other_512/car', 'other_256/ship',
         'other_800/ship', 'msar/ship', 'msar/tank']
_PROJ = np.random.default_rng(0).normal(size=(256, 24)).astype(np.float32) / np.sqrt(24)


def roi_features(fmap, box):
    """fmap tensor (256, h, w); box (N, 4) normalised xyxy -> (N, 984)."""
    u, v = mesh((np.arange(3) + .5) / 3)
    inside = sample(fmap, box_points(box, u, v))                                   # (N, 9, 256)
    ru = np.array([-.25, .5, 1.25, -.25, 1.25, -.25, .5, 1.25])
    rv = np.array([-.25, -.25, -.25, .5, .5, 1.25, 1.25, 1.25])
    ring = sample(fmap, box_points(box, ru, rv)).mean(1)
    return np.concatenate([inside.mean(1), inside.max(1), ring, (inside @ _PROJ).reshape(len(box), -1)], 1)


# ---------------------------------------------------------------- extract (server)
def extract(args):
    import torch
    from collections import defaultdict
    from error_gallery import ROOT
    from tank_feature_diagnosis import to_input, xyxy
    from tank_matching_counterfactual import Runner
    torch.set_num_threads(args.threads)
    annotations = json.load(open(ROOT / f'Annotations/{args.split}.json'))
    grouped = defaultdict(list)
    for a in annotations['annotations']:
        grouped[a['image_id']].append(a)
    images = annotations['images']
    if args.max_images and args.max_images < len(images):
        pick = np.sort(np.random.default_rng(0).choice(len(images), args.max_images, replace=False))
        images = [images[k] for k in pick]
    runner = Runner(args.run, 'cpu', args.checkpoint)
    model, inner = runner.model, runner.inner
    last = runner.default_idx
    grab = {}
    inner.lqe_layers[last].register_forward_hook(lambda m, inp, out: grab.update(final=out.detach()))
    bbox_head = model.decoder.dec_bbox_head[last]
    bbox_head.layers[0].register_forward_hook(lambda m, inp, out: grab.update(h1=bbox_head.act(out).detach()))
    inner.layers[last].register_forward_hook(lambda m, inp, out: grab.update(query=out.detach()))
    model.encoder.register_forward_hook(lambda m, inp, out: grab.update(enc=[t.detach() for t in (out[0] if isinstance(out, tuple) else out)]))
    rows = defaultdict(list)
    ids, logits_all, boxes_all = [], [], []
    with torch.no_grad():
        for n, image in enumerate(images):
            anns = grouped[image['id']]
            x, _ = to_input(ROOT / 'JPEGImages' / args.split / image['file_name'])
            inner.eval_idx = last
            result = model(x[None])
            boxes_c = result['pred_boxes'][0].float().numpy()
            boxes = xyxy(boxes_c, 1)
            final = grab['final'][0].float().numpy()
            assert np.allclose(final, result['pred_logits'][0].float().numpy(), atol=1e-4)
            if args.split == 'val':
                ids.append(image['id'])
                logits_all.append(final)
                boxes_all.append(boxes)
            sel = np.flatnonzero(sigmoid(final).max(1) >= THRESHOLD)
            if not len(sel):
                continue
            w, h = image['width'], image['height']
            gt = np.array([[a['bbox'][0] / w, a['bbox'][1] / h, (a['bbox'][0] + a['bbox'][2]) / w, (a['bbox'][1] + a['bbox'][3]) / h] for a in anns], np.float64).reshape(-1, 4)
            labels = np.array([a['category_id'] for a in anns], int)
            b = boxes[sel].astype(np.float64)
            if len(gt):
                iou = iou_xyxy(b, gt)
                t, owner = iou.max(1), iou.argmax(1)
                owner_label = labels[owner]
            else:
                t, owner, owner_label = np.zeros(len(sel)), np.full(len(sel), -1), np.full(len(sel), -1)
            g = boxes_c[sel][:, [0, 1]], np.log(np.maximum(boxes_c[sel][:, 2:], 1e-6))
            base = np.concatenate([boxes_c[sel], g[1]], 1)
            q4 = np.concatenate([g[0], g[1]], 1)
            geo = np.concatenate([base] + [q4[:, [i]] * q4[:, [j]] for i in range(4) for j in range(i, 4)], 1)
            z = final[sel]
            rows['z'].append(np.concatenate([z, z.max(1, keepdims=True), sigmoid(z.max(1, keepdims=True))], 1))
            rows['geo'].append(geo)
            rows['S5'].append(grab['query'][0, sel].float().numpy())
            rows['S4_h1'].append(grab['h1'][0, sel].float().numpy())
            rows['roi'].append(np.concatenate([roi_features(grab['enc'][k][0].float(), b) for k in (0, 1)], 1))
            rows['t'].append(t)
            rows['owner'].append(np.where(t > 0, owner, -1))
            rows['owner_label'].append(np.where(t > 0, owner_label, -1))
            rows['image'].append(np.full(len(sel), image['id']))
            rows['query'].append(sel)
            if (n + 1) % 500 == 0:
                print('images', n + 1, len(images), 'rows', sum(len(r) for r in rows['t']), flush=True)
    out = {k: np.concatenate(v).astype(np.float32 if k in ('z', 'geo', 'S5', 'S4_h1', 'roi', 't') else np.int64) for k, v in rows.items()}
    if args.split == 'val':
        np.savez_compressed(args.out / 'queries.npz', image_id=np.array(ids), logits=np.stack(logits_all).astype(np.float32),
                            boxes=np.stack(boxes_all).astype(np.float32))
    meta = dict(run=str(args.run), checkpoint=str(args.checkpoint), epoch=runner.epoch, split=args.split, images=len(images),
                rows=int(len(out['t'])), threshold=THRESHOLD, dims={k: int(out[k].shape[1]) for k in ('z', 'geo', 'S5', 'S4_h1', 'roi')})
    torch.save(dict(meta=meta, data=out), args.out / 'quality_features.pt', pickle_protocol=4)
    print('written', args.out / 'quality_features.pt', meta, flush=True)


# ---------------------------------------------------------------- analyse (server, CPU)
def fit_predict(x_train, t_train, x_test, seed=0, epochs=6):
    import torch
    torch.manual_seed(seed)
    mean, std = x_train.mean(0, keepdims=True), x_train.std(0, keepdims=True) + 1e-6
    xt = torch.from_numpy((x_train - mean) / std)
    yt = torch.from_numpy(t_train.astype(np.float32))
    net = torch.nn.Sequential(torch.nn.Linear(xt.shape[1], 256), torch.nn.ReLU(), torch.nn.Dropout(.1),
                              torch.nn.Linear(256, 256), torch.nn.ReLU(), torch.nn.Linear(256, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = epochs * (len(xt) // 2048 + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps)
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(len(xt))
        for k in range(0, len(xt), 2048):
            idx = perm[k:k + 2048]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(xt[idx])[:, 0], yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(torch.from_numpy((x_test - mean) / std))[:, 0]).numpy()


def stack(data, keys):
    return np.concatenate([data[k] for k in keys], 1)


_G = {}


def _evaluate(job):
    """Worker: fused AP of one (name, beta). Uses globals set before the fork."""
    from e1_followup import ap_arrays, ap_from, detections, evaluate, group_ap
    name, beta = job
    val, data, rows, all_images, id_of = _G['val'], _G['data'], _G['rows'], _G['images'], _G['id_of']
    d = dict(data)
    if name != 'baseline':
        q = np.full(data['logits'].shape[:2], .02, np.float64)
        q[rows[0], rows[1]] = np.clip(_G['pred'][name], .02, .98)
        d['logits'] = data['logits'] + beta * np.log(q / (1 - q))[:, :, None]
    det, npos = evaluate(val, detections(val, d))
    arr = ap_arrays(det, npos, val, all_images)
    mean, per_class = group_ap(arr)
    m = lambda ap: dict(AP=r3(np.mean(ap) * 100), AP50=r3(ap[0] * 100), AP75=r3(ap[5] * 100))  # noqa: E731
    out = dict(m(mean), per_class={val.names[c]: m(v) for c, v in per_class.items()}, cells={})
    for cell in CELLS:
        g, c = cell.split('/')[0], id_of[cell.split('/')[1]]
        a = ap_arrays(det, npos, val, [i for i in all_images if val.group[i] == g])
        out['cells'][cell] = m(ap_from(a[c])[0])
    above = {}
    for kind, label in ((1, 'duplicate'), (3, 'localisation'), (4, 'background')):
        above[label] = 0
        for c in val.classes:
            sel = det['label'] == c
            tp = sel & (det['kind'] == 0)
            if tp.any():
                above[label] += int((sel & (det['kind'] == kind) & (det['score'] > np.median(det['score'][tp]))).sum())
    out['fp_above_median_tp'] = above
    print(name, beta, out['AP'], out['AP50'], out['AP75'], above, flush=True)
    return (name, beta), out, (arr if name == 'baseline' or (beta == 1. and name.endswith('|train')) else None)


def quality_stats(pred, v):
    """How good the prediction is as a quality estimate, on val rows."""
    from e1_report import spearman
    t = v['t']
    near = t >= .1
    out = dict(mae=r3(np.abs(pred - t).mean()), mae_overlapping=r3(np.abs(pred[near] - t[near]).mean()),
               spearman_all=r3(spearman(pred, t)), spearman_overlapping=r3(spearman(pred[near], t[near])))
    # within-GT pairwise accuracy on rows with IoU >= .5 that share image and owner GT
    key = v['image'].astype(np.int64) * 1000 + v['owner']
    good = []
    order = np.argsort(key, kind='stable')
    k, tt, pp = key[order], t[order], pred[order]
    start = 0
    for end in list(np.flatnonzero(np.diff(k)) + 1) + [len(k)]:
        m = tt[start:end] >= .5
        if m.sum() >= 2 and v['owner'][order[start]] >= 0:
            a, b = np.triu_indices(int(m.sum()), 1)
            ti, pi = tt[start:end][m], pp[start:end][m]
            ok = (ti[a] != ti[b])
            if ok.any():
                good.append(((pi[a] > pi[b]) == (ti[a] > ti[b]))[ok].mean())
        start = end
    out['within_gt_pair_accuracy'] = r3(np.mean(good))
    return out


def fit(args):
    """Fit the heads (torch) and save their val predictions; evaluation runs in `fuse`, which never imports torch."""
    import torch
    torch.set_num_threads(args.threads)
    v = torch.load(args.out / 'quality_features.pt', weights_only=False)
    tr = torch.load(args.train / 'quality_features.pt', weights_only=False)
    vd, td = v['data'], tr['data']
    report = dict(meta=dict(val=v['meta'], train=tr['meta']), sets=SETS, quality={})
    pred = {}
    folds = {int(i): k % 5 for k, i in enumerate(np.random.default_rng(0).permutation(np.unique(vd['image'])))}
    fold = np.array([folds[int(i)] for i in vd['image']])
    for name, keys in SETS.items():
        x_val = stack(vd, keys)
        pred[f'{name}|train'] = fit_predict(stack(td, keys), td['t'], x_val)
        oof = np.zeros(len(x_val), np.float32)
        for f in range(5):
            oof[fold == f] = fit_predict(x_val[fold != f], vd['t'][fold != f], x_val[fold == f])
        pred[f'{name}|oof'] = oof
        for mode in ('train', 'oof'):
            report['quality'][f'{name}|{mode}'] = quality_stats(pred[f'{name}|{mode}'], vd)
            print(name, mode, report['quality'][f'{name}|{mode}'], flush=True)
    report['quality']['score_itself'] = quality_stats(vd['z'][:, 6].astype(np.float64), vd)
    print('score_itself', report['quality']['score_itself'], flush=True)
    (args.out / 'quality_fit.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    np.savez_compressed(args.out / 'quality_predictions.npz', image=vd['image'], query=vd['query'], z=vd['z'][:, :6],
                        **{'pred_' + k: p.astype(np.float32) for k, p in pred.items()})
    print('written', args.out / 'quality_predictions.npz', flush=True)


def fuse(args):
    import multiprocessing as mp
    from e1_followup import Val, paired_bootstrap
    with np.load(args.out / 'quality_predictions.npz') as z:
        saved = {k: z[k] for k in z.files}
    pred = {k[5:]: v for k, v in saved.items() if k.startswith('pred_')}
    val = Val(args.out)
    with np.load(args.out / 'queries.npz') as z:
        data = {k: z[k] for k in z.files}
    row_of = {int(i): n for n, i in enumerate(data['image_id'])}
    rows = (np.array([row_of[int(i)] for i in saved['image']]), saved['query'])
    assert np.allclose(data['logits'][rows[0], rows[1]], saved['z'], atol=1e-4)
    _G.update(val=val, data=data, rows=rows, images=sorted(val.info), id_of={n: c for c, n in val.names.items()}, pred=pred)
    jobs = [('baseline', 0.)] + [(k, b) for k in pred for b in BETAS]
    with mp.get_context('fork').Pool(args.workers) as pool:
        results = pool.map(_evaluate, jobs, chunksize=1)
    report = dict(fused={})
    arrs = {}
    for (name, beta), out, arr in results:
        report['fused'][f'{name}|beta{beta:g}' if name != 'baseline' else 'baseline'] = out
        if arr is not None:
            arrs[name] = arr
    n_images = len(_G['images'])
    best = max(('Q', 'QH', 'ROI', 'BOX', 'ALL'), key=lambda k: report['fused'][f'{k}|train|beta1']['AP'])
    boot = lambda a, b: paired_bootstrap(arrs[a], arrs[b], n_images, args.boot)  # noqa: E731
    report['primary'] = dict(best_set=best, vs_control=boot('C0|train', f'{best}|train'), vs_baseline=boot('baseline', f'{best}|train'),
                             control_vs_baseline=boot('baseline', 'C0|train'), box_vs_control=boot('C0|train', 'BOX|train'),
                             box_vs_query_only=boot('Q|train', 'BOX|train'))
    (args.out / 'quality_head_probe.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print('primary', best, report['primary'], flush=True)
    print('written', args.out / 'quality_head_probe.json', flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    e = sub.add_parser('extract')
    e.add_argument('run', type=Path)
    e.add_argument('out', type=Path)
    e.add_argument('--checkpoint', type=Path, required=True)
    e.add_argument('--split', choices=['val', 'train'], required=True)
    e.add_argument('--max-images', type=int, default=0)
    e.add_argument('--threads', type=int, default=16)
    a = sub.add_parser('fit')
    a.add_argument('out', type=Path)
    a.add_argument('--train', type=Path, required=True)
    a.add_argument('--threads', type=int, default=16)
    f = sub.add_parser('fuse')
    f.add_argument('out', type=Path)
    f.add_argument('--workers', type=int, default=13)
    f.add_argument('--boot', type=int, default=200)
    args = parser.parse_args()
    if args.command == 'extract':
        args.out.mkdir(parents=True, exist_ok=True)
    {'extract': extract, 'fit': fit, 'fuse': fuse}[args.command](args)


if __name__ == '__main__':
    main()
