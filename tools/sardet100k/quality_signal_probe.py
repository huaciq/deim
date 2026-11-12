"""D7.1: is localisation quality already present in the network, and where? Pairwise ranking probes. No training of the
detector, no change to it.

For one run's EMA checkpoint, the last decoder layer of the eval model (the layer the postprocessor uses) is read
through forward hooks on the fixed val images of chosen source x class cells. For each GT, its competing queries are
the candidates of D7 (highest-IoU same-class GT is this GT, IoU >= .5); a competing pair has at least one query with
final score >= .1 (sigmoid of the logit the postprocessor ranks). Label: which query of the pair has the higher IoU.

Signals (all from the same forward pass, last decoder layer):
  S0  box geometry (cx, cy, w, h, log w, log h of the predicted box)     control: shortcuts from box size or position
  S1  classification logit of the GT class: raw (score head) and final (raw + LQE, what the postprocessor ranks)
  S2  LQE output (the scalar added to every class logit)
  S3  LQE input: 20 edge statistics (top-4 probabilities + their mean, for 4 edges)
  S4  box regression head hidden features (dec_bbox_head of the last layer): first and second hidden layer (256 each),
      plus its output distribution (4 x (reg_max + 1) logits) for reference
  S5  decoder query feature: output of the last decoder layer (256), the input of the score head
Scalars are compared directly (higher = better). Vectors get a linear pairwise ranker P(i > j) = sigmoid(w^T (f_i - f_j)),
no bias, both orientations, standardised features, L2 penalty chosen by inner grouped CV, trained and tested with
5-fold cross-validation grouped by image (both queries of a pair, and all pairs of an image, stay in one fold). Every GT
has total weight 1, split equally over its pairs. Spearman: per GT over its competing queries (>= 3), using the
out-of-fold ranker score. 95% CI: image bootstrap of the out-of-fold results (1000 resamples). Pairs are also split
by the cosine similarity of the two queries' S5 features (< .90, .90-.98, >= .98).

Usage:
  extract RUN OUT --checkpoint PATH [--threads 16]      (server, CPU) -> OUT/probe_features.pt
  analyse FEATURES.pt [FEATURES.pt ...] --out DIR        (local) -> DIR/probe.json
  joint FEATURES.pt [...] --out DIR [--cells ...]        (local, D7.2) -> DIR/probe_joint.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_report import GROUP_ORDER, group_of, iou_xyxy, r3, sigmoid  # noqa: E402

CELLS = [('ogsod', 'tank'), ('msar', 'tank'), ('ogsod', 'bridge')]
COS_BINS = [(-1.0, .90), (.90, .98), (.98, 1.01)]


# ---------------------------------------------------------------- extract (server)
def extract(args):
    import torch
    import torch.nn.functional as F
    from error_gallery import ROOT
    from tank_feature_diagnosis import load_val, to_input, xyxy
    from tank_matching_counterfactual import Runner
    torch.set_num_threads(args.threads)
    annotations, source, grouped = load_val(args.out)
    name_of = {c['id']: c['name'] for c in annotations['categories']}
    id_of = {v: k for k, v in name_of.items()}
    want = {(g, id_of[c]) for g, c in CELLS}
    if args.all_cells:  # every source x class cell of val
        want = {(g, c) for g in GROUP_ORDER for c in name_of}
    runner = Runner(args.run, 'cpu', args.checkpoint)
    model, inner = runner.model, runner.inner
    last = runner.default_idx
    grab = {}
    lqe = inner.lqe_layers[last]
    lqe.register_forward_hook(lambda m, inp, out: grab.update(raw=inp[0].detach(), corners=inp[1].detach(), final=out.detach()))
    lqe.reg_conf.register_forward_hook(lambda m, inp, out: grab.update(quality=out.detach(), stats=inp[0].detach()))
    bbox_head = model.decoder.dec_bbox_head[last]
    bbox_head.layers[0].register_forward_hook(lambda m, inp, out: grab.update(h1=bbox_head.act(out).detach()))
    bbox_head.layers[1].register_forward_hook(lambda m, inp, out: grab.update(h2=bbox_head.act(out).detach()))
    inner.layers[last].register_forward_hook(lambda m, inp, out: grab.update(query=out.detach()))
    records = []
    images = [(i, group_of(source[i['id']], i['width'], i['height'])) for i in annotations['images']]
    if not args.all_cells:
        images = [(i, g) for i, g in images if any((g, a['category_id']) in want for a in grouped[i['id']])]
    ids, logits_all, boxes_all = [], [], []  # with --all-cells: every val image, same layout as e1_followup's queries.npz
    with torch.no_grad():
        for n, (image, g) in enumerate(images):
            anns = grouped[image['id']]
            x, _ = to_input(ROOT / 'JPEGImages/val' / image['file_name'])
            inner.eval_idx = last
            result = model(x[None])
            boxes_c = result['pred_boxes'][0].float().numpy()
            boxes = xyxy(boxes_c, 1)
            final = grab['final'][0].float().numpy()
            assert np.allclose(final, result['pred_logits'][0].float().numpy(), atol=1e-4)
            ids.append(image['id'])
            logits_all.append(final)
            boxes_all.append(boxes)
            if not anns:
                continue
            _, xy = runner.targets(image, anns)
            raw = grab['raw'][0].float().numpy()
            labels = np.array([a['category_id'] for a in anns])
            for c in np.unique(labels):
                if (g, int(c)) not in want:
                    continue
                idx = np.flatnonzero(labels == c)
                iou = iou_xyxy(boxes, xy[idx])
                owner, best = iou.argmax(1), iou.max(1)
                for j, t in enumerate(idx):
                    cand = np.flatnonzero((owner == j) & (best >= .5))
                    if len(cand) < 2:
                        continue
                    geo = np.concatenate([boxes_c[cand], np.log(np.maximum(boxes_c[cand, 2:], 1e-6))], 1)
                    records.append(dict(
                        image=image['id'], cell=f'{g}/{name_of[int(c)]}', gt=int(t), query=cand,
                        iou=iou[cand, j].astype(np.float32), score=sigmoid(final[cand, c]).astype(np.float32),
                        S0=geo.astype(np.float32), S1_raw=raw[cand, c].astype(np.float32), S1_final=final[cand, c].astype(np.float32),
                        S2=grab['quality'][0, cand, 0].float().numpy(), S3=grab['stats'][0, cand].float().numpy(),
                        S4_h1=grab['h1'][0, cand].float().numpy(), S4_h2=grab['h2'][0, cand].float().numpy(),
                        S4_out=grab['corners'][0, cand].float().numpy(), S5=grab['query'][0, cand].float().numpy()))
            if (n + 1) % 200 == 0:
                print('images', n + 1, len(images), flush=True)
    meta = dict(run=str(args.run), checkpoint=str(args.checkpoint), epoch=runner.epoch, layer=last, images=len(images))
    if args.all_cells:  # the 300 queries of the same forward pass, so that the swap oracle (D7.3) can index them by `query`
        np.savez_compressed(args.out / 'queries.npz', image_id=np.array(ids), logits=np.stack(logits_all).astype(np.float32),
                            boxes=np.stack(boxes_all).astype(np.float32))
    torch.save(dict(meta=meta, records=records), args.out / 'probe_features.pt')
    print('written', args.out / 'probe_features.pt', len(records), flush=True)


# ---------------------------------------------------------------- analyse (local)
SCALARS = ['S1_raw', 'S1_final', 'S2']
VECTORS = ['S0', 'S3', 'S4_h1', 'S4_h2', 'S4_out', 'S5']


def build_pairs(records):
    """Competing pairs: both IoU >= .5 (by construction), at least one final score >= .1, IoU differs."""
    rows = []
    for k, r in enumerate(records):
        n = len(r['iou'])
        if 'S5' in r:
            f = r['S5'] / np.maximum(np.linalg.norm(r['S5'], axis=1, keepdims=True), 1e-12)
            cos = f @ f.T
        else:  # feature files without S5 (D8): cosine bins are not computed
            cos = np.full((n, n), np.nan)
        pairs = [(a, b) for a in range(n) for b in range(a + 1, n)
                 if max(r['score'][a], r['score'][b]) >= .1 and r['iou'][a] != r['iou'][b]]
        for a, b in pairs:
            hi, lo = (a, b) if r['iou'][a] > r['iou'][b] else (b, a)
            rows.append((k, hi, lo, cos[a, b], r['iou'][hi] - r['iou'][lo]))
    return rows


def fit_ranker(D, y, w, lam):
    """Weighted L2 logistic regression without bias on pair differences (torch LBFGS)."""
    import torch
    X = torch.tensor(D, dtype=torch.float64)
    Y = torch.tensor(y, dtype=torch.float64)
    W = torch.tensor(w / w.sum(), dtype=torch.float64)
    beta = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([beta], max_iter=200, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        z = X @ beta
        loss = (W * torch.nn.functional.binary_cross_entropy_with_logits(z, Y, reduction='none')).sum() + lam * (beta ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return beta.detach().numpy()


def pair_design(records, pairs, key, mean, std):
    hi = np.stack([(records[k][key][h] - mean) / std for k, h, _, _, _ in pairs])
    lo = np.stack([(records[k][key][l] - mean) / std for k, _, l, _, _ in pairs])
    return hi - lo


def oof_vector(records, pairs, weights, key, folds, image_of_pair, lams=(1e-4, 1e-3, 1e-2, 1e-1, 1.0)):
    """Out-of-fold score (w^T f) for every query of every record, with lambda chosen by inner grouped CV."""
    score = {k: None for k in range(len(records))}
    rec_fold = np.array([folds[r['image']] for r in records])
    for f in range(5):
        train = np.flatnonzero(np.array([folds[i] != f for i in image_of_pair]))
        train_recs = np.flatnonzero(rec_fold != f)
        feats = np.concatenate([records[k][key] for k in train_recs])
        mean, std = feats.mean(0), feats.std(0) + 1e-6
        tp = [pairs[i] for i in train]
        D = pair_design(records, tp, key, mean, std)
        wt = weights[train]
        # inner CV over the remaining 4 folds
        inner_f = np.array([folds[image_of_pair[i]] for i in train])
        best, best_acc = lams[0], -1
        for lam in lams:
            correct = tot = 0.
            for g in np.unique(inner_f):
                tr, te = inner_f != g, inner_f == g
                beta = fit_ranker(np.concatenate([D[tr], -D[tr]]), np.r_[np.ones(tr.sum()), np.zeros(tr.sum())], np.r_[wt[tr], wt[tr]], lam)
                correct += (wt[te] * (D[te] @ beta > 0)).sum()
                tot += wt[te].sum()
            if correct / tot > best_acc:
                best, best_acc = lam, correct / tot
        beta = fit_ranker(np.concatenate([D, -D]), np.r_[np.ones(len(D)), np.zeros(len(D))], np.r_[wt, wt], best)
        for k in np.flatnonzero(rec_fold == f):
            score[k] = ((records[k][key] - mean) / std) @ beta
    return score


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if np.std(ra) > 0 and np.std(rb) > 0 else np.nan


def evaluate_signal(records, pairs, weights, signal):
    """Per pair: correct (1/0/0.5 for ties); per record: Spearman over competing queries."""
    correct = np.array([1. if signal[k][h] > signal[k][l] else (.5 if signal[k][h] == signal[k][l] else 0.) for k, h, l, _, _ in pairs])
    members = {}
    for k, h, l, _, _ in pairs:
        members.setdefault(k, set()).update((h, l))
    sp = {k: spearman(signal[k][sorted(m)], records[k]['iou'][sorted(m)]) for k, m in members.items() if len(m) >= 3}
    return correct, sp


def summarise(records, pairs, weights, correct, sp, image_of_pair, rng):
    images = np.unique(image_of_pair)
    by_image = {i: np.flatnonzero(image_of_pair == i) for i in images}
    rec_image = {k: records[k]['image'] for k in sp}
    sp_by_image = {}
    for k, v in sp.items():
        if np.isfinite(v):
            sp_by_image.setdefault(rec_image[k], []).append(v)
    acc = float((weights * correct).sum() / weights.sum())
    sp_all = [v for v in sp.values() if np.isfinite(v)]
    boots_a, boots_s = [], []
    for _ in range(1000):
        pick = rng.choice(images, len(images))
        idx = np.concatenate([by_image[i] for i in pick])
        boots_a.append((weights[idx] * correct[idx]).sum() / weights[idx].sum())
        s = [v for i in pick for v in sp_by_image.get(i, [])]
        boots_s.append(np.mean(s) if s else np.nan)
    out = dict(accuracy=r3(acc), accuracy_ci95=[r3(np.percentile(boots_a, 2.5)), r3(np.percentile(boots_a, 97.5))],
               accuracy_unweighted=r3(correct.mean()), spearman_mean=r3(np.mean(sp_all)) if sp_all else None,
               spearman_ci95=[r3(np.nanpercentile(boots_s, 2.5)), r3(np.nanpercentile(boots_s, 97.5))] if sp_all else None,
               gts_for_spearman=len(sp_all))
    cos = np.array([p[3] for p in pairs])
    gap = np.array([p[4] for p in pairs])
    out['by_cosine'] = {f'{lo:.2f}-{min(hi, 1):.2f}': dict(pairs=int(m.sum()), accuracy=r3((weights[m] * correct[m]).sum() / weights[m].sum()) if m.any() else None)
                        for lo, hi in COS_BINS for m in [(cos >= lo) & (cos < hi)]}
    out['iou_gap_ge010'] = dict(pairs=int((gap >= .1).sum()), accuracy=r3((weights[gap >= .1] * correct[gap >= .1]).sum() / weights[gap >= .1].sum()) if (gap >= .1).any() else None)
    return out


def analyse(args):
    import torch
    report = {}
    for path in args.features:
        data = torch.load(path, weights_only=False)
        label = args.labels[args.features.index(path)] if args.labels else Path(path).parent.name
        report[label] = dict(meta=data['meta'], cells={})
        for cell in sorted({r['cell'] for r in data['records']}):
            if args.cells and cell not in args.cells:
                continue
            records = [r for r in data['records'] if r['cell'] == cell]
            pairs = build_pairs(records)
            if len(pairs) < 50:
                report[label]['cells'][cell] = dict(pairs=len(pairs), note='too few pairs')
                continue
            per_gt = np.bincount([p[0] for p in pairs], minlength=len(records))
            weights = np.array([1. / per_gt[p[0]] for p in pairs])
            image_of_pair = np.array([records[p[0]]['image'] for p in pairs])
            rng = np.random.default_rng(0)
            images = np.unique([r['image'] for r in records])
            perm = rng.permutation(images)
            folds = {int(i): n % 5 for n, i in enumerate(perm)}
            cos = np.array([p[3] for p in pairs])
            cell_out = dict(gts=int((per_gt > 0).sum()), images=int(len(np.unique(image_of_pair))), pairs=len(pairs),
                            pairs_by_cosine={f'{lo:.2f}-{min(hi, 1):.2f}': int(((cos >= lo) & (cos < hi)).sum()) for lo, hi in COS_BINS},
                            signals={})
            for key in SCALARS:
                sig = {k: r[key] for k, r in enumerate(records)}
                correct, sp = evaluate_signal(records, pairs, weights, sig)
                cell_out['signals'][key] = summarise(records, pairs, weights, correct, sp, image_of_pair, np.random.default_rng(1))
            for key in VECTORS:
                sig = oof_vector(records, pairs, weights, key, folds, image_of_pair)
                correct, sp = evaluate_signal(records, pairs, weights, sig)
                cell_out['signals'][key] = summarise(records, pairs, weights, correct, sp, image_of_pair, np.random.default_rng(1))
                print(label, cell, key, cell_out['signals'][key]['accuracy'], flush=True)
            report[label]['cells'][cell] = cell_out
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'probe.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print('written', args.out / 'probe.json')


# ---------------------------------------------------------------- joint probes, D7.2 (local)
# Does the box head carry quality information beyond box geometry, and beyond geometry + the logit the postprocessor
# already ranks? Same pairs, folds, weights, ranker and bootstrap as `analyse`; a joint signal is the concatenation of
# its parts. S0q is a stricter geometry control: S0 plus all degree-2 monomials of (cx, cy, log w, log h).
JOINT = {
    'S0': ['S0'], 'S0q': ['S0q'], 'S4_h1': ['S4_h1'], 'S5': ['S5'],
    'S0+S4_h1': ['S0', 'S4_h1'], 'S0q+S4_h1': ['S0q', 'S4_h1'], 'S0+S5': ['S0', 'S5'],
    'S0+S1': ['S0', 'S1_final'], 'S0+S1+S4_h1': ['S0', 'S1_final', 'S4_h1'], 'S0+S1+S5': ['S0', 'S1_final', 'S5'],
}
DIFFS = [('S0+S4_h1', 'S0'), ('S0q+S4_h1', 'S0q'), ('S0+S5', 'S0'), ('S0+S4_h1', 'S0+S5'), ('S0+S1', 'S1_final'),
         ('S0+S1+S4_h1', 'S0+S1'), ('S0+S1+S5', 'S0+S1'), ('S4_h1', 'S1_final'), ('S0+S1+S4_h1', 'S1_final')]


def paired_difference(weights, a, b, image_of_pair, rng):
    """GT-weighted accuracy difference a - b on the same pairs, with a paired image bootstrap."""
    images = np.unique(image_of_pair)
    by_image = {i: np.flatnonzero(image_of_pair == i) for i in images}
    d = weights * (a - b)
    boots = []
    for _ in range(1000):
        idx = np.concatenate([by_image[i] for i in rng.choice(images, len(images))])
        boots.append(d[idx].sum() / weights[idx].sum())
    return dict(difference=r3(d.sum() / weights.sum()), ci95=[r3(np.percentile(boots, 2.5)), r3(np.percentile(boots, 97.5))],
                share_of_resamples_le0=r3(np.mean(np.array(boots) <= 0)))


ALIAS = {'S1': 'S1_final'}


def parts_of(name):
    """Components of a signal name: a JOINT entry, or any '+'-joined list of record keys (S1 = S1_final)."""
    return JOINT.get(name) or [ALIAS.get(p, p) for p in name.split('+')]


def add_joint_features(records, names):
    for r in records:
        g = r['S0'][:, [0, 1, 4, 5]].astype(np.float64)
        r['S0q'] = np.concatenate([r['S0']] + [g[:, [i]] * g[:, [j]] for i in range(4) for j in range(i, 4)], 1)
        for name in names:
            parts = parts_of(name)
            if len(parts) > 1:
                r[name] = np.concatenate([r[p] if r[p].ndim == 2 else r[p][:, None] for p in parts], 1).astype(np.float64)


def probe_setup(records):
    """Pairs, GT weights, image of each pair and the image folds, exactly as in `analyse`."""
    pairs = build_pairs(records)
    per_gt = np.bincount([p[0] for p in pairs], minlength=len(records))
    weights = np.array([1. / per_gt[p[0]] for p in pairs])
    image_of_pair = np.array([records[p[0]]['image'] for p in pairs])
    perm = np.random.default_rng(0).permutation(np.unique([r['image'] for r in records]))
    folds = {int(i): n % 5 for n, i in enumerate(perm)}
    return pairs, per_gt, weights, image_of_pair, folds


def joint(args):
    import torch
    report = {}
    names = args.signals or list(JOINT)
    for path in args.features:
        data = torch.load(path, weights_only=False)
        label = args.labels[args.features.index(path)] if args.labels else Path(path).parent.name
        report[label] = dict(meta=data['meta'], cells={})
        for cell in sorted({r['cell'] for r in data['records']}):
            if args.cells and cell not in args.cells:
                continue
            records = [r for r in data['records'] if r['cell'] == cell]
            add_joint_features(records, names)
            pairs, per_gt, weights, image_of_pair, folds = probe_setup(records)
            if len(pairs) < 50 or len(folds) < 10:
                report[label]['cells'][cell] = dict(pairs=len(pairs), images=len(folds), note='too few pairs or images')
                continue
            cell_out = dict(gts=int((per_gt > 0).sum()), images=int(len(np.unique(image_of_pair))), pairs=len(pairs), signals={}, differences={})
            correct = {}
            sig = {k: r['S1_final'] for k, r in enumerate(records)}
            correct['S1_final'], sp = evaluate_signal(records, pairs, weights, sig)
            cell_out['signals']['S1_final'] = summarise(records, pairs, weights, correct['S1_final'], sp, image_of_pair, np.random.default_rng(1))
            for name in names:
                sig = oof_vector(records, pairs, weights, name, folds, image_of_pair)
                correct[name], sp = evaluate_signal(records, pairs, weights, sig)
                cell_out['signals'][name] = summarise(records, pairs, weights, correct[name], sp, image_of_pair, np.random.default_rng(1))
                print(label, cell, name, cell_out['signals'][name]['accuracy'], flush=True)
            diffs = [tuple(d.split(':')) for d in args.diffs] if args.diffs else DIFFS
            for a, b in diffs:
                if a not in correct or b not in correct:
                    continue
                cell_out['differences'][f'{a} - {b}'] = paired_difference(weights, correct[a], correct[b], image_of_pair, np.random.default_rng(2))
                print(label, cell, a, '-', b, cell_out['differences'][f'{a} - {b}'], flush=True)
            report[label]['cells'][cell] = cell_out
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / args.name).write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print('written', args.out / args.name)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    e = sub.add_parser('extract')
    e.add_argument('run', type=Path)
    e.add_argument('out', type=Path)
    e.add_argument('--checkpoint', type=Path, required=True)
    e.add_argument('--threads', type=int, default=16)
    e.add_argument('--all-cells', action='store_true', help='every source x class cell instead of the three D7.1 cells')
    a = sub.add_parser('analyse')
    a.add_argument('features', nargs='+')
    a.add_argument('--labels', nargs='*')
    a.add_argument('--out', type=Path, required=True)
    a.add_argument('--cells', nargs='*', help='only these cells, e.g. ogsod/tank msar/tank')
    j = sub.add_parser('joint')
    j.add_argument('features', nargs='+')
    j.add_argument('--labels', nargs='*')
    j.add_argument('--out', type=Path, required=True)
    j.add_argument('--cells', nargs='*')
    j.add_argument('--name', default='probe_joint.json')
    j.add_argument('--signals', nargs='*', help='JOINT names or any "+"-joined record keys; default all of JOINT')
    j.add_argument('--diffs', nargs='*', help='paired differences as A:B (signal names); default DIFFS')
    args = parser.parse_args()
    if args.command == 'extract':
        args.out.mkdir(parents=True, exist_ok=True)
    {'extract': extract, 'analyse': analyse, 'joint': joint}[args.command](args)


if __name__ == '__main__':
    main()
