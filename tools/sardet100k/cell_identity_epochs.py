"""D6 + D7: query identity, score/IoU coupling and pairwise discrimination per source x class cell over training.

Generalises tank_identity_epochs.py (same replay, same definitions) from tank to any (source group, class) cell, and
adds, for every stage (enc, pre, dec1-4), the per-GT relation between the matcher query, the top-scoring candidate
and the best-IoU candidate. Offline replay of each saved EMA checkpoint on fixed val images, no augmentation, each
run's own matcher; union set by DEIMCriterion._get_go_indices. It describes the model at those epochs; it is not a
record of training-batch assignments. Query IDs are compared within one forward pass only.

Per GT of a chosen cell and per stage s (candidates at s: queries whose highest-IoU GT among the same-class GTs of
the image is this GT at stage s, IoU >= .1; scores are sigmoid of the GT-class logit at s):
  m = query matched to the GT at s by the matcher; top = highest-score candidate; best = highest-IoU candidate
  dS = S(m) - S(top), dIoU = IoU(m) - IoU(top)     (only when m exists)
  spearman(score, IoU) over the candidates (GTs with >= 3 candidates)
  unmatched candidates: not matched to any GT at any stage of this pass; their scores and IoUs
  pairs (decoder layers): candidate pairs with IoU >= .5 for both and at least one score >= .1 (competing pairs);
  concordance = the higher-IoU query also has the higher score; by IoU gap and by cosine similarity of the two
  queries' decoder features at that layer (concordance_all: every pair, dominated by near-zero scores)
  matcher_vs_top_cosine: feature cosine of the matcher query and the top query when they differ
D5 quantities (same_dec_all, unmatched high-score per GT, ...) are reported at the last layer as before.

Usage (repository root on the server; OUT directly under E000's independent_verify_20261106/):
  python cell_identity_epochs.py RUN OUT [--checkpoints a.pth ...] [--threads 8]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_report import group_of, iou_xyxy, r3, sigmoid, spearman  # noqa: E402
from error_gallery import ROOT  # noqa: E402
from tank_feature_diagnosis import load_val, to_input, xyxy  # noqa: E402
from tank_matching_counterfactual import Runner  # noqa: E402

STAGES = ['enc', 'pre', 'dec1', 'dec2', 'dec3', 'dec4']
DEC = ['dec1', 'dec2', 'dec3', 'dec4']
CELLS = [('ogsod', 'tank'), ('ogsod', 'bridge'), ('ogsod', 'harbor'), ('other_1000', 'ship'), ('msar', 'tank'), ('msar', 'ship')]
IOU_GAP_BINS = [.02, .05, .1, .2, 1.01]
COS_BINS = [-1, .9, .95, .98, 1.01]


def replay(run, checkpoint, out, device, cells):
    annotations, source, grouped = load_val(out)
    name_of = {c['id']: c['name'] for c in annotations['categories']}
    id_of = {v: k for k, v in name_of.items()}
    want = {(g, id_of[c]) for g, c in cells}
    runner = Runner(run, device, checkpoint)
    crit, inner, saved = runner.criterion, runner.inner, runner.saved
    images = []
    for i in annotations['images']:
        g = group_of(source[i['id']], i['width'], i['height'])
        if any((g, a['category_id']) in want for a in grouped[i['id']]):
            images.append((i, g))
    rows = []
    for image, g in images:
        anns = grouped[image['id']]
        x, _ = to_input(ROOT / 'JPEGImages/val' / image['file_name'])
        targets, xy = runner.targets(image, anns)
        outs = runner.stages(x[None].to(device))
        feats = {f'dec{k + 1}': saved[f'q{k}'][0].float().cpu().numpy() for k in range(inner.num_layers)}  # from the last pass
        ind = {s: runner.matcher({'pred_logits': outs[s][0][None], 'pred_boxes': outs[s][1][None]}, targets)['indices'][0] for s in STAGES}
        go = crit._get_go_indices([ind['dec4']], [[ind[s]] for s in ('dec1', 'dec2', 'dec3', 'pre', 'enc')])[0]
        go_q = set(go[0].tolist())
        matched_any = set()
        for s in STAGES:
            matched_any |= set(ind[s][0].tolist())
        logits = {s: outs[s][0].float().cpu().numpy() for s in STAGES}
        boxes = {s: xyxy(outs[s][1].float().cpu().numpy(), 1) for s in STAGES}
        labels = np.array([a['category_id'] for a in anns])
        for c in np.unique(labels):
            if (g, int(c)) not in want:
                continue
            idx = np.flatnonzero(labels == c)
            stage = {}
            for s in STAGES:
                iou = iou_xyxy(boxes[s], xy[idx])
                stage[s] = (sigmoid(logits[s][:, c]), iou, iou.argmax(1), iou.max(1))
            for j, t in enumerate(idx):
                ids = {s: dict(zip(ind[s][1].tolist(), ind[s][0].tolist())).get(int(t), -1) for s in STAGES}
                m4 = ids['dec4']
                row = dict(image=image['id'], cell=f'{g}/{name_of[int(c)]}', ids=ids,
                           same_dec_all=int(len({ids[s] for s in DEC}) == 1 and m4 >= 0),
                           distinct_dec=len({ids[s] for s in DEC if ids[s] >= 0}), stages={})
                for s in STAGES:
                    score, iou, owner, best_iou = stage[s]
                    cand = np.flatnonzero((owner == j) & (best_iou >= .1))
                    m = ids[s]
                    st = dict(n_cand=len(cand))
                    if len(cand):
                        sc, io = score[cand], iou[cand, j]
                        top, best = int(cand[np.argmax(sc)]), int(cand[np.argmax(io)])
                        st.update(top_is_m=int(top == m), best_is_m=int(best == m), top_is_best=int(top == best),
                                  top_score=float(sc.max()), top_iou=float(iou[top, j]), best_iou=float(io.max()))
                        if m >= 0:
                            st.update(dS=float(score[m] - score[top]), dIoU=float(iou[m, j] - iou[top, j]),
                                      m_score=float(score[m]), m_iou=float(iou[m, j]))
                        if len(cand) >= 3:
                            st['spearman'] = spearman(sc, io)
                        never = np.array([q not in matched_any for q in cand])
                        st['unmatched_scores'] = sc[never].tolist()
                        st['unmatched_ious'] = io[never].tolist()
                        st['scores'], st['ious'] = sc.tolist(), io.tolist()
                        if s in DEC:
                            good = cand[io >= .5]
                            pairs = []
                            if len(good) >= 2:
                                f = feats[s][good]
                                f = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)
                                cos = f @ f.T
                                gs, gi = score[good], iou[good, j]
                                for a in range(len(good)):
                                    for b in range(a + 1, len(good)):
                                        hi, lo = (a, b) if gi[a] >= gi[b] else (b, a)
                                        pairs.append((float(gi[hi] - gi[lo]), float(cos[a, b]), int(gs[hi] > gs[lo]), float(max(gs[a], gs[b]))))
                            st['pairs'] = pairs
                            if m >= 0 and m != top:
                                fm, ft = feats[s][m], feats[s][top]
                                st['m_top_cos'] = float(fm @ ft / max(np.linalg.norm(fm) * np.linalg.norm(ft), 1e-12))
                        if s == 'dec4':
                            st['unmatched_hi03_iou50'] = int((never & (sc >= .3) & (io >= .5)).sum())
                            st['hi03_iou50'] = int(((sc >= .3) & (io >= .5)).sum())
                            st['top_in_union'] = int(top in go_q)
                    row['stages'][s] = st
                rows.append(row)
    return runner.epoch, rows


def q(values, qs=(.1, .25, .5, .75, .9)):
    return [r3(np.quantile(values, x)) for x in qs] if len(values) else None


def summarise(rows):
    out = {}
    for cell in sorted({r['cell'] for r in rows}):
        rr = [r for r in rows if r['cell'] == cell]
        res = dict(gt=len(rr), same_dec_all=r3(np.mean([r['same_dec_all'] for r in rr])),
                   distinct_dec=r3(np.mean([r['distinct_dec'] for r in rr])), stages={})
        for s in STAGES:
            st = [r['stages'][s] for r in rr if r['stages'][s]['n_cand']]
            if not st:
                continue
            hasm = [x for x in st if 'dS' in x]
            ds, di = np.array([x['dS'] for x in hasm]), np.array([x['dIoU'] for x in hasm])
            dis = np.array([x['top_is_m'] == 0 for x in hasm])
            unm_s = np.concatenate([x['unmatched_scores'] for x in st]) if st else np.empty(0)
            unm_i = np.concatenate([x['unmatched_ious'] for x in st]) if st else np.empty(0)
            pooled_s = np.concatenate([x['scores'] for x in st])
            pooled_i = np.concatenate([x['ious'] for x in st])
            e = dict(gt_with_candidates=len(st), top_is_matcher=r3(np.mean([x['top_is_m'] for x in st])),
                     best_is_matcher=r3(np.mean([x['best_is_m'] for x in st])), top_is_best=r3(np.mean([x['top_is_best'] for x in st])),
                     dS_quantiles_10_25_50_75_90=q(ds), dIoU_quantiles_10_25_50_75_90=q(di),
                     dS_lt0=r3(np.mean(ds < 0)) if len(ds) else None, dIoU_gt0=r3(np.mean(di > 0)) if len(di) else None,
                     when_top_ne_matcher=dict(n=int(dis.sum()), dS_median=r3(np.median(ds[dis])) if dis.any() else None,
                                              dIoU_median=r3(np.median(di[dis])) if dis.any() else None,
                                              matcher_better_box=r3(np.mean(di[dis] > 0)) if dis.any() else None),
                     spearman_per_gt_median=r3(np.median([x['spearman'] for x in st if np.isfinite(x.get('spearman', np.nan))])) if any('spearman' in x for x in st) else None,
                     spearman_pooled=r3(spearman(pooled_s, pooled_i)),
                     matcher_score_median=r3(np.median([x['m_score'] for x in hasm])) if hasm else None,
                     matcher_iou_median=r3(np.median([x['m_iou'] for x in hasm])) if hasm else None,
                     top_score_median=r3(np.median([x['top_score'] for x in st])), top_iou_median=r3(np.median([x['top_iou'] for x in st])),
                     best_iou_median=r3(np.median([x['best_iou'] for x in st])),
                     unmatched=dict(per_gt=r3(len(unm_s) / len(st)), score_quantiles_50_90_99=q(unm_s, (.5, .9, .99)),
                                    ge03_per_gt=r3((unm_s >= .3).sum() / len(st)), ge03_iou50_per_gt=r3(((unm_s >= .3) & (unm_i >= .5)).sum() / len(st)),
                                    iou_median_of_ge03=r3(np.median(unm_i[unm_s >= .3])) if (unm_s >= .3).any() else None))
            if s == 'dec4':
                e['unmatched_hi03_iou50_per_gt'] = r3(np.mean([x['unmatched_hi03_iou50'] for x in st]))
                e['gt_with_2plus_hi03_iou50'] = r3(np.mean([x['hi03_iou50'] >= 2 for x in st]))
                e['top_in_union'] = r3(np.mean([x['top_in_union'] for x in st]))
            if s in DEC:
                allpairs = np.array([p for x in st for p in x.get('pairs', [])]).reshape(-1, 4)
                cosmt = [x['m_top_cos'] for x in st if 'm_top_cos' in x]
                e['matcher_vs_top_cosine'] = dict(n=len(cosmt), quantiles_10_50_90=q(cosmt, (.1, .5, .9)), ge_098=r3(np.mean(np.array(cosmt) >= .98)) if cosmt else None)
                pr = dict(n_all=len(allpairs), concordance_all=r3(allpairs[:, 2].mean()) if len(allpairs) else None)
                pairs = allpairs[allpairs[:, 3] >= .1]  # competing pairs: at least one query scores >= .1
                pr.update(n=len(pairs), concordance=r3(pairs[:, 2].mean()) if len(pairs) else None)
                if len(pairs):
                    pr['by_iou_gap'] = {f'{lo:.2f}-{hi:.2f}': dict(n=int(k.sum()), concordance=r3(pairs[k, 2].mean()) if k.any() else None)
                                        for lo, hi in zip([0] + IOU_GAP_BINS[:-1], IOU_GAP_BINS)
                                        for k in [(pairs[:, 0] >= lo) & (pairs[:, 0] < hi)]}
                    pr['by_cosine'] = {f'{lo:.2f}-{hi:.2f}': dict(n=int(k.sum()), concordance=r3(pairs[k, 2].mean()) if k.any() else None,
                                                                  iou_gap_median=r3(np.median(pairs[k, 0])) if k.any() else None)
                                       for lo, hi in zip(COS_BINS[:-1], COS_BINS[1:])
                                       for k in [(pairs[:, 1] >= lo) & (pairs[:, 1] < hi)]}
                    big = pairs[:, 0] >= .1
                    pr['iou_gap_ge010_by_cosine'] = {f'{lo:.2f}-{hi:.2f}': dict(n=int(k.sum()), concordance=r3(pairs[k, 2].mean()) if k.any() else None)
                                                     for lo, hi in zip(COS_BINS[:-1], COS_BINS[1:])
                                                     for k in [big & (pairs[:, 1] >= lo) & (pairs[:, 1] < hi)]}
                    pr['cosine_median'] = r3(np.median(pairs[:, 1]))
                e['pairs'] = pr
            res['stages'][s] = e
        out[cell] = res
    return out


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--checkpoints', nargs='*', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoints = args.checkpoints or sorted(args.run.glob('checkpoint*.pth')) + [args.run / 'best_stg2.pth']
    for ck in checkpoints:
        epoch, rows = replay(args.run, ck, args.out, args.device, CELLS)
        report = dict(epoch=epoch, checkpoint=str(ck), cells=summarise(rows))
        (args.out / f'cells_{ck.stem}.json').write_text(json.dumps(report, indent=1))
        torch.save(rows, args.out / f'cells_{ck.stem}_per_gt.pt')
        brief = {c: (v['same_dec_all'], v['stages']['dec4']['top_is_matcher'], v['stages']['dec4']['unmatched_hi03_iou50_per_gt'])
                 for c, v in report['cells'].items() if 'dec4' in v['stages']}
        print(ck.name, epoch, json.dumps(brief), flush=True)
    print('written', args.out, flush=True)


if __name__ == '__main__':
    main()
