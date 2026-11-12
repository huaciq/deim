"""D5: query identity of each tank GT over training, replayed from the saved EMA checkpoints. No training.

This is an offline replay: each saved checkpoint's EMA weights see the fixed val tank images of OGSOD and MSAR
(no augmentation), and every stage (encoder Top-K, decoder-1 pre head, decoder layers 1-4) is matched with the run's
own matcher; the union set uses DEIMCriterion._get_go_indices, exactly as in training. It describes the model's
matching state at those epochs. It is NOT a record of the assignments made on training batches; that needs the
training-time log (log_matching_stats) and is not what this script measures.

Query IDs index the 300 decoder queries of one forward pass, so they are compared across stages and layers of the
same pass only, never across checkpoints (the encoder Top-K changes with the weights).

Per tank GT and checkpoint:
  ids            matched query at enc, pre, dec1-4 (-1 when unmatched)
  same_dec_all   one query matched at all four decoder layers (definition of match_same_dec_all)
  distinct       number of distinct queries matched to the GT over the 6 stages, and over dec1-4
  enc_kept       the encoder's matched query is the dec4 matched query
  top / best     candidates (survey rule at dec4: tank-class GT of highest IoU is this GT, IoU >= .1); top = highest
                 tank score, best = highest IoU; top_is_matched = top is the dec4 matched query
  unmatched_hi   candidates with tank score >= .3 (and IoU >= .5) at dec4 that no stage matched to any GT
  traj           tank score and IoU of the dec4 matched query and of the top query at enc, pre, dec1-4

Usage (repository root on the server; OUT directly under E000's independent_verify_20261106/):
  python tank_identity_epochs.py RUN OUT [--device cpu] [--threads 12] [--checkpoints a.pth ...]
Default checkpoints: RUN/checkpoint*.pth and RUN/best_stg2.pth.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_report import iou_xyxy, r3, sigmoid  # noqa: E402
from error_gallery import ROOT  # noqa: E402
from tank_feature_diagnosis import SOURCES, TANK, load_val, to_input, xyxy  # noqa: E402
from tank_matching_counterfactual import Runner, images_of  # noqa: E402

STAGES = ['enc', 'pre', 'dec1', 'dec2', 'dec3', 'dec4']
DEC = ['dec1', 'dec2', 'dec3', 'dec4']


def replay(run, checkpoint, out, device):
    annotations, source, grouped = load_val(out)
    runner = Runner(run, device, checkpoint)
    crit = runner.criterion
    rows = []
    for image in images_of(annotations, source, grouped):
        anns = grouped[image['id']]
        x, _ = to_input(ROOT / 'JPEGImages/val' / image['file_name'])
        targets, xy = runner.targets(image, anns)
        outs = runner.stages(x[None].to(device))
        ind = {s: runner.matcher({'pred_logits': outs[s][0][None], 'pred_boxes': outs[s][1][None]}, targets)['indices'][0] for s in STAGES}
        go = crit._get_go_indices([ind['dec4']], [[ind[s]] for s in ('dec1', 'dec2', 'dec3', 'pre', 'enc')])[0]
        go_q = set(go[0].tolist())
        matched_any = set()
        for s in STAGES:
            matched_any |= set(ind[s][0].tolist())
        score = {s: sigmoid(outs[s][0].float().cpu().numpy()[:, TANK]) for s in STAGES}
        boxes = {s: xyxy(outs[s][1].float().cpu().numpy(), 1) for s in STAGES}
        labels = np.array([a['category_id'] for a in anns])
        tank = np.flatnonzero(labels == TANK)
        iou = {s: iou_xyxy(boxes[s], xy[tank]) for s in STAGES}  # Q x T
        owner, best_iou = iou['dec4'].argmax(1), iou['dec4'].max(1)
        for j, t in enumerate(tank):
            ids = {s: dict(zip(ind[s][1].tolist(), ind[s][0].tolist())).get(int(t), -1) for s in STAGES}
            m4 = ids['dec4']
            cand = np.flatnonzero((owner == j) & (best_iou >= .1))
            row = dict(image=image['id'], source=source[image['id']], gt=int(t), ids=ids,
                       same_dec_all=int(len({ids[s] for s in DEC}) == 1 and m4 >= 0),
                       distinct_all=len({ids[s] for s in STAGES if ids[s] >= 0}), distinct_dec=len({ids[s] for s in DEC if ids[s] >= 0}),
                       enc_kept=int(ids['enc'] == m4 and m4 >= 0), n_cand=len(cand))
            if len(cand):
                s4, i4 = score['dec4'][cand], iou['dec4'][cand, j]
                top, best = int(cand[np.argmax(s4)]), int(cand[np.argmax(i4)])
                never = np.array([q not in matched_any for q in cand])
                row.update(top=top, best=best, top_is_matched=int(top == m4), best_is_matched=int(best == m4),
                           top_in_union=int(top in go_q),
                           unmatched_hi03=int((never & (s4 >= .3)).sum()), unmatched_hi03_iou50=int((never & (s4 >= .3) & (i4 >= .5)).sum()),
                           hi03_iou50=int(((s4 >= .3) & (i4 >= .5)).sum()),
                           top_score=float(s4.max()), top_iou=float(iou['dec4'][top, j]), best_iou=float(i4.max()))
                for name, q in (('matched', m4), ('top', top)):
                    if q >= 0:
                        row[f'traj_{name}'] = dict(score=[float(score[s][q]) for s in STAGES], iou=[float(iou[s][q, j]) for s in STAGES])
            rows.append(row)
    return runner.epoch, rows


def summarise(rows):
    out = {}
    for src in SOURCES:
        r = [x for x in rows if x['source'] == src]
        if not r:
            continue
        c = [x for x in r if x['n_cand']]
        mean = lambda k, rr: r3(np.mean([x[k] for x in rr])) if rr else None  # noqa: E731
        res = dict(gt=len(r), with_candidates=len(c),
                   same_dec_all=mean('same_dec_all', r), distinct_queries_all_stages=mean('distinct_all', r),
                   distinct_queries_dec=mean('distinct_dec', r), enc_query_kept_to_dec4=mean('enc_kept', r),
                   top_is_matched=mean('top_is_matched', c), best_is_matched=mean('best_is_matched', c),
                   top_in_union=mean('top_in_union', c),
                   unmatched_hi03_per_gt=mean('unmatched_hi03', c), unmatched_hi03_iou50_per_gt=mean('unmatched_hi03_iou50', c),
                   gt_with_unmatched_hi03_iou50=r3(np.mean([x['unmatched_hi03_iou50'] > 0 for x in c])) if c else None,
                   hi03_iou50_per_gt=mean('hi03_iou50', c), gt_with_2plus_hi03_iou50=r3(np.mean([x['hi03_iou50'] >= 2 for x in c])) if c else None,
                   top_score_median=r3(np.median([x['top_score'] for x in c])) if c else None,
                   top_iou_median=r3(np.median([x['top_iou'] for x in c])) if c else None,
                   best_iou_median=r3(np.median([x['best_iou'] for x in c])) if c else None)
        for s in ('enc', 'pre', 'dec1', 'dec2', 'dec3'):
            res[f'same_{s}_as_dec4'] = r3(np.mean([x['ids'][s] == x['ids']['dec4'] and x['ids']['dec4'] >= 0 for x in r]))
        for name in ('matched', 'top'):
            tr = [x[f'traj_{name}'] for x in c if f'traj_{name}' in x]
            if tr:
                res[f'traj_{name}'] = dict(stages=STAGES, score_median=[r3(v) for v in np.median([t['score'] for t in tr], 0)],
                                           iou_median=[r3(v) for v in np.median([t['iou'] for t in tr], 0)])
        out[src] = res
    return out


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=12)
    parser.add_argument('--checkpoints', nargs='*', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoints = args.checkpoints or sorted(args.run.glob('checkpoint*.pth')) + [args.run / 'best_stg2.pth']
    report, per_gt = {}, {}
    for ck in checkpoints:
        epoch, rows = replay(args.run, ck, args.out, args.device)
        report[ck.name] = dict(epoch=epoch, **summarise(rows))
        per_gt[ck.name] = rows
        print(ck.name, epoch, json.dumps({k: report[ck.name]['ogsod'][k] for k in ('same_dec_all', 'top_is_matched', 'unmatched_hi03_iou50_per_gt', 'gt_with_2plus_hi03_iou50')}), flush=True)
        (args.out / 'identity.json').write_text(json.dumps(report, indent=1))
        torch.save(per_gt, args.out / 'identity_per_gt.pt')
    print('written', args.out, flush=True)


if __name__ == '__main__':
    main()
