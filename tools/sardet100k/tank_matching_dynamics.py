"""Matching statistics over training: replay each saved checkpoint's EMA weights on the fixed val tank images.

For every checkpoint of a run, the images of OGSOD and MSAR val that contain tanks are passed through the EMA model;
every stage (encoder Top-K, decoder-1 pre head, decoder layers 1..4) is matched with the run's own matcher (from its
resolved config), the union set is built with DEIMCriterion._get_go_indices, and the statistics come from
DEIMCriterion._record_matching_metrics, the same function that logs training batches (log_matching_stats).
So replay and training logs share definitions; only the images and weights differ (val, EMA, no augmentation).
Matched-query IoU quantiles per stage are added for tank GTs.

Usage (repository root on the server; OUT under E000's independent_verify_20261106/ for source.json):
  python tools/sardet100k/tank_matching_dynamics.py RUN OUT [--device cpu] [--checkpoints a.pth b.pth ...]
Default checkpoints: RUN/checkpoint*.pth and RUN/final.pth.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from error_gallery import ROOT, iou_matrix  # noqa: E402
from tank_feature_diagnosis import SOURCES, TANK, load_val, to_input, xyxy  # noqa: E402
from tank_matcher_audit import STAGES  # noqa: E402
from tank_matching_counterfactual import Runner, images_of  # noqa: E402


def replay(run, checkpoint, out, device):
    import torch
    annotations, source, grouped = load_val(out)
    runner = Runner(run, device, checkpoint)
    crit = runner.criterion
    crit.matching_stats_class = TANK
    sums = {s: defaultdict(float) for s in SOURCES}
    ious = {s: defaultdict(list) for s in SOURCES}
    for image in images_of(annotations, source, grouped):
        src = source[image['id']]
        anns = grouped[image['id']]
        x, _ = to_input(ROOT / 'JPEGImages/val' / image['file_name'])
        targets, xy = runner.targets(image, anns)
        outs = runner.stages(x[None].to(device))
        indices = {s: runner.matcher({'pred_logits': outs[s][0][None], 'pred_boxes': outs[s][1][None]}, targets)['indices'] for s in STAGES}
        go = crit._get_go_indices(indices['dec4'], [indices[s] for s in ('dec1', 'dec2', 'dec3', 'pre', 'enc')])
        outputs = dict(pred_boxes=outs['dec4'][1][None], aux_outputs=[{'pred_boxes': outs[f'dec{k}'][1][None]} for k in (1, 2, 3)],
                       pre_outputs={'pred_boxes': outs['pre'][1][None]}, enc_aux_outputs=[{'pred_boxes': outs['enc'][1][None]}])
        crit._record_matching_metrics(outputs, targets, indices['dec4'], [indices[s] for s in ('dec1', 'dec2', 'dec3', 'pre')], [indices['enc']], go)
        for name, value in crit.matching_metrics.items():
            sums[src][name] += value
        tank = [n for n, a in enumerate(anns) if a['category_id'] == TANK]
        for s in STAGES:
            q, g = indices[s][0]
            box = xyxy(outs[s][1].cpu().numpy(), 1)
            for qq, gg in zip(q.tolist(), g.tolist()):
                if gg in tank:
                    ious[src][s].append(float(iou_matrix(box[[qq]], xy[[gg]])[0, 0]))
    result = dict(checkpoint=str(checkpoint), epoch=runner.epoch)
    for src in SOURCES:
        m = sums[src]
        gt, union = max(m['match_gt_tank'], 1), max(m['match_union_tank'], 1)
        r = dict(tank_gt=m['match_gt_tank'], union_pairs_tank=m['match_union_tank'],
                 same_query_all_decoder_layers=round(m['match_same_dec_all_tank'] / gt, 4),
                 same_query_consecutive={f'dec{k}-dec{k + 1}': round(m[f'match_same_dec_{k}_{k + 1}_tank'] / gt, 4) for k in (1, 2, 3)},
                 boxpos_clsbg_per_union={f'dec{k}': round(m[f'match_boxpos_clsbg_dec{k}_tank'] / union, 4) for k in (1, 2, 3, 4)},
                 boxpos_clsbg_per_gt={f'dec{k}': round(m[f'match_boxpos_clsbg_dec{k}_tank'] / gt, 4) for k in (1, 2, 3, 4)},
                 clsbg_run2_per_union=round(m['match_clsbg_run2_tank'] / union, 4), clsbg_all_per_union=round(m['match_clsbg_all_tank'] / union, 4),
                 matched_iou={s: dict(mean=round(m[f'match_iou_{s}_tank'] / gt, 4), ge50=round(m[f'match_iou50_{s}_tank'] / gt, 4),
                                      ge75=round(m[f'match_iou75_{s}_tank'] / gt, 4),
                                      quantiles_10_25_50_75_90=np.percentile(ious[src][s], [10, 25, 50, 75, 90]).round(3).tolist() if ious[src][s] else None)
                              for s in STAGES})
        result[src] = r
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--checkpoints', nargs='*', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoints = args.checkpoints or sorted(args.run.glob('checkpoint*.pth')) + [p for p in [args.run / 'final.pth'] if p.is_file()]
    report_path = args.out / 'report.json'
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    for checkpoint in checkpoints:
        if str(checkpoint) in report:
            continue
        report[str(checkpoint)] = replay(args.run, checkpoint, args.out, args.device)
        report_path.write_text(json.dumps(report, indent=1, ensure_ascii=False))
        r = report[str(checkpoint)]
        print(r['epoch'], {s: (r[s]['matched_iou']['dec4']['mean'], r[s]['same_query_all_decoder_layers'], r[s]['boxpos_clsbg_per_union']['dec4'], r[s]['clsbg_run2_per_union']) for s in SOURCES}, flush=True)


if __name__ == '__main__':
    main()
