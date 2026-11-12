"""Follow-up to tank_single_gt_trace.py: whose box is a nearby high score, and does the final score track box quality?

Usage: python tools/sardet100k/tank_trace_checks.py RUN OUT [--device cuda:0]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from error_gallery import ROOT, iou_matrix  # noqa: E402
from tank_feature_diagnosis import INPUT, TANK, auc, load_model, load_val, to_input, xyxy  # noqa: E402
from tank_single_gt_trace import Tracer, strict_stage  # noqa: E402


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    annotations, source, grouped = load_val(args.out)
    images = {i['id']: i for i in annotations['images']}
    rows = json.load(open(args.out / 'gt_rows.json'))
    trace = json.load(open(args.out / 'trace/trace.json'))
    by = defaultdict(list)
    for r in rows:
        by[r['image_id']].append(r)
    model, _ = load_model(args.run, args.device)
    tracer = Tracer(model)
    report = dict(samples={}, groups={}, quality={})
    owners = defaultdict(list)
    quality = defaultdict(lambda: defaultdict(list))
    with torch.no_grad():
        for count, (image_id, part) in enumerate(by.items()):
            x, (w, _) = to_input(ROOT / 'JPEGImages/val' / images[image_id]['file_name'])
            _, enc, layers, _ = tracer.run(x[None].to(args.device))
            s = INPUT / w
            final = layers[-1]
            boxes = xyxy(final['box'])
            tank = final['score'][:, TANK]
            targets = [a for a in grouped[image_id] if a['category_id'] == TANK and a['bbox'][2] > 0 and a['bbox'][3] > 0]
            gt = np.array([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]] for a in targets], np.float32) * s
            all_iou = iou_matrix(boxes, gt)
            ids = [a['id'] for a in targets]
            # Score vs quality over every query that claims a tank here.
            claim = tank >= .05
            if claim.any():
                best = all_iou[claim].max(1)
                raw = final['raw'][claim, TANK]
                lqe = final['logit'][claim, TANK] - raw
                src = source[image_id]
                for key, values in (('iou', best), ('raw', raw), ('lqe', lqe), ('score', tank[claim])):
                    quality[src][key].extend(values.tolist())
                quality[src]['image'].extend([image_id] * int(claim.sum()))
            for r in part:
                if r['native_gw'] <= 0 or r['native_gh'] <= 0:
                    continue
                index = ids.index(r['gt_id'])
                near = (all_iou[:, index] >= .1) & (all_iou[:, index] < .5)
                if near.any():
                    q = int(np.flatnonzero(near)[np.argmax(tank[near])])
                    owner = int(all_iou[q].argmax())
                    owners[f"{r['status']}|{strict_stage(r)}"].append(dict(score=float(tank[q]), iou_here=float(all_iou[q, index]), owner_is_other=owner != index, owner_iou=float(all_iou[q, owner]), owner_is_tp_like=owner != index and all_iou[q, owner] >= .5))
                    wh = boxes[q, 2:] - boxes[q, :2]
                    owners[f"{r['status']}|{strict_stage(r)}"][-1].update(size_ratio=float(np.sqrt(wh.prod() / max((r['input_gw'] * r['input_gh']), 1e-6))), input_side=r['input_side'])
                for name, sample in trace['samples'].items():
                    if sample['gt_id'] == r['gt_id']:
                        report['samples'][name] = []
                        for item in sample['queries']:
                            q_ = item['query']
                            order = np.argsort(-all_iou[q_])[:2]
                            report['samples'][name].append(dict(query=q_, score=round(float(tank[q_]), 4), best_gts=[dict(gt_id=ids[o], iou=round(float(all_iou[q_, o]), 3), is_this=ids[o] == r['gt_id']) for o in order]))
            print('check', count + 1, len(by), flush=True)
    for key, part in owners.items():
        ratio = np.array([p['size_ratio'] for p in part])
        report['groups'][key] = dict(n=len(part), share_owner_other=round(float(np.mean([p['owner_is_other'] for p in part])), 3), share_owner_other_iou50=round(float(np.mean([p['owner_is_tp_like'] for p in part])), 3), median_score=round(float(np.median([p['score'] for p in part])), 3), median_iou_here=round(float(np.median([p['iou_here'] for p in part])), 3), median_size_ratio=round(float(np.median(ratio)), 3), share_box_larger=round(float((ratio > 1).mean()), 3), median_side=float(np.median([p['input_side'] for p in part])))
    for src, values in quality.items():
        iou, raw, lqe, score = (np.array(values[k]) for k in ('iou', 'raw', 'lqe', 'score'))
        good = iou >= .75
        fair = (iou >= .5) & ~good
        spearman = lambda a, b: round(float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1]), 3)
        report['quality'][src] = dict(claims=len(iou), spearman_score_iou=spearman(score, iou), spearman_raw_iou=spearman(raw, iou), spearman_lqe_iou=spearman(lqe, iou), lqe_std=round(float(lqe.std()), 3), raw_std=round(float(raw.std()), 3),
                                      lqe_median_by_iou={k: round(float(np.median(lqe[m])), 3) if m.any() else None for k, m in (('<.1', iou < .1), ('.1-.5', (iou >= .1) & (iou < .5)), ('.5-.75', fair), ('>=.75', good))},
                                      raw_median_by_iou={k: round(float(np.median(raw[m])), 3) if m.any() else None for k, m in (('<.1', iou < .1), ('.1-.5', (iou >= .1) & (iou < .5)), ('.5-.75', fair), ('>=.75', good))},
                                      share_by_iou={k: round(float(m.mean()), 3) for k, m in (('<.1', iou < .1), ('.1-.5', (iou >= .1) & (iou < .5)), ('.5-.75', fair), ('>=.75', good))},
                                      score_auc_iou50=round(auc(score[iou >= .5], score[iou < .5]), 3))
    (args.out / 'trace/checks.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=0), flush=True)


if __name__ == '__main__':
    main()
