"""How common is "high score, worse box; better box, low score" across SARDet-100K val, and where do MAL targets go?

E000 EMA weights (best_stg2), last decoder layer, all val images. For each GT of class c, its candidates are the
queries whose highest-IoU GT among the class-c GTs of the image is this GT, with IoU >= 0.1; a query's class-c score
is sigmoid(logit_c) (the score the postprocessor ranks). Per GT:
  top  = candidate with the highest class-c score;  best = candidate with the highest IoU.
The final layer is also matched with the training matcher (all GTs of every class): the matched query is the only
query with a positive MAL target for that GT at this layer; every other query has target 0 for all classes.

Severity: a swap oracle (not deployable) gives each GT's top-scoring query the best candidate's box and the best
candidate the top query's box, for that class only. Scores and the detection list (postprocessor top 300) are
unchanged, so the AP change is what ranking the right box first is worth, without fixing duplicates or background.

Usage (repository root on the server; OUT directly under E000's independent_verify_20261106/):
  collect RUN OUT [--device cpu] [--batch 4]
  analyse RUN OUT
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_report import GROUP_ORDER, group_of, iou_xyxy, r3, sigmoid, spearman  # noqa: E402


def collect(args):
    import torch
    from engine.core import YAMLConfig
    from error_gallery import ROOT
    from tank_feature_diagnosis import load_model, load_val, to_input, xyxy
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    annotations, source, grouped = load_val(out)
    model, _ = load_model(args.run, args.device)
    matcher = YAMLConfig(str(args.run / 'resolved_config.yml')).criterion.matcher
    logits_all, boxes_all, ids, matched = [], [], [], {}
    images = annotations['images']
    batch = []
    with torch.no_grad():
        for n, image in enumerate(images):
            x, _ = to_input(ROOT / 'JPEGImages/val' / image['file_name'])
            batch.append((image, x))
            if len(batch) == args.batch or n == len(images) - 1:
                result = model(torch.stack([b[1] for b in batch]).to(args.device))
                for k, (img, _) in enumerate(batch):
                    logits, boxes = result['pred_logits'][k], result['pred_boxes'][k]
                    anns = grouped[img['id']]
                    if anns:
                        w, h = img['width'], img['height']
                        xy = np.array([[a['bbox'][0] / w, a['bbox'][1] / h, (a['bbox'][0] + a['bbox'][2]) / w, (a['bbox'][1] + a['bbox'][3]) / h] for a in anns], np.float32)
                        cxcywh = np.stack([(xy[:, 0] + xy[:, 2]) / 2, (xy[:, 1] + xy[:, 3]) / 2, xy[:, 2] - xy[:, 0], xy[:, 3] - xy[:, 1]], 1)
                        targets = [dict(labels=torch.tensor([a['category_id'] for a in anns], device=args.device), boxes=torch.tensor(cxcywh, device=args.device))]
                        q, g = matcher({'pred_logits': logits[None], 'pred_boxes': boxes[None]}, targets)['indices'][0]
                        matched[img['id']] = (q.tolist(), g.tolist())
                    logits_all.append(logits.float().cpu().numpy().astype(np.float16))
                    boxes_all.append(xyxy(boxes.float().cpu().numpy(), 1).astype(np.float32))
                    ids.append(img['id'])
                batch = []
            if (n + 1) % 1000 == 0:
                print('images', n + 1, len(images), flush=True)
    np.savez_compressed(out / 'queries.npz', image_id=np.array(ids), logits=np.stack(logits_all), boxes=np.stack(boxes_all))
    (out / 'matched.json').write_text(json.dumps({str(k): v for k, v in matched.items()}))
    print('collected', flush=True)


def analyse(args):
    from faster_coco_eval import COCO, COCOeval_faster
    from error_gallery import ROOT
    from tank_feature_diagnosis import load_val
    out = args.out
    annotations, source, grouped = load_val(out)
    names = {c['id']: c['name'] for c in annotations['categories']}
    info = {i['id']: i for i in annotations['images']}
    groups = {i['id']: group_of(source[i['id']], i['width'], i['height']) for i in annotations['images']}
    # Decompress each array once: indexing an NpzFile key re-reads the whole array on every access.
    with np.load(out / 'queries.npz') as saved:
        data = {k: saved[k] for k in saved.files}
    matched = {int(k): v for k, v in json.loads((out / 'matched.json').read_text()).items()}
    rows = defaultdict(lambda: defaultdict(list))
    swap = {}  # image -> {(query, class): new box}
    pooled = defaultdict(lambda: ([], []))
    for n, image_id in enumerate(data['image_id'].tolist()):
        anns = grouped[image_id]
        if not anns:
            continue
        w, h = info[image_id]['width'], info[image_id]['height']
        gt = np.array([[a['bbox'][0] / w, a['bbox'][1] / h, (a['bbox'][0] + a['bbox'][2]) / w, (a['bbox'][1] + a['bbox'][3]) / h] for a in anns], np.float32)
        labels = np.array([a['category_id'] for a in anns])
        boxes = data['boxes'][n]
        score = sigmoid(data['logits'][n].astype(np.float32))
        q_pos, g_pos = matched.get(image_id, ([], []))
        positive = dict(zip(g_pos, q_pos))
        g = groups[image_id]
        for c in np.unique(labels):
            idx = np.flatnonzero(labels == c)
            m = iou_xyxy(boxes, gt[idx])
            owner = m.argmax(1)
            best_iou = m.max(1)
            for j, t in enumerate(idx):
                cand = np.flatnonzero((owner == j) & (best_iou >= .1))
                key = (g, names[c])
                if not len(cand):
                    rows[key]['no_candidate'].append(1.)
                    continue
                rows[key]['no_candidate'].append(0.)
                s, iou = score[cand, c], m[cand, j]
                top, best = cand[np.argmax(s)], cand[np.argmax(iou)]
                gap = m[best, j] - m[top, j]
                rows[key]['beaten_005'].append(float(gap >= .05))
                rows[key]['serious'].append(float(m[top, j] < .75 <= m[best, j]))
                rows[key]['iou_lost'].append(gap)
                rows[key]['top_iou'].append(m[top, j])
                rows[key]['best_iou'].append(m[best, j])
                rows[key]['candidates'].append(len(cand))
                pos = positive.get(int(t))
                rows[key]['best_is_mal_positive'].append(float(pos == best))
                rows[key]['top_is_mal_positive'].append(float(pos == top))
                if gap >= .05:
                    rows[key]['beaten_best_is_mal_negative'].append(float(pos != best))
                    rows[key]['beaten_score_gap'].append(float(score[top, c] - score[best, c]))
                pooled[key][0].extend(s.tolist())
                pooled[key][1].extend(iou.tolist())
                if top != best:
                    swap.setdefault(image_id, {})[(int(top), int(c))] = boxes[best]
                    swap[image_id][(int(best), int(c))] = boxes[top]

    # Mismatch statistics first, so they survive even if the AP step is interrupted.
    (out / 'mismatch_rows.json').write_text(json.dumps({'/'.join(k): {kk: [float(x) for x in vv] for kk, vv in v.items()} for k, v in rows.items()}))

    # Detection lists: postprocessor top 300 over (query, class); boxes in original pixels. Built and evaluated one
    # group at a time (only that group's images), which keeps memory and time bounded.
    gt_coco = COCO(str(ROOT / 'Annotations/val.json'))
    ap = {}
    members = defaultdict(list)
    for image_id, g in groups.items():
        members[g].append(image_id)
    row_of = {int(i): n for n, i in enumerate(data['image_id'].tolist())}
    members['all'] = list(groups)
    for g in [x for x in GROUP_ORDER if x in members] + ['all']:
        for name in ('baseline', 'swap_oracle'):
            res = []
            for image_id in members[g]:
                n = row_of[image_id]
                w, h = info[image_id]['width'], info[image_id]['height']
                score = sigmoid(data['logits'][n].astype(np.float32))
                for f in np.argsort(-score.ravel())[:300]:
                    q, c = divmod(int(f), score.shape[1])
                    b = data['boxes'][n][q] if name == 'baseline' else swap.get(image_id, {}).get((q, c), data['boxes'][n][q])
                    res.append(dict(image_id=image_id, category_id=c, score=float(score[q, c]),
                                    bbox=[float(b[0] * w), float(b[1] * h), float((b[2] - b[0]) * w), float((b[3] - b[1]) * h)]))
            e = COCOeval_faster(gt_coco, gt_coco.loadRes(res), 'bbox')
            e.params.imgIds = sorted(members[g])
            e.evaluate()
            e.accumulate()
            p, r = np.asarray(e.eval['precision']), np.asarray(e.eval['recall'])
            for k, c in enumerate(e.params.catIds):
                pk, rk = p[:, :, k, 0, -1], r[:, k, 0, -1]
                if (rk < 0).all():
                    continue
                ap[name, g, names[c]] = (float(pk[pk > -1].mean() * 100), float(pk[5][pk[5] > -1].mean() * 100))
            print('evaluated', g, name, flush=True)
    report = {}
    for key in sorted(rows, key=lambda k: (GROUP_ORDER.index(k[0]) if k[0] in GROUP_ORDER else 99, k[1])):
        v = rows[key]
        n = len(v['beaten_005'])
        base, oracle = ap.get(('baseline',) + key), ap.get(('swap_oracle',) + key)
        report['/'.join(key)] = dict(
            gt=len(v['no_candidate']), no_candidate=r3(np.mean(v['no_candidate'])), candidates_median=r3(np.median(v['candidates'])) if n else None,
            top_beaten_by_005=r3(np.mean(v['beaten_005'])) if n else None, serious_top_below_075_best_above=r3(np.mean(v['serious'])) if n else None,
            iou_lost_mean=r3(np.mean(v['iou_lost'])) if n else None, top_iou_median=r3(np.median(v['top_iou'])) if n else None,
            best_iou_median=r3(np.median(v['best_iou'])) if n else None,
            spearman_score_iou=r3(spearman(*pooled[key])) if n else None,
            mal_positive_is_best=r3(np.mean(v['best_is_mal_positive'])) if n else None, mal_positive_is_top=r3(np.mean(v['top_is_mal_positive'])) if n else None,
            beaten_best_is_mal_negative=r3(np.mean(v['beaten_best_is_mal_negative'])) if v['beaten_best_is_mal_negative'] else None,
            beaten_score_gap_median=r3(np.median(v['beaten_score_gap'])) if v['beaten_score_gap'] else None,
            ap=r3(base[0]) if base else None, ap_swap=r3(oracle[0]) if oracle else None, ap75=r3(base[1]) if base else None, ap75_swap=r3(oracle[1]) if oracle else None)
    for g in [x for x in GROUP_ORDER if x in members] + ['all']:
        cls = sorted({k[2] for k in ap if k[1] == g})
        if cls:
            report[f'{g}/mean'] = dict(ap=r3(np.mean([ap['baseline', g, c][0] for c in cls])), ap_swap=r3(np.mean([ap['swap_oracle', g, c][0] for c in cls])),
                                       ap75=r3(np.mean([ap['baseline', g, c][1] for c in cls])), ap75_swap=r3(np.mean([ap['swap_oracle', g, c][1] for c in cls])))
    # GT-weighted totals over all groups and classes.
    allv = defaultdict(list)
    for v in rows.values():
        for k in ('beaten_005', 'serious', 'iou_lost', 'best_is_mal_positive'):
            allv[k] += v[k]
    report['all_gt'] = {k: r3(np.mean(x)) for k, x in allv.items()} | dict(gt=sum(len(v['no_candidate']) for v in rows.values()))
    (out / 'report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('collect', 'analyse'))
    parser.add_argument('run', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch', type=int, default=4)
    args = parser.parse_args()
    {'collect': collect, 'analyse': analyse}[args.command](args)


if __name__ == '__main__':
    main()
