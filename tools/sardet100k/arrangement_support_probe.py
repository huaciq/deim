"""D28: does the regular arrangement of man-made targets carry detection evidence the detector does not use yet?

Oil tanks stand in tank farms, cars in parking rows, aircraft on aprons, ships along quays: targets of one class repeat
at a regular spacing, clutter peaks do not. The DETR decoder can already relate queries by self-attention, so the
question is whether that evidence is already used. Measured directly on the frozen detector's saved val predictions
(top 300 per image, E000), no features, no training.

Lattice support of a candidate box c of class k, given anchors = confident detections of class k (score >= --anchor,
duplicates removed, those overlapping c excluded), all in original pixels, s = sqrt(w h) of c:
  extension      two anchors a, b with c ~ 2a - b                (c continues the row b -> a at the same spacing)
  interpolation  two anchors a, b with c ~ (a + b) / 2           (c fills a hole in a row)
  with |c - prediction| < .3 s, anchor spacing .6 s .. 4 s (extension) or 1.2 s .. 6 s (interpolation), and anchor
  sizes within x1.5 of c.

Analyses
  1. at the same detector score, are lattice-supported candidates more often true targets than unsupported ones?
     (TP: IoU >= .5 with a GT of the class; background: IoU < .1 with every GT)
  2. GT side: share of missed GT (no detection of the class with IoU >= .5 and score >= .3) that sit at a lattice
     position predicted by the anchors, against the share of background false positives (score >= .1) that do.
  3. rescoring: logit(s') = logit(s) + beta_k * [supported], beta_k per class chosen on one half of val (by image id
     parity) and applied to the other half, both ways; COCO AP on all GT and on GT without a near-duplicate in train.

Reading, fixed before running:
  supported   on both halves (beta tuned on the other half): mAP +0.3 or more, OGSOD tank AP +1.0 or more, AP50 not
              lower; and in the score band [.1, .3) lattice-supported OGSOD tank candidates are true targets at least
              twice as often as unsupported ones. Then a layout-aware relation module in the decoder is worth designing.
  refuted     mAP gain below +0.1 on either half: the decoder already uses arrangement, or clutter shares it.
  A rescoring rule is a weak stand-in for a learned module; a negative result removes the zero-training support only.

  python tools/sardet100k/arrangement_support_probe.py OUT --pred V.npz --d12 D12 [--anchor .5] [--workers 32]
"""
import argparse
import json
import multiprocessing as mp
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_report import iou_xyxy, r3  # noqa: E402

MIN_SCORE = .02
TOL = .3
BETAS = (0., .25, .5, 1., 1.5, 2., 3.)
BANDS = ((.02, .1), (.1, .3), (.3, .5), (.5, 1.01))
STATE = {}


def dedup(boxes, scores):
    keep = []
    for i in np.argsort(-scores):
        if all(iou_xyxy(boxes[i:i + 1], boxes[j:j + 1])[0, 0] < .5 for j in keep):
            keep.append(i)
    return np.array(keep, np.int64)


def support(cand, anchors):
    """cand, anchors: xyxy. Returns the number of anchor pairs predicting each candidate (extension + interpolation)."""
    out = np.zeros(len(cand), np.int64)
    if len(anchors) < 2 or len(cand) == 0:
        return out
    ca = (anchors[:, :2] + anchors[:, 2:]) / 2
    sa = np.sqrt(np.prod(anchors[:, 2:] - anchors[:, :2], 1))
    cc = (cand[:, :2] + cand[:, 2:]) / 2
    sc = np.sqrt(np.prod(cand[:, 2:] - cand[:, :2], 1))
    overlap = iou_xyxy(cand, anchors) > .3                                              # c x A
    d = np.linalg.norm(ca[:, None] - ca[None], axis=-1)                                 # A x A
    ext = 2 * ca[:, None] - ca[None]                                                    # a, b -> 2a - b
    mid = (ca[:, None] + ca[None]) / 2
    for i in range(len(cand)):
        s = sc[i]
        ok = (~overlap[i]) & (sa > s / 1.5) & (sa < s * 1.5)
        pair = ok[:, None] & ok[None] & ~np.eye(len(anchors), dtype=bool)
        e = pair & (d > .6 * s) & (d < 4 * s) & (np.linalg.norm(ext - cc[i], axis=-1) < TOL * s)
        m = pair & (d > 1.2 * s) & (d < 6 * s) & (np.linalg.norm(mid - cc[i], axis=-1) < TOL * s)
        out[i] = e.sum() + np.triu(m).sum()
    return out


def per_image(image_id):
    p_lab, p_score, p_box = STATE['pred'][image_id]
    gts = STATE['gt'].get(image_id, [])
    g_box = np.array([g[1] for g in gts]).reshape(-1, 4)
    g_lab = np.array([g[0] for g in gts], np.int64)
    sup = np.zeros(len(p_score), np.int64)
    gt_sup = np.zeros(len(gts), np.int64)
    for k in np.unique(np.concatenate([p_lab, g_lab])):
        m = p_lab == k
        a = m & (p_score >= STATE['anchor'])
        idx = np.flatnonzero(a)
        anchors = p_box[idx[dedup(p_box[idx], p_score[idx])]] if len(idx) else np.zeros((0, 4))
        sup[m] = support(p_box[m], anchors)
        gm = g_lab == k
        if gm.any():
            gt_sup[gm] = support(g_box[gm], anchors)
    iou = iou_xyxy(p_box, g_box) if len(gts) else np.zeros((len(p_score), 0))
    same = iou * (p_lab[:, None] == g_lab[None]) if len(gts) else iou
    tp = same.max(1) >= .5 if len(gts) else np.zeros(len(p_score), bool)
    bg = iou.max(1) < .1 if len(gts) else np.ones(len(p_score), bool)
    detected = ((same >= .5) & (p_score[:, None] >= .3)).any(0) if len(gts) else np.zeros(0, bool)
    return image_id, sup, tp, bg, gt_sup, detected


def coco_eval(gt_path, dets, image_ids, novel_of=None):
    from faster_coco_eval import COCO, COCOeval_faster
    gt = COCO(str(gt_path))
    names = {c['id']: c['name'].lower() for c in gt.dataset['categories']}
    if novel_of is not None:
        for a in gt.anns.values():
            a['area'] = 1. if novel_of[a['id']] else 100.
    i, c, s, b = dets
    wh = b[:, 2:] - b[:, :2]
    results = [dict(image_id=int(x), category_id=int(y), score=float(z), bbox=[float(q[0]), float(q[1]), float(w[0]), float(w[1])], area=1.)
               for x, y, z, q, w in zip(i, c, s, b, wh)]
    dt = gt.loadRes(results)
    for d in dt.anns.values():
        d['area'] = 1.
    e = COCOeval_faster(gt, dt, 'bbox')
    e.params.imgIds = sorted(image_ids)
    e.params.areaRng = [[0., 10.]] if novel_of is not None else [[0., 1e10]]
    e.params.areaRngLbl = ['all']
    e.evaluate()
    e.accumulate()
    pr = np.asarray(e.eval['precision'])
    last = len(e.params.maxDets) - 1
    out = {}
    for k, cat in enumerate(e.params.catIds):
        p = pr[:, :, k, 0, last]
        if (p > -1).any():
            out[names[cat]] = (p[p > -1].mean() * 100, p[0][p[0] > -1].mean() * 100)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    parser.add_argument('--pred', type=Path, required=True)
    parser.add_argument('--d12', type=Path, required=True)
    parser.add_argument('--anchor', type=float, default=.5)
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    from error_gallery import ROOT
    from novel_box_anatomy import cells_of
    gt_path = ROOT / 'Annotations/val.json'
    val = json.load(open(gt_path))
    names, cell_of = cells_of(val)
    info = {i['id']: i for i in val['images']}
    sim = np.load(args.d12 / 'similarity.npz')
    novel_of = {int(a): bool(s < .9) for a, s in zip(sim['val_ann'], sim['val_sim'])}
    gt = defaultdict(list)
    for a in val['annotations']:
        x, y, w, h = a['bbox']
        gt[a['image_id']].append((a['category_id'], (x, y, x + w, y + h), a['id']))
    with np.load(args.pred) as z:
        keep = z['score'] >= MIN_SCORE
        P_img, P_lab, P_score, P_box = z['image_id'][keep], z['label'][keep].astype(np.int64), z['score'][keep].astype(np.float64), z['box'][keep].astype(np.float64)
    order = np.argsort(P_img, kind='stable')
    P_img, P_lab, P_score, P_box = P_img[order], P_lab[order], P_score[order], P_box[order]
    ids, first = np.unique(P_img, return_index=True)
    bounds = dict(zip(ids.tolist(), zip(first.tolist(), list(first[1:]) + [len(P_img)])))
    STATE.update(pred={i: (P_lab[s:e], P_score[s:e], P_box[s:e]) for i, (s, e) in bounds.items()}, gt=gt, anchor=args.anchor)
    image_ids = sorted(info)
    for i in image_ids:
        STATE['pred'].setdefault(i, (np.zeros(0, np.int64), np.zeros(0), np.zeros((0, 4))))
    with mp.get_context('fork').Pool(args.workers) as pool:
        rows = pool.map(per_image, image_ids, chunksize=16)
    sup = {r[0]: r[1] for r in rows}
    group = {}
    for i in image_ids:
        w = info[i]['width']
        group[i] = 'ogsod' if w == 256 and info[i]['file_name'].endswith('.png') else ('256jpg' if w == 256 else str(w))
    report = dict(anchor=args.anchor)

    # 1. TP rate by score band, supported vs not
    table = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))     # key -> band -> [n_sup, tp_sup, n_un, tp_un]
    for image_id, s_, tp, bg, _, _ in rows:
        lab, score, _ = STATE['pred'][image_id]
        for k in range(len(score)):
            key = f"{group[image_id]}/{names[int(lab[k])]}"
            for lo, hi in BANDS:
                if lo <= score[k] < hi:
                    t = table[key][f'{lo}-{hi}']
                    if s_[k] > 0:
                        t[0] += 1
                        t[1] += int(tp[k])
                    else:
                        t[2] += 1
                        t[3] += int(tp[k])
    report['tp_rate'] = {}
    for key in sorted(table):
        report['tp_rate'][key] = {}
        for band, (ns, ts, nu, tu) in table[key].items():
            if ns >= 30:
                report['tp_rate'][key][band] = dict(n_supported=ns, tp_supported=r3(ts / ns), n_unsupported=nu, tp_unsupported=r3(tu / max(nu, 1)))
        if report['tp_rate'][key]:
            print('TP rate', key, report['tp_rate'][key], flush=True)

    # 2. GT side
    gside = defaultdict(lambda: [0, 0, 0, 0, 0, 0])                    # missed, missed&sup, detected, detected&sup, bgFP(>=.1), bgFP&sup
    for image_id, s_, tp, bg, gt_sup, detected in rows:
        for j, (c, _, aid) in enumerate(gt[image_id]):
            key = f"{group[image_id]}/{names[c]}{'/novel' if novel_of[aid] else '/seen'}"
            g = gside[key]
            if detected[j]:
                g[2] += 1
                g[3] += int(gt_sup[j] > 0)
            else:
                g[0] += 1
                g[1] += int(gt_sup[j] > 0)
        lab, score, _ = STATE['pred'][image_id]
        for k in np.flatnonzero(bg & (score >= .1)):
            g = gside[f"{group[image_id]}/{names[int(lab[k])]}/background"]
            g[4] += 1
            g[5] += int(s_[k] > 0)
    report['gt_side'] = {}
    for key in sorted(gside):
        m, ms, d, ds, f, fs = gside[key]
        if m + d + f >= 100:
            report['gt_side'][key] = dict(missed=m, missed_at_lattice=r3(ms / max(m, 1)), detected=d, detected_at_lattice=r3(ds / max(d, 1)), background_fp=f, fp_at_lattice=r3(fs / max(f, 1)))
            print('GT side', key, report['gt_side'][key], flush=True)

    # 3. rescoring, two-fold over image id parity
    S = np.concatenate([sup[i] for i in image_ids])
    I = np.concatenate([np.full(len(STATE['pred'][i][1]), i) for i in image_ids])
    L = np.concatenate([STATE['pred'][i][0] for i in image_ids])
    Sc = np.concatenate([STATE['pred'][i][1] for i in image_ids])
    B = np.concatenate([STATE['pred'][i][2] for i in image_ids])
    logit = np.log(np.clip(Sc, 1e-6, 1 - 1e-6) / np.clip(1 - Sc, 1e-6, 1))
    halves = {'even': [i for i in image_ids if i % 2 == 0], 'odd': [i for i in image_ids if i % 2 == 1]}
    grid = {}
    for beta in BETAS:
        s2 = 1 / (1 + np.exp(-(logit + beta * (S > 0))))
        for h, ids_h in halves.items():
            grid[(beta, h)] = coco_eval(gt_path, (I, L, s2, B), ids_h)
        print('beta', beta, {h: r3(np.mean([v[0] for v in grid[(beta, h)].values()])) for h in halves}, flush=True)
    classes = list(grid[(0., 'even')])
    report['rescoring'] = {}
    for tune, test in (('even', 'odd'), ('odd', 'even')):
        best = {c: max(BETAS, key=lambda b: grid[(b, tune)][c][0]) for c in classes}
        cat_id = {n: k for k, n in names.items()}
        beta_vec = np.zeros(len(L))
        for c, b in best.items():
            beta_vec[L == cat_id[c]] = b
        s2 = 1 / (1 + np.exp(-(logit + beta_vec * (S > 0))))
        base, new = grid[(0., test)], coco_eval(gt_path, (I, L, s2, B), halves[test])
        base_n, new_n = coco_eval(gt_path, (I, L, Sc, B), halves[test], novel_of), coco_eval(gt_path, (I, L, s2, B), halves[test], novel_of)
        ogs = [i for i in halves[test] if group[i] == 'ogsod']
        base_o, new_o = coco_eval(gt_path, (I, L, Sc, B), ogs), coco_eval(gt_path, (I, L, s2, B), ogs)
        res = dict(beta=best,
                   mAP=(r3(np.mean([v[0] for v in base.values()])), r3(np.mean([v[0] for v in new.values()]))),
                   AP50=(r3(np.mean([v[1] for v in base.values()])), r3(np.mean([v[1] for v in new.values()]))),
                   per_class={c: (r3(base[c][0]), r3(new[c][0])) for c in classes},
                   novel_mAP=(r3(np.mean([v[0] for v in base_n.values()])), r3(np.mean([v[0] for v in new_n.values()]))),
                   novel_per_class={c: (r3(base_n[c][0]), r3(new_n[c][0])) for c in base_n},
                   ogsod_per_class={c: (r3(base_o[c][0]), r3(new_o[c][0])) for c in base_o})
        report['rescoring'][f'tune_{tune}_test_{test}'] = res
        print('rescoring', tune, '->', test, res, flush=True)
    (args.out / f'arrangement_support_probe_anchor{args.anchor}.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
