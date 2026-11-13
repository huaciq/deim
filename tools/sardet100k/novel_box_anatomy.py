"""D19: why are targets without a training near-duplicate localised so much worse, and what would fixing it be worth?

D12's breakdown shows that novel targets (24x24 crop correlation < .9 with every train target) are found but badly boxed:
512 aircraft AP50 71.7 / AP75 19.3 (seen AP75 70.6), OGSOD bridge 72.0 / 16.4 (seen 52.7). CPU only, saved val predictions.

  anatomy   For every GT, its highest-scoring same-class detection with IoU >= .3: log size ratio, centre offset in GT
            sizes, and the IoU that would result from correcting only the centre or only the size. Seen vs novel per cell.
  oracle    COCO AP after replacing, for detections assigned to a GT (same class, IoU >= .3), the size / the centre /
            both by the GT's, applied to detections on novel GT only, on seen GT only, or on all. Diagnostic upper bounds.
  group     No GT used: in each image and class, detections with score >= .5 vote a median width and height; every
            detection of that class in the image takes the group size (or a blend). Tests whether same-class neighbours
            carry usable size information. Also reports how consistent GT sizes are within an image.

  python tools/sardet100k/novel_box_anatomy.py OUT --d12 D12 --pred E000=V.npz [E008=...]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_report import iou_xyxy, r3  # noqa: E402


def cells_of(val):
    names = {c['id']: c['name'].lower() for c in val['categories']}
    info = {i['id']: i for i in val['images']}

    def group(i):
        if i['width'] == 256:
            return 'ogsod' if i['file_name'].endswith('.png') else '256jpg'
        return str(i['width']) if i['width'] in (224, 512, 800, 1000) else 'misc'
    return names, {a['id']: f"{group(info[a['image_id']])}/{names[a['category_id']]}" for a in val['annotations']}


def coco_ap(path, image_id, label, score, box):
    from faster_coco_eval import COCO, COCOeval_faster
    gt = COCO(str(path))
    names = {c['id']: c['name'].lower() for c in gt.dataset['categories']}
    wh = box[:, 2:] - box[:, :2]
    results = [dict(image_id=int(i), category_id=int(c), score=float(s), bbox=[float(b[0]), float(b[1]), float(w[0]), float(w[1])])
               for i, c, s, b, w in zip(image_id, label, score, box, wh)]
    e = COCOeval_faster(gt, gt.loadRes(results), 'bbox')
    e.evaluate()
    e.accumulate()
    precision = np.asarray(e.eval['precision'])
    last = len(e.params.maxDets) - 1
    out = {}
    for k, c in enumerate(e.params.catIds):
        pr = precision[:, :, k, 0, last]
        out[names[c]] = dict(AP=pr[pr > -1].mean() * 100, AP50=pr[0][pr[0] > -1].mean() * 100, AP75=pr[5][pr[5] > -1].mean() * 100)
    out['mean'] = {m: float(np.mean([v[m] for v in out.values()])) for m in ('AP', 'AP50', 'AP75')}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    parser.add_argument('--d12', type=Path, required=True)
    parser.add_argument('--pred', nargs='+', required=True)
    parser.add_argument('--skip-oracle', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    from error_gallery import ROOT
    path = ROOT / 'Annotations/val.json'
    val = json.load(open(path))
    names, cell_of = cells_of(val)
    sim = np.load(args.d12 / 'similarity.npz')
    novel_of = {int(a): bool(s < .9) for a, s in zip(sim['val_ann'], sim['val_sim'])}
    by_image = defaultdict(list)
    for a in val['annotations']:
        by_image[a['image_id']].append(a)
    report = {}

    # how consistent are GT sizes among same-class objects of one image?
    consistency = defaultdict(list)
    for anns in by_image.values():
        per_class = defaultdict(list)
        for a in anns:
            per_class[a['category_id']].append(a)
        for group in per_class.values():
            if len(group) < 3:
                for a in group:
                    consistency[cell_of[a['id']]].append((0, np.nan, novel_of[a['id']]))
                continue
            w, h = np.array([a['bbox'][2] for a in group]), np.array([a['bbox'][3] for a in group])
            mw, mh = np.median(w), np.median(h)
            for a in group:
                dev = max(abs(np.log(a['bbox'][2] / mw)), abs(np.log(a['bbox'][3] / mh)))
                consistency[cell_of[a['id']]].append((1, dev, novel_of[a['id']]))
    table = {}
    for c, rows in sorted(consistency.items()):
        rows = np.array(rows, float)
        if len(rows) < 200:
            continue
        grouped = rows[:, 0] == 1
        table[c] = dict(n=len(rows), in_group_of_3=r3(grouped.mean() * 100),
                        within_10pct_of_group_median=r3((rows[grouped, 1] < np.log(1.1)).mean() * 100) if grouped.any() else None,
                        within_20pct=r3((rows[grouped, 1] < np.log(1.2)).mean() * 100) if grouped.any() else None)
        print('GT size consistency', f'{c:18s}', table[c], flush=True)
    report['gt_size_consistency'] = table

    for spec in args.pred:
        run, npz = spec.split('=')
        with np.load(npz) as z:
            order = np.argsort(z['image_id'], kind='stable')
            p_img, p_label, p_score, p_box = z['image_id'][order], z['label'][order], z['score'][order], z['box'][order].astype(np.float64)
        starts = {int(i): (int(np.searchsorted(p_img, i, 'left')), int(np.searchsorted(p_img, i, 'right'))) for i in np.unique(p_img)}

        # assignment: each detection -> same-class GT with the highest IoU, if >= .3
        assigned = np.full(len(p_img), -1, np.int64)          # index into a flat GT list
        gt_flat, gt_box = [], []
        anatomy = defaultdict(list)
        for image_id, anns in by_image.items():
            if image_id not in starts:
                continue
            s, e = starts[image_id]
            g = np.array([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]] for a in anns])
            iou = iou_xyxy(p_box[s:e], g)
            iou[p_label[s:e][:, None] != np.array([a['category_id'] for a in anns])[None]] = 0
            best = iou.argmax(1)
            ok = iou[np.arange(e - s), best] >= .3
            base = len(gt_flat)
            assigned[s:e][ok] = base + best[ok]
            gt_flat.extend(anns)
            gt_box.extend(g)
            for k, a in enumerate(anns):
                mine = np.flatnonzero(ok & (best == k))
                if not len(mine):
                    continue
                top = mine[np.argmax(p_score[s:e][mine])]
                b, t = p_box[s + top], g[k]
                bw, bh, tw, th = b[2] - b[0], b[3] - b[1], t[2] - t[0], t[3] - t[1]
                bc, tc = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), ((t[0] + t[2]) / 2, (t[1] + t[3]) / 2)
                centre_fixed = np.array([[tc[0] - bw / 2, tc[1] - bh / 2, tc[0] + bw / 2, tc[1] + bh / 2]])
                size_fixed = np.array([[bc[0] - tw / 2, bc[1] - th / 2, bc[0] + tw / 2, bc[1] + th / 2]])
                anatomy[(cell_of[a['id']], novel_of[a['id']])].append(
                    (iou[top, k], iou_xyxy(centre_fixed, t[None])[0, 0], iou_xyxy(size_fixed, t[None])[0, 0],
                     np.log(bw / tw), np.log(bh / th), (bc[0] - tc[0]) / tw, (bc[1] - tc[1]) / th, max(tw, th)))
        gt_box = np.array(gt_box)
        rows = {}
        for (c, novel), v in sorted(anatomy.items()):
            v = np.array(v)
            if len(v) < 100:
                continue
            size_err = np.maximum(np.abs(v[:, 3]), np.abs(v[:, 4]))
            centre_err = np.maximum(np.abs(v[:, 5]), np.abs(v[:, 6]))
            rows[f"{c} {'novel' if novel else 'seen'}"] = dict(
                n=len(v), iou=r3(np.median(v[:, 0])), iou_if_centre_fixed=r3(np.median(v[:, 1])), iou_if_size_fixed=r3(np.median(v[:, 2])),
                size_bias_w=r3(np.median(v[:, 3])), size_bias_h=r3(np.median(v[:, 4])), size_err_median=r3(np.median(size_err)),
                centre_err_median=r3(np.median(centre_err)), gt_long_side=r3(np.median(v[:, 7])))
            print(run, 'anatomy', f"{c:16s} {'novel' if novel else 'seen ':5s}", rows[f"{c} {'novel' if novel else 'seen'}"], flush=True)
        report[run] = dict(anatomy=rows)
        if args.skip_oracle:
            continue

        def evaluate(tag, box):
            res = coco_ap(path, p_img, p_label, p_score, box)
            report[run].setdefault('ap', {})[tag] = {k: {m: r3(x) for m, x in v.items()} for k, v in res.items()}
            print(run, f'{tag:34s}', {m: round(v, 2) for m, v in res['mean'].items()}, ' '.join(f"{k} {v['AP']:.1f}" for k, v in res.items() if k != 'mean'), flush=True)

        evaluate('baseline', p_box)
        has = assigned >= 0
        g = gt_box[np.where(has, assigned, 0)]
        novel_det = np.array([novel_of[gt_flat[k]['id']] if k >= 0 else False for k in assigned])
        pc = np.stack([(p_box[:, 0] + p_box[:, 2]) / 2, (p_box[:, 1] + p_box[:, 3]) / 2], 1)
        pwh = p_box[:, 2:] - p_box[:, :2]
        gc = np.stack([(g[:, 0] + g[:, 2]) / 2, (g[:, 1] + g[:, 3]) / 2], 1)
        gwh = g[:, 2:] - g[:, :2]
        for scope, m in (('novel GT', has & novel_det), ('seen GT', has & ~novel_det), ('all GT', has)):
            for kind, centre, wh in (('size <- GT', pc, gwh), ('centre <- GT', gc, pwh), ('box <- GT', gc, gwh)):
                box = p_box.copy()
                box[m] = np.concatenate([centre[m] - wh[m] / 2, centre[m] + wh[m] / 2], 1)
                evaluate(f'oracle {kind} on {scope}', box)

        # group size voting, no GT
        for blend in (1., .5):
            for min_votes in (3,):
                box = p_box.copy()
                changed = 0
                for image_id, (s, e) in starts.items():
                    for c in np.unique(p_label[s:e]):
                        sel = np.flatnonzero(p_label[s:e] == c) + s
                        voters = sel[p_score[sel] >= .5]
                        if len(voters) < min_votes:
                            continue
                        vw, vh = np.median(pwh[voters, 0]), np.median(pwh[voters, 1])
                        spread = max(np.abs(np.log(pwh[voters, 0] / vw)).max(), np.abs(np.log(pwh[voters, 1] / vh)).max()) if len(voters) else 9
                        if spread > np.log(1.3):       # only groups whose confident members agree on size
                            continue
                        near = sel[(np.abs(np.log(pwh[sel, 0] / vw)) < np.log(1.5)) & (np.abs(np.log(pwh[sel, 1] / vh)) < np.log(1.5))]
                        w = blend * vw + (1 - blend) * pwh[near, 0]
                        h = blend * vh + (1 - blend) * pwh[near, 1]
                        box[near] = np.stack([pc[near, 0] - w / 2, pc[near, 1] - h / 2, pc[near, 0] + w / 2, pc[near, 1] + h / 2], 1)
                        changed += len(near)
                evaluate(f'group size vote blend {blend} ({changed} boxes)', box)
    (args.out / 'novel_box_anatomy.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
