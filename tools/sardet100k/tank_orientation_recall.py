"""Does OGSOD tank performance depend on whether the training set holds this tile in this orientation?

Usage: python tools/sardet100k/tank_orientation_recall.py RUN OUT

Training applies random horizontal flips, so a val tile counts as seen in its orientation when train holds it as is or flipped
left-right. "Rotated only" means train holds it only under another orientation, so the network never saw these pixels this way up.
Val copies of one tile under different orientations give the same targets and pixels under both conditions, a paired comparison.
Reads saved predictions and images only; nothing is retrained.
"""
import json
import multiprocessing as mp
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from error_gallery import ROOT  # noqa: E402
from tank_dihedral_layout import ORIENTATIONS, orient_array, orient_boxes, thumb_array, unit  # noqa: E402

RUN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
SAME = .98
SEEN_ORIENTATIONS = ('identity', 'flip_x')


def native_difference(task):
    a, b, name = task
    x = np.asarray(Image.open(a).convert('L'), np.float32)
    y = np.asarray(Image.open(b).convert('L'), np.float32)
    x = orient_array(x, name)
    return float(np.abs(x - y).mean()) if x.shape == y.shape else float('nan')


def binom_two_sided(k, n):
    """Exact sign test on discordant pairs."""
    from math import comb
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1., 2 * tail)


def main():
    rows = [r for r in json.load(open(OUT / 'gt_rows.json')) if r['source'] == 'ogsod']
    by = defaultdict(list)
    for r in rows:
        by[r['file_name']].append(r)
    files = sorted(by)
    train = json.load(open(ROOT / 'Annotations/train.json'))
    names = {i['id']: i['file_name'] for i in train['images'] if i['width'] == 256 and i['height'] == 256}
    train_files = sorted({names[a['image_id']] for a in train['annotations'] if a['category_id'] == 3 and a['image_id'] in names})
    with mp.Pool(32) as pool:
        val_thumbs = pool.map(thumb_array, [str(ROOT / 'JPEGImages/val' / f) for f in files], chunksize=4)
        train_thumbs = pool.map(thumb_array, [str(ROOT / 'JPEGImages/train' / f) for f in train_files], chunksize=32)
    matrix = np.stack([unit(t) for t in train_thumbs]).astype(np.float32)
    info = {}
    for f, thumb in zip(files, val_thumbs):
        scores = {name: matrix @ unit(orient_array(thumb, name)).astype(np.float32) for name in ORIENTATIONS}
        best = {name: float(v.max()) for name, v in scores.items()}
        seen = max(best[n] for n in SEEN_ORIENTATIONS)
        other = max(best[n] for n in ORIENTATIONS if n not in SEEN_ORIENTATIONS)
        if seen >= SAME:
            group = 'seen_orientation'
        elif other >= SAME:
            group = 'rotated_only'
        else:
            group = 'no_match'
        pick = max(SEEN_ORIENTATIONS, key=lambda n: best[n]) if group == 'seen_orientation' else max(ORIENTATIONS, key=lambda n: best[n])
        info[f] = dict(group=group, seen_corr=round(seen, 4), other_corr=round(other, 4), orientation=pick, train_file=train_files[int(scores[pick].argmax())])
    with mp.Pool(16) as pool:
        differences = pool.map(native_difference, [(str(ROOT / 'JPEGImages/val' / f), str(ROOT / 'JPEGImages/train' / info[f]['train_file']), info[f]['orientation']) for f in files], chunksize=2)
    for f, d in zip(files, differences):
        info[f]['native_mad'] = round(d, 2)  # Near zero: the same file. Large with high correlation: another speckle version of the tile.
    report = dict(train_tank_tiles=len(train_files), images={f: {**info[f], 'gt': len(by[f]), 'recall': round(float(np.mean([r['status'] == 'TP' for r in by[f]])), 3)} for f in files})

    groups = report['groups'] = {}
    for g in ('seen_orientation', 'rotated_only', 'no_match'):
        part = [r for f in files if info[f]['group'] == g for r in by[f] if r['native_gw'] > 0 and r['native_gh'] > 0]
        if not part:
            continue
        tp = [r['tp_iou'] for r in part if r['status'] == 'TP']
        groups[g] = dict(images=len({r['file_name'] for r in part}), gt=len(part), recall=round(float(np.mean([r['status'] == 'TP' for r in part])), 3), status=dict(Counter(r['status'] for r in part)), tp_iou=round(float(np.median(tp)), 3) if tp else None, tp_score=round(float(np.median([r['tp_score'] for r in part if r['status'] == 'TP'])), 3) if tp else None,
                         enc_prob=round(float(np.median([r['enc_prob_max_in_gt'] for r in part])), 3), side=float(np.median([r['input_side'] for r in part])), nn=round(float(np.median([r['nn_input'] for r in part])), 1),
                         by_side={f'{a}-{b}': dict(gt=sum(a <= r['input_side'] < b for r in part), recall=round(float(np.mean([r['status'] == 'TP' for r in part if a <= r['input_side'] < b])), 3) if any(a <= r['input_side'] < b for r in part) else None) for a, b in ((0, 16), (16, 24), (24, 1e5))})
    for label, test in (('exact_file_mad_lt_2', lambda f: info[f]['group'] == 'seen_orientation' and info[f]['native_mad'] < 2), ('other_version_mad_ge_2', lambda f: info[f]['group'] == 'seen_orientation' and info[f]['native_mad'] >= 2)):
        part = [r for f in files if test(f) for r in by[f] if r['native_gw'] > 0 and r['native_gh'] > 0]
        groups[label] = dict(images=len({r['file_name'] for r in part}), gt=len(part), recall=round(float(np.mean([r['status'] == 'TP' for r in part])), 3) if part else None)

    # COCO AP on each group, from the saved predictions.
    from faster_coco_eval import COCO
    from source_class_eval import long_side_gt, run_eval
    annotations = json.load(open(ROOT / 'Annotations/val.json'))
    source = {int(k): v for k, v in json.load(open(OUT.parent / 'error_gallery_val/source.json')).items()}
    file_to_id = {i['file_name']: i['id'] for i in annotations['images']}
    saved = np.load(OUT.parent / 'val_predictions.npz')
    results = [dict(image_id=int(i), category_id=int(c), score=float(s), bbox=[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]) for i, c, s, b in zip(saved['image_id'], saved['label'], saved['score'], saved['box'])]
    gt = COCO(str(ROOT / 'Annotations/val.json'))
    scales = {i['id']: 640 / max(i['width'], i['height']) for i in annotations['images']}
    long_side_gt(gt, scales, False)
    dt = gt.loadRes(results)
    long_side_gt(dt, scales, False)
    report['ap'] = {}
    for g in ('seen_orientation', 'rotated_only', 'no_match'):
        ids = [file_to_id[f] for f in files if info[f]['group'] == g]
        if ids:
            metrics = run_eval(gt, dt, ids, [3]).get((3, 'all'))
            report['ap'][g] = {k: round(v, 2) for k, v in metrics.items()}
    report['ap']['all_tank_images'] = {k: round(v, 2) for k, v in run_eval(gt, dt, [file_to_id[f] for f in files], [3])[(3, 'all')].items()}

    # Paired: val copies of one tile, matched GT by GT through the orientation that aligns them.
    def key(r):
        return np.array([[r['gt_x1'], r['gt_y1'], r['gt_x2'], r['gt_y2']]], np.float32)

    pair_rows = []
    tally = defaultdict(lambda: Counter())
    encoder_totals = defaultdict(list)
    pairs_seen = set()
    units = [unit(t) for t in val_thumbs]
    for i, a in enumerate(files):
        for j, b in enumerate(files):
            if j <= i:
                continue
            corr = {name: float(unit(orient_array(val_thumbs[i], name)) @ units[j]) for name in ORIENTATIONS}
            name = max(corr, key=corr.get)
            if corr[name] < SAME or (a, b) in pairs_seen:
                continue
            pairs_seen.add((a, b))
            target = [r for r in by[b] if r['native_gw'] > 0 and r['native_gh'] > 0]
            source_rows = [r for r in by[a] if r['native_gw'] > 0 and r['native_gh'] > 0]
            if not target or not source_rows:
                continue
            centers_b = np.array([[(r['gt_x1'] + r['gt_x2']) / 2, (r['gt_y1'] + r['gt_y2']) / 2] for r in target])
            # Boxes turned by the tile transform can sit a constant distance off, so estimate that shift first.
            turned_all = orient_boxes(np.concatenate([key(r) for r in source_rows]), name)
            centers_a = (turned_all[:, :2] + turned_all[:, 2:]) / 2
            differences_ab = np.round(centers_b[None] - centers_a[:, None]).reshape(-1, 2)
            values, counts = np.unique(differences_ab, axis=0, return_counts=True)
            label_shift = values[counts.argmax()]
            matched = []
            for r, c in zip(source_rows, centers_a + label_shift):
                d = np.abs(centers_b - c).max(1)
                if d.min() <= 1.5:
                    matched.append((r, target[int(d.argmin())]))
            pixel_a = orient_array(np.asarray(Image.open(ROOT / 'JPEGImages/val' / a).convert('L'), np.float32), name)
            pixel_b = np.asarray(Image.open(ROOT / 'JPEGImages/val' / b).convert('L'), np.float32)
            best_shift, best_score = (0, 0), -2.
            for dy in range(-5, 6):
                for dx in range(-5, 6):
                    # pixel_a moved by (dx, dy) lands on pixel_b, the same sense as label_shift.
                    p = pixel_a[max(-dy, 0):256 - max(dy, 0), max(-dx, 0):256 - max(dx, 0)].ravel()
                    q = pixel_b[max(dy, 0):256 + min(dy, 0), max(dx, 0):256 + min(dx, 0)].ravel()
                    score = float(np.corrcoef(p, q)[0, 1])
                    if score > best_score:
                        best_shift, best_score = (dx, dy), score
            kinds = tuple(sorted((info[a]['group'], info[b]['group'])))
            first_is_seen = info[a]['group'] == 'seen_orientation'
            encoder_pairs = defaultdict(list)
            for x, y in matched:
                seen_row, other_row = (x, y) if first_is_seen else (y, x)
                tally[kinds][(seen_row['status'] == 'TP', other_row['status'] == 'TP')] += 1
                encoder_pairs['seen'].append(seen_row['enc_prob_max_in_gt'])
                encoder_pairs['other'].append(other_row['enc_prob_max_in_gt'])
                encoder_totals[kinds].append((seen_row['enc_prob_max_in_gt'], other_row['enc_prob_max_in_gt'], seen_row['tp_iou'], other_row['tp_iou']))
            pair_rows.append(dict(a=a, b=b, orientation=name, corr=round(corr[name], 3), group_a=info[a]['group'], group_b=info[b]['group'], recall_a=round(float(np.mean([r['status'] == 'TP' for r in source_rows])), 3), recall_b=round(float(np.mean([r['status'] == 'TP' for r in target])), 3), matched=len(matched), of=len(source_rows),
                                  enc_a=round(float(np.median([r['enc_prob_max_in_gt'] for r in source_rows])), 3), enc_b=round(float(np.median([r['enc_prob_max_in_gt'] for r in target])), 3), mad_a=info[a]['native_mad'], mad_b=info[b]['native_mad'],
                                  label_shift=[float(v) for v in label_shift], pixel_shift=list(best_shift), pixel_corr=round(best_score, 4), label_minus_pixel=[float(label_shift[0] - best_shift[0]), float(label_shift[1] - best_shift[1])]))
    report['pairs'] = pair_rows
    report['encoder_pairs'] = {'|'.join(k): dict(n=len(v), seen_enc=round(float(np.median([x[0] for x in v])), 3), other_enc=round(float(np.median([x[1] for x in v])), 3), seen_tp_iou=round(float(np.nanmedian([x[2] for x in v])), 3) if np.isfinite([x[2] for x in v]).any() else None, other_tp_iou=round(float(np.nanmedian([x[3] for x in v])), 3) if np.isfinite([x[3] for x in v]).any() else None) for k, v in encoder_totals.items()}
    aligned = [p for p in pair_rows if p['matched'] >= .8 * p['of'] and p['pixel_corr'] >= .5]
    report['label_vs_pixel'] = dict(pairs=len(aligned), by_orientation={name: dict(n=sum(p['orientation'] == name for p in aligned), label_minus_pixel=dict(Counter(str(tuple(p['label_minus_pixel'])) for p in aligned if p['orientation'] == name))) for name in ORIENTATIONS if any(p['orientation'] == name for p in aligned)})
    report['paired_counts'] = {}
    for kinds, c in tally.items():
        hit_only_first, hit_only_second = c[(True, False)], c[(False, True)]
        report['paired_counts']['|'.join(kinds)] = dict(both_hit=c[(True, True)], first_only=hit_only_first, second_only=hit_only_second, both_miss=c[(False, False)], sign_test_p=round(binom_two_sided(hit_only_first, hit_only_first + hit_only_second), 6) if hit_only_first + hit_only_second else None)
    # GT-level traced samples: the same four targets in the rotated copy of the sample image.
    trace_file = OUT / 'trace/trace.json'
    if trace_file.exists():
        trace = json.load(open(trace_file))
        report['samples_rotated'] = {}
        lookup = {r['gt_id']: r for r in rows}
        for other in ('0036526.png', '0091189.png'):
            turned = [r for r in by[other] if r['native_gw'] > 0 and r['native_gh'] > 0]
            centers = np.array([[(r['gt_x1'] + r['gt_x2']) / 2, (r['gt_y1'] + r['gt_y2']) / 2] for r in turned])
            for name, sample in trace['samples'].items():
                r = lookup[sample['gt_id']]
                candidates = {}
                for orientation in ('rot180', 'identity'):
                    b = orient_boxes(key(r), orientation)[0]
                    d = np.abs(centers - np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])).max(1)
                    candidates[orientation] = (float(d.min()), int(d.argmin()))
                orientation = min(candidates, key=lambda o: candidates[o][0])
                distance, index = candidates[orientation]
                m = turned[index]
                report['samples_rotated'][f'{name}|{other}'] = dict(orientation=orientation, center_distance=round(distance, 2), original_status=r['status'], status=m['status'], enc_prob=round(m['enc_prob_max_in_gt'], 4), enc_prob_original=round(r['enc_prob_max_in_gt'], 4), tp_score=None if m['status'] != 'TP' else round(m['tp_score'], 3), tp_iou=None if m['status'] != 'TP' else round(m['tp_iou'], 3), best_iou=round(m['dec3_best_iou'], 3), rank=m['enc_rank_best_in_gt'], rank_original=r['enc_rank_best_in_gt'])
    (OUT / 'orientation_recall.json').write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ('train_tank_tiles', 'groups', 'ap', 'paired_counts', 'encoder_pairs', 'label_vs_pixel')}, ensure_ascii=False))
    print(json.dumps(report.get('samples_rotated'), ensure_ascii=False))
    print('--- pairs crossing the two conditions')
    for p in pair_rows:
        if p['group_a'] != p['group_b']:
            print({k: p[k] for k in ('a', 'b', 'orientation', 'corr', 'group_a', 'group_b', 'recall_a', 'recall_b', 'matched', 'of', 'enc_a', 'enc_b', 'mad_a', 'mad_b', 'label_shift', 'pixel_shift', 'pixel_corr')})
    print('non-seen images', [(f, v['group'], v['seen_corr'], v['other_corr'], v['gt'], v['recall'], v['native_mad']) for f, v in report['images'].items() if v['group'] != 'seen_orientation'])
    print('native mad quantiles, seen', np.round(np.percentile([v['native_mad'] for v in report['images'].values() if v['group'] == 'seen_orientation'], [10, 50, 90]), 2).tolist())


if __name__ == '__main__':
    main()
