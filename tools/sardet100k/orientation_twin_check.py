"""D17.2: val targets whose training twin exists only in another orientation.

D12 called a val target "seen" when its 24x24 crop correlates >= .9 with a train GT crop in the same orientation.
Here the val crop is also compared after each of the 8 flips / 90-degree rotations. Groups:
  same        twin in the same orientation (D12 "seen")
  hflip       no same-orientation twin, but a horizontally flipped one (horizontal flip is a training augmentation)
  other       twin only under one of the 6 remaining transforms (never shown to the detector in that orientation)
  none        no twin under any transform
If the detector could transfer across orientations, `other` would be detected like `same`. Reported: recall at the
operating point (score >= .3, IoU >= .5) per group and per source/class cell.

  python tools/sardet100k/orientation_twin_check.py OUT --val-pred E000=V.npz E008=V8.npz
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_report import iou_xyxy, r3  # noqa: E402
from generalisation_gap import CROP, best_match, load_split  # noqa: E402
from generalisation_probe import cell_name  # noqa: E402

TRANSFORMS = {'identity': (0, 0, 0), 'hflip': (0, 1, 0), 'vflip': (0, 0, 1), 'rot180': (0, 1, 1),
              'transpose': (1, 0, 0), 'rot90a': (1, 1, 0), 'rot90b': (1, 0, 1), 'antitranspose': (1, 1, 1)}


def transform(crops, t):
    x = crops.reshape(-1, CROP, CROP)
    if t[0]:
        x = x.transpose(0, 2, 1)
    if t[1]:
        x = x[:, :, ::-1]
    if t[2]:
        x = x[:, ::-1]
    return np.ascontiguousarray(x).reshape(len(crops), -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    parser.add_argument('--val-pred', nargs='+', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    from error_gallery import ROOT
    train, t_crops, t_anns = load_split(ROOT, 'train')
    val, v_crops, v_anns = load_split(ROOT, 'val')
    size = lambda anns: np.array([max(a['bbox'][2], a['bbox'][3]) for a in anns], np.float32)  # noqa: E731
    t_image = np.array([a['image_id'] for a in t_anns], np.int64)
    v_image = -1 - np.array([a['image_id'] for a in v_anns], np.int64)
    sims = {name: best_match(transform(v_crops, t), size(v_anns), v_image, t_crops, size(t_anns), t_image, args.device) for name, t in TRANSFORMS.items()}
    np.savez_compressed(args.out / 'd4_similarity.npz', val_ann=np.array([a['id'] for a in v_anns]), **sims)
    other = np.max([sims[k] for k in TRANSFORMS if k not in ('identity', 'hflip')], 0)
    group = np.where(sims['identity'] >= .9, 'same', np.where(sims['hflip'] >= .9, 'hflip', np.where(other >= .9, 'other', 'none')))
    names = {c['id']: c['name'] for c in val['categories']}
    width = {i['id']: i['width'] for i in val['images']}
    ext = {i['id']: i['file_name'].rsplit('.', 1)[-1] for i in val['images']}
    cell = np.array([('ogsod/' + names[a['category_id']]) if ext[a['image_id']] == 'png' else cell_name(width[a['image_id']], names[a['category_id']], 99) for a in v_anns])
    by_image = defaultdict(list)
    for k, a in enumerate(v_anns):
        by_image[a['image_id']].append(k)
    report = dict(counts={g: int((group == g).sum()) for g in ('same', 'hflip', 'other', 'none')})
    print('groups', report['counts'], flush=True)
    for spec in args.val_pred:
        name, path = spec.split('=')
        with np.load(path) as z:
            keep = z['score'] >= .3
            p_img, p_box, p_label = z['image_id'][keep], z['box'][keep].astype(np.float64), z['label'][keep]
        order = np.argsort(p_img, kind='stable')
        p_img, p_box, p_label = p_img[order], p_box[order], p_label[order]
        hit = np.zeros(len(v_anns), bool)
        for image_id, idx in by_image.items():
            s, e = np.searchsorted(p_img, image_id, 'left'), np.searchsorted(p_img, image_id, 'right')
            if s == e:
                continue
            gt = np.array([[v_anns[k]['bbox'][0], v_anns[k]['bbox'][1], v_anns[k]['bbox'][0] + v_anns[k]['bbox'][2], v_anns[k]['bbox'][1] + v_anns[k]['bbox'][3]] for k in idx])
            iou = iou_xyxy(gt, p_box[s:e])
            same = np.array([v_anns[k]['category_id'] for k in idx])[:, None] == p_label[s:e][None]
            hit[idx] = ((iou >= .5) & same).any(1)
        table = {'all': {g: dict(n=int((group == g).sum()), recall=r3(hit[group == g].mean() * 100)) for g in ('same', 'hflip', 'other', 'none')}}
        for c in np.unique(cell):
            if (cell == c).sum() >= 200:
                table[c] = {g: dict(n=int(((cell == c) & (group == g)).sum()), recall=r3(hit[(cell == c) & (group == g)].mean() * 100) if ((cell == c) & (group == g)).sum() >= 20 else None)
                            for g in ('same', 'hflip', 'other', 'none')}
        report[name] = table
        for c, row in table.items():
            print(name, f'{c:16s}', '  '.join(f"{g} {v['n']:5d} {v['recall']}" for g, v in row.items()), flush=True)
    (args.out / 'orientation_twins.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
