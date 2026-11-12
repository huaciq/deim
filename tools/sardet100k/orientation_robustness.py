"""D17: does detection depend on the image orientation the detector was trained on?

The OGSOD tank diagnosis found that flipping an image into an orientation present in train raises recall from .39 to .76
and flipping it into an absent one lowers it from .84 to .55. This tool asks the same for the whole val set: the frozen
detector is run on the 8 flips / 90-degree rotations (the dihedral group D4) of every val image, the boxes are mapped
back, and COCO AP is computed against the unchanged annotations, for all GT and for GT with / without a near-duplicate
in train (D12 similarity).

Reading, fixed before running:
  - horizontal flip is a training augmentation and serves as the control: its AP drop is the noise level;
  - if the other orientations lose less than 1 AP more than the control, the detector is already orientation-robust and
    an orientation-equivariant structure has no support;
  - if they lose 3 AP or more beyond the control, orientation is memorised; the loss on GT without a near-duplicate says
    whether it also limits generalisation (loss only on duplicated GT = it is pure duplicate recall).

  python tools/sardet100k/orientation_robustness.py predict RUN OUT --name E000
  python tools/sardet100k/orientation_robustness.py ap OUT --d12 D12 --names E000 E008
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_report import r3  # noqa: E402
from generalisation_gap import EvalImages  # noqa: E402

# (transpose, horizontal flip, vertical flip), applied to the image in this order
TRANSFORMS = {'identity': (0, 0, 0), 'hflip': (0, 1, 0), 'vflip': (0, 0, 1), 'rot180': (0, 1, 1),
              'transpose': (1, 0, 0), 'rot90a': (1, 1, 0), 'rot90b': (1, 0, 1), 'antitranspose': (1, 1, 1)}
SIZE = 640


def apply(x, t):
    if t[0]:
        x = x.transpose(-1, -2)
    if t[1]:
        x = x.flip(-1)
    if t[2]:
        x = x.flip(-2)
    return x


def boxes_back(b, t):
    """xyxy boxes in the transformed SIZE x SIZE frame -> the original frame."""
    x0, y0, x1, y1 = b.unbind(-1)
    if t[2]:
        y0, y1 = SIZE - y1, SIZE - y0
    if t[1]:
        x0, x1 = SIZE - x1, SIZE - x0
    if t[0]:
        x0, y0, x1, y1 = y0, x0, y1, x1
    import torch
    return torch.stack([x0, y0, x1, y1], -1)


def predict(args):
    import torch
    from engine.core import YAMLConfig
    from error_gallery import ROOT
    images = json.load(open(ROOT / 'Annotations/val.json'))['images']
    cfg = YAMLConfig(str(args.run / 'resolved_config.yml'))
    model, post = cfg.model, cfg.postprocessor
    model.load_state_dict(torch.load(args.run / 'best_stg2.pth', map_location='cpu', weights_only=False)['ema']['module'])
    model, post = model.to(args.device).eval(), post.to(args.device).eval()
    loader = torch.utils.data.DataLoader(EvalImages(ROOT / 'JPEGImages/val', images), batch_size=32, num_workers=8)
    rows = {k: [] for k in TRANSFORMS}
    with torch.no_grad():
        for batch, indices, widths, heights in loader:
            batch = batch.to(args.device)
            square = torch.full((len(batch), 2), float(SIZE), device=args.device)
            scale = torch.stack([widths, heights, widths, heights], 1).float().to(args.device) / SIZE
            for name, t in TRANSFORMS.items():
                for k, (index, result) in enumerate(zip(indices.tolist(), post(model(apply(batch, t)), square))):
                    n = len(result['scores'])
                    rows[name].append((np.full(n, images[index]['id']), result['labels'].cpu().numpy(), result['scores'].cpu().numpy(),
                                       (boxes_back(result['boxes'], t) * scale[k]).cpu().numpy()))
    for name, part in rows.items():
        np.savez_compressed(args.out / f'{args.name}_{name}.npz', image_id=np.concatenate([r[0] for r in part]), label=np.concatenate([r[1] for r in part]),
                            score=np.concatenate([r[2] for r in part]), box=np.concatenate([r[3] for r in part]))
    print('written', args.out, args.name, flush=True)


def evaluate(args):
    from faster_coco_eval import COCO, COCOeval_faster
    from error_gallery import ROOT
    sim = np.load(args.d12 / 'similarity.npz')
    sim_of = dict(zip(sim['val_ann'].tolist(), sim['val_sim'].tolist()))
    path = ROOT / 'Annotations/val.json'
    names = {c['id']: c['name'] for c in json.load(open(path))['categories']}
    ranges = {'all': (-2, 2), 'seen': (.9, 2), 'novel': (-2, .9)}
    report = {}
    for run in args.names:
        report[run] = {}
        for t in TRANSFORMS:
            with np.load(args.out / f'{run}_{t}.npz') as z:
                results = [dict(image_id=int(i), category_id=int(c), score=float(s), bbox=[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])], area=1.)
                           for i, c, s, b in zip(z['image_id'], z['label'], z['score'], z['box'])]
            report[run][t] = {}
            for label, (lo, hi) in ranges.items():
                gt = COCO(str(path))
                for a in gt.anns.values():
                    a['area'] = 1. if lo <= sim_of[a['id']] < hi else 100.
                dt = gt.loadRes(results)
                for d in dt.anns.values():
                    d['area'] = 1.
                e = COCOeval_faster(gt, dt, 'bbox')
                e.params.areaRng = [[0., 10.]]
                e.params.areaRngLbl = ['all']
                e.evaluate()
                e.accumulate()
                precision = np.asarray(e.eval['precision'])
                last = len(e.params.maxDets) - 1
                per_class = {}
                for k, c in enumerate(e.params.catIds):
                    pr = precision[:, :, k, 0, last]
                    if (pr > -1).any():
                        per_class[names[c]] = dict(AP=r3(pr[pr > -1].mean() * 100), AP50=r3(pr[0][pr[0] > -1].mean() * 100))
                report[run][t][label] = dict(AP=r3(np.mean([v['AP'] for v in per_class.values()])), AP50=r3(np.mean([v['AP50'] for v in per_class.values()])), per_class=per_class)
            row = report[run][t]
            print(f"{run} {t:14s} AP all {row['all']['AP']:5.1f}  seen {row['seen']['AP']:5.1f}  novel {row['novel']['AP']:5.1f}   AP50 all {row['all']['AP50']:5.1f}  novel {row['novel']['AP50']:5.1f}   "
                  + '  '.join(f"{c} {v['AP']:.1f}" for c, v in row['all']['per_class'].items()), flush=True)
    (args.out / 'orientation_robustness.json').write_text(json.dumps(report, indent=1))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('predict')
    p.add_argument('run', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--name', required=True)
    p.add_argument('--device', default='cuda:0')
    a = sub.add_parser('ap')
    a.add_argument('out', type=Path)
    a.add_argument('--d12', type=Path, required=True)
    a.add_argument('--names', nargs='+', required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    {'predict': predict, 'ap': evaluate}[args.command](args)


if __name__ == '__main__':
    main()
