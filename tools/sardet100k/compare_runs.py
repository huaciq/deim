"""Compare SARDet-100K runs on val (highest-AP epoch) and test, overall and per class.

    python tools/sardet100k/compare_runs.py --test-ann /path/to/Annotations/test.json \
        --run E000-old=RUN_DIR:TEST_EVAL_DIR --run E020=RUN_DIR:TEST_EVAL_DIR [--output result.json]

Each --run is NAME=RUN_DIR[:TEST_EVAL_DIR[:VAL_EVAL_DIR]]. Eval dirs are relative to RUN_DIR or absolute and
hold the eval.pth written by `train.py --test-only`. Val per-class AP is read from VAL_EVAL_DIR, or from
RUN_DIR/eval/latest.pth when the highest val AP is at the last epoch (otherwise it is left out). The first run is
the reference for the printed differences. Values are COCO bbox AP x 100 at maxDets=100.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

KEYS = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR100']


def metrics(path, names):
    data = torch.load(path, map_location='cpu', weights_only=False)
    p = data['params']
    precision, recall = np.asarray(data['precision']), np.asarray(data['recall'])
    area = {label: i for i, label in enumerate(p.areaRngLbl)}
    m = list(p.maxDets).index(100)
    iou = np.asarray(p.iouThrs)

    def mean(x):
        x = np.asarray(x)
        x = x[x > -1]
        return float(x.mean() * 100) if x.size else None
    full = precision[:, :, :, area['all'], m]
    out = dict(AP=mean(full), AP50=mean(full[np.isclose(iou, .5)]), AP75=mean(full[np.isclose(iou, .75)]),
               APs=mean(precision[:, :, :, area['small'], m]), APm=mean(precision[:, :, :, area['medium'], m]),
               APl=mean(precision[:, :, :, area['large'], m]), AR100=mean(recall[:, :, area['all'], m]),
               images=len(p.imgIds))
    out['per_class'] = {names.get(int(c), str(c)): mean(full[:, :, k]) for k, c in enumerate(p.catIds)}
    return out


def resolve(run, sub):
    path = Path(sub)
    path = path if path.is_absolute() else run / path
    return path / 'eval.pth' if path.is_dir() else path


def load_run(spec, names):
    name, rest = spec.split('=', 1)
    fields = rest.split(':')
    if len(fields) > 1 and len(fields[0]) == 1 and fields[1][:1] in ('/', '\\'):   # Windows drive letter
        fields = [fields[0] + ':' + fields[1]] + fields[2:]
    run = Path(fields[0])
    rows = [json.loads(line) for line in (run / 'log.txt').read_text().splitlines() if line.strip()]
    best = max(rows, key=lambda r: r['test_coco_eval_bbox'][0])
    record = dict(name=name, run=str(run), epochs=len(rows), best_epoch_one_based=best['epoch'] + 1,
                  val={k: 100 * best['test_coco_eval_bbox'][i] for i, k in enumerate(KEYS[:6])})
    record['val']['AR100'] = 100 * best['test_coco_eval_bbox'][8]
    if len(fields) > 2:
        val_eval = resolve(run, fields[2])
    elif best['epoch'] == rows[-1]['epoch']:
        val_eval = run / 'eval/latest.pth'
    else:
        val_eval = None
    if val_eval is not None and val_eval.is_file():
        record['val_per_class'] = metrics(val_eval, names)['per_class']
    if len(fields) > 1:
        record['test'] = metrics(resolve(run, fields[1]), names)
    return record


def _cells(values, fmt):
    return ''.join('%9s' % (fmt % v if v is not None else '-') for v in values)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', action='append', required=True)
    parser.add_argument('--test-ann', required=True, help='annotation file used only for category names')
    parser.add_argument('--output')
    args = parser.parse_args(argv)
    names = {int(c['id']): c['name'] for c in json.loads(Path(args.test_ann).read_text())['categories']}
    classes = list(names.values())
    records = [load_run(spec, names) for spec in args.run]
    reference = records[0]
    for split in ('val', 'test'):
        rows = [r for r in records if split in r]
        if not rows:
            continue
        print('== ' + split + (' (best val epoch)' if split == 'val' else ''))
        print('%-10s' % '' + ''.join('%8s' % k for k in KEYS) + '  |' + ''.join('%9s' % c for c in classes))
        ref_cls = (reference.get('val_per_class', {}) if split == 'val'
                   else reference.get('test', {}).get('per_class', {}))
        for r in rows:
            cls = r.get('val_per_class', {}) if split == 'val' else r['test']['per_class']
            print('%-10s' % r['name'] + ''.join('%8.2f' % r[split][k] for k in KEYS) + '  |'
                  + _cells([cls.get(c) for c in classes], '%.2f'))
            if r is not reference and split in reference:
                deltas = [cls[c] - ref_cls[c] if cls.get(c) is not None and ref_cls.get(c) is not None else None
                          for c in classes]
                print('%-10s' % '  delta' + ''.join('%+8.2f' % (r[split][k] - reference[split][k]) for k in KEYS)
                      + '  |' + _cells(deltas, '%+.2f'))
    print('best val epochs (name, epoch, total):',
          [(r['name'], r['best_epoch_one_based'], r['epochs']) for r in records])
    if args.output:
        Path(args.output).write_text(json.dumps(records, indent=2))
    return records


if __name__ == '__main__':
    main()
