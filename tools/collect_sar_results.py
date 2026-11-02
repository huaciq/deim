"""Collect every finished or running detection run into one results ledger.

Read-only over experiment roots: each run directory holding log.txt and
resolved_config.yml becomes one row. Val metrics come from the per-epoch JSONL
log (best val AP selects the checkpoint, as in the solver); test metrics come
from independent --test-only evaluations (any eval.pth below the run).
Outputs runs.csv, test_evaluations.csv, ledger.json and ledger.md in a new
directory. Nothing inside the experiment roots is modified.
"""

import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import re
import sys

import yaml

STAT_NAMES = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100']
METRICS = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR100']
MODULE_SECTIONS = ['HGNetv2', 'HybridEncoder', 'DFINETransformer', 'DEIMCriterion']
# Official detector settings, not methods: HGNetv2 learnable affine blocks and focal loss.
BASE_SWITCHES = {'use_focal_loss', 'use_lab'}
VARIANT_KEYS = {'use_rem': ['rem_variant', 'rem_residual_mode'], 'use_ost_da': ['ost_da_version'],
                'use_slqc': ['slqc_mode'], 'use_dsg': ['dsg_mode']}
NAME_PREFIXES = ['deim_m_sar_aircraft_deim_hgnetv2_m_sar_aircraft_', 'deim_m_sar_aircraft_',
                 'deim_hgnetv2_m_sar_aircraft_']
SPLIT_BY_IMAGES = {442: 'val', 879: 'test', 3047: 'train'}


def percent(value):
    return None if value is None else round(float(value) * 100, 2)


def read_text(path):
    try:
        return path.read_text(encoding='utf-8', errors='replace').strip()
    except OSError:
        return None


def epoch_rows(path):
    rows = []
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        if line.startswith('{'):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line being written while the run is active
            if 'test_coco_eval_bbox' in row:
                rows.append(row)
    return rows


def method_name(cfg):
    """Every enabled use_* switch of the model/criterion sections, plus variants."""
    parts = []
    for section in MODULE_SECTIONS:
        block = cfg.get(section) or {}
        for key, value in block.items():
            if key.startswith('use_') and value is True and key not in BASE_SWITCHES:
                variants = [str(block[k]) for k in VARIANT_KEYS.get(key, []) if k in block]
                name = key[4:] + (f'[{",".join(variants)}]' if variants else '')
                if name not in parts:  # e.g. use_bccq in both decoder and criterion
                    parts.append(name)
    s2 = (cfg.get('HybridEncoder') or {}).get('s2_injection_mode')
    if s2:
        parts.append(f's2[{s2}]')
    return '+'.join(parts) if parts else 'baseline'


def protocol_of(cfg):
    train = cfg.get('train_dataloader', {})
    collate = train.get('collate_fn', {})
    ann = train.get('dataset', {}).get('ann_file', '')
    key = dict(dataset=Path(ann).parent.parent.name or ann, epochs=cfg.get('epoches'),
               warmup=cfg.get('warmup_iter'), flat=cfg.get('flat_epoch'), no_aug=cfg.get('no_aug_epoch'),
               stop_epoch=collate.get('stop_epoch'), batch=train.get('total_batch_size'),
               eval_size=cfg.get('eval_spatial_size'), multiscale_repeat=collate.get('base_size_repeat'),
               policy=train.get('dataset', {}).get('transforms', {}).get('policy', {}).get('epoch'),
               lr=(cfg.get('optimizer') or {}).get('lr'))
    scale = 'ms' if key['multiscale_repeat'] else 'fixed'
    size = key['eval_size'][0] if isinstance(key['eval_size'], list) else key['eval_size']
    label = f"{key['epochs']}e/b{key['batch']}/{scale}{size}"
    return key, label


def run_status(run, rows, epochs):
    last = rows[-1]['epoch'] if rows else None
    if last is not None and epochs is not None and last >= epochs - 1:
        return 'complete'
    pid = read_text(run / 'launcher.pid')
    if pid and pid.isdigit():
        try:
            os.kill(int(pid), 0)
            return 'running'
        except (OSError, PermissionError):
            pass
    return 'stopped' if rows else 'no_epochs'


def selected_checkpoint(rows, stop_epoch):
    """Best val AP epoch; the solver stores it in best_stg2 only after stop_epoch."""
    best = max(rows, key=lambda row: row['test_coco_eval_bbox'][0])
    stage = 'best_stg2' if stop_epoch is not None and best['epoch'] >= stop_epoch else 'best_stg1'
    return best, stage


def coco_metrics(evaluation):
    """AP family and per-class AP from a saved COCOeval.eval dict (T,R,K,A,M)."""
    import numpy as np

    params = evaluation['params']
    precision = np.asarray(evaluation['precision'])
    recall = np.asarray(evaluation['recall'])
    labels = list(getattr(params, 'areaRngLbl', ['all', 'small', 'medium', 'large']))
    max_dets = list(getattr(params, 'maxDets', [1, 10, 100]))
    m = max_dets.index(100) if 100 in max_dets else len(max_dets) - 1
    ious = np.asarray(getattr(params, 'iouThrs', np.linspace(.5, .95, 10)))

    def mean(values):
        valid = values[values > -1]
        return round(float(valid.mean()) * 100, 2) if valid.size else None

    area = {name: labels.index(name) for name in ('all', 'small', 'medium', 'large') if name in labels}
    full = precision[:, :, :, area['all'], m]
    result = dict(AP=mean(full), AP50=mean(full[np.isclose(ious, .5)]),
                  AP75=mean(full[np.isclose(ious, .75)]),
                  APs=mean(precision[:, :, :, area['small'], m]) if 'small' in area else None,
                  APm=mean(precision[:, :, :, area['medium'], m]) if 'medium' in area else None,
                  APl=mean(precision[:, :, :, area['large'], m]) if 'large' in area else None,
                  AR100=mean(recall[:, :, area['all'], m]))
    categories = list(getattr(params, 'catIds', []))
    result['per_class_AP'] = {str(c): mean(full[:, :, k]) for k, c in enumerate(categories)}
    result['images'] = len(getattr(params, 'imgIds', []) or [])
    return result


def test_evaluations(run, load_eval):
    found = []
    for path in sorted(run.rglob('eval.pth')):
        folder = path.parent
        console = read_text(folder / 'console.log') or ''
        resumed = re.findall(r'Resume checkpoint from (\S+)', console)
        checkpoint = Path(resumed[-1]).stem if resumed else folder.name
        entry = dict(run=str(run), eval_dir=str(folder), checkpoint=checkpoint)
        try:
            metrics = load_eval(path)
        except Exception as error:  # report unreadable evaluations instead of skipping silently
            entry.update(split='unreadable', error=f'{type(error).__name__}: {error}')
            found.append(entry)
            continue
        split = SPLIT_BY_IMAGES.get(metrics['images'])
        if split is None:
            split = 'test' if 'test.json' in console else 'val' if 'val.json' in console else 'unknown'
        entry.update(split=split, **metrics)
        found.append(entry)
    return found


def default_load_eval(path):
    import torch
    return coco_metrics(torch.load(path, map_location='cpu', weights_only=False))


def short_name(name):
    for prefix in NAME_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def find_runs(roots, max_depth):
    runs = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            print(f'skip missing root {root}', file=sys.stderr)
            continue
        base = len(root.parts)
        for current, folders, files in os.walk(root):
            here = Path(current)
            if 'log.txt' in files and 'resolved_config.yml' in files:
                runs.append(here)
                folders[:] = []  # evaluations below a run are read by test_evaluations
            elif len(here.parts) - base >= max_depth:
                folders[:] = []
    return sorted(set(r.resolve() for r in runs))


def collect_run(run, load_eval):
    cfg = yaml.safe_load((run / 'resolved_config.yml').read_text(encoding='utf-8')) or {}
    rows = epoch_rows(run / 'log.txt')
    key, label = protocol_of(cfg)
    metadata = {}
    if (run / 'run_metadata.json').is_file():
        try:
            metadata = json.loads((run / 'run_metadata.json').read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            metadata = {}
    config_source = read_text(run / 'config_path.txt') or metadata.get('source_config') or ''
    record = dict(run=str(run), name=short_name(run.name), method=method_name(cfg),
                  config=Path(config_source).name if config_source else '',
                  dataset=key['dataset'], protocol=label, protocol_key=json.dumps(key, sort_keys=True),
                  seed=cfg.get('seed', metadata.get('seed')), global_batch=key['batch'],
                  git_revision=(read_text(run / 'git_revision.txt') or metadata.get('current_git_revision') or '')[:7],
                  status=run_status(run, rows, cfg.get('epoches')),
                  last_epoch=rows[-1]['epoch'] if rows else None, epochs=cfg.get('epoches'))
    if rows:
        best, stage = selected_checkpoint(rows, key['stop_epoch'])
        stats = best['test_coco_eval_bbox']
        record.update(best_epoch=best['epoch'], selected=stage,
                      selected_file_exists=(run / f'{stage}.pth').is_file(),
                      **{f'val_{n}': percent(stats[i]) for i, n in enumerate(STAT_NAMES) if i < len(stats)})
    evaluations = test_evaluations(run, load_eval)
    tests = [e for e in evaluations if e.get('split') == 'test']
    chosen = [e for e in tests if e['checkpoint'] == record.get('selected')]
    record['test_evaluations'] = len(tests)
    if chosen:
        record.update({f'test_{n}': chosen[-1].get(n) for n in METRICS})
    return record, evaluations


def add_baseline_deltas(records):
    """Val/test AP minus the mean of complete baselines with identical protocol and seed."""
    groups = defaultdict(list)
    for r in records:
        if r['method'] == 'baseline' and r['status'] == 'complete' and r.get('val_AP') is not None:
            groups[(r['protocol_key'], r['seed'])].append(r)
    for r in records:
        bases = groups.get((r['protocol_key'], r['seed']), [])
        r['baseline_runs'] = len(bases)
        for split in ('val', 'test'):
            values = [b[f'{split}_AP'] for b in bases if b.get(f'{split}_AP') is not None]
            if values and r.get(f'{split}_AP') is not None and r['method'] != 'baseline':
                r[f'd_{split}_AP'] = round(r[f'{split}_AP'] - sum(values) / len(values), 2)


def fmt(value):
    if value is None or value == '':
        return '—'
    if isinstance(value, float):
        return f'{value:.2f}'
    return str(value)


def signed(value):
    return '—' if value is None else f'{value:+.2f}'


def write_markdown(path, records, evaluations, roots):
    lines = ['# Results ledger', '',
             f'Roots: {", ".join(map(str, roots))}. Runs: {len(records)}.', '',
             'Val = best per-epoch val AP (selects the checkpoint, EMA, as logged). '
             'Test = independent --test-only of that selected checkpoint. '
             'Δ = minus the mean of complete baselines with the same protocol and seed. '
             'All values ×100.', '']
    groups = defaultdict(list)
    for r in records:
        groups[(r['dataset'], r['protocol'], r['protocol_key'])].append(r)
    for (dataset, protocol, key), rows in sorted(groups.items(), key=lambda item: item[0][:2]):
        lines += [f'## {dataset} · {protocol}', '', f'<sub>{key}</sub>', '',
                  '| method | config | seed | status | best ep / last | val AP | AP50 | AP75 | APs | APm | APl | ΔAP val '
                  '| test AP | AP50 | AP75 | ΔAP test | rev | run |',
                  '|' + '---|' * 18]
        for r in sorted(rows, key=lambda r: (r['method'] != 'baseline', r['method'], r['seed'] or 0)):
            lines.append('| ' + ' | '.join([
                r['method'], r['config'] or '—', fmt(r['seed']), r['status'],
                f"{fmt(r.get('best_epoch'))} / {fmt(r.get('last_epoch'))}",
                *[fmt(r.get(f'val_{n}')) for n in ('AP', 'AP50', 'AP75', 'APs', 'APm', 'APl')],
                signed(r.get('d_val_AP')),
                *[fmt(r.get(f'test_{n}')) for n in ('AP', 'AP50', 'AP75')],
                signed(r.get('d_test_AP')), r['git_revision'] or '—', r['name']]) + ' |')
        lines.append('')
    lines += ['## All independent evaluations', '',
              '| run | eval dir | checkpoint | split | AP | AP50 | AP75 | APs | APm | APl | AR100 |',
              '|' + '---|' * 11]
    for e in evaluations:
        lines.append('| ' + ' | '.join([short_name(Path(e['run']).name), Path(e['eval_dir']).name,
                                        e['checkpoint'], e['split'],
                                        *[fmt(e.get(n)) for n in METRICS]]) + ' |')
    gaps = [r for r in records if r['status'] == 'complete' and r.get('test_AP') is None]
    if gaps:
        lines += ['', '## Complete runs without a test evaluation of the selected checkpoint', '']
        lines += [f"- {r['name']}: selected {r.get('selected')} (epoch {r.get('best_epoch')}), "
                  f"file exists={r.get('selected_file_exists')}" for r in gaps]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_csv(path, rows, skip=()):
    fields = []
    for row in rows:
        fields += [k for k in row if k not in fields and k not in skip]
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def collect(roots, output, max_depth=3, load_eval=default_load_eval):
    records, evaluations = [], []
    for run in find_runs(roots, max_depth):
        try:
            record, found = collect_run(run, load_eval)
        except Exception as error:
            print(f'skip {run}: {type(error).__name__}: {error}', file=sys.stderr)
            continue
        records.append(record)
        evaluations += found
    add_baseline_deltas(records)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / 'runs.csv', records)
    write_csv(output / 'test_evaluations.csv', evaluations, skip=('per_class_AP',))
    (output / 'ledger.json').write_text(json.dumps(dict(runs=records, evaluations=evaluations),
                                                   indent=2, ensure_ascii=False), encoding='utf-8')
    write_markdown(output / 'ledger.md', records, evaluations, roots)
    return records, evaluations


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--root', action='append', required=True, help='experiment root; repeatable')
    parser.add_argument('--output', required=True, help='new directory for the ledger files')
    parser.add_argument('--max-depth', type=int, default=3, help='directory depth searched for runs')
    args = parser.parse_args()
    records, evaluations = collect(args.root, Path(args.output), args.max_depth)
    print(f'{len(records)} runs, {len(evaluations)} evaluations -> {args.output}/ledger.md')


if __name__ == '__main__':
    main()
