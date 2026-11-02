"""Evaluate the val-selected checkpoint of finished SAR-AIRcraft runs on test.json.

For each run the checkpoint is chosen by best per-epoch val AP (best_stg2 when
that epoch is at or after stop_epoch, else best_stg1); test AP never changes
the choice. Runs are evaluated one after another on GPU 0,1 with the run's own
resolved_config.yml, only the val image folder and annotation file pointed at
the test split. Each evaluation writes RUN/test_eval_<stage>_<batch>/ and is
skipped when that stage already has a test evaluation, so rerunning the same
command continues an interrupted batch. summary.{tsv,json,md} in --output are
rewritten after every run.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.collect_sar_results import (default_load_eval, epoch_rows, find_runs, method_name, percent,
                                       protocol_of, selected_checkpoint, short_name)

COLUMNS = ['name', 'method', 'epochs', 'stage', 'best_epoch', 'val_AP', 'val_AP50', 'val_AP75',
           'test_AP', 'test_AP50', 'test_AP75', 'test_APm', 'test_APl', 'test_AR100', 'status', 'eval_dir', 'run']


def plan_run(run):
    cfg = yaml.safe_load((run / 'resolved_config.yml').read_text(encoding='utf-8')) or {}
    key, label = protocol_of(cfg)
    rows = epoch_rows(run / 'log.txt')
    entry = dict(run=str(run), name=short_name(run.name), method=method_name(cfg), dataset=key['dataset'],
                 protocol=label, epochs=cfg.get('epoches'), seed=cfg.get('seed', 0))
    if not rows or rows[-1]['epoch'] < cfg.get('epoches', 0) - 1:
        entry['status'] = f"incomplete (last epoch {rows[-1]['epoch'] if rows else None})"
        return entry
    best, stage = selected_checkpoint(rows, key['stop_epoch'])
    stats = best['test_coco_eval_bbox']
    entry.update(stage=stage, best_epoch=best['epoch'], val_AP=percent(stats[0]),
                 val_AP50=percent(stats[1]), val_AP75=percent(stats[2]))
    entry['status'] = 'planned' if (run / f'{stage}.pth').is_file() else f'missing {stage}.pth'
    return entry


def existing_eval(run, stage):
    for path in sorted(run.glob(f'test_eval_{stage}_*/eval.pth')):
        return path
    return None


def evaluate(entry, data_root, batch):
    run = Path(entry['run'])
    eval_dir = run / f'test_eval_{entry["stage"]}_{batch}'
    attempt = 1
    while eval_dir.exists():  # keep a failed attempt's log when the batch is rerun
        attempt += 1
        eval_dir = run / f'test_eval_{entry["stage"]}_{batch}_attempt{attempt}'
    eval_dir.mkdir()
    command = ['torchrun', '--standalone', '--nproc_per_node=2', 'train.py',
               '-c', str(run / 'resolved_config.yml'), '-r', str(run / f'{entry["stage"]}.pth'),
               '--test-only', '--seed', str(entry['seed']),
               '--output-dir', str(eval_dir), '--summary-dir', str(eval_dir / 'summary'),
               '-u', f'val_dataloader.dataset.img_folder={data_root}',
               f'val_dataloader.dataset.ann_file={data_root}/COCO/test.json']
    (eval_dir / 'command.json').write_text(json.dumps(command, indent=2), encoding='utf-8')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1', PYTHONUNBUFFERED='1')
    print(f'EVAL {entry["name"]} {entry["stage"]} -> {eval_dir}', flush=True)
    with (eval_dir / 'console.log').open('w', encoding='utf-8') as log:
        code = subprocess.call(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL)
    return eval_dir, code


def attach_metrics(entry, path, load_eval):
    metrics = load_eval(path)
    entry['eval_dir'] = str(path.parent)
    entry['test_images'] = metrics['images']
    entry['test_per_class_AP'] = metrics['per_class_AP']
    for name in ('AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR100'):
        entry[f'test_{name}'] = metrics[name]
    entry['status'] = 'done' if metrics['images'] == 879 else f'done ({metrics["images"]} images, expected 879)'


def write_summary(output, entries):
    (output / 'summary.json').write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = ['\t'.join(COLUMNS)]
    lines += ['\t'.join('' if e.get(c) is None else str(e.get(c)) for c in COLUMNS) for e in entries]
    (output / 'summary.tsv').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    md = ['| protocol | method | run | stage / epoch | val AP | test AP | AP50 | AP75 | APm | APl | AR100 | status |',
          '|' + '---|' * 12]
    for e in sorted(entries, key=lambda e: (e.get('protocol', ''), e['method'] != 'baseline', e['method'])):
        values = [e.get(k) for k in ('val_AP', 'test_AP', 'test_AP50', 'test_AP75', 'test_APm', 'test_APl', 'test_AR100')]
        md.append('| ' + ' | '.join([e.get('protocol', ''), e['method'], e['name'],
                                     f"{e.get('stage', '—')} / {e.get('best_epoch', '—')}",
                                     *['—' if v is None else f'{v:.2f}' for v in values], e['status']]) + ' |')
    (output / 'summary.md').write_text('\n'.join(md) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--root', action='append', default=[], help='experiment root searched for runs; repeatable')
    parser.add_argument('--run', action='append', default=[], help='explicit run directory; repeatable')
    parser.add_argument('--data-root', default='/home/zxy/sar/datasets/SAR-AIRcraft-1.0')
    parser.add_argument('--dataset', default='SAR-AIRcraft-1.0', help='keep runs whose train annotations are under this dataset')
    parser.add_argument('--output', required=True, help='batch directory for plan and summary files')
    parser.add_argument('--plan-only', action='store_true', help='write plan.tsv and stop without evaluating')
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    batch = output.name
    runs = find_runs(args.root, 3) + [Path(r).resolve() for r in args.run]
    entries = [plan_run(run) for run in dict.fromkeys(runs)]
    entries = [e for e in entries if e['dataset'] == args.dataset]
    write_summary(output, entries)
    (output / 'plan.tsv').write_text((output / 'summary.tsv').read_text(encoding='utf-8'), encoding='utf-8')
    print(f'{len(entries)} {args.dataset} runs; {sum(e["status"] == "planned" for e in entries)} planned', flush=True)
    if args.plan_only:
        return

    for entry in entries:
        if entry['status'] != 'planned':
            continue
        run = Path(entry['run'])
        path = existing_eval(run, entry['stage'])
        if path is None:
            eval_dir, code = evaluate(entry, args.data_root, batch)
            path = eval_dir / 'eval.pth'
            if code != 0 or not path.is_file():
                entry.update(status=f'failed (exit {code})', eval_dir=str(eval_dir))
                write_summary(output, entries)
                continue
        attach_metrics(entry, path, default_load_eval)
        write_summary(output, entries)
        print(f'DONE {entry["name"]} test AP {entry["test_AP"]}', flush=True)
    print('ALL FINISHED', flush=True)


if __name__ == '__main__':
    main()
