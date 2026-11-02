"""Functional checks for the read-only results ledger collector."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import yaml

from tools.collect_sar_results import coco_metrics, collect, method_name


def config(epochs=102, stop=90, **sections):
    cfg = dict(seed=0, epoches=epochs, warmup_iter=2000, flat_epoch=49, no_aug_epoch=12,
               eval_spatial_size=[640, 640], optimizer=dict(lr=3.5e-4),
               HybridEncoder=dict(use_encoder_idx=[2]), DFINETransformer={}, DEIMCriterion={},
               train_dataloader=dict(total_batch_size=28,
                                     dataset=dict(ann_file='/d/SAR-AIRcraft-1.0/COCO/train.json',
                                                  transforms=dict(policy=dict(epoch=[4, 49, 90]))),
                                     collate_fn=dict(stop_epoch=stop, base_size_repeat=6)))
    for section, values in sections.items():
        cfg[section].update(values)
    return cfg


def evaluation(images, ap):
    return dict(AP=ap, AP50=90., AP75=70., APs=50., APm=60., APl=55., AR100=80.,
                per_class_AP={'0': ap}, images=images)


class CollectResultsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='deim_ledger_')
        self.root = Path(self.temporary.name).resolve()  # runs are reported as resolved paths

    def tearDown(self):
        self.temporary.cleanup()

    def run_dir(self, name, cfg, aps, start=0):
        run = self.root / 'experiments' / name
        run.mkdir(parents=True)
        (run / 'resolved_config.yml').write_text(yaml.safe_dump(cfg), encoding='utf-8')
        rows = [dict(epoch=start + i, test_coco_eval_bbox=[ap, .9, .7, .5, .6, .55, .3, .6, .8])
                for i, ap in enumerate(aps)]
        (run / 'log.txt').write_text('\n'.join(map(json.dumps, rows)) + '\n{"partial', encoding='utf-8')
        (run / 'git_revision.txt').write_text('028c31cabcdef\n', encoding='utf-8')
        return run

    def add_eval(self, run, folder, checkpoint, images):
        target = run / folder
        target.mkdir(parents=True)
        (target / 'eval.pth').write_bytes(b'placeholder')
        (target / 'console.log').write_text(f'Resume checkpoint from {run}/{checkpoint}.pth\n'
                                            f'IMAGES={images}\n', encoding='utf-8')

    def test_selection_split_and_deltas(self):
        # Both end at epoch 101; baseline peaks after stop_epoch, REM before it.
        base = self.run_dir('deim_m_sar_aircraft_base_seed0_x', config(), [.60, .645, .63], start=99)
        rem = self.run_dir('deim_m_sar_aircraft_rem_seed0_y', config(HybridEncoder=dict(use_rem=True)),
                           [.66] + [.64] * 13, start=88)
        short = self.run_dir('deim_m_sar_aircraft_base58_z', config(epochs=58, stop=50), [.62, .631], start=48)
        (base / 'best_stg2.pth').write_bytes(b'')
        self.add_eval(base, 'test_eval_a', 'best_stg2', 879)
        self.add_eval(base, 'test_eval/checkpoint0047', 'checkpoint0047', 879)
        self.add_eval(rem, 'test_eval_b', 'best_stg1', 879)
        self.add_eval(rem, 'val_eval', 'best_stg1', 442)

        def load(path):
            text = (path.parent / 'console.log').read_text()
            images = int(text.split('IMAGES=')[1])
            ap = {base: 45.7, rem: 46.5}[path.parent.parent if path.parent.parent.name.startswith('deim') else
                                          path.parent.parent.parent]
            return evaluation(images, ap)

        records, evaluations = collect([self.root / 'experiments', self.root / 'missing'],
                                       self.root / 'out', load_eval=load)
        by = {Path(r['run']).name: r for r in records}
        b, r, s = by[base.name], by[rem.name], by[short.name]
        self.assertEqual((b['method'], b['best_epoch'], b['selected'], b['status']), ('baseline', 100, 'best_stg2', 'complete'))
        self.assertEqual((b['val_AP'], b['test_AP'], b['test_evaluations']), (64.5, 45.7, 2))
        self.assertEqual((r['method'], r['best_epoch'], r['selected'], r['status']),
                         ('rem', 88, 'best_stg1', 'complete'))
        self.assertEqual((r['d_val_AP'], r['d_test_AP']), (1.5, 0.8))
        self.assertEqual((s['status'], s['baseline_runs']), ('stopped', 0))
        self.assertNotEqual(s['protocol_key'], b['protocol_key'])
        self.assertEqual(b['git_revision'], '028c31c')
        self.assertEqual(sorted(e['split'] for e in evaluations), ['test', 'test', 'test', 'val'])
        ledger = (self.root / 'out' / 'ledger.md').read_text(encoding='utf-8')
        self.assertIn('SAR-AIRcraft-1.0 · 102e/b28/ms640', ledger)
        self.assertIn('SAR-AIRcraft-1.0 · 58e/b28/ms640', ledger)
        self.assertIn('+1.50', ledger)
        self.assertTrue((self.root / 'out' / 'runs.csv').is_file())

    def test_method_name_variants(self):
        cfg = config(HybridEncoder=dict(use_rem=True, rem_residual_mode='per_band'),
                     DFINETransformer=dict(use_bccq=True),
                     DEIMCriterion=dict(use_wfdr=True, use_focal_loss=True, use_bccq=True))
        cfg['HGNetv2'] = dict(use_lab=True)
        self.assertEqual(method_name(cfg), 'rem[per_band]+bccq+wfdr')
        baseline = config()
        baseline['HGNetv2'] = dict(use_lab=True)
        self.assertEqual(method_name(baseline), 'baseline')

    def test_coco_metrics_from_precision(self):
        # T=10 IoUs, R=101, K=2 classes, A=4 areas, M=3 maxDets; class 1 absent in small.
        precision = np.full((10, 101, 2, 4, 3), .5)
        precision[:, :, 0, :, :] = .7
        precision[0, :, :, 0, 2] = .9
        precision[:, :, 1, 1, :] = -1
        recall = np.full((10, 2, 4, 3), .8)
        params = SimpleNamespace(areaRngLbl=['all', 'small', 'medium', 'large'], maxDets=[1, 10, 100],
                                 iouThrs=np.linspace(.5, .95, 10), catIds=[0, 1], imgIds=list(range(879)))
        m = coco_metrics(dict(params=params, precision=precision, recall=recall))
        self.assertAlmostEqual(m['AP'], round((.9 * 2 + 9 * (.7 + .5)) / 20 * 100, 2))
        self.assertEqual((m['AP50'], m['AP75'], m['APs'], m['AR100']), (90., 60., 70., 80.))
        self.assertEqual((m['per_class_AP']['0'], m['images']), (72., 879))


if __name__ == '__main__':
    unittest.main()
