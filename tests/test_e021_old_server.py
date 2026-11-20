"""Old-server E021 and plan-rebuild configs differ from the eight-GPU versions only in paths.

Run on CPU: python -m unittest tests.test_e021_old_server -v
"""
import json
import tempfile
import unittest
from pathlib import Path

from engine.core.yaml_utils import load_config
from tools.sardet100k import plan_summary

FOLDER = Path(__file__).resolve().parents[1] / 'configs/deim_dfine'
META = {'__include__', 'output_dir', 'experiment_protocol_document', 'experiment_baseline_revision'}
DATA = {'train_dataloader.dataset.img_folder', 'train_dataloader.dataset.ann_file', 'train_dataloader.dataset.plan_dir',
        'val_dataloader.dataset.img_folder', 'val_dataloader.dataset.ann_file'}


def flat(cfg, prefix=''):
    out = {}
    for key, value in cfg.items():
        out.update(flat(value, f'{prefix}{key}.') if isinstance(value, dict) else {f'{prefix}{key}': value})
    return out


def load(name):
    return load_config(str(FOLDER / name))


def changed(a, b):
    fa, fb = flat(a), flat(b)
    return {k for k in set(fa) | set(fb) if fa.get(k, '<none>') != fb.get(k, '<none>')} - META


class Configs(unittest.TestCase):
    e010 = load('deim_hgnetv2_m_sardet100k_region_dense_sar_36e_b32.yml')
    e010_old = load('deim_hgnetv2_m_sardet100k_region_dense_sar_36e_b32_old_server.yml')
    e021 = load('deim_hgnetv2_m_sardet100k_imagenet_init_full_model_36e_b32.yml')
    e021_old = load('deim_hgnetv2_m_sardet100k_imagenet_init_full_model_36e_b32_old_server.yml')
    e000_old = load('deim_hgnetv2_m_sardet100k_36e_b32_nozoom_nocrop_old_server.yml')

    def test_only_paths_change(self):
        self.assertEqual(changed(self.e010, self.e010_old), DATA)
        self.assertEqual(changed(self.e021, self.e021_old), DATA | {'HGNetv2.local_model_dir'})

    def test_old_server_paths(self):
        for cfg in (self.e010_old, self.e021_old):
            for split in ('train_dataloader', 'val_dataloader'):
                for key in ('img_folder', 'ann_file'):
                    self.assertEqual(cfg[split]['dataset'][key], self.e000_old[split]['dataset'][key])
            self.assertTrue(cfg['train_dataloader']['dataset']['plan_dir'].startswith('/data2/zxy/sar/experiments/region_banks/'))
            self.assertTrue(cfg['output_dir'].startswith('/data2/zxy/sar/experiments/'))
        self.assertEqual(self.e021_old['HGNetv2']['local_model_dir'], '/home/zxy/sar/pretrain/hgnetv2/')
        self.assertTrue(self.e021_old['HGNetv2']['pretrained'])

    def test_v3_builder_accepts_the_rebuilt_v2_base(self):
        # tools/build_region_dense_adaptive.py requires the v2 settings to equal the v3 settings minus adaptive_partition.
        v2 = self.e010_old['train_dataloader']['dataset']['plan_settings']
        v3 = dict(self.e021_old['train_dataloader']['dataset']['plan_settings'])
        self.assertTrue(v3.pop('adaptive_partition'))
        self.assertEqual(v2, v3)
        self.assertEqual(self.e021_old['train_dataloader']['dataset']['type'], 'RegionDenseAdaptiveCocoDetection')
        self.assertEqual(self.e021_old['train_dataloader']['total_batch_size'],
                         self.e010_old['train_dataloader']['total_batch_size'])


class PlanSummary(unittest.TestCase):
    def test_reads_metadata_and_report(self):
        meta = dict(version=3, source_images=10, output_canvases_per_epoch=8, optimizer_steps_per_epoch=2,
                    settings={'seed': 0})
        report = dict(epochs={'4': {'routes': {'-1': 5}, 'gt': 9}, '19': {'routes': {'-1': 6}, 'gt': 7}},
                      completed_source_passes=[{'every_source_exactly_once': True}], remaining_pass_sources_used=3,
                      total_source_exposures=40, min_instance_exposure=3, max_instance_exposure=4,
                      total_class_counts=[1, 2], feature_selection_fallbacks=0, adaptive_feature_fallbacks=0)
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'plan_metadata.json').write_text(json.dumps(meta))
            Path(folder, 'plan_report.json').write_text(json.dumps(report))
            out = plan_summary.main(['--plan-dir', folder])
        self.assertEqual(out['planned_epochs'], [4, 19])
        self.assertEqual(out['first_epoch_gt'], 9)
        self.assertEqual(out['last_epoch_routes'], {'-1': 6})
        self.assertTrue(out['every_pass_complete'])
        self.assertEqual(out['adaptive_feature_fallbacks'], 0)
        self.assertNotIn('feature_valid_regions', out)


if __name__ == '__main__':
    unittest.main()
