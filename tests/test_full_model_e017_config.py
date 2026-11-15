"""E017 must be exactly E016 + E013-v2's orientation fusion, and exactly E013-v2 + E016's training data."""

import unittest
from pathlib import Path

from engine.core.yaml_utils import load_config

FOLDER = Path(__file__).resolve().parents[1] / 'configs/deim_dfine'
META = {'__include__', 'output_dir', 'experiment_protocol_document', 'experiment_baseline_revision'}


def flat(cfg, prefix=''):
    out = {}
    for key, value in cfg.items():
        out.update(flat(value, f'{prefix}{key}.') if isinstance(value, dict) else {f'{prefix}{key}': value})
    return out


def load(name):
    return flat(load_config(str(FOLDER / name)))


def changed(a, b):
    return {k for k in set(a) | set(b) if a.get(k, '<none>') != b.get(k, '<none>')} - META


class FullModelConfigTest(unittest.TestCase):
    full = load('deim_hgnetv2_m_sardet100k_orientation_fusion_v2_region_dense_adaptive_36e_b32.yml')

    def test_adds_only_orientation_fusion_to_e016(self):
        diff = changed(load('deim_hgnetv2_m_sardet100k_region_dense_adaptive_36e_b32.yml'), self.full)
        self.assertTrue(diff)
        self.assertTrue(all(k.startswith('DEIM.orientation_fusion.') for k in diff), diff)

    def test_orientation_block_equals_e013_v2(self):
        e013 = load('deim_hgnetv2_m_sardet100k_orientation_fusion_v2_36e_b32.yml')
        keys = {k for k in e013 if k.startswith('DEIM.orientation_fusion.')}
        self.assertTrue(keys)
        self.assertEqual({k: self.full[k] for k in keys}, {k: e013[k] for k in keys})

    def test_differs_from_e013_v2_only_in_training_data(self):
        diff = changed(load('deim_hgnetv2_m_sardet100k_orientation_fusion_v2_36e_b32.yml'), self.full)
        self.assertTrue(all(k.startswith(('train_dataloader.dataset.', 'train_dataloader.collate_fn.mixup_prob'))
                            for k in diff), diff)
        self.assertEqual(self.full['train_dataloader.dataset.type'], 'RegionDenseAdaptiveCocoDetection')
        self.assertTrue(self.full['train_dataloader.dataset.plan_settings.adaptive_partition'])
        self.assertTrue(self.full['output_dir'].startswith('/root/zxy/experiments/'))


if __name__ == '__main__':
    unittest.main()
