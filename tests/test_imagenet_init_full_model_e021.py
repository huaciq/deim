"""E021 must be E017 with E019's backbone initialisation, i.e. E021 - E019 changes exactly what E017 - E000 changes.

Run on CPU: python -m unittest tests.test_imagenet_init_full_model_e021 -v
"""
import unittest
from pathlib import Path

from engine.core.yaml_utils import load_config

FOLDER = Path(__file__).resolve().parents[1] / 'configs/deim_dfine'
META = {'__include__', 'output_dir', 'experiment_protocol_document', 'experiment_baseline_revision'}
INIT = {'HGNetv2.pretrained', 'HGNetv2.local_model_dir'}


def flat(cfg, prefix=''):
    out = {}
    for key, value in cfg.items():
        out.update(flat(value, f'{prefix}{key}.') if isinstance(value, dict) else {f'{prefix}{key}': value})
    return out


def load(name):
    return flat(load_config(str(FOLDER / name)))


def changed(a, b):
    return {k for k in set(a) | set(b) if a.get(k, '<none>') != b.get(k, '<none>')} - META


class Config(unittest.TestCase):
    e000 = load('deim_hgnetv2_m_sardet100k_36e_b32_nozoom_nocrop.yml')
    e017 = load('deim_hgnetv2_m_sardet100k_orientation_fusion_v2_region_dense_adaptive_36e_b32.yml')
    e019 = load('deim_hgnetv2_m_sardet100k_imagenet_init_36e_b32.yml')
    e021 = load('deim_hgnetv2_m_sardet100k_imagenet_init_full_model_36e_b32.yml')

    def test_differs_from_e017_only_in_backbone_initialisation(self):
        self.assertEqual(changed(self.e017, self.e021), INIT)
        for key in INIT:
            self.assertEqual(self.e021[key], self.e019[key], key)

    def test_method_changes_over_e019_equal_those_of_e017_over_e000(self):
        method = changed(self.e000, self.e017)
        self.assertTrue(any(k.startswith('DEIM.orientation_fusion.') for k in method))
        self.assertEqual(self.e017['train_dataloader.dataset.type'], 'RegionDenseAdaptiveCocoDetection')
        self.assertEqual(changed(self.e019, self.e021), method)
        for key in method:
            self.assertEqual(self.e021.get(key, '<none>'), self.e017.get(key, '<none>'), key)

    def test_output_and_plan_paths(self):
        self.assertTrue(self.e021['output_dir'].startswith('/root/zxy/experiments/'))
        self.assertNotEqual(self.e021['output_dir'], self.e017['output_dir'])
        self.assertEqual(self.e021['train_dataloader.dataset.plan_dir'],
                         '/root/zxy/experiments/region_banks/sardet100k_train_region_dense_sar_v3_seed0_b32')


if __name__ == '__main__':
    unittest.main()
