"""E007 must differ from the old-server E000 protocol only in Mosaic/MixUp probability and the VFL loss."""

import unittest
from pathlib import Path

from engine.core.yaml_utils import load_config

FOLDER = Path(__file__).resolve().parents[1] / 'configs/deim_dfine'
IGNORED = {'__include__', 'output_dir', 'experiment_protocol_document'}
EXPECTED = {
    'train_dataloader.dataset.transforms.mosaic_prob': 0.,
    'train_dataloader.collate_fn.mixup_prob': 0.,
    'DEIMCriterion.losses': ['vfl', 'boxes', 'local'],
    'DEIMCriterion.gamma': 2.0,
}


def flat(cfg, prefix=''):
    out = {}
    for key, value in cfg.items():
        out.update(flat(value, f'{prefix}{key}.') if isinstance(value, dict) else {f'{prefix}{key}': value})
    return out


class NoMalNoDenseO2OConfigTest(unittest.TestCase):
    def test_only_the_two_mechanisms_change(self):
        base = flat(load_config(str(FOLDER / 'deim_hgnetv2_m_sardet100k_36e_b32_nozoom_nocrop_old_server.yml')))
        run = flat(load_config(str(FOLDER / 'deim_hgnetv2_m_sardet100k_no_mal_no_dense_o2o_36e_b32_old_server.yml')))
        changed = {k for k in set(base) | set(run) if base.get(k) != run.get(k)} - IGNORED
        self.assertEqual(changed, set(EXPECTED))
        for key, value in EXPECTED.items():
            self.assertEqual(run[key], value, key)
        self.assertEqual(run['DEIMCriterion.weight_dict']['loss_vfl'] if 'DEIMCriterion.weight_dict' in run
                         else run['DEIMCriterion.weight_dict.loss_vfl'], 1)
        self.assertEqual(run['DEIMCriterion.alpha'], 0.75)
        self.assertTrue(run['output_dir'].startswith('/data2/zxy/sar/experiments/'))


if __name__ == '__main__':
    unittest.main()
