"""E008 must differ from the E000 protocol only in the Mosaic and MixUp probabilities (Dense O2O off)."""

import unittest
from pathlib import Path

from engine.core.yaml_utils import load_config

FOLDER = Path(__file__).resolve().parents[1] / 'configs/deim_dfine'
IGNORED = {'__include__', 'output_dir', 'experiment_baseline_revision'}
EXPECTED = {
    'train_dataloader.dataset.transforms.mosaic_prob': 0.,
    'train_dataloader.collate_fn.mixup_prob': 0.,
}


def flat(cfg, prefix=''):
    out = {}
    for key, value in cfg.items():
        out.update(flat(value, f'{prefix}{key}.') if isinstance(value, dict) else {f'{prefix}{key}': value})
    return out


class NoDenseO2OConfigTest(unittest.TestCase):
    def test_only_dense_o2o_changes(self):
        base = flat(load_config(str(FOLDER / 'deim_hgnetv2_m_sardet100k_36e_b32_nozoom_nocrop.yml')))
        run = flat(load_config(str(FOLDER / 'deim_hgnetv2_m_sardet100k_no_dense_o2o_36e_b32.yml')))
        changed = {k for k in set(base) | set(run) if base.get(k) != run.get(k)} - IGNORED
        self.assertEqual(changed, set(EXPECTED))
        for key, value in EXPECTED.items():
            self.assertEqual(run[key], value, key)
        self.assertTrue(run['output_dir'].startswith('/root/zxy/experiments/'))


if __name__ == '__main__':
    unittest.main()
