"""CPU configuration and LR regression for the SAR-AIRcraft 58-epoch protocol."""
import copy
from pathlib import Path
import unittest

import yaml

from engine.core.yaml_utils import load_config
from engine.optim.lr_scheduler import flat_cosine_schedule


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / 'configs' / 'deim_dfine'
PARENT_NAME = 'deim_hgnetv2_m_sar_aircraft_102e_b28.yml'
CHILD_NAME = 'deim_hgnetv2_m_sar_aircraft_58e_b28.yml'
SCHEDULE_OVERRIDES = {
    'epoches': 58,
    'warmup_iter': 432,
    'flat_epoch': 29,
    'no_aug_epoch': 8,
    'train_dataloader.dataset.transforms.policy.epoch': [4, 29, 50],
    'train_dataloader.collate_fn.mixup_epochs': [4, 29],
    'train_dataloader.collate_fn.stop_epoch': 50,
}


def config_value(config, path):
    for key in path.split('.'):
        config = config[key]
    return config


def without_protocol_overrides(config):
    config = copy.deepcopy(config)
    # load_config retains this source directive; it is not a runtime setting.
    config.pop('__include__', None)
    for path in [*SCHEDULE_OVERRIDES, 'output_dir', 'experiment_protocol_document']:
        keys = path.split('.')
        section = config
        for key in keys[:-1]:
            section = section[key]
        section.pop(keys[-1], None)
    return config


class Sar58eProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = load_config(str(CONFIG_DIR / PARENT_NAME), {})
        cls.config = load_config(str(CONFIG_DIR / CHILD_NAME), {})

    def test_single_parent_and_protocol_archive(self):
        source = yaml.safe_load((CONFIG_DIR / CHILD_NAME).read_text(encoding='utf-8'))
        self.assertEqual(source['__include__'], ['./' + PARENT_NAME])
        self.assertEqual(self.config['output_dir'],
                         '/root/zxy/experiments/deim_m_sar_aircraft_58e_b28/seed0')
        self.assertEqual(self.config['experiment_protocol_document'],
                         'docs/sar_aircraft_deim_m_58e_protocol.md')
        self.assertTrue((ROOT / self.config['experiment_protocol_document']).is_file())

    def test_resolved_schedule_and_inherited_policy_ops(self):
        for path, expected in SCHEDULE_OVERRIDES.items():
            with self.subTest(path=path):
                self.assertEqual(config_value(self.config, path), expected)
        self.assertEqual(
            self.config['train_dataloader']['dataset']['transforms']['policy']['ops'],
            ['Mosaic', 'RandomPhotometricDistort'])

    def test_all_other_resolved_settings_match_102e_baseline(self):
        self.assertEqual(without_protocol_overrides(self.config),
                         without_protocol_overrides(self.baseline))

    def test_flat_cosine_boundaries_at_108_iterations_per_epoch(self):
        config = self.config
        iter_per_epoch = 108
        total_iter = config['epoches'] * iter_per_epoch
        flat_iter = config['flat_epoch'] * iter_per_epoch
        no_aug_iter = config['no_aug_epoch'] * iter_per_epoch
        self.assertEqual((total_iter, flat_iter, total_iter - no_aug_iter),
                         (6264, 3132, 5400))
        for step, expected in [
            (0, 0.0),
            (216, 0.0000875),
            (432, 0.00035),
            (3132, 0.00035),
            (5400, 0.000175),
            (5401, 0.000175),
            (6263, 0.000175),
            (6264, 0.000175),
        ]:
            with self.subTest(step=step):
                actual = flat_cosine_schedule(
                    total_iter=total_iter, warmup_iter=config['warmup_iter'],
                    flat_iter=flat_iter, no_aug_iter=no_aug_iter, current_iter=step,
                    init_lr=config['optimizer']['lr'],
                    min_lr=config['optimizer']['lr'] * config['lr_gamma'])
                self.assertAlmostEqual(actual, expected, places=12)


if __name__ == '__main__':
    unittest.main()
