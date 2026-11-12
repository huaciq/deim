import copy
import unittest

from engine.core.yaml_utils import load_config


class RegionDenseMixupConfigTests(unittest.TestCase):
    def test_only_mixup_differs_from_region_dense(self):
        base = load_config('configs/deim_dfine/deim_hgnetv2_m_sardet100k_region_dense_36e_b32.yml', {})
        new = load_config('configs/deim_dfine/deim_hgnetv2_m_sardet100k_region_dense_mixup_36e_b32.yml', {})
        original = load_config('configs/deim_dfine/deim_hgnetv2_m_sardet100k_36e_b32_nozoom_nocrop.yml', {})
        for config in (base, new, original):
            config.pop('__include__', None)
        collate = new['train_dataloader']['collate_fn']
        # MixUp is exactly as in the original DEIM protocol (E000): probability and epoch window.
        self.assertEqual(collate['mixup_prob'], original['train_dataloader']['collate_fn']['mixup_prob'])
        self.assertEqual(collate['mixup_epochs'], original['train_dataloader']['collate_fn']['mixup_epochs'])
        self.assertEqual(base['train_dataloader']['collate_fn']['mixup_prob'], 0.)
        # Mosaic stays off and RegionDense is unchanged.
        self.assertEqual(new['train_dataloader']['dataset']['transforms']['mosaic_prob'], 0.)
        edited = copy.deepcopy(new)
        edited['train_dataloader']['collate_fn']['mixup_prob'] = 0.
        for key in ('output_dir', 'experiment_protocol_document'):
            edited[key] = base[key]
        self.assertEqual(edited, base)


if __name__ == '__main__':
    unittest.main()
