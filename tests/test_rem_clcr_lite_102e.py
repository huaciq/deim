"""The 102-epoch REM + CLCR-Lite config differs from each single module only by the other module."""

from pathlib import Path
import unittest

from engine.core.yaml_utils import load_config


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'configs/deim_dfine'
SKIP = ('__include__', 'output_dir', 'experiment_protocol_document')


def resolved(name):
    config = load_config(str(DIRECTORY / f'deim_hgnetv2_m_sar_aircraft_{name}_b28_old_server.yml'), {})
    return {k: v for k, v in config.items() if k not in SKIP}, config


class RemClcrLite102Tests(unittest.TestCase):
    def test_combination_adds_only_the_other_module(self):
        combo, raw = resolved('rem_clcr_lite_102e')
        rem, _ = resolved('rem_102e')
        lite, _ = resolved('clcr_lite_102e')
        self.assertTrue((ROOT / raw['experiment_protocol_document']).is_file())
        self.assertEqual(combo['epoches'], 102)
        self.assertTrue(combo['HybridEncoder']['use_rem'])
        self.assertEqual(combo['HybridEncoder'].get('rem_feat_indices', [0]), [0])
        transformer = combo['DFINETransformer']
        self.assertEqual((transformer['use_clcr_lite'], transformer['clcr_lite_detach_inputs'],
                          transformer['clcr_lite_start_epoch'], transformer['use_clcr']), (True, False, 0, False))
        lite['HybridEncoder'] = dict(lite['HybridEncoder'], use_rem=True)
        self.assertEqual(combo, lite)
        for key in ('use_clcr', 'use_clcr_lite', 'clcr_lite_detach_inputs', 'clcr_lite_start_epoch'):
            transformer.pop(key)
            rem['DFINETransformer'].pop(key, None)
        self.assertEqual(combo, rem)


if __name__ == '__main__':
    unittest.main()
