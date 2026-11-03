"""Protocol v2 stage 1 configs change only resolution-bound settings of the frozen 102-epoch protocol."""

from pathlib import Path
import unittest

from engine.core.yaml_utils import load_config
from engine.data.dataloader import generate_scales


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'configs/deim_dfine'
SKIP = ('__include__', 'output_dir', 'experiment_protocol_document')


def resolved(name):
    config = load_config(str(DIRECTORY / f'deim_hgnetv2_m_sar_aircraft_{name}_old_server.yml'), {})
    return {k: v for k, v in config.items() if k not in SKIP}, config


class ProtocolV2StageOneTests(unittest.TestCase):
    def test_1024_baseline_changes_only_resolution_bound_settings(self):
        new, raw = resolved('1024_102e_b12')
        old, _ = resolved('102e_b28')
        self.assertTrue((ROOT / raw['experiment_protocol_document']).is_file())
        self.assertEqual(new['eval_spatial_size'], [1024, 1024])
        train, val = new['train_dataloader'], new['val_dataloader']
        self.assertEqual(train['total_batch_size'], 12)
        ops = {op['type']: op for op in train['dataset']['transforms']['ops']}
        self.assertEqual((ops['Mosaic']['output_size'], ops['Resize']['size']), (512, [1024, 1024]))
        self.assertEqual(val['dataset']['transforms']['ops'][0], dict(type='Resize', size=[1024, 1024]))
        self.assertEqual(generate_scales(train['collate_fn']['base_size'], train['collate_fn']['base_size_repeat'])[0], 768)
        self.assertEqual(max(generate_scales(1024, 6)), 1280)
        self.assertAlmostEqual(new['optimizer']['lr'], round(3.5e-4 * (12 / 28) ** .5, 5))
        self.assertAlmostEqual(new['optimizer']['params'][0]['lr'], new['optimizer']['lr'] / 10)
        self.assertEqual(new['warmup_iter'], round(2000 / -(-3047 // 28) * -(-3047 // 12), -1))

        # Restore the resolution-bound settings: everything else must equal the frozen protocol.
        for key in ('eval_spatial_size', 'optimizer', 'warmup_iter'):
            new[key] = old[key]
        for loader in ('train_dataloader', 'val_dataloader'):
            new[loader] = dict(new[loader], total_batch_size=old[loader]['total_batch_size'])
            new[loader]['dataset'] = dict(new[loader]['dataset'], transforms=old[loader]['dataset']['transforms'])
        new['train_dataloader']['collate_fn'] = dict(new['train_dataloader']['collate_fn'], base_size=640)
        self.assertEqual(new, old)

    def test_rem_1024_adds_only_rem(self):
        rem, _ = resolved('rem_1024_102e_b12')
        base, _ = resolved('1024_102e_b12')
        self.assertTrue(rem['HybridEncoder']['use_rem'])
        rem['HybridEncoder'] = dict(rem['HybridEncoder'])
        rem['HybridEncoder'].pop('use_rem')
        base['HybridEncoder'].pop('use_rem', None)
        self.assertEqual(rem, base)


if __name__ == '__main__':
    unittest.main()
