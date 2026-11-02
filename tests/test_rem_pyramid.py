"""Targeted checks for REM on several projected encoder levels (REM-Pyramid)."""

from pathlib import Path
import unittest

import torch

from engine.core.yaml_utils import load_config
from engine.deim.hybrid_encoder import HybridEncoder
from engine.solver.det_engine import _module_training_metrics


ROOT = Path(__file__).resolve().parents[1]
KWARGS = dict(in_channels=[32, 64, 128], hidden_dim=32, nhead=4, dim_feedforward=64, depth_mult=.34)


def features():
    return [torch.randn(2, channels, side, side) for channels, side in [(32, 8), (64, 4), (128, 2)]]


class RemPyramidTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def build(self, **extra):
        torch.manual_seed(83)
        return HybridEncoder(**KWARGS, **extra).eval()

    def test_default_rem_keeps_reported_keys_and_no_extra_levels(self):
        single = self.build(use_rem=True)
        explicit = self.build(use_rem=True, rem_feat_indices=[0])
        self.assertEqual(len(single.rem_levels), 0)
        self.assertEqual(single.state_dict().keys(), explicit.state_dict().keys())
        self.assertFalse(any(name.startswith('rem_levels') for name in single.state_dict()))

    def test_pyramid_starts_as_identity_and_shares_stride8_initialization(self):
        baseline = self.build()
        single = self.build(use_rem=True)
        torch.manual_seed(83)
        pyramid = HybridEncoder(**KWARGS, use_rem=True, rem_feat_indices=[0, 1, 2]).eval()
        after_pyramid = torch.get_rng_state()
        torch.manual_seed(83)
        HybridEncoder(**KWARGS)
        torch.testing.assert_close(after_pyramid, torch.get_rng_state(), rtol=0, atol=0)
        self.assertEqual(sorted(pyramid.rem_levels), ['1', '2'])
        for name, value in single.state_dict().items():
            torch.testing.assert_close(pyramid.state_dict()[name], value, rtol=0, atol=0)
        inputs = features()
        with torch.no_grad():
            reference = baseline(inputs)
            initial = pyramid(inputs)
        for actual, expected in zip(initial, reference):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_every_level_is_applied_receives_gradients_and_is_logged(self):
        pyramid = self.build(use_rem=True, rem_feat_indices=[0, 1, 2]).train()
        calls = []
        modules = [pyramid.rem, pyramid.rem_levels['1'], pyramid.rem_levels['2']]
        hooks = [m.register_forward_hook(lambda module, args, result: calls.append(tuple(args[0].shape)))
                 for m in modules]
        with torch.no_grad():
            for value, module in zip((.2, .3, .4), modules):
                module.alpha.fill_(value)
        outputs = pyramid(features())
        for hook in hooks:
            hook.remove()
        self.assertEqual(calls, [(2, 32, 8, 8), (2, 32, 4, 4), (2, 32, 2, 2)])
        sum(value.square().mean() for value in outputs).backward()
        for module in modules:
            self.assertIsNotNone(module.alpha.grad)
            self.assertGreater(float(module.lh[1][0].weight.grad.abs().sum()), 0.)
        owner = torch.nn.Module()
        owner.encoder = pyramid
        metrics = _module_training_metrics(owner)
        self.assertAlmostEqual(metrics['rem_alpha'], .2, places=6)
        self.assertAlmostEqual(metrics['rem_p4_alpha'], .3, places=6)
        self.assertAlmostEqual(metrics['rem_p5_alpha'], .4, places=6)

    def test_invalid_levels_are_rejected(self):
        for indices in ([0, 0], [3], [-1]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                self.build(use_rem=True, rem_feat_indices=indices)

    def test_configs_change_only_rem_levels_relative_to_rem_58e(self):
        directory = ROOT / 'configs/deim_dfine'
        skip = ('__include__', 'output_dir', 'experiment_protocol_document')
        rem = load_config(str(directory / 'deim_hgnetv2_m_sar_aircraft_rem_58e_b28_old_server.yml'), {})
        for name, indices in (('rem_pyramid', [0, 1, 2]), ('rem_s8s16', [0, 1])):
            with self.subTest(name=name):
                config = load_config(str(directory / f'deim_hgnetv2_m_sar_aircraft_{name}_58e_b28_old_server.yml'), {})
                self.assertEqual(config['HybridEncoder'].pop('rem_feat_indices'), indices)
                self.assertTrue((ROOT / config['experiment_protocol_document']).is_file())
                self.assertNotEqual(config['output_dir'], rem['output_dir'])
                self.assertEqual({k: v for k, v in config.items() if k not in skip},
                                 {k: v for k, v in rem.items() if k not in skip})


if __name__ == '__main__':
    unittest.main()
