"""Targeted REM transform, gradient, encoder routing and protocol checks."""

from pathlib import Path
import unittest

import torch

from engine.core.yaml_utils import load_config
from engine.deim.hybrid_encoder import HybridEncoder
from engine.deim.radar_enhancement import HaarDWT2D, HaarIDWT2D, RadarEnhancementModule
from engine.solver.det_engine import _module_training_metrics


ROOT = Path(__file__).resolve().parents[1]


class RadarEnhancementTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(71)

    def test_haar_coefficients_keep_signs_and_channels_separate(self):
        features = torch.tensor([[[[1., 2.], [3., 4.]], [[2., 2.], [2., 2.]]]])
        dwt = HaarDWT2D()
        bands = dwt(features)
        for band, expected in zip(bands, [[5., 4.], [-2., 0.], [-1., 0.], [0., 0.]]):
            torch.testing.assert_close(band.flatten(), torch.tensor(expected))
        self.assertEqual(list(dwt.parameters()), [])

    def test_haar_even_and_odd_roundtrip_and_input_gradients(self):
        for shape in [(2, 3, 8, 10), (2, 3, 7, 9)]:
            with self.subTest(shape=shape):
                features = torch.randn(shape, requires_grad=True)
                restored = HaarIDWT2D()(HaarDWT2D()(features), shape[-2:])
                torch.testing.assert_close(restored, features, rtol=1e-6, atol=1e-6)
                restored.sum().backward()
                torch.testing.assert_close(features.grad, torch.ones_like(features))

    def test_identity_then_all_four_branches_receive_gradients_with_autocast(self):
        for amp in [False, True]:
            with self.subTest(amp=amp):
                module = RadarEnhancementModule(8)
                features = torch.randn(2, 8, 9, 11, requires_grad=True)
                with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
                    initial = module(features)
                torch.testing.assert_close(initial, features, rtol=0, atol=0)
                self.assertEqual(initial.dtype, features.dtype)
                initial.square().mean().backward()
                self.assertNotEqual(float(module.alpha.grad), 0.)
                module.zero_grad(set_to_none=True)
                with torch.no_grad():
                    module.alpha.fill_(.2)
                with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
                    result = module(features)
                    loss = result.square().mean()
                self.assertEqual(result.shape, features.shape)
                loss.backward()
                for name in ['ll', 'lh', 'hl', 'hh']:
                    gradients = [parameter.grad for parameter in getattr(module, name).parameters()]
                    self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all()
                                        for gradient in gradients), name)
                    self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0., name)

    def test_encoder_inserts_rem_before_fusion_without_changing_default_scales(self):
        kwargs = dict(in_channels=[32, 64, 128], hidden_dim=32, nhead=4,
                      dim_feedforward=64, depth_mult=.34)
        torch.manual_seed(73)
        baseline = HybridEncoder(**kwargs).eval()
        original_rng = torch.get_rng_state()
        torch.manual_seed(73)
        adapted = HybridEncoder(**kwargs, use_rem=True).eval()
        torch.testing.assert_close(torch.get_rng_state(), original_rng, rtol=0, atol=0)
        self.assertIsNone(baseline.rem)
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(value, adapted.state_dict()[name], rtol=0, atol=0)
        features = [torch.randn(2, channels, side, side)
                    for channels, side in [(32, 8), (64, 4), (128, 2)]]
        with torch.no_grad():
            reference = baseline(features)
            initial = adapted(features)
        for actual, expected in zip(initial, reference):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        calls = []
        hook = adapted.rem.register_forward_hook(lambda module, args, result: calls.append(args[0].shape))
        with torch.no_grad():
            adapted.rem.alpha.fill_(.2)
        outputs = adapted(features)
        hook.remove()
        self.assertEqual(calls, [torch.Size([2, 32, 8, 8])])
        self.assertEqual([tuple(value.shape) for value in outputs],
                         [(2, 32, 8, 8), (2, 32, 4, 4), (2, 32, 2, 2)])
        self.assertGreater(float((outputs[0] - reference[0]).detach().abs().max()), 0.)
        sum(value.square().mean() for value in outputs).backward()
        self.assertIsNotNone(adapted.rem.lh[1][0].weight.grad)
        owner = torch.nn.Module()
        owner.encoder = adapted
        self.assertAlmostEqual(_module_training_metrics(owner)['rem_alpha'], .2, places=6)

    def test_config_adds_only_rem_to_old_server_58e_baseline(self):
        directory = ROOT / 'configs/deim_dfine'
        baseline = load_config(str(directory / 'deim_hgnetv2_m_sar_aircraft_58e_b28_old_server.yml'), {})
        config = load_config(str(directory / 'deim_hgnetv2_m_sar_aircraft_rem_58e_b28_old_server.yml'), {})
        self.assertTrue(config['HybridEncoder'].pop('use_rem'))
        self.assertFalse(config['HybridEncoder'].get('use_dsg', False))
        self.assertFalse(config['DFINETransformer'].get('use_scs', False))
        self.assertFalse(config['DFINETransformer'].get('use_slqc', False))
        self.assertFalse(config['DFINETransformer'].get('use_mpq', False))
        self.assertTrue((ROOT / config.pop('experiment_protocol_document')).is_file())
        self.assertNotEqual(config['output_dir'], baseline['output_dir'])
        for candidate in [config, baseline]:
            candidate.pop('__include__', None)
            candidate.pop('output_dir')
            candidate.pop('experiment_protocol_document', None)
        self.assertEqual(config, baseline)


if __name__ == '__main__':
    unittest.main()
