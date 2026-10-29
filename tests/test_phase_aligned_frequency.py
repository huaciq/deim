"""Targeted PACFG spectral, gradient, encoder and protocol checks."""

import math
from pathlib import Path
import unittest

import torch

from engine.core.yaml_utils import load_config
from engine.deim.hybrid_encoder import HybridEncoder
from engine.deim.phase_aligned_frequency import PhaseAlignedCrossFrequencyGuidance
from engine.solver.det_engine import _module_training_metrics


ROOT = Path(__file__).resolve().parents[1]


class PhaseAlignedFrequencyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(96)

    def test_circular_phase_alignment_and_amplitude_consistency_on_known_signals(self):
        module = PhaseAlignedCrossFrequencyGuidance(1, 1)
        with torch.no_grad():
            for parameter in module.frequency_gate.parameters():
                parameter.zero_()  # A fixed .5 gate makes the expected step analytic.
        theta = torch.arange(16, dtype=torch.float32) * (2 * math.pi / 16)
        low = theta.cos()[None, None, None].expand(1, 1, 8, 16)
        prior = (theta + math.pi / 3).cos()[None, None, None].expand_as(low)
        enhanced = module._enhance_low(low, prior)
        source_bin = torch.fft.fft2(low, norm='ortho')[0, 0, 0, 1]
        target_bin = torch.fft.fft2(enhanced, norm='ortho')[0, 0, 0, 1]
        expected_step = (math.pi / 4) * .5 * math.sin(math.pi / 3)
        self.assertAlmostEqual(float((target_bin * source_bin.conj()).angle().detach()), expected_step, places=5)
        torch.testing.assert_close(target_bin.abs(), source_bin.abs(), rtol=1e-5, atol=1e-5)
        stronger = module._enhance_low(low, low * 2)
        stronger_bin = torch.fft.fft2(stronger, norm='ortho')[0, 0, 0, 1]
        self.assertGreater(float(stronger_bin.abs().detach()), float(source_bin.abs()))
        self.assertLess(float(stronger_bin.abs().detach()), float(source_bin.abs()) * 2)
        self.assertAlmostEqual(float((stronger_bin * source_bin.conj()).angle().detach()), 0., places=5)
        torch.testing.assert_close(module._enhance_low(low, low), low, rtol=1e-5, atol=1e-6)

    def test_zero_spectrum_and_signed_feature_gradients_are_finite(self):
        module = PhaseAlignedCrossFrequencyGuidance(8, 4)
        with torch.no_grad():
            module.alpha.fill_(.2)
        for zero in [True, False]:
            with self.subTest(zero=zero):
                features = (torch.zeros(2, 8, 7, 11) if zero else torch.randn(2, 8, 7, 11)).requires_grad_()
                module.zero_grad(set_to_none=True)
                output = module(features)
                if zero:
                    torch.testing.assert_close(output, features, rtol=0, atol=0)
                loss = (output - torch.randn_like(output)).square().mean()
                loss.backward()
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue(torch.isfinite(features.grad).all())
                for name, parameter in module.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_identity_then_branch_learning_with_autocast_and_odd_sizes(self):
        for amp in [False, True]:
            with self.subTest(amp=amp):
                module = PhaseAlignedCrossFrequencyGuidance(16, 8)
                features = torch.randn(2, 16, 9, 11, requires_grad=True)
                with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
                    initial = module(features)
                self.assertEqual(initial.dtype, features.dtype)
                torch.testing.assert_close(initial, features, rtol=0, atol=0)
                initial.square().mean().backward()
                self.assertNotEqual(float(module.alpha.grad), 0.)
                module.zero_grad(set_to_none=True)
                with torch.no_grad():
                    module.alpha.fill_(.2)
                with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
                    output = module(features)
                    loss = output.square().mean()
                self.assertEqual(output.shape, features.shape)
                loss.backward()
                for name in ['reduce', 'prior', 'frequency_gate', 'subband_gate', 'local_refine', 'expand']:
                    gradients = [parameter.grad for parameter in getattr(module, name).parameters()]
                    self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all()
                                        for gradient in gradients), name)
                    self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0., name)
        bf16 = torch.randn(2, 16, 7, 9, dtype=torch.bfloat16)
        self.assertEqual(module(bf16).dtype, bf16.dtype)

    def test_encoder_preserves_initial_state_and_routes_pacfg_before_fusion(self):
        kwargs = dict(in_channels=[32, 64, 128], hidden_dim=32, nhead=4,
                      dim_feedforward=64, depth_mult=.34)
        torch.manual_seed(97)
        baseline = HybridEncoder(**kwargs).eval()
        original_rng = torch.get_rng_state()
        torch.manual_seed(97)
        adapted = HybridEncoder(**kwargs, use_pacfg=True, pacfg_bottleneck=8).eval()
        torch.testing.assert_close(torch.get_rng_state(), original_rng, rtol=0, atol=0)
        self.assertIsNone(baseline.pacfg)
        for name, expected in baseline.state_dict().items():
            torch.testing.assert_close(adapted.state_dict()[name], expected, rtol=0, atol=0)
        features = [torch.randn(2, channels, side, side)
                    for channels, side in [(32, 8), (64, 4), (128, 2)]]
        with torch.no_grad():
            expected, initial = baseline(features), adapted(features)
        for reference, actual in zip(expected, initial):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        calls = []
        hook = adapted.pacfg.register_forward_hook(lambda module, args, result: calls.append(args[0].shape))
        with torch.no_grad():
            adapted.pacfg.alpha.fill_(.2)
        outputs = adapted(features)
        hook.remove()
        self.assertEqual(calls, [torch.Size([2, 32, 8, 8])])
        self.assertEqual([tuple(value.shape) for value in outputs],
                         [(2, 32, 8, 8), (2, 32, 4, 4), (2, 32, 2, 2)])
        self.assertGreater(float((outputs[0] - expected[0]).detach().abs().max()), 0.)
        sum(value.square().mean() for value in outputs).backward()
        self.assertTrue(torch.isfinite(adapted.pacfg.frequency_gate[0].weight.grad).all())
        owner = torch.nn.Module()
        owner.encoder = adapted
        self.assertAlmostEqual(_module_training_metrics(owner)['pacfg_alpha'], .2, places=6)

    def test_config_preserves_58e_baseline_and_enables_only_this_adapter(self):
        directory = ROOT / 'configs/deim_dfine'
        baseline = load_config(str(directory / 'deim_hgnetv2_m_sar_aircraft_58e_b28_old_server.yml'), {})
        config = load_config(str(directory / 'deim_hgnetv2_m_sar_aircraft_pacfg_58e_b28_old_server.yml'), {})
        self.assertTrue(config['HybridEncoder'].pop('use_pacfg'))
        self.assertEqual(config['HybridEncoder'].pop('pacfg_bottleneck'), 64)
        for flag in ['use_rem', 'use_dsg']:
            self.assertFalse(config['HybridEncoder'].get(flag, False))
        self.assertFalse(config['HGNetv2'].get('use_crossgate_dbdm', False))
        for flag in ['use_scs', 'use_slqc', 'use_mpq']:
            self.assertFalse(config['DFINETransformer'].get(flag, False))
        self.assertTrue((ROOT / config.pop('experiment_protocol_document')).is_file())
        self.assertNotEqual(config['output_dir'], baseline['output_dir'])
        for candidate in [config, baseline]:
            candidate.pop('__include__', None)
            candidate.pop('output_dir')
            candidate.pop('experiment_protocol_document', None)
        self.assertEqual(config, baseline)


if __name__ == '__main__':
    unittest.main()
