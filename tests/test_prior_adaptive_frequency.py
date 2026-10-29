"""Targeted PAFP filtering, adaptation, gradient and encoder checks."""

from pathlib import Path
import unittest

import torch

from engine.core.yaml_utils import load_config
from engine.deim.hybrid_encoder import HybridEncoder
from engine.deim.prior_adaptive_frequency import GaussianExpert, PriorAdaptiveFrequencyPurification
from engine.solver.det_engine import _module_training_metrics


ROOT = Path(__file__).resolve().parents[1]


class PriorAdaptiveFrequencyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(109)

    def test_gaussian_experts_preserve_constants_signs_and_learn_sigma(self):
        peaks = []
        for kernel_size, sigma in [(3, .6), (5, 1.), (7, 1.4)]:
            with self.subTest(kernel_size=kernel_size):
                expert = GaussianExpert(kernel_size, sigma)
                kernel = expert.kernel()
                self.assertTrue((kernel > 0).all())
                torch.testing.assert_close(kernel.sum(), torch.tensor(1.))
                constant = torch.full((2, 3, 5, 7), .75)
                torch.testing.assert_close(expert(constant), constant)
                impulse = torch.zeros(1, 2, 11, 11)
                impulse[0, 0, 5, 5], impulse[0, 1, 5, 5] = 1., -1.
                output = expert(impulse)
                torch.testing.assert_close(output[0, 0], -output[0, 1])
                torch.testing.assert_close(output[0, 0].sum(), torch.tensor(1.))
                peaks.append(float(output[0, 0, 5, 5].detach()))
                output.square().mean().backward()
                self.assertTrue(torch.isfinite(expert.raw_sigma.grad))
                self.assertNotEqual(float(expert.raw_sigma.grad), 0.)
        self.assertGreater(peaks[0], peaks[1])
        self.assertGreater(peaks[1], peaks[2])

    def test_confident_details_are_retained_while_weak_details_are_smoothed(self):
        module = PriorAdaptiveFrequencyPurification(1, 1)
        with torch.no_grad():
            for head in [module.spatial_prior, module.channel_prior, module.threshold_head, module.expert_head]:
                for parameter in head.parameters():
                    parameter.zero_()
        low = torch.zeros(1, 1, 15, 15)
        details = torch.zeros(1, 3, 1, 15, 15)
        details[..., 4, 4] = 4.
        details[..., 10, 10] = .1
        purified = module._purify_high(low, details)
        # Fixed .5 spatial/channel priors give .25 modulation. The test asks
        # whether purification treats confidence differently beyond that gain.
        strong = float(purified[0, 0, 0, 4, 4].detach()) / (4 * .25)
        weak = float(purified[0, 0, 0, 10, 10].detach()) / (.1 * .25)
        self.assertGreater(strong, .99)
        self.assertGreater(weak, 0.)
        self.assertLess(weak, .6)
        torch.testing.assert_close(module._purify_high(low, -details), -purified)

    def test_initial_identity_then_all_adaptive_branches_learn_with_autocast(self):
        for amp in [False, True]:
            with self.subTest(amp=amp):
                module = PriorAdaptiveFrequencyPurification(16, 8)
                features = torch.randn(2, 16, 9, 11, requires_grad=True)
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
                    output = module(features)
                    loss = output.square().mean()
                self.assertEqual(output.shape, features.shape)
                loss.backward()
                for name in ['reduce', 'spatial_prior', 'channel_prior', 'threshold_head',
                             'expert_head', 'gaussians', 'refine', 'expand']:
                    gradients = [parameter.grad for parameter in getattr(module, name).parameters()]
                    self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all()
                                        for gradient in gradients), name)
                    self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0., name)

    def test_zero_features_remain_finite_with_nonzero_residual_and_odd_sizes(self):
        module = PriorAdaptiveFrequencyPurification(8, 4)
        with torch.no_grad():
            module.alpha.fill_(.2)
        features = torch.zeros(2, 8, 7, 9, requires_grad=True)
        output = module(features)
        torch.testing.assert_close(output, features, rtol=0, atol=0)
        (output - torch.randn_like(output)).square().mean().backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        for name, parameter in module.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertEqual(module(features.detach().bfloat16()).dtype, torch.bfloat16)

    def test_encoder_routes_pafp_and_preserves_baseline_initial_state(self):
        kwargs = dict(in_channels=[32, 64, 128], hidden_dim=32, nhead=4,
                      dim_feedforward=64, depth_mult=.34)
        torch.manual_seed(110)
        baseline = HybridEncoder(**kwargs).eval()
        original_rng = torch.get_rng_state()
        torch.manual_seed(110)
        adapted = HybridEncoder(**kwargs, use_pafp=True, pafp_bottleneck=8).eval()
        torch.testing.assert_close(torch.get_rng_state(), original_rng, rtol=0, atol=0)
        self.assertIsNone(baseline.pafp)
        for name, reference in baseline.state_dict().items():
            torch.testing.assert_close(adapted.state_dict()[name], reference, rtol=0, atol=0)
        features = [torch.randn(2, channels, side, side)
                    for channels, side in [(32, 8), (64, 4), (128, 2)]]
        with torch.no_grad():
            reference, initial = baseline(features), adapted(features)
        for original, actual in zip(reference, initial):
            torch.testing.assert_close(actual, original, rtol=0, atol=0)
        calls = []
        hook = adapted.pafp.register_forward_hook(lambda module, args, result: calls.append(args[0].shape))
        with torch.no_grad():
            adapted.pafp.alpha.fill_(.2)
        outputs = adapted(features)
        hook.remove()
        self.assertEqual(calls, [torch.Size([2, 32, 8, 8])])
        self.assertEqual([tuple(value.shape) for value in outputs],
                         [(2, 32, 8, 8), (2, 32, 4, 4), (2, 32, 2, 2)])
        self.assertGreater(float((outputs[0] - reference[0]).detach().abs().max()), 0.)
        sum(value.square().mean() for value in outputs).backward()
        self.assertTrue(torch.isfinite(adapted.pafp.threshold_head[1].weight.grad).all())
        owner = torch.nn.Module()
        owner.encoder = adapted
        self.assertAlmostEqual(_module_training_metrics(owner)['pafp_alpha'], .2, places=6)

    def test_config_adds_only_pafp_and_preserves_the_complete_58e_protocol(self):
        directory = ROOT / 'configs/deim_dfine'
        baseline = load_config(str(directory / 'deim_hgnetv2_m_sar_aircraft_58e_b28_old_server.yml'), {})
        config = load_config(str(directory / 'deim_hgnetv2_m_sar_aircraft_pafp_58e_b28_old_server.yml'), {})
        self.assertTrue(config['HybridEncoder'].pop('use_pafp'))
        self.assertEqual(config['HybridEncoder'].pop('pafp_bottleneck'), 64)
        for flag in ['use_rem', 'use_dsg', 'use_pacfg']:
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
