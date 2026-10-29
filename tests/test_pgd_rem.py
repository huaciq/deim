"""PGD-REM's specified signal path, learning behavior and 58-epoch protocol."""

import math
from pathlib import Path
import unittest

import torch
from torch import nn

from engine.core.yaml_utils import load_config
from engine.deim.hybrid_encoder import HybridEncoder
from engine.deim.radar_enhancement import (
    HaarDWT2D, HaarIDWT2D, LowFrequencyStructureGate, PriorGatedDirectionalREM,
    RadarEnhancementModule,
)
from engine.solver.det_engine import _module_training_metrics


ROOT = Path(__file__).resolve().parents[1]
ENCODER_ARGS = dict(in_channels=[32, 64, 128], hidden_dim=32, nhead=4,
                    dim_feedforward=64, depth_mult=.34)


class PgdRemTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(131)

    def test_full_width_single_transform_and_original_directional_chains(self):
        module = PriorGatedDirectionalREM(256).eval()
        reference = RadarEnhancementModule(256)
        for name in ['lh', 'hl', 'hh']:
            self.assertEqual(str(getattr(module, name)), str(getattr(reference, name)))
        self.assertEqual((module.ll.in_channels, module.ll.out_channels), (256, 256))
        self.assertEqual((module.prior_gate.reduce.in_channels,
                          module.prior_gate.reduce.out_channels), (256, 64))
        self.assertEqual(module.prior_gate.smooth.groups, 64)
        self.assertEqual(module.prior_gate.smooth.kernel_size, (5, 5))
        self.assertEqual(sum(isinstance(layer, HaarDWT2D) for layer in module.modules()), 1)
        self.assertEqual(sum(isinstance(layer, HaarIDWT2D) for layer in module.modules()), 1)
        self.assertEqual([name for name, parameter in module.named_parameters()
                          if parameter.ndim == 0], ['alpha'])

        for height, width in [(8, 10), (9, 11)]:
            calls = {'dwt': [], 'idwt': [], 'gates': []}
            handles = [
                module.dwt.register_forward_hook(
                    lambda layer, args, output: calls['dwt'].append(output)),
                module.idwt.register_forward_pre_hook(
                    lambda layer, args: calls['idwt'].append(args[0])),
                module.prior_gate.register_forward_hook(
                    lambda layer, args, output: calls['gates'].append(output)),
            ]
            try:
                with torch.no_grad():
                    features = torch.randn(1, 256, height, width)
                    result = module(features)
                torch.testing.assert_close(result, features, rtol=0, atol=0)
                for kind in calls:
                    self.assertEqual(len(calls[kind]), 1, kind)
                    for band in calls[kind][0]:
                        self.assertEqual(band.shape, (1, 256, (height + 1) // 2, (width + 1) // 2))
            finally:
                for handle in handles:
                    handle.remove()

    def test_each_gate_modulates_its_signed_band_before_one_residual_addition(self):
        module = PriorGatedDirectionalREM(1, prior_channels=1).eval()
        for name in ['ll', 'lh', 'hl', 'hh']:
            setattr(module, name, nn.Identity())
        with torch.no_grad():
            module.alpha.fill_(.2)
            module.prior_gate.gate.weight.zero_()
            module.prior_gate.gate.bias.copy_(torch.tensor([-math.log(3.), math.log(3.), 0.]))
        features = torch.tensor([[[[1., 2.], [3., 7.]]]])
        # Haar coefficients are (6.5, -3.5, -2.5, 1.5). Gates are (.75, 1.25, 1).
        # Analytic inverse after gating preserves LL and the signed band order.
        reconstructed = torch.tensor([[[[1.125, 2.75], [2.25, 6.875]]]])
        torch.testing.assert_close(module(features), features + .2 * reconstructed)

    def test_low_frequency_comparison_controls_spatial_per_channel_gates(self):
        gate = LowFrequencyStructureGate(2, prior_channels=2)
        with torch.no_grad():
            gate.reduce.weight.copy_(torch.eye(2).reshape(2, 2, 1, 1))
            gate.reduce.bias.zero_()
            gate.gate.weight.zero_()
            gate.gate.bias.zero_()
            # LH channel 0 reads only prior-minus-LL; the other outputs stay neutral.
            gate.gate.weight[0, 2, 0, 0] = 10.
        constant = torch.ones(1, 2, 7, 9)
        torch.testing.assert_close(gate.smooth(constant), constant)
        ll = torch.zeros_like(constant)
        ll[0, 0, 3, 4] = 2.
        lh, hl, hh = gate(ll)
        self.assertLess(lh[0, 0, 3, 4].item(), 1.)
        self.assertGreater(lh[0, 0, 3, 3].item(), 1.)
        torch.testing.assert_close(lh[:, 1], torch.ones_like(lh[:, 1]))
        for unchanged in [hl, hh]:
            torch.testing.assert_close(unchanged, torch.ones_like(unchanged))
        for value in [lh, hl, hh]:
            self.assertTrue(((value >= .5) & (value <= 1.5)).all())

    def test_zero_alpha_starts_learning_and_then_all_branches_receive_gradients(self):
        for amp, dtype in [(False, torch.float32), (True, torch.float32), (True, torch.bfloat16)]:
            with self.subTest(amp=amp, dtype=dtype):
                module = PriorGatedDirectionalREM(8, prior_channels=4).train()
                features = torch.randn(2, 8, 9, 11, dtype=dtype, requires_grad=True)
                optimizer = torch.optim.SGD(module.parameters(), lr=.1)
                with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
                    initial = module(features)
                torch.testing.assert_close(initial, features, rtol=0, atol=0)
                self.assertEqual(initial.dtype, dtype)
                initial.float().square().mean().backward()
                self.assertTrue(torch.isfinite(module.alpha.grad))
                self.assertNotEqual(module.alpha.grad.item(), 0.)
                optimizer.step()
                self.assertNotEqual(module.alpha.item(), 0.)
                optimizer.zero_grad(set_to_none=True)
                features.grad = None
                with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
                    result = module(features)
                self.assertEqual(result.shape, features.shape)
                self.assertEqual(result.dtype, dtype)
                result.float().square().mean().backward()
                for name, parameter in module.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                for branch in [module.ll, module.lh, module.hl, module.hh,
                               module.prior_gate.reduce, module.prior_gate.smooth, module.prior_gate.gate]:
                    self.assertGreater(sum(parameter.grad.abs().sum().item()
                                           for parameter in branch.parameters()), 0.)
                self.assertTrue(torch.isfinite(features.grad).all())

    def test_encoder_preserves_baseline_initialization_and_routes_stride8_to_fpn(self):
        torch.manual_seed(137)
        baseline = HybridEncoder(**ENCODER_ARGS).eval()
        baseline_rng = torch.get_rng_state()
        torch.manual_seed(137)
        adapted = HybridEncoder(**ENCODER_ARGS, use_pgd_rem=True, pgd_rem_prior_channels=8).eval()
        torch.testing.assert_close(torch.get_rng_state(), baseline_rng, rtol=0, atol=0)
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(adapted.state_dict()[name], value, rtol=0, atol=0)
        missing, unexpected = adapted.load_state_dict(baseline.state_dict(), strict=False)
        self.assertTrue(missing)
        self.assertTrue(all(name.startswith('pgd_rem.') for name in missing))
        self.assertEqual(unexpected, [])
        for name in ['rem', 'pacfg', 'pafp']:
            self.assertIsNone(getattr(adapted, name))
        features = [torch.randn(2, channels, height, width)
                    for channels, height, width in [(32, 12, 20), (64, 6, 10), (128, 3, 5)]]
        with torch.no_grad():
            reference = baseline(features)
            initial = adapted(features)
            for actual, expected in zip(initial, reference):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            adapted.pgd_rem.alpha.fill_(.2)

        projections, calls, fpn_inputs = {}, [], []
        handles = [
            adapted.input_proj[0].register_forward_hook(
                lambda layer, args, output: projections.__setitem__(0, output)),
            adapted.input_proj[1].register_forward_hook(
                lambda layer, args, output: projections.__setitem__(1, output)),
            adapted.pgd_rem.register_forward_hook(
                lambda layer, args, output: calls.append((args[0], output))),
        ]
        handles += [block.register_forward_pre_hook(
            lambda layer, args: fpn_inputs.append(args[0])) for block in adapted.fpn_blocks]
        try:
            outputs = adapted(features)
            self.assertEqual(len(calls), 1)
            self.assertIs(calls[0][0], projections[0])
            torch.testing.assert_close(fpn_inputs[0][:, 32:], projections[1], rtol=0, atol=0)
            torch.testing.assert_close(fpn_inputs[1][:, 32:], calls[0][1], rtol=0, atol=0)
            self.assertEqual([tuple(value.shape) for value in outputs],
                             [(2, 32, 12, 20), (2, 32, 6, 10), (2, 32, 3, 5)])
            sum(value.square().mean() for value in outputs).backward()
        finally:
            for handle in handles:
                handle.remove()
        self.assertGreater(adapted.pgd_rem.prior_gate.reduce.weight.grad.abs().sum().item(), 0.)
        owner = nn.Module()
        owner.encoder = adapted
        self.assertAlmostEqual(_module_training_metrics(owner)['pgd_rem_alpha'], .2, places=6)

    def test_config_changes_only_adapter_and_run_paths(self):
        directory = ROOT / 'configs/deim_dfine'
        baseline = load_config(str(directory / 'deim_hgnetv2_m_sar_aircraft_58e_b28_old_server.yml'), {})
        config = load_config(str(directory / 'deim_hgnetv2_m_sar_aircraft_pgd_rem_58e_b28_old_server.yml'), {})
        encoder = config['HybridEncoder']
        self.assertTrue(encoder.pop('use_pgd_rem'))
        self.assertEqual(encoder.pop('pgd_rem_prior_channels'), 64)
        self.assertEqual(encoder['hidden_dim'], 256)
        self.assertEqual(encoder['feat_strides'], [8, 16, 32])
        for flag in ['use_rem', 'use_pacfg', 'use_pafp', 'use_dsg']:
            self.assertFalse(encoder.get(flag, False))
        self.assertEqual(config['epoches'], 58)
        self.assertEqual(config['train_dataloader']['total_batch_size'], 28)
        self.assertTrue((ROOT / config['experiment_protocol_document']).is_file())
        self.assertNotEqual(config['output_dir'], baseline['output_dir'])
        for candidate in [config, baseline]:
            candidate.pop('__include__', None)
            candidate.pop('output_dir')
            candidate.pop('experiment_protocol_document')
        self.assertEqual(config, baseline)


if __name__ == '__main__':
    unittest.main()
