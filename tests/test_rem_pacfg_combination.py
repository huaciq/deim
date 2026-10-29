"""REM/PACFG composition: matched initialization, connected gradients and protocol."""

from pathlib import Path
import unittest

import torch

from engine.core.yaml_utils import load_config
from engine.deim.hybrid_encoder import HybridEncoder
from engine.solver.det_engine import _module_training_metrics


ROOT = Path(__file__).resolve().parents[1]
ENCODER_ARGS = dict(in_channels=[32, 64, 128], hidden_dim=32, nhead=4,
                    dim_feedforward=64, depth_mult=.34)


class RemPacfgCombinationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(117)

    def test_combination_preserves_baseline_and_single_adapter_initialization(self):
        models, rng_states = [], []
        for flags in [{}, {'use_rem': True}, {'use_pacfg': True},
                      {'use_rem': True, 'use_pacfg': True}]:
            torch.manual_seed(117)
            models.append(HybridEncoder(**ENCODER_ARGS, pacfg_bottleneck=8, **flags).eval())
            rng_states.append(torch.get_rng_state())
        baseline, rem_only, pacfg_only, combined = models
        for state in rng_states[1:]:
            torch.testing.assert_close(state, rng_states[0], rtol=0, atol=0)
        for reference in [baseline, rem_only, pacfg_only]:
            for name, expected in reference.state_dict().items():
                torch.testing.assert_close(combined.state_dict()[name], expected, rtol=0, atol=0)
        features = [torch.randn(2, channels, side, side)
                    for channels, side in [(32, 8), (64, 4), (128, 2)]]
        with torch.no_grad():
            reference, initial = baseline(features), combined(features)
        for expected, actual in zip(reference, initial):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_serial_path_trains_both_adapters_and_keeps_all_three_output_scales(self):
        for amp in [False, True]:
            with self.subTest(cpu_autocast=amp):
                model = HybridEncoder(**ENCODER_ARGS, use_rem=True, use_pacfg=True,
                                      pacfg_bottleneck=8).eval()
                features = [torch.randn(2, channels, side, side, requires_grad=True)
                            for channels, side in [(32, 8), (64, 4), (128, 2)]]
                with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
                    initial = model(features)
                    initial_loss = sum(value.square().mean() for value in initial)
                initial_loss.backward()
                for adapter in [model.rem, model.pacfg]:
                    self.assertTrue(torch.isfinite(adapter.alpha.grad))
                    self.assertNotEqual(float(adapter.alpha.grad), 0.)
                model.zero_grad(set_to_none=True)
                for feature in features:
                    feature.grad = None
                with torch.no_grad():
                    model.rem.alpha.fill_(.2)
                    model.pacfg.alpha.fill_(.2)
                rem_outputs, pacfg_inputs = [], []
                handles = [
                    model.rem.register_forward_hook(lambda module, args, output: rem_outputs.append(output)),
                    model.pacfg.register_forward_pre_hook(lambda module, args: pacfg_inputs.append(args[0])),
                ]
                try:
                    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
                        outputs = model(features)
                        loss = sum(value.square().mean() for value in outputs)
                    self.assertEqual(len(rem_outputs), 1)
                    self.assertEqual(len(pacfg_inputs), 1)
                    self.assertIs(pacfg_inputs[0], rem_outputs[0])
                    self.assertEqual([tuple(value.shape) for value in outputs],
                                     [(2, 32, 8, 8), (2, 32, 4, 4), (2, 32, 2, 2)])
                    self.assertGreater(float((outputs[0] - initial[0]).detach().abs().max()), 0.)
                    loss.backward()
                finally:
                    for handle in handles:
                        handle.remove()
                for adapter in [model.rem, model.pacfg]:
                    for name, parameter in adapter.named_parameters():
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    gradient_sum = sum(float(parameter.grad.abs().sum())
                                       for name, parameter in adapter.named_parameters() if name != 'alpha')
                    self.assertGreater(gradient_sum, 0.)
                self.assertTrue(torch.isfinite(features[0].grad).all())
                self.assertGreater(float(features[0].grad.abs().sum()), 0.)
                owner = torch.nn.Module()
                owner.encoder = model
                metrics = _module_training_metrics(owner)
                self.assertAlmostEqual(metrics['rem_alpha'], .2, places=6)
                self.assertAlmostEqual(metrics['pacfg_alpha'], .2, places=6)

    def test_four_ablation_cells_share_the_frozen_58e_protocol(self):
        directory = ROOT / 'configs/deim_dfine'
        common = None
        outputs = set()
        for suffix, enabled in [('', set()), ('rem_', {'use_rem'}),
                                ('pacfg_', {'use_pacfg'}), ('rem_pacfg_', {'use_rem', 'use_pacfg'})]:
            config = load_config(str(directory / f'deim_hgnetv2_m_sar_aircraft_{suffix}58e_b28_old_server.yml'), {})
            encoder = config['HybridEncoder']
            actual = {flag for flag in ['use_rem', 'use_pacfg'] if encoder.pop(flag, False)}
            self.assertEqual(actual, enabled)
            if 'use_pacfg' in enabled:
                self.assertEqual(encoder.pop('pacfg_bottleneck'), 64)
            self.assertEqual(config['epoches'], 58)
            self.assertEqual(config['train_dataloader']['total_batch_size'], 28)
            self.assertTrue((ROOT / config.pop('experiment_protocol_document')).is_file())
            outputs.add(config.pop('output_dir'))
            config.pop('__include__', None)
            if common is None:
                common = config
            else:
                self.assertEqual(config, common)
        self.assertEqual(len(outputs), 4)


if __name__ == '__main__':
    unittest.main()
