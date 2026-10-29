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
                      {'use_rem': True, 'use_pacfg': True},
                      {'use_rem': True, 'use_pacfg': True, 'pacfg_feat_index': 1}]:
            torch.manual_seed(117)
            models.append(HybridEncoder(**ENCODER_ARGS, pacfg_bottleneck=8, **flags).eval())
            rng_states.append(torch.get_rng_state())
        baseline, rem_only, pacfg_only, combined, split = models
        for state in rng_states[1:]:
            torch.testing.assert_close(state, rng_states[0], rtol=0, atol=0)
        for adapted in [combined, split]:
            for reference in [baseline, rem_only, pacfg_only]:
                for name, expected in reference.state_dict().items():
                    torch.testing.assert_close(adapted.state_dict()[name], expected, rtol=0, atol=0)
        features = [torch.randn(2, channels, side, side)
                    for channels, side in [(32, 8), (64, 4), (128, 2)]]
        with torch.no_grad():
            reference = baseline(features)
            for adapted in [combined, split]:
                for expected, actual in zip(reference, adapted(features)):
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

    def test_split_route_feeds_independent_enhanced_levels_into_fpn_with_gradients(self):
        for amp in [False, True]:
            with self.subTest(cpu_autocast=amp):
                model = HybridEncoder(**ENCODER_ARGS, use_rem=True, use_pacfg=True,
                                      pacfg_bottleneck=8, pacfg_feat_index=1).eval()
                features = [torch.randn(2, channels, height, width, requires_grad=True)
                            for channels, height, width in [(32, 12, 20), (64, 6, 10), (128, 3, 5)]]
                with torch.no_grad():
                    model.rem.alpha.fill_(.2)
                    model.pacfg.alpha.fill_(.2)
                projections, rem_calls, pacfg_calls, fpn_inputs = {}, [], [], []
                handles = [
                    model.input_proj[0].register_forward_hook(
                        lambda module, args, output: projections.__setitem__(0, output)),
                    model.input_proj[1].register_forward_hook(
                        lambda module, args, output: projections.__setitem__(1, output)),
                    model.rem.register_forward_hook(
                        lambda module, args, output: rem_calls.append((args[0], output))),
                    model.pacfg.register_forward_hook(
                        lambda module, args, output: pacfg_calls.append((args[0], output))),
                ]
                handles += [block.register_forward_pre_hook(
                    lambda module, args: fpn_inputs.append(args[0])) for block in model.fpn_blocks]
                try:
                    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
                        outputs = model(features)
                        loss = sum(value.square().mean() for value in outputs)
                    self.assertEqual((len(rem_calls), len(pacfg_calls), len(fpn_inputs)), (1, 1, 2))
                    self.assertIs(rem_calls[0][0], projections[0])
                    self.assertIs(pacfg_calls[0][0], projections[1])
                    self.assertEqual(tuple(rem_calls[0][1].shape), (2, 32, 12, 20))
                    self.assertEqual(tuple(pacfg_calls[0][1].shape), (2, 32, 6, 10))
                    self.assertEqual(pacfg_calls[0][1].dtype, projections[1].dtype)
                    torch.testing.assert_close(fpn_inputs[0][:, 32:], pacfg_calls[0][1], rtol=0, atol=0)
                    torch.testing.assert_close(fpn_inputs[1][:, 32:], rem_calls[0][1], rtol=0, atol=0)
                    self.assertEqual([tuple(value.shape) for value in outputs],
                                     [(2, 32, 12, 20), (2, 32, 6, 10), (2, 32, 3, 5)])
                    loss.backward()
                finally:
                    for handle in handles:
                        handle.remove()
                for adapter in [model.rem, model.pacfg]:
                    for name, parameter in adapter.named_parameters():
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    self.assertGreater(sum(float(parameter.grad.abs().sum())
                                           for name, parameter in adapter.named_parameters() if name != 'alpha'), 0.)
                for feature in features[:2]:
                    self.assertTrue(torch.isfinite(feature.grad).all())
                    self.assertGreater(float(feature.grad.abs().sum()), 0.)

    def test_split_config_only_moves_pacfg_and_changes_run_paths(self):
        directory = ROOT / 'configs/deim_dfine'
        serial = load_config(str(directory / 'deim_hgnetv2_m_sar_aircraft_rem_pacfg_58e_b28_old_server.yml'), {})
        split = load_config(str(directory / 'deim_hgnetv2_m_sar_aircraft_rem_s8_pacfg_s16_58e_b28_old_server.yml'), {})
        self.assertEqual(serial['HybridEncoder'].get('pacfg_feat_index', 0), 0)
        self.assertEqual(split['HybridEncoder'].pop('pacfg_feat_index'), 1)
        self.assertEqual(split['HybridEncoder']['feat_strides'], [8, 16, 32])
        self.assertNotEqual(serial['output_dir'], split['output_dir'])
        for candidate in [serial, split]:
            self.assertTrue((ROOT / candidate.pop('experiment_protocol_document')).is_file())
            candidate.pop('__include__', None)
            candidate.pop('output_dir')
        self.assertEqual(serial, split)

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
