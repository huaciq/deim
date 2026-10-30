"""SPI morphology/CoV, residual learning and three-scale encoder integration."""

import copy
import io
import math
from pathlib import Path
import unittest

import torch
from torch import nn
import torch.nn.functional as F

from engine.core import YAMLConfig
from engine.core.workspace import create
from engine.core.yaml_utils import load_config
from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.dfine_decoder import DFINETransformer
from engine.deim.hybrid_encoder import HybridEncoder
from engine.deim.matcher import HungarianMatcher
from engine.deim.scattering_pattern_isolation import ScatteringPatternIsolation
from engine.deim.scattering_physics import SPIModule
from engine.optim.ema import ModelEMA
from engine.solver.det_engine import _module_training_metrics, _to_float32


ROOT = Path(__file__).resolve().parents[1]
ENCODER_ARGS = dict(in_channels=[32, 64, 128], hidden_dim=32, nhead=4,
                    dim_feedforward=64, depth_mult=.34)


def features():
    return [torch.randn(2, c, h, w) for c, h, w in [(32, 8, 12), (64, 4, 6), (128, 2, 3)]]


class SpiModuleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(347)

    def test_specified_full_width_architecture_and_channelwise_zero_residuals(self):
        module = SPIModule()
        self.assertEqual((module.channels, module.k, module.eps), (256, 3, 1e-5))
        for norm in [module.norm_peak, module.norm_cov]:
            self.assertIsInstance(norm, nn.GroupNorm)
            self.assertEqual((norm.num_groups, norm.num_channels), (32, 256))
        self.assertEqual(module.phys_gate[0].weight.shape, (256, 512, 1, 1))
        self.assertIsNone(module.phys_gate[0].bias)
        self.assertIsInstance(module.phys_gate[1], nn.BatchNorm2d)
        self.assertIsInstance(module.phys_gate[2], nn.Sigmoid)
        for branch, kernel in [(module.bridge_h, (1, 5)), (module.bridge_v, (5, 1))]:
            self.assertEqual(branch.kernel_size, kernel)
            self.assertEqual((branch.in_channels, branch.out_channels, branch.groups), (256, 256, 256))
            self.assertEqual(branch.dilation, (1, 1))
            self.assertIsNone(branch.bias)
        self.assertIsInstance(module.bridge_pw[0], nn.BatchNorm2d)
        self.assertIsInstance(module.bridge_pw[1], nn.SiLU)
        self.assertEqual(module.bridge_pw[2].weight.shape, (256, 256, 1, 1))
        for parameter in [module.alpha, module.beta]:
            self.assertEqual(parameter.shape, (1, 256, 1, 1))
            self.assertEqual(parameter.abs().sum().item(), 0.)
        self.assertEqual(sum(p.numel() for p in module.parameters()), 201728)

    def test_white_top_hat_selects_isolated_peak_and_removes_extended_plateau(self):
        module = SPIModule(32)
        isolated = torch.zeros(1, 32, 9, 11)
        isolated[:, :, 4, 5] = 9.
        peak, cov = module._physical_statistics(isolated)
        torch.testing.assert_close(peak, isolated, rtol=0, atol=0)
        expected = math.sqrt(8. + module.eps) / (1. + module.eps)
        torch.testing.assert_close(cov[:, :, 4, 5], torch.full((1, 32), expected))
        plateau = torch.zeros_like(isolated)
        plateau[:, :, 2:7, 3:8] = 9.
        extended, _ = module._physical_statistics(plateau)
        torch.testing.assert_close(extended, torch.zeros_like(extended), rtol=0, atol=0)

    def test_cov_uses_absolute_mean_and_original_zero_including_boundary_average(self):
        module = SPIModule(32)
        for sign in [1., -1.]:
            constant = torch.full((1, 32, 9, 11), sign * 2.)
            peak, cov = module._physical_statistics(constant)
            torch.testing.assert_close(peak, torch.zeros_like(peak), rtol=0, atol=0)
            center = math.sqrt(module.eps) / (2. + module.eps)
            corner = math.sqrt(16 / 9 - (8 / 9) ** 2 + module.eps) / (8 / 9 + module.eps)
            torch.testing.assert_close(cov[..., 4, 5], torch.full((1, 32), center))
            torch.testing.assert_close(cov[..., 0, 0], torch.full((1, 32), corner))
        _, zero_cov = module._physical_statistics(torch.zeros(1, 32, 9, 11))
        torch.testing.assert_close(zero_cov, torch.full_like(zero_cov, math.sqrt(module.eps) / module.eps))

    def test_both_residual_terms_match_channelwise_formula_and_bridge_orthogonal_neighbors(self):
        module = SPIModule(32).eval()
        with torch.no_grad():
            module.phys_gate[0].weight.zero_()  # A_phys = sigmoid(0) = .5.
            module.bridge_h.weight.fill_(.2)
            module.bridge_v.weight.fill_(.2)
            module.bridge_pw[0].running_var.fill_(1. - module.bridge_pw[0].eps)
            module.bridge_pw[2].weight.copy_(torch.eye(32).reshape(32, 32, 1, 1))
            module.alpha.copy_(torch.linspace(-.2, .3, 32).reshape(1, 32, 1, 1))
            module.beta.copy_(torch.linspace(.1, .4, 32).reshape(1, 32, 1, 1))
        x = torch.zeros(1, 32, 9, 11)
        x[:, :, 4, 5] = 1.
        salient = x * .5
        horizontal = F.pad(salient, (2, 2, 0, 0))
        vertical = F.pad(salient, (0, 0, 2, 2))
        connected = sum(horizontal[..., offset:offset + 11] for offset in range(5)) * .2
        connected += sum(vertical[..., offset:offset + 9, :] for offset in range(5)) * .2
        expected = x + module.alpha * salient + module.beta * F.silu(connected)
        actual = module(x)
        torch.testing.assert_close(actual, expected)
        self.assertGreater(actual[0, 0, 4, 3].item(), 0.)
        self.assertGreater(actual[0, 0, 2, 5].item(), 0.)
        self.assertEqual(actual[0, 0, 2, 3].item(), 0.)

    def test_identity_and_learning_work_in_fp32_and_amp_without_changing_feature_dtype(self):
        for amp, dtype in [(False, torch.float32), (True, torch.float32),
                           (True, torch.bfloat16), (True, torch.float16)]:
            with self.subTest(amp=amp, dtype=dtype):
                module = SPIModule(32).train()
                x = torch.randn(2, 32, 9, 11, dtype=dtype, requires_grad=True)
                optimizer = torch.optim.SGD(module.parameters(), lr=.1)
                for step in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    x.grad = None
                    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
                        actual = module(x)
                    self.assertEqual(actual.dtype, dtype)
                    self.assertEqual(actual.shape, x.shape)
                    if not step:
                        torch.testing.assert_close(actual, x, rtol=0, atol=0)
                    actual.float().square().mean().backward()
                    for name, parameter in module.named_parameters():
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    for residual in [module.alpha, module.beta]:
                        self.assertGreater(residual.grad.abs().sum().item(), 0.)
                    if step:
                        for branch in [module.norm_peak, module.norm_cov, module.phys_gate,
                                       module.bridge_h, module.bridge_v, module.bridge_pw]:
                            self.assertGreater(sum(p.grad.abs().sum().item() for p in branch.parameters()), 0.)
                    self.assertTrue(torch.isfinite(x.grad).all())
                    optimizer.step()

    def test_fp32_statistics_avoid_half_square_overflow_and_handle_zero_features(self):
        module = SPIModule(32).train()
        large = (torch.randn(2, 32, 9, 11) * 1000.).half().requires_grad_()
        self.assertTrue(torch.isinf(large.square()).any())
        for x in [large, torch.zeros_like(large, requires_grad=True)]:
            records = []
            handle = module.norm_cov.register_forward_pre_hook(lambda layer, args: records.append(args[0]))
            try:
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    actual = module(x)
            finally:
                handle.remove()
            self.assertEqual(records[0].dtype, torch.float32)
            self.assertTrue(torch.isfinite(records[0]).all())
            torch.testing.assert_close(actual, x, rtol=0, atol=0)
            actual.float().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_encoder_keeps_common_initialization_rng_outputs_and_old_spi_option(self):
        torch.manual_seed(349)
        baseline = HybridEncoder(**copy.deepcopy(ENCODER_ARGS))
        expected_rng = torch.get_rng_state()
        torch.manual_seed(349)
        adapted = HybridEncoder(**copy.deepcopy(ENCODER_ARGS), use_spi_module=True)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(adapted.state_dict()[name], value, rtol=0, atol=0)
        missing, unexpected = adapted.load_state_dict(baseline.state_dict(), strict=False)
        self.assertTrue(missing)
        self.assertTrue(all(name.startswith('spi_modules.') for name in missing))
        self.assertFalse(unexpected)
        self.assertIsNone(adapted.spi)
        self.assertEqual(len(adapted.spi_modules), 3)
        self.assertEqual(len({id(block) for block in adapted.spi_modules}), 3)
        x = features()
        for training in [True, False]:
            baseline.train(training)
            adapted.train(training)
            expected, actual = baseline(x), adapted(x)
            self.assertIsInstance(actual, list)
            for new, old in zip(actual, expected):
                torch.testing.assert_close(new, old, rtol=0, atol=0)
        legacy = HybridEncoder(**copy.deepcopy(ENCODER_ARGS), use_spi=True)
        self.assertIsInstance(legacy.spi, ScatteringPatternIsolation)
        self.assertEqual(len(legacy.spi_modules), 0)

    def test_all_projected_levels_run_before_transformer_and_reach_fpn(self):
        model = HybridEncoder(**copy.deepcopy(ENCODER_ARGS), use_spi_module=True).eval()
        projections, reconstructed, transformer_inputs, fpn_inputs, order, handles = {}, {}, [], [], [], []
        for index in range(3):
            model.spi_modules[index].alpha.data.fill_(.1)
            model.spi_modules[index].beta.data.fill_(.2)
            handles.append(model.input_proj[index].register_forward_hook(
                lambda layer, args, result, index=index: projections.__setitem__(index, result)))
            def capture(layer, args, result, index=index):
                self.assertIs(args[0], projections[index])
                reconstructed[index] = result
                order.append(index)
            handles.append(model.spi_modules[index].register_forward_hook(capture))
        def encoder_input(layer, args):
            transformer_inputs.append(args[0])
            order.append('transformer')
        handles.append(model.encoder[0].register_forward_pre_hook(encoder_input))
        handles.extend(block.register_forward_pre_hook(lambda layer, args: fpn_inputs.append(args[0])) for block in model.fpn_blocks)
        try:
            result = model(features())
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(order, [0, 1, 2, 'transformer'])
        torch.testing.assert_close(transformer_inputs[0], reconstructed[2].flatten(2).permute(0, 2, 1), rtol=0, atol=0)
        torch.testing.assert_close(fpn_inputs[0][:, 32:], reconstructed[1], rtol=0, atol=0)
        torch.testing.assert_close(fpn_inputs[1][:, 32:], reconstructed[0], rtol=0, atol=0)
        self.assertEqual([tuple(value.shape) for value in result], [(2, 32, 8, 12), (2, 32, 4, 6), (2, 32, 2, 3)])

    def test_real_detection_losses_train_all_scales_with_lite_wfdr_and_amp(self):
        for amp in [False, True]:
            with self.subTest(amp=amp):
                owner = nn.Module()
                owner.encoder = HybridEncoder(**copy.deepcopy(ENCODER_ARGS), use_spi_module=True)
                owner.decoder = DFINETransformer(num_classes=7, hidden_dim=32, feat_channels=[32] * 3,
                                                 num_queries=5, num_layers=3, eval_idx=1, nhead=4, num_points=2,
                                                 dim_feedforward=64, reg_max=32, num_denoising=4,
                                                 eval_spatial_size=None, use_clcr_lite=True)
                loss_fn = DEIMCriterion(
                    HungarianMatcher(dict(cost_class=2, cost_bbox=5, cost_giou=2), use_focal_loss=True),
                    dict(loss_mal=1, loss_bbox=5, loss_giou=2, loss_fgl=.15, loss_ddf=1.5, loss_wfdr=.15),
                    ['mal', 'boxes', 'local'], num_classes=7, reg_max=32, use_wfdr=True)
                groups = YAMLConfig.get_optim_params(dict(type='AdamW', params=[dict(params='.*norm.*', weight_decay=0)]), owner)
                optimizer = torch.optim.AdamW(groups, lr=1e-3)
                ids = [id(p) for group in groups for p in group['params']]
                self.assertTrue(all(ids.count(id(p)) == 1 for p in owner.encoder.spi_modules.parameters()))
                x = features()
                targets = [dict(labels=torch.tensor([3]), boxes=torch.tensor([[.4, .5, .3, .4]])),
                           dict(labels=torch.tensor([1, 6]), boxes=torch.tensor([[.3, .4, .2, .3], [.7, .6, .3, .2]]))]
                for step in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
                        output = owner.decoder(owner.encoder(x), targets)
                    loss = loss_fn(_to_float32(output), targets)
                    self.assertTrue(all(torch.isfinite(value) for value in loss.values()))
                    self.assertIn('loss_wfdr_dn_0', loss)
                    sum(loss.values()).backward()
                    for adapter in owner.encoder.spi_modules:
                        for name, parameter in adapter.named_parameters():
                            self.assertIsNotNone(parameter.grad, name)
                            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                        self.assertGreater(adapter.alpha.grad.abs().sum().item(), 0.)
                        self.assertGreater(adapter.beta.grad.abs().sum().item(), 0.)
                        if step:
                            for branch in [adapter.norm_peak, adapter.norm_cov, adapter.phys_gate,
                                           adapter.bridge_h, adapter.bridge_v, adapter.bridge_pw]:
                                self.assertGreater(sum(p.grad.abs().sum().item() for p in branch.parameters()), 0.)
                    optimizer.step()

    def test_checkpoint_ema_and_deployment_preserve_active_multiscale_spi(self):
        model = HybridEncoder(**copy.deepcopy(ENCODER_ARGS), use_spi_module=True).eval()
        for index, adapter in enumerate(model.spi_modules):
            adapter.alpha.data.fill_(.1 * (index + 1))
            adapter.beta.data.fill_(.2)
        ema = ModelEMA(model, decay=.5, warmups=0)
        for adapter in model.spi_modules:
            adapter.alpha.data.add_(.1)
        ema.update(model)
        for index, adapter in enumerate(ema.module.spi_modules):
            torch.testing.assert_close(adapter.alpha, torch.full_like(adapter.alpha, .1 * (index + 1) + .05))
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        restored = HybridEncoder(**copy.deepcopy(ENCODER_ARGS), use_spi_module=True).eval()
        restored.load_state_dict(torch.load(stream, weights_only=True))
        x = features()
        with torch.no_grad():
            expected, actual = model(x), restored(x)
        for new, old in zip(actual, expected):
            torch.testing.assert_close(new, old, rtol=0, atol=0)
        for layer in restored.modules():
            if hasattr(layer, 'convert_to_deploy'):
                layer.convert_to_deploy()
        self.assertEqual(len(restored.spi_modules), 3)
        with torch.no_grad():
            deployed = restored(x)
        for new, old in zip(deployed, expected):
            torch.testing.assert_close(new, old, rtol=1e-4, atol=1e-5)

    def test_metrics_track_both_channel_scales_at_each_pyramid_level(self):
        owner = nn.Module()
        owner.encoder = HybridEncoder(**copy.deepcopy(ENCODER_ARGS), use_spi_module=True)
        for index, adapter in enumerate(owner.encoder.spi_modules):
            adapter.alpha.data.fill_(-.1 * (index + 1))
            adapter.beta.data.fill_(.2 * (index + 1))
        actual = _module_training_metrics(owner)
        self.assertEqual(len(actual), 6)
        for index in range(3):
            self.assertAlmostEqual(actual[f'spi_module_alpha_p{index+3}_mean_abs'], .1 * (index + 1), places=6)
            self.assertAlmostEqual(actual[f'spi_module_beta_p{index+3}_mean_abs'], .2 * (index + 1), places=6)

    def test_configs_enable_three_full_width_modules_and_preserve_reference_protocol(self):
        folder = ROOT / 'configs/deim_dfine'
        outputs = set()
        for suffix, parent in [('spi_module', '58e_b28'), ('spi_module_wfdr', 'wfdr_58e_b28')]:
            path = folder / f'deim_hgnetv2_m_sar_aircraft_{suffix}_58e_b28_old_server.yml'
            config = load_config(str(path), {})
            reference = load_config(str(folder / f'deim_hgnetv2_m_sar_aircraft_{parent}_old_server.yml'), {})
            self.assertTrue(config['HybridEncoder'].pop('use_spi_module'))
            self.assertFalse(config['HybridEncoder'].get('use_spi', False))
            actual = create('HybridEncoder', YAMLConfig(str(path)).global_cfg)
            self.assertEqual(len(actual.spi_modules), 3)
            self.assertEqual(actual.feat_strides, [8, 16, 32])
            self.assertTrue(all(adapter.channels == 256 for adapter in actual.spi_modules))
            self.assertEqual(sum(p.numel() for p in actual.spi_modules.parameters()), 605184)
            self.assertTrue((ROOT / config['experiment_protocol_document']).is_file())
            self.assertNotIn(config['output_dir'], outputs)
            outputs.add(config['output_dir'])
            for candidate in [config, reference]:
                candidate.pop('__include__', None)
                candidate.pop('output_dir')
                candidate.pop('experiment_protocol_document')
            self.assertEqual(config, reference)


if __name__ == '__main__':
    unittest.main()
