"""REM basis/strip controls: actual transforms, matched budgets and detector gradients."""

import copy
from pathlib import Path
import unittest

import torch
from torch import nn

from engine.core import YAMLConfig
from engine.core.workspace import create
from engine.core.yaml_utils import load_config
from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.dfine_decoder import DFINETransformer
from engine.deim.hybrid_encoder import HybridEncoder
from engine.deim.matcher import HungarianMatcher
from engine.deim.radar_enhancement import (
    DepthToSpace2D, HaarDWT2D, HaarIDWT2D, RadarEnhancementModule, SpaceToDepth2D,
)
from engine.optim.ema import ModelEMA
from engine.solver.det_engine import _module_training_metrics, _to_float32


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('space_to_depth', 'haar_square')
ENCODER_ARGS = dict(in_channels=[32, 64, 128], hidden_dim=32, nhead=8,
                    dim_feedforward=64, depth_mult=.34)


class Gain(nn.Module):
    def __init__(self, gain):
        super().__init__()
        self.gain = gain

    def forward(self, value):
        return value * self.gain


class RemMechanismTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(443)

    def test_spatial_phases_are_raw_pixels_with_exact_even_odd_roundtrip_and_gradients(self):
        x = torch.tensor([[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]])
        phases = SpaceToDepth2D()(x)
        for phase, expected in zip(phases, ([1., 5.], [2., 6.], [3., 7.], [4., 8.])):
            torch.testing.assert_close(phase.flatten(), torch.tensor(expected), rtol=0, atol=0)
        self.assertFalse(torch.equal(phases[0], HaarDWT2D()(x)[0]))
        self.assertEqual(list(SpaceToDepth2D().parameters()), [])
        for shape in ((2, 3, 8, 10), (2, 3, 7, 9), (1, 2, 1, 3)):
            with self.subTest(shape=shape):
                features = torch.randn(shape, requires_grad=True)
                restored = DepthToSpace2D()(SpaceToDepth2D()(features), shape[-2:])
                torch.testing.assert_close(restored, features, rtol=0, atol=0)
                restored.sum().backward()
                torch.testing.assert_close(features.grad, torch.ones_like(features), rtol=0, atol=0)

    def test_spatial_control_routes_four_phase_branches_back_to_their_original_positions(self):
        module = RadarEnhancementModule(1, variant='space_to_depth')
        for index, name in enumerate(('ll', 'lh', 'hl', 'hh'), 1):
            setattr(module, name, Gain(index))
        module.alpha.data.fill_(1.)
        features = torch.tensor([[[[1., 2.], [3., 4.]]]])
        expected = features + torch.tensor([[[[1., 4.], [9., 16.]]]])
        torch.testing.assert_close(module(features), expected, rtol=0, atol=0)
        self.assertFalse(any(isinstance(layer, (HaarDWT2D, HaarIDWT2D)) for layer in module.modules()))

    def test_square_control_matches_depth_branch_rf_and_nearly_matches_capacity(self):
        reference = RadarEnhancementModule(256)
        spatial = RadarEnhancementModule(256, variant='space_to_depth')
        square = RadarEnhancementModule(256, variant='haar_square')
        counts = [sum(p.numel() for p in module.parameters()) for module in (reference, spatial, square)]
        self.assertEqual(counts, [2823681, 2823681, 2825321])
        self.assertLess(abs(counts[2] / counts[0] - 1), .001)
        conv_counts = [sum(layer.weight.numel() for layer in module.modules() if isinstance(layer, nn.Conv2d))
                       for module in (reference, spatial, square)]
        self.assertEqual(conv_counts, [2818048, 2818048, 2820096])
        for module in (reference, square):
            self.assertEqual(sum(isinstance(layer, nn.Conv2d) for layer in module.modules()), 11)
            self.assertEqual(sum(isinstance(layer, nn.BatchNorm2d) for layer in module.modules()), 11)
            for branch in (module.lh, module.hl):
                kernels = [layer.kernel_size for layer in branch.modules() if isinstance(layer, nn.Conv2d)]
                self.assertEqual(tuple(1 + sum(k[axis] - 1 for k in kernels) for axis in (0, 1)), (5, 5))
        for branch in (square.lh, square.hl):
            self.assertEqual((branch[1][0].in_channels, branch[1][0].out_channels), (256, 154))
            self.assertEqual((branch[2][0].in_channels, branch[2][0].out_channels), (154, 256))
            self.assertEqual([block[0].kernel_size for block in branch], [(1, 1), (3, 3), (1, 1), (3, 3)])

    def test_common_branch_initialization_and_rng_match_the_reported_rem(self):
        torch.manual_seed(449)
        reference = RadarEnhancementModule(32)
        expected_rng = torch.get_rng_state()
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                torch.manual_seed(449)
                module = RadarEnhancementModule(32, variant=variant)
                torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
                for name, tensor in reference.state_dict().items():
                    if variant == 'space_to_depth' and name.startswith(('dwt.', 'idwt.')):
                        continue
                    if variant == 'haar_square' and name.startswith(('lh.1.', 'lh.2.', 'hl.1.', 'hl.2.')):
                        continue
                    torch.testing.assert_close(module.state_dict()[name], tensor, rtol=0, atol=0)

    def test_initial_identity_and_two_step_branch_learning_in_fp32_and_bf16_amp(self):
        for variant in VARIANTS:
            for amp in (False, True):
                with self.subTest(variant=variant, amp=amp):
                    module = RadarEnhancementModule(8, variant=variant)
                    features = torch.randn(2, 8, 9, 11, requires_grad=True)
                    optimizer = torch.optim.SGD(module.parameters(), lr=.1)
                    for step in range(2):
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
                            output = module(features)
                        self.assertEqual(output.dtype, features.dtype)
                        self.assertEqual(output.shape, features.shape)
                        if step == 0:
                            torch.testing.assert_close(output, features, rtol=0, atol=0)
                        output.square().mean().backward()
                        self.assertTrue(torch.isfinite(module.alpha.grad))
                        self.assertNotEqual(module.alpha.grad.item(), 0.)
                        for name in ('ll', 'lh', 'hl', 'hh'):
                            gradients = [p.grad for p in getattr(module, name).parameters()]
                            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
                            if step:
                                self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0.)
                            else:
                                self.assertEqual(sum(g.abs().sum().item() for g in gradients), 0.)
                        optimizer.step()

    def test_encoder_keeps_common_initialization_rng_and_stride8_insertion(self):
        torch.manual_seed(457)
        baseline = HybridEncoder(**ENCODER_ARGS).eval()
        expected_rng = torch.get_rng_state()
        features = [torch.randn(2, c, h, w) for c, h, w in ((32, 8, 12), (64, 4, 6), (128, 2, 3))]
        with torch.no_grad():
            expected = baseline(features)
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                torch.manual_seed(457)
                adapted = HybridEncoder(**ENCODER_ARGS, use_rem=True, rem_variant=variant).eval()
                torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
                for name, tensor in baseline.state_dict().items():
                    torch.testing.assert_close(adapted.state_dict()[name], tensor, rtol=0, atol=0)
                calls = []
                handle = adapted.rem.register_forward_pre_hook(lambda module, args: calls.append(args[0].shape))
                with torch.no_grad():
                    actual = adapted(features)
                handle.remove()
                self.assertEqual(calls, [torch.Size([2, 32, 8, 12])])
                for result, reference in zip(actual, expected):
                    torch.testing.assert_close(result, reference, rtol=0, atol=0)
                adapted.rem.alpha.data.fill_(.2)
                owner = nn.Module()
                owner.encoder = adapted
                self.assertAlmostEqual(_module_training_metrics(owner)['rem_alpha'], .2, places=6)

    def test_actual_four_layer_detector_trains_controls_through_aux_and_dn_with_original_losses(self):
        for variant, amp in (('space_to_depth', False), ('haar_square', True)):
            with self.subTest(variant=variant, amp=amp):
                encoder = HybridEncoder(**ENCODER_ARGS, use_rem=True, rem_variant=variant)
                decoder = DFINETransformer(num_classes=7, hidden_dim=32, feat_channels=[32] * 3,
                    num_queries=5, num_layers=4, eval_idx=-1, nhead=8, num_points=[3, 6, 3],
                    dim_feedforward=64, reg_max=32, num_denoising=4, eval_spatial_size=None)
                criterion = DEIMCriterion(
                    HungarianMatcher(dict(cost_class=2, cost_bbox=5, cost_giou=2), use_focal_loss=True),
                    dict(loss_mal=1, loss_bbox=5, loss_giou=2, loss_fgl=.15, loss_ddf=1.5),
                    ['mal', 'boxes', 'local'], num_classes=7, reg_max=32)
                features = [torch.randn(2, c, h, w) for c, h, w in ((32, 8, 12), (64, 4, 6), (128, 2, 3))]
                targets = [dict(labels=torch.tensor([3]), boxes=torch.tensor([[.4, .5, .3, .4]])),
                    dict(labels=torch.tensor([1, 6]), boxes=torch.tensor([[.3, .4, .2, .3], [.7, .6, .3, .2]]))]
                optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
                for step in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
                        output = decoder(encoder(features), targets)
                    losses = criterion(_to_float32(output), targets)
                    self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
                    self.assertFalse(any('wfdr' in name for name in losses))
                    self.assertIn('loss_fgl_aux_2', losses)
                    self.assertIn('loss_mal_dn_3', losses)
                    # Zero-initialized FDR heads have identical teacher/student
                    # distributions; original D-FINE skips DDF on that step.
                    if step:
                        self.assertIn('loss_ddf_aux_0', losses)
                    else:
                        self.assertNotIn('loss_ddf_aux_0', losses)
                    sum(losses.values()).backward()
                    self.assertTrue(torch.isfinite(encoder.rem.alpha.grad))
                    self.assertNotEqual(encoder.rem.alpha.grad.item(), 0.)
                    self.assertGreater(decoder.dec_bbox_head[-1].layers[-1].weight.grad.abs().sum().item(), 0.)
                    if step:
                        for branch in (encoder.rem.ll, encoder.rem.lh, encoder.rem.hl, encoder.rem.hh):
                            self.assertGreater(sum(p.grad.abs().sum().item() for p in branch.parameters()), 0.)
                    optimizer.step()

    def test_checkpoint_and_ema_restore_each_controls_transform_and_learned_residual(self):
        features = [torch.randn(2, c, h, w) for c, h, w in ((32, 8, 12), (64, 4, 6), (128, 2, 3))]
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                encoder = HybridEncoder(**ENCODER_ARGS, use_rem=True, rem_variant=variant).eval()
                ema = ModelEMA(encoder, decay=.75, warmups=0)
                encoder.rem.alpha.data.fill_(.4)
                ema.update(encoder)
                self.assertAlmostEqual(ema.module.rem.alpha.item(), .1, places=6)
                restored = HybridEncoder(**ENCODER_ARGS, use_rem=True, rem_variant=variant).eval()
                restored.load_state_dict(ema.state_dict()['module'], strict=True)
                for name, tensor in ema.module.state_dict().items():
                    torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)
                with torch.no_grad():
                    for result, expected in zip(restored(features), ema.module(features)):
                        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-7)
                self.assertEqual(restored.rem.variant, variant)

    def test_factory_configs_change_only_the_requested_variant_and_independent_outputs(self):
        folder = ROOT / 'configs/deim_dfine'
        reference = load_config(str(folder / 'deim_hgnetv2_m_sar_aircraft_rem_58e_b28_old_server.yml'), {})
        outputs = {reference['output_dir']}
        for suffix, variant in (('rem_space_to_depth', 'space_to_depth'), ('rem_square', 'haar_square')):
            with self.subTest(variant=variant):
                path = folder / f'deim_hgnetv2_m_sar_aircraft_{suffix}_58e_b28_old_server.yml'
                config = load_config(str(path), {})
                actual = create('HybridEncoder', YAMLConfig(str(path)).global_cfg)
                self.assertEqual(actual.rem.variant, variant)
                self.assertEqual(actual.rem.lh[0][0].in_channels, 256)
                self.assertEqual(config['HybridEncoder'].pop('rem_variant'), variant)
                self.assertFalse(config['DEIMCriterion'].get('use_wfdr', False))
                self.assertTrue((ROOT / config['experiment_protocol_document']).is_file())
                self.assertNotIn(config['output_dir'], outputs)
                outputs.add(config['output_dir'])
                expected = copy.deepcopy(reference)
                for item in (config, expected):
                    for key in ('__include__', 'output_dir', 'experiment_protocol_document'):
                        item.pop(key, None)
                self.assertEqual(config, expected)

    def test_invalid_variant_is_rejected_and_legacy_default_rem_remains_explicit(self):
        self.assertEqual(RadarEnhancementModule(8).variant, 'haar_directional')
        with self.assertRaisesRegex(ValueError, 'variant'):
            RadarEnhancementModule(8, variant='invalid')
        with self.assertRaisesRegex(ValueError, 'variant'):
            HybridEncoder(**ENCODER_ARGS, use_rem=True, rem_variant='invalid')


if __name__ == '__main__':
    unittest.main()
