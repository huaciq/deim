"""REM point-strip: compact support, axial routing, learning and frozen protocol."""

import copy
import io
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
from engine.deim.radar_enhancement import HaarDWT2D, HaarIDWT2D, PointStripDualPath, RadarEnhancementModule
from engine.optim.ema import ModelEMA
from engine.solver.det_engine import _module_training_metrics, _to_float32


ROOT = Path(__file__).resolve().parents[1]
ENCODER_ARGS = dict(in_channels=[32, 64, 128], hidden_dim=32, nhead=8,
                    dim_feedforward=64, depth_mult=.34)
CONFIG = ROOT / 'configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_rem_point_strip_58e_b28_old_server.yml'


class PointStripTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(733)

    def test_impulse_has_only_the_requested_axial_support_and_a_direct_point_shortcut(self):
        module = RadarEnhancementModule(1, act=None, variant='point_strip').eval()
        impulse = torch.zeros(1, 1, 11, 11)
        impulse[..., 5, 5] = 1.
        for name, expected_kernel in (('lh', (1, 3)), ('hl', (3, 1))):
            branch = getattr(module, name)
            self.assertIsInstance(branch, PointStripDualPath)
            self.assertEqual([layer.kernel_size for layer in branch.modules() if isinstance(layer, nn.Conv2d)],
                             [(1, 1), expected_kernel])
            with torch.no_grad():
                for layer in branch.modules():
                    if isinstance(layer, nn.Conv2d):
                        layer.weight.fill_(1.)
                    elif isinstance(layer, nn.BatchNorm2d):
                        layer.weight.fill_((1 + layer.eps) ** .5)
                        layer.bias.zero_()
                branch.beta.fill_(1.)
                actual = branch(impulse)
                expected = torch.zeros_like(impulse)
                if name == 'lh':
                    expected[..., 5, 4:7] = torch.tensor([1., 2., 1.])
                else:
                    expected[..., 4:7, 5] = torch.tensor([1., 2., 1.])
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
                branch.beta.zero_()
                torch.testing.assert_close(branch(impulse), impulse, rtol=1e-6, atol=1e-6)

    def test_reuses_retained_weights_preserves_rng_and_reduces_actual_parameters_and_macs(self):
        torch.manual_seed(739)
        reference = RadarEnhancementModule(256)
        expected_rng = torch.get_rng_state()
        torch.manual_seed(739)
        adapted = RadarEnhancementModule(256, variant='point_strip')
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        for name in ('dwt', 'idwt', 'll', 'hh'):
            for key, value in getattr(reference, name).state_dict().items():
                torch.testing.assert_close(getattr(adapted, name).state_dict()[key], value, rtol=0, atol=0)
        for name in ('lh', 'hl'):
            for old_index, new_name in ((0, 'point'), (1, 'strip')):
                for key, value in getattr(reference, name)[old_index].state_dict().items():
                    torch.testing.assert_close(getattr(getattr(adapted, name), new_name).state_dict()[key], value, rtol=0, atol=0)
            self.assertEqual(getattr(adapted, name).beta.item(), 1.)
            self.assertTrue(getattr(adapted, name).beta.requires_grad)
        self.assertEqual(sum(p.numel() for p in reference.parameters()), 2823681)
        self.assertEqual(sum(p.numel() for p in adapted.parameters()), 1248771)
        self.assertEqual(sum(isinstance(layer, nn.Conv2d) for layer in adapted.modules()), 7)
        self.assertEqual(sum(layer.weight.numel() for layer in adapted.modules() if isinstance(layer, nn.Conv2d)), 1245184)
        self.assertAlmostEqual(1245184 * 40 * 40 / 1e9, 1.9922944)
        self.assertEqual(sum(isinstance(layer, HaarDWT2D) for layer in adapted.modules()), 1)
        self.assertEqual(sum(isinstance(layer, HaarIDWT2D) for layer in adapted.modules()), 1)

    def test_one_analysis_synthesis_pair_preserves_odd_shape_and_identity_initialization(self):
        for shape in ((2, 8, 8, 10), (2, 8, 9, 11), (1, 8, 1, 3)):
            with self.subTest(shape=shape):
                module = RadarEnhancementModule(8, variant='point_strip').eval()
                features = torch.randn(shape)
                calls = []
                handles = [layer.register_forward_hook(lambda m, a, o, name=name: calls.append(name))
                           for name, layer in (('dwt', module.dwt), ('idwt', module.idwt))]
                with torch.no_grad():
                    output = module(features)
                self.assertEqual(calls, ['dwt', 'idwt'])
                self.assertEqual(output.dtype, features.dtype)
                torch.testing.assert_close(output, features, rtol=0, atol=0)
                calls.clear()
                with torch.no_grad():
                    module.alpha.fill_(.2)
                    output = module(features)
                self.assertEqual(calls, ['dwt', 'idwt'])
                self.assertEqual(output.shape, features.shape)
                self.assertTrue(torch.isfinite(output).all())
                self.assertGreater((output - features).abs().sum().item(), 0.)
                for handle in handles:
                    handle.remove()

    def test_both_paths_and_betas_learn_after_zero_alpha_step_in_fp32_and_bf16(self):
        for amp in (False, True):
            with self.subTest(amp=amp):
                module = RadarEnhancementModule(8, variant='point_strip')
                optimizer = torch.optim.SGD(module.parameters(), lr=.1)
                features = torch.randn(2, 8, 9, 11)
                for step in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
                        output = module(features)
                        loss = output.square().mean()
                    self.assertEqual(output.dtype, features.dtype)
                    if not step:
                        torch.testing.assert_close(output, features, rtol=0, atol=0)
                    loss.backward()
                    self.assertTrue(torch.isfinite(module.alpha.grad))
                    self.assertNotEqual(module.alpha.grad.item(), 0.)
                    for name in ('lh', 'hl'):
                        branch = getattr(module, name)
                        for path in (branch.point, branch.strip):
                            gradients = [p.grad for p in path.parameters()]
                            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
                            if step:
                                self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0.)
                        self.assertTrue(torch.isfinite(branch.beta.grad))
                        self.assertEqual(branch.beta.grad.dtype, torch.float32)
                        if step:
                            self.assertNotEqual(branch.beta.grad.item(), 0.)
                        else:
                            self.assertEqual(branch.beta.grad.item(), 0.)
                    optimizer.step()

    def test_encoder_routes_stride8_and_logs_all_three_scales_without_altering_common_initialization(self):
        torch.manual_seed(743)
        baseline = HybridEncoder(**ENCODER_ARGS).eval()
        expected_rng = torch.get_rng_state()
        torch.manual_seed(743)
        encoder = HybridEncoder(**ENCODER_ARGS, use_rem=True, rem_variant='point_strip').eval()
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(encoder.state_dict()[key], value, rtol=0, atol=0)
        features = [torch.randn(2, c, h, w) for c, h, w in ((32, 8, 12), (64, 4, 6), (128, 2, 3))]
        calls = []
        handle = encoder.rem.register_forward_pre_hook(lambda m, a: calls.append(tuple(a[0].shape)))
        with torch.no_grad():
            actual, expected = encoder(features), baseline(features)
        handle.remove()
        self.assertEqual(calls, [(2, 32, 8, 12)])
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        owner = nn.Module()
        owner.encoder = encoder
        self.assertEqual(_module_training_metrics(owner), dict(rem_alpha=0., rem_beta_lh=1., rem_beta_hl=1.))

    def test_real_four_layer_decoder_and_original_mal_fgl_ddf_losses_train_new_branches(self):
        encoder = HybridEncoder(**ENCODER_ARGS, use_rem=True, rem_variant='point_strip')
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
            with torch.autocast('cpu', dtype=torch.bfloat16):
                output = decoder(encoder(features), targets)
            losses = criterion(_to_float32(output), targets)
            self.assertTrue(all(torch.isfinite(loss) for loss in losses.values()))
            self.assertIn('loss_mal_dn_3', losses)
            self.assertIn('loss_fgl_aux_2', losses)
            self.assertEqual('loss_ddf_aux_0' in losses, bool(step))
            self.assertFalse(any(any(name in key for name in ('wfdr', 'margin', 'asym')) for key in losses))
            sum(losses.values()).backward()
            self.assertTrue(torch.isfinite(encoder.rem.alpha.grad))
            if step:
                for name in ('lh', 'hl'):
                    branch = getattr(encoder.rem, name)
                    self.assertNotEqual(branch.beta.grad.item(), 0.)
                    for path in (branch.point, branch.strip):
                        self.assertGreater(sum(p.grad.abs().sum().item() for p in path.parameters()), 0.)
            optimizer.step()

    def test_checkpoint_and_ema_restore_point_strip_weights_and_betas(self):
        encoder = HybridEncoder(**ENCODER_ARGS, use_rem=True, rem_variant='point_strip').eval()
        ema = ModelEMA(encoder, decay=.75, warmups=0)
        with torch.no_grad():
            encoder.rem.alpha.fill_(.4)
            encoder.rem.lh.beta.fill_(.2)
            encoder.rem.hl.beta.fill_(.6)
        ema.update(encoder)
        self.assertAlmostEqual(ema.module.rem.alpha.item(), .1, places=6)
        self.assertAlmostEqual(ema.module.rem.lh.beta.item(), .8, places=6)
        self.assertAlmostEqual(ema.module.rem.hl.beta.item(), .9, places=6)
        stream = io.BytesIO()
        torch.save(ema.state_dict(), stream)
        stream.seek(0)
        restored = HybridEncoder(**ENCODER_ARGS, use_rem=True, rem_variant='point_strip').eval()
        restored.load_state_dict(torch.load(stream, map_location='cpu', weights_only=False)['module'], strict=True)
        features = [torch.randn(2, c, h, w) for c, h, w in ((32, 8, 12), (64, 4, 6), (128, 2, 3))]
        with torch.no_grad():
            for actual, expected in zip(restored(features), ema.module(features)):
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    def test_factory_and_full_resolved_protocol_change_only_requested_variant_and_output(self):
        reference = load_config(str(ROOT / 'configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_rem_58e_b28_old_server.yml'), {})
        config = load_config(str(CONFIG), {})
        encoder = create('HybridEncoder', YAMLConfig(str(CONFIG)).global_cfg)
        self.assertEqual(encoder.rem.variant, 'point_strip')
        self.assertEqual(encoder.rem.residual_mode, 'shared')
        self.assertEqual(encoder.rem.lh.point[0].in_channels, 256)
        self.assertEqual(encoder.rem.lh.strip[0].out_channels, 256)
        self.assertEqual(sum(p.numel() for p in encoder.rem.parameters()), 1248771)
        self.assertEqual(config['epoches'], 58)
        self.assertEqual(config['train_dataloader']['total_batch_size'], 28)
        self.assertEqual(config['train_dataloader']['dataset']['transforms']['policy']['epoch'], [4, 29, 50])
        self.assertEqual(config['train_dataloader']['collate_fn']['stop_epoch'], 50)
        self.assertEqual(config['warmup_iter'], 432)
        self.assertFalse({'RandomZoomOut', 'RandomIoUCrop'} & {op['type'] for op in config['train_dataloader']['dataset']['transforms']['ops']})
        self.assertTrue((ROOT / config['experiment_protocol_document']).is_file())
        self.assertNotEqual(config['output_dir'], reference['output_dir'])
        self.assertEqual(config['HybridEncoder'].pop('rem_variant'), 'point_strip')
        expected = copy.deepcopy(reference)
        for item in (config, expected):
            for key in ('__include__', 'output_dir', 'experiment_protocol_document'):
                item.pop(key, None)
        self.assertEqual(config, expected)


if __name__ == '__main__':
    unittest.main()
