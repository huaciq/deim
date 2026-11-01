"""Independent REM band scales: Haar routing, learning and frozen protocols."""

import copy
import io
from pathlib import Path
import unittest

import torch
from torch import nn

from engine.core import YAMLConfig
from engine.core.yaml_utils import load_config
from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.dfine_decoder import DFINETransformer
from engine.deim.hybrid_encoder import HybridEncoder
from engine.deim.matcher import HungarianMatcher
from engine.deim.radar_enhancement import HaarDWT2D, HaarIDWT2D, RadarEnhancementModule
from engine.optim.ema import ModelEMA
from engine.solver.det_engine import _module_training_metrics, _to_float32


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'configs/deim_dfine'
BANDS = ('ll', 'lh', 'hl', 'hh')
ENCODER_ARGS = dict(in_channels=[32, 64, 128], hidden_dim=32, nhead=8,
                    dim_feedforward=64, depth_mult=.34)


def normalized(config):
    result = copy.deepcopy(config)
    for key in ('__include__', 'output_dir', 'experiment_protocol_document'):
        result.pop(key, None)
    return result


def features():
    return [torch.randn(2, c, h, w) for c, h, w in ((32, 8, 12), (64, 4, 6), (128, 2, 3))]


class RemPerBandTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(901)

    def test_common_initialization_rng_and_legacy_shared_checkpoint_are_preserved(self):
        shared = RadarEnhancementModule(8).eval()
        state = torch.get_rng_state()
        torch.manual_seed(901)
        separate = RadarEnhancementModule(8, residual_mode='per_band').eval()
        torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)
        self.assertEqual(shared.residual_mode, 'shared')
        self.assertEqual(shared.alpha.shape, torch.Size([]))
        self.assertEqual(separate.alpha.shape, torch.Size([4]))
        self.assertEqual(shared.state_dict().keys(), separate.state_dict().keys())
        for name, value in shared.state_dict().items():
            if name != 'alpha':
                torch.testing.assert_close(value, separate.state_dict()[name], rtol=0, atol=0)
        restored = RadarEnhancementModule(8)
        restored.load_state_dict(shared.state_dict(), strict=True)
        x = torch.randn(2, 8, 7, 9)
        for module in (shared, separate):
            torch.testing.assert_close(module(x), x, rtol=0, atol=0)

    def test_equal_band_scales_reproduce_shared_residual_and_its_scale_gradient(self):
        for size in ((8, 10), (7, 9)):
            with self.subTest(size=size):
                shared = RadarEnhancementModule(4).double().eval()
                separate = RadarEnhancementModule(4, residual_mode='per_band').double().eval()
                state = shared.state_dict()
                state['alpha'] = torch.full((4,), .17, dtype=torch.float64)
                separate.load_state_dict(state, strict=True)
                shared.alpha.data.fill_(.17)
                x = torch.randn(2, 4, *size, dtype=torch.float64)
                upstream = torch.randn_like(x)
                reference = shared(x)
                result = separate(x)
                torch.testing.assert_close(result, reference, rtol=1e-12, atol=1e-12)
                (reference * upstream).sum().backward()
                (result * upstream).sum().backward()
                torch.testing.assert_close(separate.alpha.grad.sum(), shared.alpha.grad,
                                           rtol=1e-12, atol=1e-12)
                for name, parameter in shared.named_parameters():
                    if name != 'alpha':
                        torch.testing.assert_close(dict(separate.named_parameters())[name].grad,
                                                   parameter.grad, rtol=1e-11, atol=1e-12)

    def test_independent_scales_route_signed_bands_through_one_analysis_and_synthesis(self):
        module = RadarEnhancementModule(1, residual_mode='per_band')
        for name in BANDS:
            setattr(module, name, nn.Identity())
        x = torch.tensor([[[[1., 2.], [3., 5.]]]])
        original_bands = HaarDWT2D()(x)
        calls = []
        handles = [module.dwt.register_forward_hook(lambda *args: calls.append('dwt')),
                   module.idwt.register_forward_hook(lambda *args: calls.append('idwt'))]
        for index in range(4):
            module.alpha.data.zero_()
            module.alpha.data[index] = -.25 if index % 2 else .5
            coefficient = module.alpha[index].detach()
            selected = tuple(band * coefficient if i == index else torch.zeros_like(band)
                             for i, band in enumerate(original_bands))
            expected = x + HaarIDWT2D()(selected)
            torch.testing.assert_close(module(x), expected, rtol=0, atol=0)
        for handle in handles:
            handle.remove()
        self.assertEqual(calls, ['dwt', 'idwt'] * 4)

    def test_zero_identity_then_all_scales_and_branches_learn_in_float32_and_autocast(self):
        for amp in (False, True):
            with self.subTest(amp=amp):
                module = RadarEnhancementModule(8, residual_mode='per_band')
                optimizer = torch.optim.SGD(module.parameters(), lr=.1)
                x = torch.randn(2, 8, 9, 11)
                target = torch.randn_like(x)
                for step in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
                        result = module(x)
                    self.assertEqual(result.dtype, x.dtype)
                    if step == 0:
                        torch.testing.assert_close(result, x, rtol=0, atol=0)
                    (result - target).square().mean().backward()
                    self.assertTrue(torch.isfinite(module.alpha.grad).all())
                    self.assertTrue((module.alpha.grad != 0).all())
                    for name in BANDS:
                        gradients = [p.grad for p in getattr(module, name).parameters()]
                        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
                        magnitude = sum(g.abs().sum().item() for g in gradients)
                        if step == 0:
                            self.assertEqual(magnitude, 0.)
                        else:
                            self.assertGreater(magnitude, 0.)
                    optimizer.step()

    def test_encoder_placement_common_rng_outputs_and_four_training_metrics(self):
        torch.manual_seed(903)
        shared = HybridEncoder(**ENCODER_ARGS, use_rem=True).eval()
        state = torch.get_rng_state()
        torch.manual_seed(903)
        separate = HybridEncoder(**ENCODER_ARGS, use_rem=True, rem_residual_mode='per_band').eval()
        torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)
        for name, value in shared.state_dict().items():
            if name != 'rem.alpha':
                torch.testing.assert_close(value, separate.state_dict()[name], rtol=0, atol=0)
        source = features()
        with torch.no_grad():
            actual, expected = separate(source), shared(source)
        for x, y in zip(actual, expected):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        observed = []
        h = separate.rem.register_forward_pre_hook(lambda module, args: observed.append(args[0].detach().clone()))
        separate.rem.alpha.data.copy_(torch.tensor([.1, -.2, .3, .4]))
        separate(source)
        h.remove()
        torch.testing.assert_close(observed[0], separate.input_proj[0](source[0]), rtol=0, atol=0)
        owner = nn.Module()
        owner.encoder = separate
        metrics = _module_training_metrics(owner)
        self.assertEqual(set(metrics), {f'rem_alpha_{band}' for band in BANDS})
        for name, value in zip(BANDS, (.1, -.2, .3, .4)):
            self.assertAlmostEqual(metrics[f'rem_alpha_{name}'], value, places=6)
        owner.encoder = shared
        self.assertEqual(_module_training_metrics(owner), {'rem_alpha': 0.})

    def test_original_four_layer_losses_aux_dn_and_regression_train_the_new_scales(self):
        for amp in (False, True):
            with self.subTest(amp=amp):
                encoder = HybridEncoder(**ENCODER_ARGS, use_rem=True, rem_residual_mode='per_band')
                decoder = DFINETransformer(num_classes=7, hidden_dim=32, feat_channels=[32] * 3,
                    num_queries=5, num_layers=4, eval_idx=-1, nhead=8, num_points=[3, 6, 3],
                    dim_feedforward=64, reg_max=32, num_denoising=4, eval_spatial_size=None)
                criterion = DEIMCriterion(
                    HungarianMatcher(dict(cost_class=2, cost_bbox=5, cost_giou=2), use_focal_loss=True),
                    dict(loss_mal=1, loss_bbox=5, loss_giou=2, loss_fgl=.15, loss_ddf=1.5),
                    ['mal', 'boxes', 'local'], num_classes=7, reg_max=32)
                source = features()
                targets = [dict(labels=torch.tensor([3]), boxes=torch.tensor([[.4, .5, .3, .4]])),
                    dict(labels=torch.tensor([1, 6]), boxes=torch.tensor([[.3, .4, .2, .3], [.7, .6, .3, .2]]))]
                optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
                for step in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
                        output = decoder(encoder(source), targets)
                    losses = criterion(_to_float32(output), targets)
                    self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
                    self.assertIn('loss_fgl_aux_2', losses)
                    self.assertIn('loss_mal_dn_3', losses)
                    self.assertFalse(any('wfdr' in key for key in losses))
                    if step:
                        self.assertIn('loss_ddf_aux_0', losses)
                    sum(losses.values()).backward()
                    self.assertTrue(torch.isfinite(encoder.rem.alpha.grad).all())
                    self.assertTrue((encoder.rem.alpha.grad != 0).all())
                    self.assertGreater(decoder.dec_bbox_head[-1].layers[-1].weight.grad.abs().sum().item(), 0.)
                    if step:
                        for band in BANDS:
                            self.assertGreater(sum(p.grad.abs().sum().item()
                                for p in getattr(encoder.rem, band).parameters()), 0.)
                    optimizer.step()

    def test_optimizer_checkpoint_and_ema_keep_all_four_learned_scales(self):
        module = RadarEnhancementModule(4, residual_mode='per_band').eval()
        optimizer = torch.optim.AdamW(module.parameters(), lr=.01)
        ema = ModelEMA(module, decay=.75, warmups=0)
        x = torch.randn(2, 4, 7, 9)
        module(x).square().mean().backward()
        optimizer.step()
        learned = module.alpha.detach().clone()
        ema.update(module)
        torch.testing.assert_close(ema.module.alpha, learned * .25, rtol=0, atol=0)
        checkpoint = io.BytesIO()
        torch.save(dict(model=module.state_dict(), optimizer=optimizer.state_dict(), ema=ema.state_dict()), checkpoint)
        checkpoint.seek(0)
        saved = torch.load(checkpoint, weights_only=True)
        restored = RadarEnhancementModule(4, residual_mode='per_band').eval()
        restored.load_state_dict(saved['model'], strict=True)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=.01)
        restored_optimizer.load_state_dict(saved['optimizer'])
        restored_ema = ModelEMA(restored, decay=.75, warmups=0)
        restored_ema.load_state_dict(saved['ema'])
        for target, reference in ((restored, module), (restored_ema.module, ema.module)):
            torch.testing.assert_close(target(x), reference(x), rtol=0, atol=0)
        for owner, optim in ((module, optimizer), (restored, restored_optimizer)):
            optim.zero_grad(set_to_none=True)
            owner(x).square().mean().backward()
            optim.step()
        torch.testing.assert_close(restored.alpha, module.alpha, rtol=0, atol=0)

    def test_102e_server_paths_and_original_rem_add_only_the_requested_settings(self):
        base = load_config(str(FOLDER / 'deim_hgnetv2_m_sar_aircraft_102e_b28.yml'), {})
        server = load_config(str(FOLDER / 'deim_hgnetv2_m_sar_aircraft_102e_b28_old_server.yml'), {})
        expected = normalized(base)
        for loader, split in (('train_dataloader', 'train'), ('val_dataloader', 'val')):
            expected[loader]['dataset']['img_folder'] = '/home/zxy/sar/datasets/SAR-AIRcraft-1.0'
            expected[loader]['dataset']['ann_file'] = f'/home/zxy/sar/datasets/SAR-AIRcraft-1.0/COCO/{split}.json'
        self.assertEqual(normalized(server), expected)
        shared = load_config(str(FOLDER / 'deim_hgnetv2_m_sar_aircraft_rem_102e_b28_old_server.yml'), {})
        self.assertTrue(shared['HybridEncoder'].pop('use_rem'))
        self.assertEqual(normalized(shared), normalized(server))
        self.assertEqual(server['epoches'], 102)
        self.assertEqual((server['warmup_iter'], server['flat_epoch'], server['no_aug_epoch']), (2000, 49, 12))
        self.assertEqual(server['train_dataloader']['dataset']['transforms']['policy']['epoch'], [4, 49, 90])
        self.assertEqual(server['train_dataloader']['collate_fn']['mixup_epochs'], [4, 49])
        self.assertEqual(server['train_dataloader']['collate_fn']['stop_epoch'], 90)

    def test_both_schedules_keep_protocol_and_real_factory_has_only_three_extra_parameters(self):
        outputs = set()
        for epochs in (58, 102):
            original = FOLDER / f'deim_hgnetv2_m_sar_aircraft_rem_{epochs}e_b28_old_server.yml'
            updated = FOLDER / f'deim_hgnetv2_m_sar_aircraft_rem_per_band_{epochs}e_b28_old_server.yml'
            reference = load_config(str(original), {})
            config = load_config(str(updated), {})
            self.assertEqual(config['HybridEncoder'].pop('rem_residual_mode'), 'per_band')
            self.assertEqual(normalized(config), normalized(reference))
            self.assertFalse(config['DEIMCriterion'].get('use_wfdr', False))
            self.assertFalse(config['DFINETransformer'].get('use_clcr_lite', False))
            self.assertEqual(config['train_dataloader']['total_batch_size'], 28)
            self.assertEqual(config['eval_spatial_size'], [640, 640])
            ops = {op['type'] for op in config['train_dataloader']['dataset']['transforms']['ops']}
            self.assertFalse({'RandomZoomOut', 'RandomIoUCrop'} & ops)
            self.assertTrue((ROOT / config['experiment_protocol_document']).is_file())
            for item in (reference, config):
                self.assertNotIn(item['output_dir'], outputs)
                outputs.add(item['output_dir'])
        counts = []
        for suffix, mode in (('rem', 'shared'), ('rem_per_band', 'per_band')):
            model = YAMLConfig(str(FOLDER / f'deim_hgnetv2_m_sar_aircraft_{suffix}_102e_b28_old_server.yml')).model
            self.assertEqual(model.encoder.rem.residual_mode, mode)
            self.assertEqual(model.encoder.rem.variant, 'haar_directional')
            counts.append(sum(p.numel() for p in model.parameters() if p.requires_grad))
            self.assertEqual(sum(p.numel() for p in model.encoder.rem.parameters()),
                             2823681 if mode == 'shared' else 2823684)
        self.assertEqual(counts, [22301250, 22301253])

    def test_invalid_residual_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'residual_mode'):
            RadarEnhancementModule(8, residual_mode='invalid')
        with self.assertRaisesRegex(ValueError, 'residual_mode'):
            HybridEncoder(**ENCODER_ARGS, use_rem=True, rem_residual_mode='invalid')


if __name__ == '__main__':
    unittest.main()
