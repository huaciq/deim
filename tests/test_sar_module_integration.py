"""Regression checks for the new branches' actual detector integration."""
import copy
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

from engine.core.yaml_utils import load_config
from engine.core import YAMLConfig
from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.dfine_decoder import DFINETransformer
from engine.deim.hybrid_encoder import HybridEncoder
from engine.deim.matcher import HungarianMatcher
from engine.solver.det_engine import _module_training_metrics


def decoder(**extra):
    torch.manual_seed(31)
    return DFINETransformer(num_classes=7, hidden_dim=32, feat_channels=[32, 32, 32],
                            num_queries=5, num_layers=2, nhead=4, num_points=2,
                            dim_feedforward=64, reg_max=8, num_denoising=4,
                            eval_spatial_size=None, slqc_token_dim=8,
                            slqc_roi_size=2, **extra)


def inputs():
    torch.manual_seed(32)
    features = [torch.randn(2, 32, side, side, requires_grad=True) for side in [4, 2, 1]]
    targets = [dict(labels=torch.tensor([3]), boxes=torch.tensor([[.5, .5, .35, .3]]),
                    scs_signature=torch.rand(1, 28), scs_valid=torch.ones(1, dtype=torch.bool))
               for _ in range(2)]
    return features, targets


def paired_forward(model, features, targets):
    torch.manual_seed(33)
    return model(features, targets)


class SarModuleIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_encoder_dsg_preserves_initial_state_rng_outputs_and_scales(self):
        kwargs = dict(in_channels=[32, 64, 128], hidden_dim=32, nhead=4,
                      dim_feedforward=64, depth_mult=.34)
        torch.manual_seed(30)
        base = HybridEncoder(**kwargs)
        original_rng = torch.get_rng_state()
        torch.manual_seed(30)
        adapted = HybridEncoder(**kwargs, use_dsg=True, dsg_bottleneck=8)
        torch.testing.assert_close(torch.get_rng_state(), original_rng, rtol=0, atol=0)
        for name, value in base.state_dict().items():
            torch.testing.assert_close(value, adapted.state_dict()[name], rtol=0, atol=0)
        self.assertIsNone(adapted.p2_inject)
        features = [torch.randn(2, channels, side, side) for channels, side in [(32, 8), (64, 4), (128, 2)]]
        base_outputs, adapted_outputs = base(features), adapted(features)
        self.assertEqual(len(adapted_outputs), 3)
        for reference, actual in zip(base_outputs, adapted_outputs):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        sum(o.square().mean() for o in adapted_outputs).backward()
        self.assertIsNotNone(adapted.dsg.alpha.grad)
        self.assertNotEqual(float(adapted.dsg.alpha.grad), 0)

    def test_decoder_initial_state_seed_and_outputs_match_with_slqc(self):
        base = decoder()
        original_rng = torch.get_rng_state()
        adapted = decoder(use_slqc=True)
        torch.testing.assert_close(torch.get_rng_state(), original_rng, rtol=0, atol=0)
        for name, value in base.state_dict().items():
            torch.testing.assert_close(value, adapted.state_dict()[name], rtol=0, atol=0)
        features, targets = inputs()
        for training in [True, False]:
            base.train(training)
            adapted.train(training)
            reference = paired_forward(base, features, targets)
            actual = paired_forward(adapted, features, targets)
            for name in ['pred_logits', 'pred_boxes', 'pred_corners']:
                torch.testing.assert_close(actual[name], reference[name], rtol=0, atol=0)
            self.assertNotIn('pred_scs', actual)

    def test_slqc_changes_only_final_classification_preserving_lqe_dn_aux_teacher(self):
        base, adapted = decoder(), decoder(use_slqc=True)
        for model in [base, adapted]:
            for layer in model.decoder.lqe_layers:
                layer.reg_conf.layers[-1].bias.data.fill_(.25)
        adapted.slqc.alpha.data.fill_(.4)
        features, targets = inputs()
        residuals = []
        hook = adapted.slqc.register_forward_hook(lambda module, args, result:residuals.append(result))
        reference = paired_forward(base, features, targets)
        actual = paired_forward(adapted, features, targets)
        hook.remove()
        correction = F.linear(residuals[0], adapted.dec_score_head[-1].weight, bias=None)
        self.assertGreater(float(correction.abs().max()), 0)
        torch.testing.assert_close(actual['pred_logits'], reference['pred_logits'] + correction)
        for name in ['pred_boxes', 'pred_corners', 'ref_points']:
            torch.testing.assert_close(actual[name], reference[name], rtol=0, atol=0)
        for branch in ['aux_outputs', 'dn_outputs', 'enc_aux_outputs']:
            for original, changed in zip(reference[branch], actual[branch]):
                for name in ['pred_logits', 'pred_boxes', 'teacher_logits']:
                    if name in original:
                        torch.testing.assert_close(changed[name], original[name], rtol=0, atol=0)

    def test_eval_without_gt_and_deploy_retain_slqc(self):
        model = decoder(use_slqc=True).eval()
        model.slqc.alpha.data.fill_(.3)
        features, _ = inputs()
        with torch.no_grad():
            reference = model(features)
            deployed = copy.deepcopy(model)
            for layer in deployed.modules():
                if hasattr(layer, 'convert_to_deploy'):
                    layer.convert_to_deploy()
            actual = deployed(features)
        self.assertIsNotNone(deployed.slqc)
        for name in ['pred_logits', 'pred_boxes', 'pred_corners']:
            torch.testing.assert_close(actual[name], reference[name], rtol=0, atol=0)

    def test_detection_and_scs_losses_backpropagate_to_new_modules(self):
        encoder = HybridEncoder(in_channels=[32, 64, 128], hidden_dim=32, nhead=4,
                                dim_feedforward=64, depth_mult=.34, use_dsg=True, dsg_bottleneck=8)
        model = decoder(use_slqc=True, use_scs=True, scs_hidden_dim=16)
        encoder.dsg.alpha.data.fill_(.1)
        model.slqc.alpha.data.fill_(.1)
        _, targets = inputs()
        features = [torch.randn(2, c, side, side) for c, side in [(32, 8), (64, 4), (128, 2)]]
        outputs = model(encoder(features), targets)
        criterion = DEIMCriterion(
            HungarianMatcher(dict(cost_class=2, cost_bbox=5, cost_giou=2), use_focal_loss=True),
            dict(loss_mal=1, loss_bbox=5, loss_giou=2, loss_fgl=.15, loss_ddf=1.5,
                 loss_scs_sig=1, loss_scs_pair=.1),
            ['mal', 'boxes', 'local'], num_classes=7, reg_max=8, use_scs=True, scs_pairs=[[3, 4, 1]])
        losses = criterion(outputs, targets)
        self.assertIn('loss_scs_sig', losses)
        self.assertIn('loss_scs_pair', losses)
        self.assertTrue(all(torch.isfinite(value).all() for value in losses.values()))
        sum(losses.values()).backward()
        for prefix, module in [('dsg', encoder.dsg), ('slqc', model.slqc), ('scs', model.scs_head)]:
            for name, parameter in module.named_parameters():
                self.assertIsNotNone(parameter.grad, prefix + '.' + name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), prefix + '.' + name)

    def test_variants_preserve_protocol_and_declared_module_flags(self):
        directory = Path('configs/deim_dfine/sar_aircraft_modules')
        baseline = load_config('configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_102e_b28.yml', {})
        old_scs = load_config('configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_scs_m1_102e_b28.yml', {})
        outputs = set()
        configs = list(directory.glob('*.yml'))
        self.assertEqual(len(configs), 10)
        for path in configs:
            with self.subTest(config=path.name):
                cfg = load_config(str(path), {})
                for key in ['optimizer', 'epoches', 'seed', 'warmup_iter', 'flat_epoch', 'no_aug_epoch',
                            'lrsheduler', 'lr_gamma', 'eval_spatial_size', 'use_amp', 'use_ema',
                            'amp_dtype', 'amp_loss_fp32', 'amp_fp32_retry', 'val_dataloader']:
                    self.assertEqual(cfg[key], baseline[key], key)
                for key in ['dataset', 'total_batch_size']:
                    self.assertEqual(cfg['train_dataloader'][key], baseline['train_dataloader'][key])
                for key, value in baseline['train_dataloader']['collate_fn'].items():
                    self.assertEqual(cfg['train_dataloader']['collate_fn'][key], value)
                self.assertNotIn(cfg['output_dir'], outputs)
                outputs.add(cfg['output_dir'])
                scs_expected = path.stem.startswith('scs_')
                self.assertEqual(cfg['DFINETransformer'].get('use_scs', False), scs_expected)
                self.assertEqual(cfg['DFINETransformer'].get('use_slqc', False), 'slqc' in path.stem)
                self.assertEqual(cfg['HybridEncoder'].get('use_dsg', False), 'dsg' in path.stem)
                if scs_expected:
                    self.assertEqual(cfg['DEIMCriterion'], old_scs['DEIMCriterion'])
                    self.assertEqual(cfg['train_dataloader']['collate_fn'], old_scs['train_dataloader']['collate_fn'])

    def test_optional_residual_metrics_work_with_wrapped_models(self):
        core = torch.nn.Module()
        core.encoder, core.decoder = torch.nn.Identity(), torch.nn.Identity()
        self.assertEqual(_module_training_metrics(core), {})
        core.encoder = HybridEncoder(in_channels=[32, 64, 128], hidden_dim=32, nhead=4,
                                     dim_feedforward=64, depth_mult=.34, use_dsg=True, dsg_bottleneck=8)
        core.decoder = decoder(use_slqc=True)
        self.assertEqual(_module_training_metrics(core), dict(dsg_alpha=0., slqc_alpha=0.))
        core.encoder.dsg.alpha.data.fill_(.25)
        core.decoder.slqc.alpha.data.fill_(-.5)
        self.assertEqual(_module_training_metrics(torch.nn.DataParallel(core)),
                         dict(dsg_alpha=.25, slqc_alpha=-.5))

    def test_default_config_loading_does_not_leak_variant_switches(self):
        directory = Path('configs/deim_dfine/sar_aircraft_modules')
        dsg = YAMLConfig(str(directory / 'dsg.yml')).yaml_cfg
        slqc = YAMLConfig(str(directory / 'slqc.yml')).yaml_cfg
        combined = YAMLConfig(str(directory / 'scs_slqc_dsg.yml')).yaml_cfg
        baseline = YAMLConfig('configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_102e_b28.yml').yaml_cfg
        self.assertTrue(dsg['HybridEncoder']['use_dsg'])
        self.assertFalse(slqc['HybridEncoder']['use_dsg'])
        self.assertTrue(combined['DFINETransformer']['use_scs'])
        self.assertNotIn('use_dsg', baseline['HybridEncoder'])
        self.assertNotIn('use_slqc', baseline['DFINETransformer'])
        self.assertNotIn('use_scs', baseline['DEIMCriterion'])


if __name__ == '__main__':
    unittest.main()
