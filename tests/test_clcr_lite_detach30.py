"""Delayed detached calibration: gradient boundaries, true freeze, EMA and epoch routing."""

import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn

from engine.core import YAMLConfig
from engine.core.yaml_utils import load_config
from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.dfine_decoder import CLCRLite, DFINETransformer
from engine.deim.hybrid_encoder import HybridEncoder
from engine.deim.matcher import HungarianMatcher
from engine.optim.ema import ModelEMA
from engine.solver.det_engine import (
    _discard_inactive_clcr_lite_gradients, _module_training_metrics, _set_clcr_lite_epoch,
    train_one_epoch,
)


ROOT = Path(__file__).resolve().parents[1]
DECODER_ARGS = dict(num_classes=7, hidden_dim=32, feat_channels=[32] * 3,
                    num_queries=5, num_layers=4, eval_idx=-1, nhead=8,
                    num_points=[3, 6, 3], dim_feedforward=64, reg_max=32,
                    num_denoising=4, eval_spatial_size=None)


def make_decoder(**extra):
    args = copy.deepcopy(DECODER_ARGS)
    args.update(use_clcr_lite=True, clcr_lite_detach_inputs=True, clcr_lite_start_epoch=30)
    args.update(extra)
    return DFINETransformer(**args)


class ToyDetector(nn.Module):
    """Exercise the actual epoch/optimizer loop with a real calibration head."""
    def __init__(self):
        super().__init__()
        self.score_head = nn.Linear(3, 7)
        self.box_head = nn.Linear(3, 4)
        self.decoder = nn.Module()
        self.decoder.decoder = nn.Module()
        self.decoder.decoder.clcr_lite = CLCRLite(detach_inputs=True, start_epoch=30)

    def forward(self, samples, targets=None):
        scores = self.score_head(samples).unsqueeze(1)
        boxes = self.box_head(samples).sigmoid().unsqueeze(1)
        edge = scores.new_ones((*scores.shape[:2], 20))
        correction = self.decoder.decoder.clcr_lite(scores, edge, boxes)
        return dict(pred_logits=scores + correction, pred_boxes=boxes)


class ToyCriterion(nn.Module):
    def forward(self, outputs, targets, **kwargs):
        return dict(loss_labels=outputs['pred_logits'].square().mean(),
                    loss_boxes=outputs['pred_boxes'].square().mean())


class ActualDetector(nn.Module):
    def __init__(self, lite=True):
        super().__init__()
        self.encoder = HybridEncoder(in_channels=[32, 64, 128], hidden_dim=32, nhead=8,
                                     dim_feedforward=64, depth_mult=.34, use_rem=True)
        self.decoder = make_decoder(use_clcr_lite=lite)

    def forward(self, features, targets=None):
        return self.decoder(self.encoder(features), targets)


def criterion():
    return DEIMCriterion(
        HungarianMatcher(dict(cost_class=2, cost_bbox=5, cost_giou=2), use_focal_loss=True),
        dict(loss_mal=1, loss_bbox=5, loss_giou=2, loss_fgl=.15, loss_ddf=1.5),
        ['mal', 'boxes', 'local'], num_classes=7, reg_max=32)


class Detach30Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(503)

    def test_all_eight_inputs_are_detached_while_the_calibration_head_learns(self):
        scores = torch.randn(2, 3, 7, requires_grad=True)
        edge = torch.rand(2, 3, 20, requires_grad=True)
        boxes = torch.rand(2, 3, 4, requires_grad=True)
        head = CLCRLite(detach_inputs=True)
        head.calibration[-1].weight.data.normal_(std=.2)
        evidence = []
        handle = head.calibration[0].register_forward_pre_hook(lambda module, args: evidence.append(args[0]))
        head(scores, edge, boxes).square().sum().backward()
        handle.remove()
        self.assertEqual(evidence[0].shape, (2, 3, 7, 8))
        self.assertFalse(evidence[0].requires_grad)
        for value in (scores, edge, boxes):
            self.assertIsNone(value.grad)
        for parameter in head.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0.)

    def test_detach_keeps_the_direct_logit_gradient_and_can_still_change_matching(self):
        head = CLCRLite(detach_inputs=True)
        with torch.no_grad():
            head.calibration[0].weight.zero_()
            head.calibration[0].bias.zero_()
            head.calibration[0].weight[0, 5] = 1.
            head.calibration[0].bias[0] = 3.
            head.calibration[-1].weight.zero_()
            head.calibration[-1].weight[0, 0] = 2.
        scores = torch.full((1, 2, 7), -5.)
        scores[0, :, 0] = torch.tensor([1., .9])
        scores.requires_grad_()
        boxes = torch.tensor([[[.5, .5, .1, .3], [.5, .5, .8, .3]]])
        calibrated = scores + head(scores, torch.ones(1, 2, 20), boxes)
        matcher = HungarianMatcher(dict(cost_class=1, cost_bbox=0, cost_giou=0), use_focal_loss=True)
        targets = [dict(labels=torch.tensor([0]), boxes=torch.tensor([[.5, .5, .3, .3]]))]
        original = matcher(dict(pred_logits=scores, pred_boxes=boxes), targets)['indices'][0][0]
        changed = matcher(dict(pred_logits=calibrated, pred_boxes=boxes), targets)['indices'][0][0]
        self.assertEqual(original.item(), 0)
        self.assertEqual(changed.item(), 1)
        base_grad = torch.autograd.grad(scores.sigmoid().sum(), scores, retain_graph=True)[0]
        calibrated.sigmoid().sum().backward()
        self.assertGreater(scores.grad.abs().sum().item(), 0.)
        self.assertFalse(torch.allclose(scores.grad, base_grad))

    def test_epoch_29_is_exactly_zero_and_epoch_30_enables_the_residual_in_train_and_eval(self):
        head = CLCRLite(detach_inputs=True, start_epoch=30)
        head.calibration[-1].bias.data.fill_(.4)
        inputs = (torch.randn(1, 2, 7), torch.rand(1, 2, 20), torch.rand(1, 2, 4))
        for training in (True, False):
            head.train(training)
            head.set_epoch(29)
            with patch.object(head, 'input_statistics', side_effect=AssertionError('inactive branch must not read evidence')):
                torch.testing.assert_close(head(*inputs), torch.zeros_like(inputs[0]), rtol=0, atol=0)
            head.set_epoch(30)
            self.assertTrue(head.active)
            torch.testing.assert_close(head(*inputs), torch.full_like(inputs[0], .4), rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in head.parameters()), 162)

    def test_inactive_head_has_ddp_connected_zero_grads_but_no_adamw_update_or_state(self):
        head = CLCRLite(detach_inputs=True, start_epoch=30)
        optimizer = torch.optim.AdamW(head.parameters(), lr=.01, weight_decay=.2)
        inputs = (torch.randn(2, 3, 7), torch.rand(2, 3, 20), torch.rand(2, 3, 4))
        initial = {name: value.detach().clone() for name, value in head.named_parameters()}
        for epoch in (0, 29, 30, 31):
            head.set_epoch(epoch)
            optimizer.zero_grad(set_to_none=True)
            (head(*inputs) - 1.).square().mean().backward()
            self.assertTrue(all(p.grad is not None for p in head.parameters()))
            if epoch < 30:
                self.assertTrue(all(p.grad.abs().sum() == 0 for p in head.parameters()))
            else:
                self.assertGreater(head.calibration[-1].weight.grad.abs().sum().item(), 0.)
                if epoch > 30:
                    self.assertGreater(head.calibration[0].weight.grad.abs().sum().item(), 0.)
                    self.assertGreater(head.beta.grad.abs().item(), 0.)
            head.discard_inactive_gradients()
            optimizer.step()
            if epoch < 30:
                self.assertFalse(optimizer.state)
                for name, parameter in head.named_parameters():
                    torch.testing.assert_close(parameter, initial[name], rtol=0, atol=0)

    def test_real_training_loop_synchronizes_ema_and_skips_inactive_adamw_for_fp32_and_amp(self):
        for amp in (False, True):
            with self.subTest(amp=amp):
                model = ToyDetector()
                head = model.decoder.decoder.clcr_lite
                ema = ModelEMA(model, decay=.75, warmups=0)
                optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.2)
                initial = {name: p.detach().clone() for name, p in head.named_parameters()}
                score_weight = model.score_head.weight.detach().clone()
                targets = [dict(labels=torch.tensor([0]), boxes=torch.tensor([[.5, .5, .3, .3]]))] * 2
                data = [(torch.randn(2, 3), targets)]
                scaler = torch.amp.GradScaler('cpu') if amp else None
                for epoch in (29, 30):
                    stats = train_one_epoch(False, None, model, ToyCriterion(), data, optimizer,
                        torch.device('cpu'), epoch, max_norm=.1, print_freq=100,
                        ema=ema, scaler=scaler, amp_dtype='bfloat16', amp_loss_fp32=True)
                    self.assertEqual(head.active, epoch >= 30)
                    self.assertEqual(ema.module.decoder.decoder.clcr_lite.active, head.active)
                    self.assertEqual(stats['clcr_lite_active'], float(epoch >= 30))
                    if epoch < 30:
                        for name, parameter in head.named_parameters():
                            torch.testing.assert_close(parameter, initial[name], rtol=0, atol=0)
                            self.assertNotIn(parameter, optimizer.state)
                        self.assertFalse(torch.equal(model.score_head.weight, score_weight))
                    else:
                        self.assertGreater(head.calibration[-1].weight.abs().sum().item(), 0.)

    def test_warmup_rem_detector_outputs_losses_and_common_gradients_match_rem_alone(self):
        torch.manual_seed(509)
        baseline = ActualDetector(lite=False)
        torch.manual_seed(509)
        model = ActualDetector()
        _set_clcr_lite_epoch(model, 29)
        features = [torch.randn(2, c, h, w) for c, h, w in ((32, 8, 12), (64, 4, 6), (128, 2, 3))]
        targets = [dict(labels=torch.tensor([3]), boxes=torch.tensor([[.4, .5, .3, .4]])),
                   dict(labels=torch.tensor([1, 6]), boxes=torch.tensor([[.3, .4, .2, .3], [.7, .6, .3, .2]]))]
        torch.manual_seed(521)
        reference = baseline(features, targets)
        torch.manual_seed(521)
        actual = model(features, targets)
        for key in ('pred_logits', 'pred_boxes', 'pred_corners'):
            torch.testing.assert_close(actual[key], reference[key], rtol=0, atol=0)
        for group in ('aux_outputs', 'dn_outputs'):
            for new, old in zip(actual[group], reference[group]):
                torch.testing.assert_close(new['pred_logits'], old['pred_logits'], rtol=0, atol=0)
        expected_losses, losses = criterion()(reference, targets), criterion()(actual, targets)
        self.assertEqual(losses.keys(), expected_losses.keys())
        for key in losses:
            torch.testing.assert_close(losses[key], expected_losses[key], rtol=0, atol=0)
        sum(expected_losses.values()).backward()
        sum(losses.values()).backward()
        _discard_inactive_clcr_lite_gradients(model)
        parameters = dict(model.named_parameters())
        for name, parameter in baseline.named_parameters():
            if parameter.grad is None:
                self.assertIsNone(parameters[name].grad)
            else:
                torch.testing.assert_close(parameters[name].grad, parameter.grad, rtol=1e-6, atol=1e-7)
        self.assertTrue(all(p.grad is None for p in model.decoder.decoder.clcr_lite.parameters()))

    def test_checkpoint_ema_and_deploy_preserve_activation_without_an_epoch_argument(self):
        model = make_decoder(eval_idx=1).eval()
        head = model.decoder.clcr_lite
        head.calibration[-1].bias.data.fill_(.25)
        features = [torch.randn(2, 32, h, w) for h, w in ((8, 12), (4, 6), (2, 3))]
        for epoch in (29, 30):
            head.set_epoch(epoch)
            ema = ModelEMA(model, decay=.9, warmups=0)
            restored = make_decoder(eval_idx=1).eval()
            restored.load_state_dict(ema.state_dict()['module'], strict=True)
            self.assertEqual(restored.decoder.clcr_lite.active, epoch >= 30)
            with torch.no_grad():
                expected = model(features)
            restored.convert_to_deploy()
            restored.decoder.convert_to_deploy()
            with torch.no_grad():
                actual = restored(features)
            for key in ('pred_logits', 'pred_boxes', 'pred_corners'):
                torch.testing.assert_close(actual[key], expected[key], rtol=1e-6, atol=1e-7)

    def test_new_configs_preserve_the_original_lite_rem_and_58e_settings(self):
        folder = ROOT / 'configs/deim_dfine'
        reference = load_config(str(folder / 'deim_hgnetv2_m_sar_aircraft_clcr_lite_58e_b28_old_server.yml'), {})
        outputs = {reference['output_dir']}
        for suffix, delayed in (('rem_clcr_lite', False), ('rem_clcr_lite_detach30', True)):
            path = folder / f'deim_hgnetv2_m_sar_aircraft_{suffix}_58e_b28_old_server.yml'
            cfg = YAMLConfig(str(path))
            head = cfg.model.decoder.decoder.clcr_lite
            self.assertEqual(sum(p.numel() for p in head.parameters()), 162)
            self.assertEqual(head.detach_inputs, delayed)
            self.assertEqual(head.start_epoch, 30 if delayed else 0)
            self.assertEqual(head.active, not delayed)
            config = load_config(str(path), {})
            self.assertTrue(config['HybridEncoder'].pop('use_rem'))
            if delayed:
                self.assertTrue(config['DFINETransformer'].pop('clcr_lite_detach_inputs'))
                self.assertEqual(config['DFINETransformer'].pop('clcr_lite_start_epoch'), 30)
            self.assertNotIn(config['output_dir'], outputs)
            outputs.add(config['output_dir'])
            self.assertTrue((ROOT / config['experiment_protocol_document']).is_file())
            expected = copy.deepcopy(reference)
            for item in (config, expected):
                for key in ('__include__', 'output_dir', 'experiment_protocol_document'):
                    item.pop(key, None)
            self.assertEqual(config, expected)

    def test_default_state_dict_stays_compatible_and_invalid_thresholds_are_rejected(self):
        original = CLCRLite()
        self.assertEqual(len(original.state_dict()), 5)
        copy_head = CLCRLite()
        copy_head.load_state_dict(original.state_dict(), strict=True)
        self.assertTrue(copy_head.active)
        self.assertFalse(copy_head.detach_inputs)
        self.assertEqual(len(CLCRLite(start_epoch=30).state_dict()), 6)
        for start in (-1, 1.5):
            with self.assertRaises(ValueError):
                CLCRLite(start_epoch=start)


if __name__ == '__main__':
    unittest.main()
