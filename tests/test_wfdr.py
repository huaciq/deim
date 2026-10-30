"""Nonuniform W1 values/gradients, real FGL targets and auxiliary/DN routing."""

import copy
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import wasserstein_distance
import torch
from torch import nn
import torch.nn.functional as F

from engine.core import YAMLConfig
from engine.core.yaml_utils import load_config
from engine.deim.box_ops import box_cxcywh_to_xyxy
from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.dfine_decoder import DFINETransformer
from engine.deim.dfine_utils import bbox2distance, translate_gt, weighting_function
from engine.deim.hybrid_encoder import HybridEncoder
from engine.deim.matcher import HungarianMatcher
from engine.deim.wasserstein_fdr import WassersteinFDRLoss


ROOT = Path(__file__).resolve().parents[1]


def criterion(enabled=True, sdcp=False, strength=.5):
    weights = dict(loss_mal=1, loss_bbox=5, loss_giou=2, loss_fgl=.15, loss_ddf=1.5)
    if enabled:
        weights['loss_wfdr'] = .15
    if sdcp:
        weights.update(loss_bbox=0., loss_asym_box=5., loss_margin=.1)
    return DEIMCriterion(
        HungarianMatcher(dict(cost_class=2, cost_bbox=5, cost_giou=2), use_focal_loss=True),
        weights, ['mal', 'boxes', 'local'], num_classes=7, reg_max=32,
        use_wfdr=enabled, wfdr_weight=strength, use_sdcp=sdcp)


def fdr_grid():
    return weighting_function(32, torch.tensor([.5]), torch.tensor([4.]))


def point_logits(index, bins=33):
    logits = torch.full((1, bins), -100.)
    logits[0, index] = 100.
    return logits


class WassersteinFdrTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(331)

    def test_exact_nonuniform_w1_matches_independent_optimal_transport_reference(self):
        grid = fdr_grid()
        logits = torch.randn(5, 33)
        physical = torch.tensor([-5., -1.13, 0., .47, 6.])
        label, wr, wl = translate_gt(physical, 32, torch.tensor([4.]), torch.tensor([.5]))
        q = torch.zeros(5, 33)
        left = label.long()
        q.scatter_add_(1, left[:, None], wl[:, None])
        q.scatter_add_(1, (left + 1)[:, None], wr[:, None])
        probabilities = logits.softmax(-1)
        expected = torch.tensor([
            wasserstein_distance(grid.numpy(), grid.numpy(), p.numpy(), target.numpy())
            for p, target in zip(probabilities, q)], dtype=torch.float32)
        weight = torch.tensor([1., .7, 0., .3, .9])
        loss = WassersteinFDRLoss()
        actual = loss(logits, label, grid, weight, weight_left=wl, weight_right=wr)
        torch.testing.assert_close(actual, .5 * (expected * weight).mean())
        torch.testing.assert_close(loss(logits, label, grid, weight, weight_left=wl, weight_right=wr, avg_factor=2.5),
                                   .5 * (expected * weight).sum() / 2.5)

    def test_equal_wrong_mass_farther_away_has_distance_scaled_loss_and_gradient(self):
        grid = torch.tensor([-4., -1., 0., .5, 4.])
        values, gradients = [], []
        for wrong in [3, 4]:
            logits = torch.full((1, 5), -100.)
            logits[0, 2], logits[0, wrong] = math.log(.8), math.log(.2)
            logits.requires_grad_()
            loss = WassersteinFDRLoss(4, 1.)(logits, torch.tensor([2.]), grid)
            loss.backward()
            distance = grid[wrong] - grid[2]
            torch.testing.assert_close(loss, .2 * distance)
            torch.testing.assert_close(logits.grad[0, wrong], .16 * distance)
            values.append(loss.detach())
            gradients.append(logits.grad[0, wrong])
        torch.testing.assert_close(values[1], values[0] * 8)
        torch.testing.assert_close(gradients[1], gradients[0] * 8)
        # CE alone assigns the same loss to these two equally likely wrong peaks.
        near = point_logits(3, 5)
        far = point_logits(4, 5)
        torch.testing.assert_close(F.cross_entropy(near, torch.tensor([2])),
                                   F.cross_entropy(far, torch.tensor([2])))

    def test_identical_two_bin_distribution_has_zero_w1_without_gaussian_target(self):
        probability = torch.tensor([[0., .75, .25, 0., 0.]])
        value = WassersteinFDRLoss(4)(probability.log(), torch.tensor([1.25]),
                                     torch.tensor([-4., -2., 0., 1., 4.]))
        torch.testing.assert_close(value, torch.tensor(0.), atol=1e-7, rtol=0)

    def test_last_interval_final_endpoint_and_clipped_endpoints_are_preserved(self):
        grid = fdr_grid()
        module = WassersteinFDRLoss(32, 1.)
        logits = point_logits(32).expand(3, -1)
        target = torch.tensor([31.75, 32., 100.])
        actual = module(logits, target, grid, avg_factor=1)
        torch.testing.assert_close(actual, .25 * (grid[32] - grid[31]))
        self.assertEqual(module(point_logits(32), torch.tensor([32.]), grid).item(), 0.)
        self.assertEqual(module(point_logits(0), torch.tensor([-10.]), grid).item(), 0.)

    def test_dfine_left_indices_reuse_physical_interpolation_weights_not_index_fraction(self):
        grid = fdr_grid()
        y = grid[21] * .65 + grid[22] * .35
        label, wr, wl = translate_gt(y[None], 32, torch.tensor([4.]), torch.tensor([.5]))
        self.assertEqual(label.item(), 21.)
        torch.testing.assert_close(wr, torch.tensor([.35]))
        q = torch.zeros(1, 33)
        q[0, 21], q[0, 22] = wl[0], wr[0]
        module = WassersteinFDRLoss(32, 1.)
        value = module(q.log(), label, grid, weight_left=wl, weight_right=wr)
        torch.testing.assert_close(value, torch.tensor(0.), atol=1e-7, rtol=0)
        self.assertGreater(module(q.log(), label, grid).item(), 0.)
        # translate_gt uses 31.9 with right weight 1 above the supported range.
        high, high_right, high_left = translate_gt(torch.tensor([100.]), 32, torch.tensor([4.]), torch.tensor([.5]))
        self.assertEqual(high.long().item(), 31)
        self.assertEqual(module(point_logits(32), high, grid, weight_left=high_left, weight_right=high_right).item(), 0.)

    def test_amp_extreme_logits_and_empty_matches_have_finite_connected_gradients(self):
        for amp, dtype in [(False, torch.float32), (True, torch.bfloat16)]:
            with self.subTest(amp=amp):
                logits = torch.full((4, 33), -1000., dtype=dtype)
                logits[:, 0] = 1000.
                logits.requires_grad_()
                target = torch.tensor([0., 12.4, 31.9, 32.], requires_grad=True)
                grid = fdr_grid().requires_grad_()
                weight = torch.tensor([1., .5, .3, 0.], requires_grad=True)
                with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
                    value = WassersteinFDRLoss()(logits, target, grid, weight)
                self.assertEqual(value.dtype, torch.float32)
                self.assertTrue(torch.isfinite(value))
                value.backward()
                self.assertTrue(torch.isfinite(logits.grad).all())
                self.assertIsNone(target.grad)
                self.assertIsNone(grid.grad)
                self.assertIsNone(weight.grad)
        logits = torch.empty(0, 33, requires_grad=True)
        module = WassersteinFDRLoss()
        for normalizer in [None, 1.]:
            value = module(logits, torch.empty(0), fdr_grid(), avg_factor=normalizer)
            self.assertEqual(value.item(), 0.)
            value.backward()
        self.assertIsNotNone(logits.grad)

    def test_zero_iou_edges_have_zero_gradient_and_weights_do_not_renormalize_the_mean(self):
        logits = torch.randn(2, 33, requires_grad=True)
        target = torch.tensor([8.2, 20.6])
        module = WassersteinFDRLoss()
        first = module(logits[:1], target[:1], fdr_grid())
        weighted = module(logits, target, fdr_grid(), torch.tensor([1., 0.]))
        torch.testing.assert_close(weighted, first / 2)
        weighted.backward()
        self.assertGreater(logits.grad[0].abs().sum().item(), 0.)
        torch.testing.assert_close(logits.grad[1], torch.zeros(33))

    def test_real_loss_local_preserves_dfl_and_distillation_while_adding_weighted_w1(self):
        boxes = torch.tensor([[[.5, .5, .35, .3], [.7, .4, .2, .3]]])
        targets = [dict(labels=torch.tensor([0, 2]), boxes=torch.tensor([[.48, .5, .3, .3], [.68, .4, .25, .2]]))]
        indices = [(torch.arange(2), torch.arange(2))]
        output = dict(pred_boxes=boxes, pred_corners=torch.randn(1, 2, 132, requires_grad=True),
                      ref_points=boxes.clone(), up=torch.tensor([.5]), reg_scale=torch.tensor([4.]),
                      teacher_corners=torch.randn(1, 2, 132), teacher_logits=torch.randn(1, 2, 7))
        baseline, enabled = criterion(False), criterion()
        old = baseline.loss_local(output, targets, indices, 2.)
        new = enabled.loss_local(output, targets, indices, 2.)
        self.assertEqual(set(new), {'loss_fgl', 'loss_ddf', 'loss_wfdr'})
        for name in old:
            torch.testing.assert_close(new[name], old[name], rtol=0, atol=0)
        labels, wr, wl = bbox2distance(boxes.flatten(0, 1), box_cxcywh_to_xyxy(targets[0]['boxes']),
                                      32, output['reg_scale'], output['up'])
        torch.testing.assert_close(enabled.fgl_targets[0], labels)
        # Independently integrate the physical-coordinate target with SciPy.
        probabilities = output['pred_corners'].detach().reshape(-1, 33).softmax(-1).numpy()
        grid = fdr_grid().numpy()
        distances = []
        for p, left, right_weight, left_weight in zip(probabilities, labels.long(), wr, wl):
            q = np.zeros(33)
            q[left.item()], q[left.item() + 1] = left_weight.item(), right_weight.item()
            distances.append(wasserstein_distance(grid, grid, p, q))
        from engine.deim.box_ops import box_iou
        ious = box_iou(box_cxcywh_to_xyxy(boxes[0]), box_cxcywh_to_xyxy(targets[0]['boxes']))[0].diag()
        expected = .5 * (torch.tensor(distances, dtype=torch.float32) * ious.repeat_interleave(4)).sum() / 2
        torch.testing.assert_close(new['loss_wfdr'], expected)
        new['loss_wfdr'].backward()
        self.assertGreater(output['pred_corners'].grad.abs().sum().item(), 0.)

    def test_sdcp_noise_tolerance_also_gates_wfdr_so_it_does_not_restore_strict_supervision(self):
        target = torch.tensor([[.5, .5, .3, .3]])
        prediction = target.clone()
        prediction[:, 0] += .5 / 640
        corners = torch.randn(1, 1, 132, requires_grad=True)
        output = dict(pred_boxes=prediction[None], pred_corners=corners, ref_points=target[None],
                      up=torch.tensor([.5]), reg_scale=torch.tensor([4.]), sdcp_image_size=torch.tensor([640., 640.]))
        targets = [dict(labels=torch.tensor([0]), boxes=target)]
        indices = [(torch.tensor([0]), torch.tensor([0]))]
        gated = criterion(sdcp=True).loss_local(output, targets, indices, 1)['loss_wfdr']
        ordinary = criterion().loss_local(output, targets, indices, 1)['loss_wfdr']
        self.assertEqual(gated.item(), 0.)
        self.assertGreater(ordinary.item(), 0.)
        gated.backward()
        torch.testing.assert_close(corners.grad, torch.zeros_like(corners))

    def test_actual_training_routes_all_fdr_layers_and_dn_with_lite_pgd_and_optional_sdcp(self):
        for sdcp in [False, True]:
            with self.subTest(sdcp=sdcp):
                encoder = HybridEncoder(in_channels=[32, 64, 128], hidden_dim=32, nhead=4,
                                        dim_feedforward=64, depth_mult=.34, use_pgd_rem=True, pgd_rem_prior_channels=8)
                model = DFINETransformer(num_classes=7, hidden_dim=32, feat_channels=[32] * 3, num_queries=5,
                                         num_layers=3, eval_idx=1, nhead=4, num_points=2, dim_feedforward=64,
                                         reg_max=32, num_denoising=4, eval_spatial_size=None,
                                         use_clcr_lite=True, use_sdcp=sdcp)
                loss_fn = criterion(sdcp=sdcp)
                features = [torch.randn(2, c, h, w) for c, h, w in [(32, 8, 12), (64, 4, 6), (128, 2, 3)]]
                targets = [dict(labels=torch.tensor([3]), boxes=torch.tensor([[.4, .5, .3, .4]])),
                           dict(labels=torch.tensor([1, 6]), boxes=torch.tensor([[.3, .4, .2, .3], [.7, .6, .3, .2]]))]
                optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(model.parameters()), lr=1e-3)
                calls = []
                handle = loss_fn.wfdr_loss.register_forward_pre_hook(lambda module, args, kwargs: calls.append((args, kwargs)), with_kwargs=True)
                try:
                    for step in range(2):
                        optimizer.zero_grad(set_to_none=True)
                        calls.clear()
                        output = model(encoder(features), targets, image_size=(64, 96))
                        losses = loss_fn(output, targets)
                        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
                        self.assertEqual({name for name in losses if name.startswith('loss_wfdr')},
                                         {'loss_wfdr', 'loss_wfdr_aux_0', 'loss_wfdr_aux_1',
                                          'loss_wfdr_dn_0', 'loss_wfdr_dn_1', 'loss_wfdr_dn_2'})
                        self.assertEqual(len(calls), 6)
                        self.assertTrue(all(args[0].shape[-1] == 33 and kwargs['avg_factor'] > 0 for args, kwargs in calls))
                        for args, kwargs in calls:
                            torch.testing.assert_close(kwargs['weight_left'] + kwargs['weight_right'], torch.ones_like(kwargs['weight_left']))
                            torch.testing.assert_close(args[2], fdr_grid())
                        self.assertGreater(sum(value for name, value in losses.items() if name.startswith('loss_wfdr')).item(), 0.)
                        sum(losses.values()).backward()
                        self.assertGreater(model.dec_bbox_head[-1].layers[-1].weight.grad.abs().sum().item(), 0.)
                        self.assertGreater(model.decoder.clcr_lite.calibration[-1].weight.grad.abs().sum().item(), 0.)
                        self.assertNotEqual(encoder.pgd_rem.alpha.grad.item(), 0.)
                        if step:
                            self.assertGreater(encoder.pgd_rem.prior_gate.reduce.weight.grad.abs().sum().item(), 0.)
                        optimizer.step()
                finally:
                    handle.remove()

    def test_disabled_and_zero_strength_preserve_existing_loss_and_parameter_state(self):
        torch.manual_seed(337)
        baseline = criterion(False)
        before = torch.get_rng_state()
        enabled = criterion()
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
        self.assertEqual(list(enabled.state_dict()), list(baseline.state_dict()))
        self.assertEqual(sum(p.numel() for p in enabled.parameters()), sum(p.numel() for p in baseline.parameters()))
        self.assertIsNone(baseline.wfdr_loss)
        output = dict(pred_boxes=torch.tensor([[[.5, .5, .3, .3]]]), pred_corners=torch.randn(1, 1, 132),
                      ref_points=torch.tensor([[[.5, .5, .3, .3]]]), up=torch.tensor([.5]), reg_scale=torch.tensor([4.]))
        targets = [dict(labels=torch.tensor([0]), boxes=torch.tensor([[.5, .5, .3, .3]]))]
        indices = [(torch.tensor([0]), torch.tensor([0]))]
        old = baseline.loss_local(output, targets, indices, 1)
        zero = criterion(strength=0.).loss_local(output, targets, indices, 1)
        self.assertEqual(zero['loss_wfdr'].item(), 0.)
        torch.testing.assert_close(zero['loss_fgl'], old['loss_fgl'], rtol=0, atol=0)

    def test_new_configs_change_only_wfdr_and_explicit_combination_flags(self):
        folder = ROOT / 'configs/deim_dfine'
        baseline = load_config(str(folder / 'deim_hgnetv2_m_sar_aircraft_58e_b28_old_server.yml'), {})
        outputs = {baseline['output_dir']}
        for prefix, combined in [('wfdr', False), ('pgd_rem_clcr_lite_wfdr', True)]:
            path = folder / f'deim_hgnetv2_m_sar_aircraft_{prefix}_58e_b28_old_server.yml'
            configured = YAMLConfig(str(path)).criterion
            self.assertIsInstance(configured.wfdr_loss, WassersteinFDRLoss)
            self.assertEqual(configured.wfdr_loss.loss_weight, .5)
            config = load_config(str(path), {})
            settings = config['DEIMCriterion']
            self.assertTrue(settings.pop('use_wfdr'))
            self.assertEqual(settings.pop('wfdr_weight'), .5)
            self.assertEqual(settings['weight_dict'].pop('loss_wfdr'), settings['weight_dict']['loss_fgl'])
            if combined:
                self.assertTrue(config['HybridEncoder'].pop('use_pgd_rem'))
                self.assertEqual(config['HybridEncoder'].pop('pgd_rem_prior_channels'), 64)
                self.assertFalse(config['DFINETransformer'].pop('use_clcr'))
                self.assertTrue(config['DFINETransformer'].pop('use_clcr_lite'))
            self.assertTrue((ROOT / config['experiment_protocol_document']).is_file())
            self.assertNotIn(config['output_dir'], outputs)
            outputs.add(config['output_dir'])
            reference = copy.deepcopy(baseline)
            for item in [config, reference]:
                item.pop('__include__', None)
                item.pop('output_dir')
                item.pop('experiment_protocol_document')
            self.assertEqual(config, reference)


if __name__ == '__main__':
    unittest.main()
