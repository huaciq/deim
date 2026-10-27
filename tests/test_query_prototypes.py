"""MPQ objective, gradient isolation, decoder routing, and protocol invariants."""
import copy
from unittest import TestCase, main

import torch
import torch.nn.functional as F

from engine.core.yaml_utils import load_config
from engine.deim.query_prototypes import QueryPrototypeHead, MatchedQueryPrototypeLoss
from engine.deim.dfine_decoder import DFINETransformer
from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.matcher import HungarianMatcher
from engine.core import YAMLConfig


def fixture():
    # Two class-0 samples of different quality, class 1, and a rejected class 2.
    boxes = torch.tensor([[[.2, .2, .2, .2], [.5, .5, .16, .16],
                           [.8, .8, .2, .2], [.1, .9, .02, .02]]], requires_grad=True)
    targets = [dict(labels=torch.tensor([0, 0, 1, 2]),
                    boxes=torch.tensor([[.2, .2, .2, .2], [.5, .5, .2, .2],
                                        [.8, .8, .2, .2], [.8, .1, .2, .2]]))]
    indices = [(torch.arange(4), torch.arange(4))]
    return boxes, targets, indices


class QueryPrototypeTests(TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(6)

    def test_head_matches_cosine_and_soft_assignment_definitions_in_fp32(self):
        q = torch.randn(2, 5, 8)
        for counts in [[1, 1, 1], [1, 1, 4]]:
            head = QueryPrototypeHead(8, 3, 4, counts)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                output = head(q)
            z = F.normalize(head.projection(head.norm(q)), dim=-1)
            similarities = z @ F.normalize(head.prototypes, dim=-1).T
            expected = []
            for r in similarities.split(counts, dim=-1):
                expected.append(((r / .1).softmax(-1) * r).sum(-1) / .1)
            torch.testing.assert_close(output['pred_mpq_logits'], torch.stack(expected, -1))
            self.assertEqual(output['pred_mpq_logits'].dtype, torch.float32)
            self.assertEqual(output['mpq_assignments'].shape, (2, 5, sum(counts)))

    def test_class_means_quality_weights_and_box_gradient_isolation(self):
        head = QueryPrototypeHead(8, 3, 4)
        query = torch.randn(1, 4, 8, requires_grad=True)
        output = head(query)
        boxes, targets, indices = fixture()
        output['pred_boxes'] = boxes
        loss = MatchedQueryPrototypeLoss(3, min_iou=.6)
        value = loss(output, targets, indices, epoch=54)
        ce = F.cross_entropy(output['pred_mpq_logits'][0], targets[0]['labels'], reduction='none')
        # Second IoU = .16^2 / .2^2 = .64, therefore weight = .64^2.
        expected = ((ce[0] + ce[1] * .64 ** 2) / (1 + .64 ** 2) + ce[2]) / 2
        torch.testing.assert_close(value, expected)
        value.backward()
        self.assertIsNone(boxes.grad)
        self.assertGreater(query.grad.abs().sum().item(), 0)
        for p in head.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        self.assertEqual(float(loss.last_metrics['mpq_valid_c0']), 2)
        self.assertEqual(float(loss.last_metrics['mpq_valid_c1']), 1)
        self.assertEqual(float(loss.last_metrics['mpq_valid_c2']), 0)

    def test_inactive_rejected_and_empty_matches_keep_all_gradients_connected(self):
        for mode in ['inactive', 'rejected', 'empty']:
            head = QueryPrototypeHead(8, 3, 4, [1, 1, 4])
            query = torch.randn(1, 4, 8, requires_grad=True)
            output = head(query)
            boxes, targets, indices = fixture()
            output['pred_boxes'] = boxes
            if mode == 'rejected':
                targets[0]['boxes'][:, :2] = -1
            if mode == 'empty':
                targets = [dict(labels=torch.empty(0, dtype=torch.long), boxes=torch.empty(0, 4))]
                indices = [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))]
            loss = MatchedQueryPrototypeLoss(3)
            value = loss(output, targets, indices, epoch=0 if mode == 'inactive' else 54)
            self.assertEqual(value.item(), 0)
            value.backward()
            for p in head.parameters():
                self.assertIsNotNone(p.grad, mode)
                self.assertEqual(p.grad.abs().sum().item(), 0, mode)
            self.assertEqual(query.grad.abs().sum().item(), 0)

    def test_activation_and_five_epoch_ramp_follow_logical_epoch(self):
        loss = MatchedQueryPrototypeLoss(3)
        self.assertEqual(loss.ramp(48, 99, 100), 0)
        self.assertEqual(loss.ramp(49), 0)
        self.assertAlmostEqual(loss.ramp(49, 50, 100), .1)
        self.assertAlmostEqual(loss.ramp(51, 50, 100), .5)
        self.assertEqual(loss.ramp(54), 1)
        self.assertEqual(loss.ramp(90), 1)

    def test_optional_head_preserves_seed_detector_outputs_and_dn_routing(self):
        kwargs = dict(num_classes=7, hidden_dim=32, feat_channels=[32] * 3,
                      num_queries=5, num_layers=2, nhead=4, num_points=2,
                      dim_feedforward=64, reg_max=8, num_denoising=4,
                      eval_spatial_size=None)
        torch.manual_seed(30)
        baseline = DFINETransformer(**copy.deepcopy(kwargs))
        original_rng = torch.get_rng_state()
        torch.manual_seed(30)
        adapted = DFINETransformer(**copy.deepcopy(kwargs), use_mpq=True, mpq_projection_dim=8)
        torch.testing.assert_close(torch.get_rng_state(), original_rng, rtol=0, atol=0)
        features = [torch.randn(2, 32, s, s) for s in [4, 2, 1]]
        targets = [dict(labels=torch.tensor([3]), boxes=torch.tensor([[.5, .5, .3, .3]])) for _ in range(2)]
        for training in [True, False]:
            baseline.train(training); adapted.train(training)
            torch.manual_seed(31)
            reference = baseline(features, targets if training else None)
            torch.manual_seed(31)
            actual = adapted(features, targets if training else None)
            for name in ['pred_logits', 'pred_boxes', 'pred_corners']:
                torch.testing.assert_close(reference[name], actual[name], rtol=0, atol=0)
            if training:
                self.assertEqual(actual['pred_mpq_logits'].shape, (2, 5, 7))
                self.assertNotIn('pred_mpq_logits', actual['dn_outputs'][0])
            else:
                self.assertNotIn('pred_mpq_logits', actual)
        adapted.convert_to_deploy()
        self.assertIsNone(adapted.mpq_head)

    def test_two_configs_keep_frozen_protocol_and_isolate_other_centers(self):
        baseline = load_config('configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_102e_b28.yml', {})
        single = load_config('configs/deim_dfine/sar_aircraft_mpq/mpq_single.yml', {})
        other4 = load_config('configs/deim_dfine/sar_aircraft_mpq/mpq_other4.yml', {})
        for key, value in baseline.items():
            if key not in ['__include__', 'output_dir', 'DFINETransformer', 'DEIMCriterion', 'HybridEncoder']:
                self.assertEqual(single[key], value, key)
        for section in ['DFINETransformer', 'DEIMCriterion', 'HybridEncoder']:
            for key, value in baseline[section].items():
                if key == 'weight_dict':
                    for k, v in value.items(): self.assertEqual(single[section][key][k], v)
                else:
                    self.assertEqual(single[section][key], value)
        self.assertEqual(single['DFINETransformer']['mpq_prototype_counts'], [1] * 7)
        self.assertEqual(other4['DFINETransformer']['mpq_prototype_counts'], [1] * 6 + [4])
        other4['DFINETransformer']['mpq_prototype_counts'] = [1] * 7
        for key in single:
            if key not in ['__include__', 'output_dir']:
                self.assertEqual(single[key], other4[key], key)

    def test_actual_criterion_keeps_detector_losses_and_optimizes_auxiliary_head(self):
        model = DFINETransformer(num_classes=7, hidden_dim=32, feat_channels=[32] * 3,
                                 num_queries=5, num_layers=2, nhead=4, num_points=2,
                                 dim_feedforward=64, reg_max=8, num_denoising=4,
                                 eval_spatial_size=None, use_mpq=True, mpq_projection_dim=8)
        features = [torch.randn(2, 32, s, s) for s in [4, 2, 1]]
        targets = [dict(labels=torch.tensor([3]), boxes=torch.tensor([[.5, .5, .3, .3]])) for _ in range(2)]
        output = model(features, targets)
        for b, target in enumerate(targets):
            target['boxes'] = output['pred_boxes'][b, :1].detach().clone()
        weights = dict(loss_mal=1, loss_bbox=5, loss_giou=2, loss_fgl=.15, loss_ddf=1.5)
        def criterion(enabled):
            return DEIMCriterion(
                HungarianMatcher(dict(cost_class=2, cost_bbox=5, cost_giou=2), use_focal_loss=True),
                dict(weights, loss_mpq=.05), ['mal', 'boxes', 'local'],
                num_classes=7, reg_max=8, use_mpq=enabled)
        reference = criterion(False)(output, targets)
        actual = criterion(True)(output, targets, epoch=54)
        self.assertGreater(actual['loss_mpq'].item(), 0)
        for name, value in reference.items():
            torch.testing.assert_close(actual[name], value)
        groups = YAMLConfig.get_optim_params(dict(type='AdamW', params=[dict(params='.*norm.*', weight_decay=0)]), model)
        parameter_ids = {id(p) for group in groups for p in group['params']}
        self.assertTrue(all(id(p) in parameter_ids for p in model.mpq_head.parameters()))
        sum(actual.values()).backward()
        for p in model.mpq_head.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())


if __name__ == '__main__':
    main()
