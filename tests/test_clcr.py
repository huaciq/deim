"""CLCR mathematics, local geometry, decoder/DN routing and training gradients."""

import copy
import math
from pathlib import Path
import unittest

import torch
from torch import nn
import torch.nn.functional as F

from engine.core import YAMLConfig
from engine.core.yaml_utils import load_config
from engine.deim.dfine_decoder import CLCR, DFINETransformer, LQE
from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.hybrid_encoder import HybridEncoder
from engine.deim.matcher import HungarianMatcher
from engine.solver.det_engine import _module_training_metrics


ROOT = Path(__file__).resolve().parents[1]
DECODER_ARGS = dict(num_classes=7, hidden_dim=32, feat_channels=[32] * 3,
                    num_queries=5, num_layers=3, eval_idx=1, nhead=4,
                    num_points=2, dim_feedforward=64, reg_max=8,
                    num_denoising=4, eval_spatial_size=None)


def decoder(clcr=False):
    return DFINETransformer(**copy.deepcopy(DECODER_ARGS), use_clcr=clcr, clcr_hidden_dim=16)


def fixture():
    features = [torch.randn(2, 32, h, w) for h, w in [(8, 12), (4, 6), (2, 3)]]
    targets = [
        dict(labels=torch.tensor([3]), boxes=torch.tensor([[.4, .5, .3, .4]])),
        dict(labels=torch.tensor([1, 6]), boxes=torch.tensor([[.3, .4, .2, .3], [.7, .6, .3, .2]])),
    ]
    return features, targets


class ClcrTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(211)

    def assert_tree_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_tree_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for value, reference in zip(actual, expected):
                self.assert_tree_equal(value, reference)
        else:
            self.assertEqual(actual, expected)

    def test_competition_matches_all_four_definitions_including_ties(self):
        probabilities = torch.tensor([[[.7, .6, .3, .2, .1, .05, .05], [.2] * 7]])
        scores = torch.logit(probabilities).requires_grad_()
        actual = CLCR.competition_statistics(scores)
        margins = []
        for index in range(7):
            others = torch.cat((probabilities[..., :index], probabilities[..., index + 1:]), dim=-1)
            margins.append(probabilities[..., index] - others.max(-1).values)
        normalized = probabilities / probabilities.sum(-1, keepdim=True)
        entropy = -(normalized * normalized.log()).sum(-1, keepdim=True) / math.log(7)
        expected = torch.stack((probabilities, torch.stack(margins, -1),
                                probabilities / (probabilities.sum(-1, keepdim=True) + 1e-6),
                                entropy.expand_as(probabilities)), dim=-1)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.shape, (1, 2, 7, 4))
        self.assertGreater(actual[0, 0, 0, 1].item(), 0.)
        self.assertLess(actual[0, 0, 1, 1].item(), 0.)
        torch.testing.assert_close(actual[0, 1, :, 1], torch.zeros(7))
        actual.square().sum().backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertGreater(scores.grad.abs().sum().item(), 0.)

    def test_entropy_and_gradients_remain_finite_for_extreme_amp_logits(self):
        scores = torch.tensor([[[-1000.] * 7, [1000., -1000., -1000., -1000., -1000., -1000., -1000.]]],
                              dtype=torch.bfloat16, requires_grad=True)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            statistics = CLCR.competition_statistics(scores)
        self.assertEqual(statistics.dtype, torch.float32)
        self.assertTrue(torch.isfinite(statistics).all())
        torch.testing.assert_close(statistics[0, 0, :, 3], torch.ones(7))
        torch.testing.assert_close(statistics[0, 1, :, 3], torch.zeros(7))
        statistics.sum().backward()
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_lqe_keeps_scalar_geometry_and_uses_raw_class_specific_competition(self):
        lqe = LQE(4, 64, 2, 8)
        lqe.clcr = CLCR(8, 20)
        # Make the learned residual exactly the class margin, keeping a nonzero LQE.
        lqe.clcr.calibration = nn.Linear(40, 1, bias=False)
        with torch.no_grad():
            lqe.reg_conf.layers[-1].bias.fill_(.7)
            lqe.clcr.beta.fill_(.3)
            lqe.clcr.calibration.weight.zero_()
            lqe.clcr.calibration.weight[0, 1] = 1.
        scores = torch.tensor([[[1.2, .8, -.2, -.4, -1., -2., -3.]]])
        corners = torch.randn(1, 1, 36)
        boxes = torch.tensor([[[.5, .5, .4, .3]]])
        feature = torch.randn(1, 8, 8, 12)
        records = []
        hook = lqe.clcr.calibration.register_forward_pre_hook(lambda layer, args: records.append(args[0]))
        try:
            result = lqe(scores, corners, boxes, feature)
        finally:
            hook.remove()
        margin = CLCR.competition_statistics(scores)[..., 1]
        torch.testing.assert_close(result, scores + .7 + .3 * margin)
        self.assertGreater((result - scores - .7)[0, 0, 0].item(), 0.)
        self.assertLess((result - scores - .7)[0, 0, 1].item(), 0.)
        top = corners.reshape(1, 1, 4, 9).softmax(-1).topk(4, -1).values
        edge = torch.cat((top, top.mean(-1, keepdim=True)), -1).reshape(1, 1, 20)
        torch.testing.assert_close(records[0][..., -20:], edge.unsqueeze(2).expand(-1, -1, 7, -1))
        permutation = torch.tensor([3, 0, 6, 2, 5, 1, 4])
        torch.testing.assert_close(lqe(scores[..., permutation], corners, boxes, feature),
                                   result[..., permutation])

    def test_core_ring_geometry_pooling_and_batch_isolation(self):
        module = CLCR(2, 20)
        height, width = 80, 120
        yy, xx = torch.meshgrid((torch.arange(height) + .5) / height,
                                (torch.arange(width) + .5) / width, indexing='ij')
        compact = ((xx > .3) & (xx < .7) & (yy > .3) & (yy < .7)).float()
        feature = torch.stack((compact, compact * 2)).unsqueeze(0).repeat(2, 1, 1, 1)
        feature[1] += 10.
        boxes = torch.tensor([[[.5, .5, .5, .5]], [[.5, .5, .5, .5]]])
        inside, ring = module.pool_local_evidence(boxes, feature)
        torch.testing.assert_close(inside, torch.tensor([[[1., 2.]], [[11., 12.]]]))
        torch.testing.assert_close(ring, torch.tensor([[[0., 0.]], [[10., 10.]]]))
        self.assertEqual(module.sample_offsets.shape, (21, 2))
        self.assertEqual(torch.unique(module.sample_offsets[9:], dim=0).shape[0], 12)
        torch.testing.assert_close(module.sample_offsets[:9].abs().max(), torch.tensor(.375))
        torch.testing.assert_close(module.sample_offsets[9:].abs().amax(-1), torch.full((12,), .525))

        # Linear spatial channels verify x/y ordering and independent query centers.
        ramp = torch.stack((xx, yy)).unsqueeze(0)
        multiple = torch.tensor([[[.3, .6, .2, .3], [.7, .4, .3, .2]]])
        for pooled in module.pool_local_evidence(multiple, ramp):
            torch.testing.assert_close(pooled, multiple[..., :2])
        border_boxes = torch.tensor([[[0., .5, .3, .3], [1.1, 1.1, .2, .2]]])
        for pooled in module.pool_local_evidence(border_boxes, torch.ones_like(ramp) * 3):
            torch.testing.assert_close(pooled, torch.full((1, 2, 2), 3.))

    def test_zero_beta_preserves_lqe_then_all_evidence_paths_learn_with_amp(self):
        for amp, dtype in [(False, torch.float32), (True, torch.bfloat16)]:
            with self.subTest(amp=amp):
                lqe = LQE(4, 64, 2, 8)
                with torch.no_grad():
                    lqe.reg_conf.layers[-1].weight.normal_(std=.1)
                reference = copy.deepcopy(lqe)
                lqe.clcr = CLCR(8, 20, hidden_dim=16)
                scores = torch.randn(2, 3, 7, dtype=dtype, requires_grad=True)
                corners = torch.randn(2, 3, 36, dtype=dtype, requires_grad=True)
                boxes = torch.tensor([[[.45, .55, .4, .3]] * 3] * 2, dtype=dtype, requires_grad=True)
                feature = torch.randn(2, 8, 12, 16, dtype=dtype, requires_grad=True)
                optimizer = torch.optim.SGD(lqe.parameters(), lr=.1)
                with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
                    initial = lqe(scores, corners, boxes, feature)
                    original = reference(scores, corners)
                torch.testing.assert_close(initial, original, rtol=0, atol=0)
                self.assertEqual(initial.dtype, dtype)
                initial.float().square().mean().backward()
                self.assertNotEqual(lqe.clcr.beta.grad.item(), 0.)
                for parameter in lqe.clcr.parameters():
                    self.assertIsNotNone(parameter.grad)
                optimizer.step()
                self.assertNotEqual(lqe.clcr.beta.item(), 0.)
                optimizer.zero_grad(set_to_none=True)
                for value in [scores, corners, boxes, feature]:
                    value.grad = None
                with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
                    result = lqe(scores, corners, boxes, feature)
                result.float().square().mean().backward()
                for name, parameter in lqe.clcr.named_parameters():
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                for branch in [lqe.clcr.local_mlp, lqe.clcr.calibration]:
                    self.assertGreater(sum(p.grad.abs().sum().item() for p in branch.parameters()), 0.)
                for value in [scores, corners, boxes, feature]:
                    self.assertTrue(torch.isfinite(value.grad).all())
                    self.assertGreater(value.grad.abs().sum().item(), 0.)

    def test_decoder_preserves_initialization_outputs_aux_and_denoising_paths(self):
        torch.manual_seed(223)
        baseline = decoder()
        baseline_rng = torch.get_rng_state()
        torch.manual_seed(223)
        adapted = decoder(True)
        torch.testing.assert_close(torch.get_rng_state(), baseline_rng, rtol=0, atol=0)
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(adapted.state_dict()[name], value, rtol=0, atol=0)
        missing, unexpected = adapted.load_state_dict(baseline.state_dict(), strict=False)
        self.assertTrue(missing)
        self.assertTrue(all('.clcr.' in name for name in missing))
        self.assertEqual(unexpected, [])
        features, targets = fixture()
        for training in [True, False]:
            baseline.train(training)
            adapted.train(training)
            torch.manual_seed(227)
            expected = baseline(features, targets if training else None)
            torch.manual_seed(227)
            actual = adapted(features, targets if training else None)
            self.assert_tree_equal(actual, expected)

    def test_every_active_layer_uses_current_boxes_and_stride8_and_changes_only_logits(self):
        model = decoder(True)
        features, targets = fixture()
        for training in [True, False]:
            model.train(training)
            for lqe in model.decoder.lqe_layers:
                lqe.clcr.beta.data.zero_()
            torch.manual_seed(229)
            reference = model(features, targets if training else None)
            records, handles = {}, []
            for index, lqe in enumerate(model.decoder.lqe_layers):
                with torch.no_grad():
                    lqe.clcr.beta.fill_(.2)
                    lqe.clcr.calibration.layers[-1].weight.zero_()
                    lqe.clcr.calibration.layers[-1].bias.fill_(index + 1.)
                handles.append(lqe.clcr.register_forward_pre_hook(
                    lambda layer, args, index=index: records.__setitem__(index, args)))
            try:
                torch.manual_seed(229)
                actual = model(features, targets if training else None)
            finally:
                for handle in handles:
                    handle.remove()
            selected = model.num_layers - 1 if training else model.eval_idx
            self.assertEqual(set(records), set(range(model.num_layers)) if training else {selected})
            for key in ['pred_boxes', 'pred_corners', 'ref_points']:
                self.assert_tree_equal(actual[key], reference[key])
            torch.testing.assert_close(actual['pred_logits'] - reference['pred_logits'],
                                       torch.full_like(actual['pred_logits'], .2 * (selected + 1)))
            for index, args in records.items():
                torch.testing.assert_close(args[3], features[0], rtol=0, atol=0)
                if training:
                    regular = actual if index == selected else actual['aux_outputs'][index]
                    expected_boxes = torch.cat((actual['dn_outputs'][index]['pred_boxes'], regular['pred_boxes']), 1)
                else:
                    expected_boxes = actual['pred_boxes']
                torch.testing.assert_close(args[2], expected_boxes, rtol=0, atol=0)
            if training:
                for group in ['pre_outputs', 'dn_pre_outputs', 'enc_aux_outputs']:
                    self.assert_tree_equal(actual[group], reference[group])
                for index, (new, old) in enumerate(zip(actual['dn_outputs'], reference['dn_outputs'])):
                    torch.testing.assert_close(new['pred_logits'] - old['pred_logits'],
                                               torch.full_like(new['pred_logits'], .2 * (index + 1)))
            owner = nn.Module()
            owner.decoder = model
            metrics = _module_training_metrics(owner)
            for index in range(model.num_layers):
                self.assertAlmostEqual(metrics[f'clcr_beta_l{index}'], .2, places=6)

    def test_existing_detection_loss_trains_clcr_and_combined_pgd_rem(self):
        owner = nn.Module()
        owner.encoder = HybridEncoder(in_channels=[32, 64, 128], hidden_dim=32, nhead=4,
                                      dim_feedforward=64, depth_mult=.34,
                                      use_pgd_rem=True, pgd_rem_prior_channels=8)
        owner.decoder = decoder(True)
        features = [torch.randn(2, channels, h, w) for channels, h, w in
                    [(32, 8, 12), (64, 4, 6), (128, 2, 3)]]
        _, targets = fixture()
        criterion = DEIMCriterion(
            HungarianMatcher(dict(cost_class=2, cost_bbox=5, cost_giou=2), use_focal_loss=True),
            dict(loss_mal=1, loss_bbox=5, loss_giou=2, loss_fgl=.15, loss_ddf=1.5),
            ['mal', 'boxes', 'local'], num_classes=7, reg_max=8)
        groups = YAMLConfig.get_optim_params(dict(type='AdamW', params=[dict(params='.*norm.*', weight_decay=0)]), owner)
        optimizer = torch.optim.AdamW(groups, lr=1e-3)
        parameter_ids = {id(p) for group in groups for p in group['params']}
        heads = [lqe.clcr for lqe in owner.decoder.decoder.lqe_layers]
        self.assertTrue(all(id(p) in parameter_ids for head in heads for p in head.parameters()))
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = owner.decoder(owner.encoder(features), targets)
            losses = criterion(output, targets)
            self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
            self.assertTrue(any(name.endswith('_dn_0') for name in losses))
            self.assertTrue(any(name.endswith('_aux_0') for name in losses))
            self.assertFalse(any('clcr' in name for name in losses))
            sum(losses.values()).backward()
            for head in heads:
                for name, parameter in head.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertNotEqual(head.beta.grad.item(), 0.)
                if step:
                    self.assertGreater(head.local_mlp.layers[0].weight.grad.abs().sum().item(), 0.)
            self.assertNotEqual(owner.encoder.pgd_rem.alpha.grad.item(), 0.)
            if step:
                self.assertGreater(owner.encoder.pgd_rem.prior_gate.reduce.weight.grad.abs().sum().item(), 0.)
            optimizer.step()

    def test_checkpoint_roundtrip_and_deploy_keep_active_calibration(self):
        model = decoder(True).eval()
        for lqe in model.decoder.lqe_layers:
            lqe.clcr.beta.data.fill_(.3)
        features, _ = fixture()
        with torch.no_grad():
            expected = model(features)
        restored = decoder(True).eval()
        restored.load_state_dict(model.state_dict())
        with torch.no_grad():
            self.assert_tree_equal(restored(features), expected)
        restored.convert_to_deploy()
        restored.decoder.convert_to_deploy()
        self.assertEqual(len(restored.decoder.lqe_layers), restored.eval_idx + 1)
        self.assertAlmostEqual(restored.decoder.lqe_layers[-1].clcr.beta.item(), .3, places=6)
        with torch.no_grad():
            actual = restored(features)
        for key in ['pred_logits', 'pred_boxes', 'pred_corners']:
            torch.testing.assert_close(actual[key], expected[key])

    def test_configs_keep_the_same_protocol_and_isolate_single_and_combined_modules(self):
        directory = ROOT / 'configs/deim_dfine'
        baseline = load_config(str(directory / 'deim_hgnetv2_m_sar_aircraft_58e_b28_old_server.yml'), {})
        outputs = {baseline['output_dir']}
        for suffix, combined in [('clcr', False), ('pgd_rem_clcr', True)]:
            config = load_config(str(directory / f'deim_hgnetv2_m_sar_aircraft_{suffix}_58e_b28_old_server.yml'), {})
            self.assertTrue(config['DFINETransformer'].pop('use_clcr'))
            self.assertEqual(config['DFINETransformer'].pop('clcr_hidden_dim'), 64)
            if combined:
                self.assertTrue(config['HybridEncoder'].pop('use_pgd_rem'))
                self.assertEqual(config['HybridEncoder'].pop('pgd_rem_prior_channels'), 64)
            self.assertEqual(config['epoches'], 58)
            self.assertEqual(config['train_dataloader']['total_batch_size'], 28)
            self.assertTrue((ROOT / config['experiment_protocol_document']).is_file())
            self.assertNotIn(config['output_dir'], outputs)
            outputs.add(config['output_dir'])
            reference = copy.deepcopy(baseline)
            for candidate in [config, reference]:
                candidate.pop('__include__', None)
                candidate.pop('output_dir')
                candidate.pop('experiment_protocol_document')
            self.assertEqual(config, reference)


if __name__ == '__main__':
    unittest.main()
