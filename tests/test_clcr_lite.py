"""CLCR-Lite statistics, learnable zero residual, sampling removal and shared routing."""

import copy
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn

from engine.core import YAMLConfig
from engine.core.yaml_utils import load_config
from engine.deim.dfine_decoder import BoxLocalEvidenceSampler, CLCR, CLCRLite, DFINETransformer, LQE
from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.hybrid_encoder import HybridEncoder
from engine.deim.matcher import HungarianMatcher
from engine.solver.det_engine import _module_training_metrics
from engine.optim.ema import ModelEMA


ROOT = Path(__file__).resolve().parents[1]
DECODER_ARGS = dict(num_classes=7, hidden_dim=32, feat_channels=[32] * 3,
                    num_queries=5, num_layers=4, eval_idx=1, nhead=4,
                    num_points=2, dim_feedforward=64, reg_max=8,
                    num_denoising=4, eval_spatial_size=None)


def decoder(lite=True, sdcp=False):
    return DFINETransformer(**copy.deepcopy(DECODER_ARGS), use_clcr_lite=lite, use_sdcp=sdcp)


def fixture():
    features = [torch.randn(2, 32, h, w) for h, w in [(8, 12), (4, 6), (2, 3)]]
    targets = [dict(labels=torch.tensor([3]), boxes=torch.tensor([[.4, .5, .3, .4]])),
               dict(labels=torch.tensor([1, 6]), boxes=torch.tensor([[.3, .4, .2, .3], [.7, .6, .3, .2]]))]
    return features, targets


def edge_statistics(corners):
    top = corners.reshape(*corners.shape[:2], 4, 9).softmax(-1).topk(4, -1).values
    return torch.cat((top, top.mean(-1, keepdim=True)), -1).flatten(-2)


class ClcrLiteTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(281)

    def assert_tree_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for name in expected:
                self.assert_tree_equal(actual[name], expected[name])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for value, reference in zip(actual, expected):
                self.assert_tree_equal(value, reference)
        else:
            self.assertEqual(actual, expected)

    def test_eight_statistics_match_competition_four_edge_summary_and_all_scale_priors(self):
        p = torch.tensor([[[.7, .6, .3, .2, .1, .05, .05], [.2] * 7]])
        corners = torch.randn(1, 2, 36)
        edge = edge_statistics(corners)
        boxes = torch.tensor([[[.5, .5, .2, .4], [.3, .6, .7, .5]]])
        actual = CLCRLite.input_statistics(torch.logit(p), edge, boxes)
        expected = torch.empty(1, 2, 7, 8)
        normalized = p / p.sum(-1, keepdim=True)
        entropy = -(normalized * normalized.log()).sum(-1) / math.log(7)
        for query in range(2):
            means = [edge[0, query, index * 5 + 4].item() for index in range(4)]
            avg = sum(means) / 4
            std = math.sqrt(sum((value - avg) ** 2 for value in means) / 4)
            w, h = boxes[0, query, 2:].tolist()
            for category in range(7):
                other = max(p[0, query, index].item() for index in range(7) if index != category)
                expected[0, query, category] = torch.tensor([
                    p[0, query, category], p[0, query, category] - other, entropy[0, query],
                    avg, std, math.log(w), math.log(h), math.log(w / h)])
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual[0, 1, :, 1], torch.zeros(7))
        self.assertEqual(actual.shape[-1], 8)
        # log(area) in the compact formula is recoverable from the two log sizes.
        torch.testing.assert_close(actual[..., 5] + actual[..., 6],
                                   boxes[..., 2:].prod(-1).log()[..., None].expand(-1, -1, 7))

    def test_extreme_entropy_equal_edges_and_tiny_boxes_remain_finite_under_amp(self):
        scores = torch.tensor([[[-1000.] * 7, [1000., -1000., -1000., -1000., -1000., -1000., -1000.]]],
                              dtype=torch.bfloat16, requires_grad=True)
        corners = torch.zeros(1, 2, 36, dtype=torch.bfloat16, requires_grad=True)
        boxes = torch.tensor([[[.5, .5, 0., 1e-12], [.5, .5, .4, .2]]],
                             dtype=torch.bfloat16, requires_grad=True)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            statistics = CLCRLite.input_statistics(scores, edge_statistics(corners), boxes)
        self.assertEqual(statistics.dtype, torch.float32)
        self.assertTrue(torch.isfinite(statistics).all())
        torch.testing.assert_close(statistics[0, 0, :, 2], torch.ones(7))
        torch.testing.assert_close(statistics[0, 1, :, 2], torch.zeros(7))
        torch.testing.assert_close(statistics[..., 4], torch.zeros(1, 2, 7))
        statistics.square().mean().backward()
        for value in [scores, corners, boxes]:
            self.assertTrue(torch.isfinite(value.grad).all())

    def test_exact_architecture_parameter_budget_and_shared_head_across_four_layers(self):
        lite = CLCRLite()
        self.assertIsInstance(lite.calibration[0], nn.Linear)
        self.assertIsInstance(lite.calibration[1], nn.SiLU)
        self.assertIsInstance(lite.calibration[2], nn.Linear)
        self.assertEqual(lite.calibration[0].weight.shape, (16, 8))
        self.assertEqual(lite.calibration[2].weight.shape, (1, 16))
        self.assertEqual(sum(p.numel() for p in lite.parameters()), 162)
        self.assertEqual(lite.beta.item(), 1.)
        self.assertGreater(lite.calibration[0].weight.abs().sum().item(), 0.)
        self.assertEqual(lite.calibration[2].weight.abs().sum().item(), 0.)
        self.assertEqual(lite.calibration[2].bias.item(), 0.)
        self.assertFalse(hasattr(lite, 'sample_offsets'))
        self.assertFalse(hasattr(lite, 'local_mlp'))
        model, baseline = decoder(), decoder(False)
        self.assertIsInstance(model.decoder.clcr_lite, CLCRLite)
        self.assertEqual(len([key for key in model.state_dict() if '.clcr_lite.' in key]), 5)
        self.assertTrue(all(lqe.clcr is None for lqe in model.decoder.lqe_layers))
        self.assertIsNone(model.decoder.sdcp_sampler)
        self.assertEqual(sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in baseline.parameters()), 162)
        self.assertEqual(sum(p.numel() for p in CLCR(256, 20).parameters()) * 4, 146248)
        with self.assertRaisesRegex(ValueError, 'alternative'):
            DFINETransformer(**copy.deepcopy(DECODER_ARGS), use_clcr=True, use_clcr_lite=True)

    def test_lqe_keeps_all_twenty_edge_inputs_and_adds_class_specific_raw_logit_correction(self):
        lqe = LQE(4, 64, 2, 8)
        lite = CLCRLite()
        lite.calibration = nn.Linear(8, 1, bias=False)
        with torch.no_grad():
            lqe.reg_conf.layers[-1].bias.fill_(.7)
            lite.beta.fill_(.3)
            lite.calibration.weight.zero_()
            lite.calibration.weight[0, 1] = 1.
        scores = torch.randn(1, 2, 7)
        corners = torch.randn(1, 2, 36)
        boxes = torch.tensor([[[.4, .6, .3, .2], [.5, .5, .7, .6]]])
        recorded = []
        hook = lqe.reg_conf.register_forward_pre_hook(lambda layer, args: recorded.append(args[0]))
        try:
            with patch('torch.nn.functional.grid_sample', side_effect=AssertionError('Lite must not sample')):
                actual = lqe(scores, corners, boxes, clcr_lite=lite)
        finally:
            hook.remove()
        torch.testing.assert_close(recorded[0], edge_statistics(corners))
        self.assertEqual(recorded[0].shape[-1], 20)
        margin = CLCR.competition_statistics(scores)[..., 1]
        torch.testing.assert_close(actual, scores + .7 + .3 * margin)
        order = torch.tensor([2, 1, 5, 4, 6, 3, 0])
        torch.testing.assert_close(lqe(scores[..., order], corners, boxes, clcr_lite=lite), actual[..., order])

    def test_zero_output_starts_learning_then_all_statistics_and_layers_receive_gradients(self):
        for amp, dtype in [(False, torch.float32), (True, torch.bfloat16)]:
            with self.subTest(amp=amp):
                module = CLCRLite()
                scores = torch.randn(2, 3, 7, dtype=dtype, requires_grad=True)
                corners = torch.randn(2, 3, 36, dtype=dtype, requires_grad=True)
                boxes = torch.tensor([[[.5, .5, .4, .3]] * 3] * 2, dtype=dtype, requires_grad=True)
                optimizer = torch.optim.SGD(module.parameters(), lr=.1)
                for step in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    for value in [scores, corners, boxes]:
                        value.grad = None
                    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
                        result = module(scores, edge_statistics(corners), boxes)
                    self.assertEqual(result.dtype, dtype)
                    if step == 0:
                        torch.testing.assert_close(result, torch.zeros_like(scores), rtol=0, atol=0)
                    (result.float() - 1.).square().mean().backward()
                    self.assertGreater(module.calibration[-1].weight.grad.abs().sum().item(), 0.)
                    for name, parameter in module.named_parameters():
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    if step:
                        self.assertNotEqual(module.beta.grad.item(), 0.)
                        self.assertGreater(module.calibration[0].weight.grad.abs().sum().item(), 0.)
                        for value in [scores, corners, boxes]:
                            self.assertGreater(value.grad.abs().sum().item(), 0.)
                            self.assertTrue(torch.isfinite(value.grad).all())
                    optimizer.step()

    def test_initialization_outputs_and_rng_match_baseline_with_no_local_sampling(self):
        torch.manual_seed(283)
        baseline = decoder(False)
        expected_rng = torch.get_rng_state()
        torch.manual_seed(283)
        adapted = decoder()
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(adapted.state_dict()[name], value, rtol=0, atol=0)
        missing, unexpected = adapted.load_state_dict(baseline.state_dict(), strict=False)
        self.assertTrue(missing)
        self.assertTrue(all('.clcr_lite.' in name for name in missing))
        self.assertFalse(unexpected)
        features, targets = fixture()
        for training in [True, False]:
            baseline.train(training)
            adapted.train(training)
            torch.manual_seed(293)
            reference = baseline(features, targets if training else None)
            with patch.object(BoxLocalEvidenceSampler, 'pool_local_evidence', side_effect=AssertionError('unexpected local sampling')):
                torch.manual_seed(293)
                actual = adapted(features, targets if training else None)
            self.assert_tree_equal(actual, reference)

    def test_active_layers_use_current_boxes_including_dn_and_only_change_logits(self):
        model = decoder()
        lite = model.decoder.clcr_lite
        features, targets = fixture()
        for training in [True, False]:
            model.train(training)
            lite.calibration[-1].bias.data.zero_()
            torch.manual_seed(307)
            expected = model(features, targets if training else None)
            lite.calibration[-1].bias.data.fill_(.25)
            records, layer_arguments, handles = [], [], []
            handles.append(lite.register_forward_pre_hook(lambda layer, args: records.append(args)))
            for lqe in model.decoder.lqe_layers:
                handles.append(lqe.register_forward_pre_hook(
                    lambda layer, args, kwargs: layer_arguments.append((args, kwargs)), with_kwargs=True))
            try:
                with patch.object(BoxLocalEvidenceSampler, 'pool_local_evidence', side_effect=AssertionError('unexpected local sampling')):
                    torch.manual_seed(307)
                    actual = model(features, targets if training else None)
            finally:
                for handle in handles:
                    handle.remove()
            self.assertEqual(len(records), model.num_layers if training else 1)
            self.assertTrue(all(args[3] is None and kwargs['clcr_lite'] is lite for args, kwargs in layer_arguments))
            for name in ['pred_boxes', 'pred_corners', 'ref_points']:
                self.assert_tree_equal(actual[name], expected[name])
            torch.testing.assert_close(actual['pred_logits'] - expected['pred_logits'], torch.full_like(actual['pred_logits'], .25))
            for index, args in enumerate(records):
                if training:
                    regular = actual if index == model.num_layers - 1 else actual['aux_outputs'][index]
                    boxes = torch.cat((actual['dn_outputs'][index]['pred_boxes'], regular['pred_boxes']), 1)
                else:
                    boxes = actual['pred_boxes']
                torch.testing.assert_close(args[2], boxes, rtol=0, atol=0)
            if training:
                for name in ['pre_outputs', 'dn_pre_outputs', 'enc_aux_outputs']:
                    self.assert_tree_equal(actual[name], expected[name])
                for new, old in zip(actual['dn_outputs'], expected['dn_outputs']):
                    torch.testing.assert_close(new['pred_logits'] - old['pred_logits'], torch.full_like(new['pred_logits'], .25))

    def test_actual_detection_losses_optimize_shared_head_with_pgd_rem_and_optional_sdcp(self):
        for sdcp in [False, True]:
            with self.subTest(sdcp=sdcp):
                owner = nn.Module()
                owner.encoder = HybridEncoder(in_channels=[32, 64, 128], hidden_dim=32, nhead=4,
                                              dim_feedforward=64, depth_mult=.34, use_pgd_rem=True, pgd_rem_prior_channels=8)
                owner.decoder = decoder(sdcp=sdcp)
                lite = owner.decoder.decoder.clcr_lite
                features = [torch.randn(2, channels, h, w) for channels, h, w in [(32, 8, 12), (64, 4, 6), (128, 2, 3)]]
                _, targets = fixture()
                weights = dict(loss_mal=1, loss_bbox=5, loss_giou=2, loss_fgl=.15, loss_ddf=1.5)
                if sdcp:
                    weights.update(loss_bbox=0, loss_asym_box=5, loss_margin=.1)
                criterion = DEIMCriterion(
                    HungarianMatcher(dict(cost_class=2, cost_bbox=5, cost_giou=2), use_focal_loss=True),
                    weights, ['mal', 'boxes', 'local'], num_classes=7, reg_max=8, use_sdcp=sdcp)
                groups = YAMLConfig.get_optim_params(dict(type='AdamW', params=[dict(params='.*norm.*', weight_decay=0)]), owner)
                optimizer = torch.optim.AdamW(groups, lr=1e-3)
                ids = [id(parameter) for group in groups for parameter in group['params']]
                self.assertTrue(all(ids.count(id(parameter)) == 1 for parameter in lite.parameters()))
                for step in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    output = owner.decoder(owner.encoder(features), targets, image_size=(64, 96))
                    losses = criterion(output, targets)
                    self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
                    self.assertFalse(any('clcr' in name for name in losses))
                    self.assertEqual('loss_margin' in losses, sdcp)
                    sum(losses.values()).backward()
                    self.assertGreater(lite.calibration[-1].weight.grad.abs().sum().item(), 0.)
                    for parameter in lite.parameters():
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertNotEqual(owner.encoder.pgd_rem.alpha.grad.item(), 0.)
                    if step:
                        self.assertGreater(lite.calibration[0].weight.grad.abs().sum().item(), 0.)
                        self.assertNotEqual(lite.beta.grad.item(), 0.)
                        self.assertGreater(owner.encoder.pgd_rem.prior_gate.reduce.weight.grad.abs().sum().item(), 0.)
                    optimizer.step()
                metrics = _module_training_metrics(owner)
                self.assertAlmostEqual(metrics['clcr_lite_beta'], lite.beta.item())
                self.assertFalse(any(name.startswith('clcr_beta_') for name in metrics))

    def test_checkpoint_and_deploy_preserve_shared_calibration_and_outputs(self):
        model = decoder().eval()
        lite = model.decoder.clcr_lite
        with torch.no_grad():
            lite.calibration[-1].weight.normal_(std=.1)
            lite.beta.fill_(.7)
        features, _ = fixture()
        with torch.no_grad():
            expected = model(features)
        restored = decoder().eval()
        restored.load_state_dict(model.state_dict())
        shared = restored.decoder.clcr_lite
        with torch.no_grad():
            self.assert_tree_equal(restored(features), expected)
        restored.convert_to_deploy()
        restored.decoder.convert_to_deploy()
        self.assertIs(restored.decoder.clcr_lite, shared)
        self.assertEqual(sum(p.numel() for p in shared.parameters()), 162)
        with torch.no_grad():
            actual = restored(features)
        for name in ['pred_logits', 'pred_boxes', 'pred_corners']:
            torch.testing.assert_close(actual[name], expected[name])

    def test_shared_parameters_have_one_state_entry_and_exactly_one_ema_update(self):
        model = decoder()
        lite = model.decoder.clcr_lite
        ema = ModelEMA(model, decay=.5, warmups=0)
        self.assertEqual(len([name for name in ema.module.state_dict() if '.clcr_lite.' in name]), 5)
        with torch.no_grad():
            lite.calibration[-1].weight.fill_(.2)
        ema.update(model)
        torch.testing.assert_close(ema.module.decoder.clcr_lite.calibration[-1].weight,
                                   torch.full_like(lite.calibration[-1].weight, .1), rtol=0, atol=0)

    def test_three_new_configs_keep_their_baseline_protocol_and_original_clcr_config(self):
        directory = ROOT / 'configs/deim_dfine'
        outputs = set()
        for suffix, pgd, sdcp in [('clcr_lite', False, False), ('pgd_rem_clcr_lite', True, False), ('pgd_rem_clcr_lite_sdcp', True, True)]:
            config = load_config(str(directory / f'deim_hgnetv2_m_sar_aircraft_{suffix}_58e_b28_old_server.yml'), {})
            base = 'sdcp_58e_b28' if sdcp else '58e_b28'
            reference = load_config(str(directory / f'deim_hgnetv2_m_sar_aircraft_{base}_old_server.yml'), {})
            self.assertFalse(config['DFINETransformer'].pop('use_clcr'))
            self.assertTrue(config['DFINETransformer'].pop('use_clcr_lite'))
            if pgd:
                self.assertTrue(config['HybridEncoder'].pop('use_pgd_rem'))
                self.assertEqual(config['HybridEncoder'].pop('pgd_rem_prior_channels'), 64)
            self.assertTrue((ROOT / config['experiment_protocol_document']).is_file())
            self.assertNotIn(config['output_dir'], outputs)
            outputs.add(config['output_dir'])
            for candidate in [config, reference]:
                candidate.pop('__include__', None)
                candidate.pop('output_dir')
                candidate.pop('experiment_protocol_document')
            self.assertEqual(config, reference)
        original = load_config(str(directory / 'deim_hgnetv2_m_sar_aircraft_clcr_58e_b28_old_server.yml'), {})
        self.assertTrue(original['DFINETransformer']['use_clcr'])
        self.assertFalse(original['DFINETransformer'].get('use_clcr_lite', False))


if __name__ == '__main__':
    unittest.main()
