"""SDCP cosine/margin definitions, pixel-tolerant supervision and joint routing."""

import copy
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F

from engine.core.yaml_utils import load_config
from engine.deim.deim import DEIM
from engine.deim.dfine_decoder import BoxLocalEvidenceSampler, DFINETransformer, SDCPCosineClassifier
from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.hybrid_encoder import HybridEncoder
from engine.deim.matcher import HungarianMatcher


ROOT = Path(__file__).resolve().parents[1]
DECODER_ARGS = dict(num_classes=7, hidden_dim=32, feat_channels=[32] * 3,
                    num_queries=5, num_layers=3, eval_idx=1, nhead=4, num_points=2,
                    dim_feedforward=64, reg_max=8, num_denoising=4, eval_spatial_size=None)


def decoder(enabled=True, combined=False):
    return DFINETransformer(**copy.deepcopy(DECODER_ARGS), use_sdcp=enabled,
                            use_clcr=combined, clcr_hidden_dim=16, sdcp_aspect_dim=8)


def criterion(enabled=True):
    weights = dict(loss_mal=1, loss_bbox=5, loss_giou=2, loss_fgl=.15, loss_ddf=1.5)
    if enabled:
        weights.update(loss_bbox=0., loss_asym_box=5., loss_margin=.1)
    return DEIMCriterion(
        HungarianMatcher(dict(cost_class=2, cost_bbox=5, cost_giou=2), use_focal_loss=True),
        weights, ['mal', 'boxes', 'local'], num_classes=7, reg_max=8, use_sdcp=enabled)


def fixture():
    features = [torch.randn(2, 32, h, w) for h, w in [(8, 12), (4, 6), (2, 3)]]
    targets = [dict(labels=torch.tensor([3]), boxes=torch.tensor([[.4, .5, .3, .4]])),
               dict(labels=torch.tensor([1, 6]), boxes=torch.tensor([[.3, .4, .2, .3], [.7, .6, .3, .2]]))]
    return features, targets


class SdcpTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(251)

    def test_classifier_matches_cosine_formula_and_prototype_magnitude_invariance(self):
        weight = torch.randn(7, 8)
        head = SDCPCosineClassifier(weight, feature_dim=6, aspect_dim=4, temperature=10.)
        query = torch.randn(2, 3, 8)
        inside = torch.randn(2, 3, 6)
        boxes = torch.tensor([[[.5, .5, .3, .2]] * 3] * 2)
        size = torch.tensor([640., 800.])
        aspect = head.aspect_embedding(head.log_aspect_ratio(boxes, size))
        fused = query + head.local_projection(torch.cat((inside, aspect), -1))
        expected = 10 * F.normalize(fused, dim=-1) @ F.normalize(weight, dim=-1).T
        result = head(query, inside, boxes, size)
        torch.testing.assert_close(result, expected)
        self.assertFalse(hasattr(head, 'bias'))
        with torch.no_grad():
            head.weight.mul_(torch.arange(1, 8)[:, None])
        torch.testing.assert_close(head(query, inside, boxes, size), result)
        self.assertLessEqual(result.abs().max().item(), 10.)
        with torch.no_grad():
            head.local_projection.weight.zero_()
            head.local_projection.bias.zero_()
        torch.testing.assert_close(head(query * 4, inside, boxes, size), head(query, inside, boxes, size))

    def test_aspect_and_box_relative_core_are_invariant_to_uniform_input_resizing(self):
        sampler = BoxLocalEvidenceSampler()
        boxes = torch.tensor([[[.4, .6, .2, .3], [.7, .3, .25, .2]]])
        pooled = []
        for side in [80, 100]:
            yy, xx = torch.meshgrid((torch.arange(side) + .5) / side,
                                    (torch.arange(side) + .5) / side, indexing='ij')
            pooled.append(sampler.pool_local_evidence(boxes, torch.stack((xx, yy))[None])[0])
        torch.testing.assert_close(pooled[0], pooled[1])
        head = SDCPCosineClassifier(torch.randn(7, 8), 2, aspect_dim=4)
        query = torch.randn(1, 2, 8)
        torch.testing.assert_close(head(query, pooled[0], boxes, torch.tensor([640., 640.])),
                                   head(query, pooled[1], boxes, torch.tensor([800., 800.])))
        rectangular_box = torch.tensor([[[.5, .5, .2, .25]]])
        torch.testing.assert_close(head.log_aspect_ratio(rectangular_box, torch.tensor([640., 800.])),
                                   torch.zeros(1, 1, 1))
        torch.testing.assert_close(head.log_aspect_ratio(rectangular_box, torch.tensor([640., 1600.])),
                                   torch.full((1, 1, 1), math.log(2)))

    def test_cosine_head_amp_gradients_reach_query_core_aspect_and_prototypes(self):
        for amp, dtype in [(False, torch.float32), (True, torch.bfloat16)]:
            with self.subTest(amp=amp):
                head = SDCPCosineClassifier(torch.randn(7, 8), 6, aspect_dim=4)
                query = torch.randn(2, 3, 8, dtype=dtype, requires_grad=True)
                inside = torch.randn(2, 3, 6, dtype=dtype, requires_grad=True)
                boxes = torch.tensor([[[.5, .5, .3, .2]] * 3] * 2, dtype=dtype, requires_grad=True)
                with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
                    scores = head(query, inside, boxes, torch.tensor([640., 640.]))
                self.assertEqual(scores.dtype, dtype)
                scores.float().square().mean().backward()
                for value in [query, inside, boxes]:
                    self.assertTrue(torch.isfinite(value.grad).all())
                    self.assertGreater(value.grad.abs().sum().item(), 0.)
                for name, parameter in head.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(head.aspect_embedding.layers[0].weight.grad.abs().sum().item(), 0.)
                self.assertGreater(head.weight.grad.abs().sum().item(), 0.)

    def test_margin_uses_strongest_other_class_only_matched_queries_and_positive_mean(self):
        probability = torch.tensor([[[.8, .1, .05, .05, .05, .05, .05],
                                      [.6, .3, .1, .05, .05, .05, .05],
                                      [.45, .1, .5, .05, .05, .05, .05],
                                      [.99, .01, .01, .01, .01, .01, .01]]])
        logits = torch.logit(probability).requires_grad_()
        targets = [dict(labels=torch.tensor([0, 1, 2]))]
        indices = [(torch.arange(3), torch.arange(3))]
        value = criterion().loss_sdcp_margin({'sdcp_logits': logits}, targets, indices)
        self.assertAlmostEqual(value.item(), (0. + .5 ** 2 + .15 ** 2) / 3, places=6)
        value.backward()
        torch.testing.assert_close(logits.grad[0, 0], torch.zeros(7))
        torch.testing.assert_close(logits.grad[0, 3], torch.zeros(7))
        self.assertLess(logits.grad[0, 1, 1].item(), 0.)
        self.assertGreater(logits.grad[0, 1, 0].item(), 0.)
        torch.testing.assert_close(logits.grad[0, 1, 2:], torch.zeros(5))

    def test_empty_margin_is_finite_and_keeps_gradient_path(self):
        logits = torch.randn(2, 3, 7, requires_grad=True)
        empty = torch.empty(0, dtype=torch.long)
        targets = [dict(labels=empty), dict(labels=empty)]
        value = criterion().loss_sdcp_margin({'sdcp_logits': logits}, targets, [(empty, empty)] * 2)
        self.assertEqual(value.item(), 0.)
        value.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_pixel_dead_zone_and_smooth_transition_scale_with_actual_image_size(self):
        loss = criterion()
        target = torch.tensor([[.5, .5, .4, .3]])
        previous = None
        for height, width in [(640, 640), (800, 800), (640, 800)]:
            within = target.clone()
            within[:, 0] += .75 / width
            within[:, 1] += .75 / height
            within.requires_grad_()
            penalty, fgl, giou = loss.sdcp_boundary_terms(within, target, [height, width])
            self.assertEqual(penalty.sum().item(), 0.)
            self.assertEqual(fgl.sum().item(), 0.)
            self.assertEqual(giou.sum().item(), 0.)
            penalty.sum().backward()
            torch.testing.assert_close(within.grad, torch.zeros_like(within))
            outside = target.clone()
            outside[:, 0] += 1.5 / width
            penalty, fgl, giou = loss.sdcp_boundary_terms(outside, target, [height, width])
            pixels = penalty * torch.tensor([width, height, width, height])
            torch.testing.assert_close(pixels, torch.tensor([[.125, 0., .125, 0.]]), atol=1e-5, rtol=1e-4)
            self.assertFalse(fgl.requires_grad)
            self.assertFalse(giou.requires_grad)
            if previous is not None:
                torch.testing.assert_close(pixels, previous, atol=1e-5, rtol=1e-4)
            previous = pixels

    def test_over_width_penalty_is_1p2_only_on_horizontal_edges(self):
        loss = criterion()
        target = torch.tensor([[.5, .5, .4, .3]])
        wide, narrow = target.clone(), target.clone()
        wide[:, 2] += .04
        narrow[:, 2] -= .04
        wide.requires_grad_()
        narrow.requires_grad_()
        wide_penalty, wide_fgl, _ = loss.sdcp_boundary_terms(wide, target, [640, 640])
        narrow_penalty, narrow_fgl, _ = loss.sdcp_boundary_terms(narrow, target, [640, 640])
        torch.testing.assert_close(wide_penalty, narrow_penalty * 1.2, atol=1e-7, rtol=1e-5)
        torch.testing.assert_close(wide_fgl, torch.tensor([[1.2, 0., 1.2, 0.]]))
        torch.testing.assert_close(narrow_fgl, torch.tensor([[1., 0., 1., 0.]]))
        wide_penalty.sum().backward()
        narrow_penalty.sum().backward()
        self.assertGreater(wide.grad[0, 2].item(), 0.)
        self.assertLess(narrow.grad[0, 2].item(), 0.)
        wide_tall = wide.detach().clone()
        wide_tall[:, 3] += .04
        _, weights, _ = loss.sdcp_boundary_terms(wide_tall, target, [640, 640])
        torch.testing.assert_close(weights, torch.tensor([[1.2, 1., 1.2, 1.]]))

    def test_l1_is_replaced_and_fgl_giou_also_reduce_label_noise_gradients(self):
        enabled, baseline = criterion(), criterion(False)
        target = torch.tensor([[.5, .5, .3, .3]])
        source = target.clone()
        source[:, 0] += .5 / 640
        source.requires_grad_()
        corners = torch.randn(1, 1, 36, requires_grad=True)
        output = dict(pred_boxes=source[None], pred_corners=corners, ref_points=target[None],
                      up=torch.tensor([.5]), reg_scale=torch.tensor([4.]), sdcp_image_size=torch.tensor([640, 640]))
        targets = [dict(boxes=target, labels=torch.tensor([0]))]
        indices = [(torch.tensor([0]), torch.tensor([0]))]
        losses = enabled.loss_boxes(output, targets, indices, 1)
        self.assertNotIn('loss_bbox', losses)
        self.assertEqual(losses['loss_asym_box'].item(), 0.)
        self.assertEqual(losses['loss_giou'].item(), 0.)
        fgl = enabled.loss_local(output, targets, indices, 1)['loss_fgl']
        self.assertEqual(fgl.item(), 0.)
        self.assertGreater(baseline.loss_boxes(output, targets, indices, 1)['loss_bbox'].item(), 0.)
        self.assertGreater(baseline.loss_local(output, targets, indices, 1)['loss_fgl'].item(), 0.)
        (sum(losses.values()) + fgl).backward()
        torch.testing.assert_close(source.grad, torch.zeros_like(source))
        torch.testing.assert_close(corners.grad, torch.zeros_like(corners))

    def test_decoder_reuses_local_evidence_and_exports_raw_non_dn_cosine_logits(self):
        model = decoder(combined=True)
        features, targets = fixture()
        for lqe in model.decoder.lqe_layers:
            with torch.no_grad():
                lqe.reg_conf.layers[-1].bias.fill_(.7)
                lqe.clcr.beta.fill_(.2)
                lqe.clcr.calibration.layers[-1].weight.zero_()
                lqe.clcr.calibration.layers[-1].bias.fill_(2.)
        sampler = model.decoder.sdcp_sampler
        original = sampler.pool_local_evidence
        with patch.object(sampler, 'pool_local_evidence', wraps=original) as sampling:
            with patch('engine.deim.dfine_decoder.CLCR.pool_local_evidence', side_effect=AssertionError('duplicate sampling')):
                output = model(features, targets, image_size=(65, 97))
            self.assertEqual(sampling.call_count, model.num_layers + 1)
        self.assertEqual(output['sdcp_logits'].shape, (2, 5, 7))
        self.assertLessEqual(output['sdcp_logits'].abs().max().item(), 10.)
        torch.testing.assert_close(output['pred_logits'] - output['sdcp_logits'],
                                   torch.full_like(output['pred_logits'], 1.1))
        torch.testing.assert_close(output['sdcp_image_size'], torch.tensor([65., 97.]))
        self.assertNotIn('sdcp_logits', output['dn_outputs'][0])
        self.assertEqual(len(output['dn_outputs']), 3)
        model.eval()
        with patch.object(sampler, 'pool_local_evidence', wraps=original) as sampling:
            with torch.no_grad():
                inference = model(features, image_size=(65, 97))
            self.assertEqual(sampling.call_count, 1)
        self.assertNotIn('sdcp_logits', inference)
        self.assertNotIn('sdcp_image_size', inference)

    def test_common_initialization_and_classifier_weight_names_are_preserved(self):
        torch.manual_seed(257)
        baseline = decoder(False)
        expected_rng = torch.get_rng_state()
        torch.manual_seed(257)
        adapted = decoder()
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        adapted_state = adapted.state_dict()
        for name, value in baseline.state_dict().items():
            if name.startswith('dec_score_head.') and name.endswith('.bias'):
                self.assertNotIn(name, adapted_state)
            else:
                torch.testing.assert_close(adapted_state[name], value, rtol=0, atol=0)
        common = {name: value for name, value in baseline.state_dict().items() if name in adapted_state}
        missing, unexpected = adapted.load_state_dict(common, strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(all('aspect_embedding' in name or 'local_projection' in name for name in missing))

    def test_image_wrapper_forwards_actual_not_rounded_backbone_size(self):
        class CapturingDecoder(nn.Module):
            use_sdcp = True

            def forward(self, features, targets=None, image_size=None):
                return {'seen_size': image_size}

        model = DEIM(nn.Identity(), nn.Identity(), CapturingDecoder())
        result = model(torch.zeros(1, 3, 65, 97))
        self.assertEqual(tuple(result['seen_size']), (65, 97))

    def test_joint_training_losses_reach_all_three_modules_and_all_supervised_box_paths(self):
        encoder = HybridEncoder(in_channels=[32, 64, 128], hidden_dim=32, nhead=4,
                                dim_feedforward=64, depth_mult=.34,
                                use_pgd_rem=True, pgd_rem_prior_channels=8)
        model = decoder(combined=True)
        loss_fn = criterion()
        features = [torch.randn(2, channels, h, w) for channels, h, w in
                    [(32, 8, 12), (64, 4, 6), (128, 2, 3)]]
        _, targets = fixture()
        optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(model.parameters()), lr=1e-3)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = model(encoder(features), targets, image_size=(64, 96))
            losses = loss_fn(output, targets)
            self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
            self.assertEqual([name for name in losses if name.startswith('loss_margin')], ['loss_margin'])
            self.assertFalse(any(name.startswith('loss_bbox') for name in losses))
            for suffix in ['', '_aux_0', '_pre', '_enc_0', '_dn_0', '_dn_pre']:
                self.assertIn('loss_asym_box' + suffix, losses)
            for group in ['aux_outputs', 'enc_aux_outputs', 'dn_outputs']:
                for branch in output[group]:
                    torch.testing.assert_close(branch['sdcp_image_size'], torch.tensor([64., 96.]))
            sum(losses.values()).backward()
            for head in model.dec_score_head:
                for name, parameter in head.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(head.local_projection.weight.grad.abs().sum().item(), 0.)
                self.assertGreater(head.weight.grad.abs().sum().item(), 0.)
            self.assertNotEqual(encoder.pgd_rem.alpha.grad.item(), 0.)
            for lqe in model.decoder.lqe_layers:
                self.assertNotEqual(lqe.clcr.beta.grad.item(), 0.)
                if step:
                    self.assertGreater(lqe.clcr.local_mlp.layers[0].weight.grad.abs().sum().item(), 0.)
            if step:
                self.assertGreater(encoder.pgd_rem.prior_gate.reduce.weight.grad.abs().sum().item(), 0.)
            optimizer.step()

    def test_restoring_and_deployment_keep_cosine_head_and_clcr(self):
        model = decoder(combined=True).eval()
        for lqe in model.decoder.lqe_layers:
            lqe.clcr.beta.data.fill_(.2)
        features, _ = fixture()
        with torch.no_grad():
            expected = model(features, image_size=(64, 96))
        restored = decoder(combined=True).eval()
        restored.load_state_dict(model.state_dict())
        restored.convert_to_deploy()
        restored.decoder.convert_to_deploy()
        with torch.no_grad():
            actual = restored(features, image_size=(64, 96))
        for name in ['pred_logits', 'pred_boxes', 'pred_corners']:
            torch.testing.assert_close(actual[name], expected[name])
        self.assertIsInstance(restored.dec_score_head[restored.eval_idx], SDCPCosineClassifier)

    def test_single_and_triple_configs_only_change_sdcp_and_explicit_module_flags(self):
        directory = ROOT / 'configs/deim_dfine'
        baseline = load_config(str(directory / 'deim_hgnetv2_m_sar_aircraft_58e_b28_old_server.yml'), {})
        for suffix, triple in [('sdcp', False), ('pgd_rem_clcr_sdcp', True)]:
            config = load_config(str(directory / f'deim_hgnetv2_m_sar_aircraft_{suffix}_58e_b28_old_server.yml'), {})
            self.assertTrue(config['DFINETransformer'].pop('use_sdcp'))
            self.assertEqual(config['DFINETransformer'].pop('sdcp_aspect_dim'), 16)
            self.assertEqual(config['DFINETransformer'].pop('sdcp_temperature'), 10.)
            loss = config['DEIMCriterion']
            self.assertTrue(loss.pop('use_sdcp'))
            self.assertEqual(loss.pop('sdcp_margin'), .2)
            self.assertEqual(loss.pop('sdcp_epsilon_px'), 1.)
            self.assertEqual(loss.pop('sdcp_lambda_over'), .2)
            self.assertEqual(loss['weight_dict']['loss_bbox'], 0.)
            loss['weight_dict']['loss_bbox'] = baseline['DEIMCriterion']['weight_dict']['loss_bbox']
            self.assertEqual(loss['weight_dict'].pop('loss_asym_box'), 5.)
            self.assertEqual(loss['weight_dict'].pop('loss_margin'), .1)
            if triple:
                self.assertTrue(config['DFINETransformer'].pop('use_clcr'))
                self.assertEqual(config['DFINETransformer'].pop('clcr_hidden_dim'), 64)
                self.assertTrue(config['HybridEncoder'].pop('use_pgd_rem'))
                self.assertEqual(config['HybridEncoder'].pop('pgd_rem_prior_channels'), 64)
            self.assertEqual(config['epoches'], 58)
            self.assertEqual(config['train_dataloader']['total_batch_size'], 28)
            self.assertTrue((ROOT / config['experiment_protocol_document']).is_file())
            reference = copy.deepcopy(baseline)
            self.assertNotEqual(config['output_dir'], reference['output_dir'])
            for candidate in [config, reference]:
                candidate.pop('__include__', None)
                candidate.pop('output_dir')
                candidate.pop('experiment_protocol_document')
            self.assertEqual(config, reference)


if __name__ == '__main__':
    unittest.main()
