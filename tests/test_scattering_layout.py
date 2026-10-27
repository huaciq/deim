"""Functional tests for local sampling, layout, AMP and SLQC gradient paths."""
import unittest

import torch

from engine.deim.scattering_layout import ScatteringLayoutQueryClassifier


class ScatteringLayoutTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    @staticmethod
    def identity_projectors(module):
        with torch.no_grad():
            for projection in module.feature_projs:
                projection.weight.zero_()
                projection.bias.zero_()
                for c in range(min(projection.in_channels, projection.out_channels)):
                    projection.weight[c, c, 0, 0] = 1

    def test_roi_bin_centers_level_order_and_batch_isolation(self):
        module = ScatteringLayoutQueryClassifier(hidden_dim=2, token_dim=2, roi_size=4,
                                                  num_levels=2, alpha_init=1)
        self.identity_projectors(module)
        ramp = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4).repeat(2, 2, 1, 1)
        ramp[1] += 100
        boxes = torch.tensor([[[.5, .5, 1., 1.]], [[.5, .5, 1., 1.]]])
        tokens, valid = module.sample_roi_tokens(boxes, [ramp, ramp + 1000])
        self.assertEqual(tokens.shape, (2, 1, 32, 2))
        torch.testing.assert_close(tokens[0, 0, :16, 0], torch.arange(16, dtype=torch.float32))
        torch.testing.assert_close(tokens[1, 0, :16, 0], torch.arange(16, dtype=torch.float32) + 100)
        torch.testing.assert_close(tokens[0, 0, 16:, 0], torch.arange(16, dtype=torch.float32) + 1000)
        torch.testing.assert_close(valid, torch.ones(2, 1, 1))

    def test_border_extension_and_fully_outside_reliability(self):
        module = ScatteringLayoutQueryClassifier(hidden_dim=8, token_dim=4, roi_size=4,
                                                  num_levels=1, alpha_init=1)
        queries = torch.randn(1, 2, 8)
        boxes = torch.tensor([[[-.5, .5, .25, .5], [0., .5, .5, .5]]])
        features = [torch.randn(1, 8, 8, 8)]
        _, valid = module.sample_roi_tokens(boxes, features)
        torch.testing.assert_close(valid[0, 0], torch.tensor([0.]))
        torch.testing.assert_close(valid[0, 1], torch.tensor([.5]))
        delta = module(queries, boxes, features)
        torch.testing.assert_close(delta[0, 0], torch.zeros(8), rtol=0, atol=0)
        self.assertTrue(torch.isfinite(delta).all())

    def test_multiple_query_grids_preserve_roi_order(self):
        module = ScatteringLayoutQueryClassifier(hidden_dim=2, token_dim=2, roi_size=4,
                                                  num_levels=1, alpha_init=1)
        self.identity_projectors(module)
        ramp = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8).repeat(1, 2, 1, 1)
        boxes = torch.tensor([[[.25, .25, .5, .5], [.75, .75, .5, .5]]])
        tokens, _ = module.sample_roi_tokens(boxes, [ramp])
        torch.testing.assert_close(tokens[0, 0, :, 0], ramp[0, 0, :4, :4].flatten())
        torch.testing.assert_close(tokens[0, 1, :, 0], ramp[0, 0, 4:, 4:].flatten())

    def test_content_control_is_permutation_invariant_layout_is_sensitive(self):
        torch.manual_seed(8)
        queries = torch.randn(1, 2, 16)
        tokens = torch.randn(1, 2, 32, 8)
        permutation = torch.arange(32).flip(0)
        for mode in ['roi', 'layout']:
            module = ScatteringLayoutQueryClassifier(hidden_dim=16, token_dim=8,
                                                      mode=mode, alpha_init=1)
            original = module.classify_tokens(queries, tokens)
            permuted = module.classify_tokens(queries, tokens[:, :, permutation])
            if mode == 'roi':
                torch.testing.assert_close(original, permuted, atol=2e-6, rtol=2e-5)
                self.assertIsNone(module.layout_encoder)
            else:
                self.assertGreater(float((original - permuted).abs().max()), 1e-5)

    def test_detached_boxes_and_gradient_to_queries_features_all_parameters(self):
        for mode in ['layout', 'roi']:
            torch.manual_seed(9)
            module = ScatteringLayoutQueryClassifier(hidden_dim=16, token_dim=8,
                                                      mode=mode, alpha_init=.4)
            queries = torch.randn(2, 3, 16, requires_grad=True)
            boxes = torch.tensor([[[.5, .5, .4, .4]] * 3] * 2, requires_grad=True)
            features = [torch.randn(2, 16, side, side, requires_grad=True) for side in [8, 4]]
            delta = module(queries, boxes, features)
            self.assertEqual(delta.shape, queries.shape)
            delta.square().sum().backward()
            self.assertIsNone(boxes.grad)
            self.assertGreater(float(queries.grad.abs().sum()), 0)
            for feature in features:
                self.assertGreater(float(feature.grad.abs().sum()), 0)
                self.assertTrue(torch.isfinite(feature.grad).all())
            for name, parameter in module.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_zero_scale_identity_and_scale_can_learn(self):
        torch.manual_seed(10)
        module = ScatteringLayoutQueryClassifier(hidden_dim=16, token_dim=8)
        queries = torch.randn(1, 3, 16)
        boxes = torch.tensor([[[.5, .5, .4, .4]] * 3])
        features = [torch.randn(1, 16, side, side) for side in [8, 4]]
        delta = module(queries, boxes, features)
        torch.testing.assert_close(delta, torch.zeros_like(queries), rtol=0, atol=0)
        delta.sum().backward()
        self.assertNotEqual(float(module.alpha.grad), 0)
        self.assertTrue(all(p.grad is not None for p in module.parameters()))
        self.assertGreater(float(module.up_proj.weight.abs().sum()), 0)

    def test_empty_queries_keep_ddp_parameter_paths(self):
        module = ScatteringLayoutQueryClassifier(hidden_dim=16, token_dim=8)
        queries = torch.empty(2, 0, 16, requires_grad=True)
        boxes = torch.empty(2, 0, 4, requires_grad=True)
        features = [torch.randn(2, 16, side, side, requires_grad=True) for side in [8, 4]]
        delta = module(queries, boxes, features)
        self.assertEqual(delta.shape, (2, 0, 16))
        delta.sum().backward()
        self.assertIsNone(boxes.grad)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()))

    def test_amp_uses_finite_fp32_sampling_and_returns_query_dtype(self):
        module = ScatteringLayoutQueryClassifier(hidden_dim=16, token_dim=8, alpha_init=.5)
        queries = (torch.randn(1, 3, 16) * 1000).bfloat16().requires_grad_()
        boxes = torch.tensor([[[.5, .5, .4, .4]] * 3]).bfloat16().requires_grad_()
        features = [(torch.randn(1, 16, side, side) * 1000).bfloat16().requires_grad_() for side in [8, 4]]
        with torch.autocast('cpu', dtype=torch.bfloat16):
            delta = module(queries, boxes, features)
        self.assertEqual(delta.dtype, queries.dtype)
        self.assertTrue(torch.isfinite(delta).all())
        delta.float().square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()))


if __name__ == '__main__':
    unittest.main()
