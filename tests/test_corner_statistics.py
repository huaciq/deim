import ast
import copy
import inspect
import textwrap
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from tools.corner_signal_analysis import corr, match_edges
from tools.export_corner_stats import distribution_moments, projection_parameters
import engine.deim.dfine_decoder as decoder_module
from engine.deim.dfine_utils import distance2bbox, weighting_function


class CornerStatisticsTests(unittest.TestCase):
    def test_export_projection_keeps_parameter_shape(self):
        decoder = SimpleNamespace(reg_max=32, up=torch.tensor([.5]), reg_scale=torch.tensor([4.]))
        reg_max, _, scale, project = projection_parameters(decoder)
        self.assertEqual((reg_max, scale, project.numel()), (32, 4., 33))
        self.assertEqual((project.min().item(), project.max().item()), (-4., 4.))

    def test_diagnostic_eval_keys_preserve_original_logits_and_boxes(self):
        # Compare the two-key pre-change forward with the current forward on
        # the same decoder weights/features, including the real decoding path.
        tree = ast.parse(textwrap.dedent(inspect.getsource(decoder_module.DFINETransformer.forward)))
        old_forward = copy.deepcopy(tree.body[0])
        old_forward.name = 'old_forward'
        changed = 0
        for node in ast.walk(old_forward):
            if isinstance(node, ast.If) and node.orelse and isinstance(node.orelse[0], ast.Assign):
                value = node.orelse[0].value
                if isinstance(value, ast.Dict) and [getattr(k, 'value', None) for k in value.keys] == [
                        'pred_logits', 'pred_boxes', 'pred_corners', 'ref_points', 'up', 'reg_scale']:
                    value.keys, value.values = value.keys[:2], value.values[:2]
                    changed += 1
        self.assertEqual(changed, 1)
        namespace = dict(vars(decoder_module))
        exec(compile(ast.Module(body=[old_forward], type_ignores=[]), '<old_forward>', 'exec'), namespace)
        torch.manual_seed(0)
        model = decoder_module.DFINETransformer(num_classes=6, hidden_dim=32, num_queries=10,
                feat_channels=[32, 32, 32], num_layers=3, dim_feedforward=64,
                num_denoising=0, eval_spatial_size=[64, 64]).eval()
        features = [torch.randn(1, 32, size, size) for size in [8, 4, 2]]
        with torch.no_grad():
            old = namespace['old_forward'](model, features)
            new = model(features)
        for key in ['pred_logits', 'pred_boxes']:
            torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)
        project = weighting_function(model.reg_max, model.up, model.reg_scale)
        mean, _ = distribution_moments(new['pred_corners'], project)
        torch.testing.assert_close(distance2bbox(new['ref_points'], mean, model.reg_scale),
                                   new['pred_boxes'], rtol=0, atol=1e-6)

    def test_moments_use_nonuniform_signed_projection(self):
        # Equal mass at -4 and +1 has mean -1.5 and standard deviation 2.5.
        project = torch.tensor([-4., 0., 1.])
        corners = torch.tensor([0., -1000., 0.]).repeat(4).reshape(1, 1, -1)
        mean, std = distribution_moments(corners, project)
        torch.testing.assert_close(mean, torch.full((1, 1, 4), -1.5))
        torch.testing.assert_close(std, torch.full((1, 1, 4), 2.5))

    def test_valid_negative_projected_targets_are_not_excluded(self):
        # These targets are -1/3 on every edge. Rejecting negative values would
        # discard every pair even though all targets lie in actual [-4,4] support.
        n = 40
        sigma = np.linspace(.1, .8, n)
        width = .25 + .015 * sigma
        z = dict(image_id=np.arange(n), label=np.zeros(n, int), score=np.ones(n),
                 box=np.column_stack([np.full(n, .5), np.full(n, .5), width, width]),
                 ref=np.tile([.5, .5, .3, .3], (n, 1)), std=np.repeat(sigma[:, None], 4, axis=1),
                 project=np.linspace(-4, 4, 33), reg_scale=np.array(4.), reg_max=np.array(32))
        gt = dict(images=[dict(id=i, width=256, height=256) for i in range(n)],
                  annotations=[dict(image_id=i, category_id=0, bbox=[96, 96, 64, 64]) for i in range(n)])
        _, matched_sigma, error, _, _, counts = match_edges(z, gt)
        self.assertEqual(counts['retained'], n)
        self.assertEqual(counts['valid_pairs_rejected_by_document_index_filter'], n)
        self.assertAlmostEqual(corr(matched_sigma[:, 0], np.abs(error[:, 0])), 1.)


if __name__ == '__main__':
    unittest.main()
