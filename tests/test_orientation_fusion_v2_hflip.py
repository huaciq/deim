"""E018 (horizontal-flip view control) must equal E013-v2 except for the second view.

Run on CPU: python -m unittest tests.test_orientation_fusion_v2_hflip -v
"""
import unittest
from pathlib import Path

import torch

from engine.core import YAMLConfig
from engine.core.yaml_utils import load_config
from engine.deim.orientation_projection_fusion import ProjectedOrientationFusion

FOLDER = Path(__file__).resolve().parents[1] / 'configs/deim_dfine'
E013 = FOLDER / 'deim_hgnetv2_m_sardet100k_orientation_fusion_v2_36e_b32.yml'
E018 = FOLDER / 'deim_hgnetv2_m_sardet100k_orientation_fusion_v2_hflip_view_36e_b32.yml'
META = {'__include__', 'output_dir', 'experiment_protocol_document', 'experiment_baseline_revision'}


def flat(cfg, prefix=''):
    out = {}
    for key, value in cfg.items():
        out.update(flat(value, f'{prefix}{key}.') if isinstance(value, dict) else {f'{prefix}{key}': value})
    return out


class Config(unittest.TestCase):
    def test_only_the_view_differs_from_e013_v2(self):
        a, b = flat(load_config(str(E013))), flat(load_config(str(E018)))
        diff = {k for k in set(a) | set(b) if a.get(k, '<none>') != b.get(k, '<none>')} - META
        self.assertEqual(diff, {'DEIM.orientation_fusion.views'})
        self.assertEqual(a['DEIM.orientation_fusion.views'], ['vflip'])
        self.assertEqual(b['DEIM.orientation_fusion.views'], ['hflip'])
        self.assertNotEqual(a['output_dir'], b['output_dir'])
        self.assertTrue(b['output_dir'].startswith('/root/zxy/experiments/'))
        # The training pipeline still flips horizontally at random, which is what makes this view redundant.
        ops = [op['type'] for op in load_config(str(E018))['train_dataloader']['dataset']['transforms']['ops']]
        self.assertIn('RandomHorizontalFlip', ops)


class Module(unittest.TestCase):
    def test_default_is_still_the_e013_v2_vertical_view(self):
        fusion = ProjectedOrientationFusion(hidden_dim=8, num_levels=1, groups=2, checkpoint_views=False)
        self.assertEqual(fusion.view_name, 'vflip')

    def test_rejects_anything_but_one_flip_view(self):
        for views in (('vflip', 'hflip'), ('rot180',), ()):
            with self.assertRaises(ValueError):
                ProjectedOrientationFusion(views=views)

    def test_hflip_view_is_encoded_aligned_and_its_invalid_column_masked(self):
        fusion = ProjectedOrientationFusion(views=('hflip',), hidden_dim=8, num_levels=1, groups=2,
                                            checkpoint_views=False).eval()
        with torch.no_grad():
            fusion.gates[0][-1].bias[:2] = -100   # gate prefers the second view everywhere
            fusion.gates[0][-1].bias[2:] = 100
        images = torch.rand(2, 3, 64, 64)
        original, flipped = torch.randn(2, 8, 8, 8), torch.randn(2, 8, 8, 8)
        seen = []

        def encode(x):
            seen.append(x)
            return [original if len(seen) == 1 else flipped]
        out = fusion(images, encode)[0]
        torch.testing.assert_close(seen[1], fusion.view(images, 'hflip'), atol=0, rtol=0)
        torch.testing.assert_close(seen[1][..., 3:], images.flip(-1)[..., :-3], atol=0, rtol=0)  # 3-pixel shift
        aligned = fusion.align(flipped, 'hflip')
        torch.testing.assert_close(out[..., :, 0], original[..., :, 0], atol=0, rtol=0)
        torch.testing.assert_close(out[..., :, 1:], aligned[..., :, 1:], atol=0, rtol=0)
        self.assertEqual(sorted(fusion.last_metrics), ['orientation_hflip_p3'])

    def test_gt_centre_metric_uses_the_hflip_name(self):
        fusion = ProjectedOrientationFusion(views=('hflip',), hidden_dim=8, num_levels=1, groups=2,
                                            checkpoint_views=False).train()
        targets = [{'boxes': torch.tensor([[.5, .5, .1, .1]])}, {'boxes': torch.zeros(0, 4)}]
        fusion(torch.rand(2, 3, 32, 32), lambda x: [torch.randn(2, 8, 4, 4)], targets=targets)
        self.assertEqual(sorted(fusion.last_metrics), ['orientation_hflip_p3', 'orientation_hflip_p3_gt_center'])


class Model(unittest.TestCase):
    def test_same_parameters_and_the_e013_v2_initial_state(self):
        torch.manual_seed(0)
        e013 = YAMLConfig(str(E013)).model
        torch.manual_seed(0)
        e018 = YAMLConfig(str(E018)).model
        self.assertEqual(e018.orientation_fusion.view_name, 'hflip')
        a, b = e013.state_dict(), e018.state_dict()
        self.assertEqual(list(a), list(b))
        for name in a:
            torch.testing.assert_close(a[name], b[name], atol=0, rtol=0, msg=name)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
