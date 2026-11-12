"""Targeted checks for orientation-shared encoding (E013). Run on CPU: python -m unittest tests.test_orientation_fusion -v"""
import math
import unittest
from pathlib import Path

import torch

from engine.core import YAMLConfig
from engine.core.yaml_utils import load_config
from engine.deim.orientation_fusion import VIEWS, OrientationSharedFusion
from engine.solver._solver import BaseSolver

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/deim_dfine/deim_hgnetv2_m_sardet100k_orientation_fusion_36e_b32.yml'
BASELINE = ROOT / 'configs/deim_dfine/deim_hgnetv2_m_sardet100k_36e_b32_nozoom_nocrop.yml'
IGNORED = {'__include__', 'output_dir', 'experiment_baseline_revision', 'experiment_protocol_document'}
GENERATED = {'decoder.anchors', 'decoder.valid_mask'}


def flat(cfg, prefix=''):
    out = {}
    for key, value in cfg.items():
        out.update(flat(value, f'{prefix}{key}.') if isinstance(value, dict) else {f'{prefix}{key}': value})
    return out


def build(path):
    torch.manual_seed(0)
    return YAMLConfig(str(path)).model


class Config(unittest.TestCase):
    def test_only_the_fusion_block_is_added_to_the_baseline(self):
        base, run = flat(load_config(str(BASELINE))), flat(load_config(str(CONFIG)))
        changed = {k for k in set(base) | set(run) if base.get(k) != run.get(k)} - IGNORED
        self.assertTrue(changed)
        self.assertTrue(all(k.startswith('DEIM.orientation_fusion.') for k in changed), changed)
        self.assertEqual(run['DEIM.orientation_fusion.views'], ['vflip'])


class Module(unittest.TestCase):
    def test_views_are_undone_by_align_without_shifts(self):
        x = torch.randn(2, 3, 16, 16)
        for name in VIEWS:
            f = OrientationSharedFusion(views=(name,), hidden_dim=8, num_levels=1, groups=2, image_shift=0, cell_shift=0)
            self.assertTrue(torch.equal(f.align(f.view(x, name), name), x), name)

    def test_initial_gate_is_content_independent_and_favours_the_image_itself(self):
        f = OrientationSharedFusion(views=('vflip',), hidden_dim=8, num_levels=2, groups=2, image_shift=0, cell_shift=0).eval()
        a, b = torch.randn(3, 8, 6, 6), torch.randn(3, 8, 6, 6)
        # an "encoder" that returns the same two maps for the image and for its view
        out = f(torch.zeros(3, 3, 12, 12), lambda images: [torch.cat([a, b.flip(-2)])] * 2)
        w = math.exp(2.) / (math.exp(2.) + 1)
        for level in out:
            self.assertTrue(torch.allclose(level, w * a + (1 - w) * b, atol=1e-5))
        self.assertTrue(torch.allclose(f.last_weights, torch.tensor([[w, 1 - w]] * 2), atol=1e-5))

    def test_one_encoder_sees_all_views_and_gets_gradients_from_each(self):
        torch.manual_seed(0)
        conv = torch.nn.Conv2d(3, 8, 3, padding=1)
        seen = []

        def encode(images):
            seen.append(images.shape[0])
            return [conv(images)]
        f = OrientationSharedFusion(views=('vflip', 'hflip'), hidden_dim=8, num_levels=1, groups=2)
        images = torch.rand(2, 3, 16, 16)
        f(images, encode)[0].sum().backward()
        self.assertEqual(seen, [6])                      # one pass, 3 views x 2 images
        self.assertGreater(conv.weight.grad.abs().sum().item(), 0)
        # The last gate layer starts at zero; its bias and the stacked features still receive gradients.
        self.assertIsNotNone(f.gates[0][-1].bias.grad)

    def test_gradient_checkpointing_gives_the_same_output_and_gradients(self):
        torch.manual_seed(0)
        conv = torch.nn.Conv2d(3, 8, 3, padding=1)
        images = torch.rand(2, 3, 16, 16)
        results = []
        for flag in (False, True):
            torch.manual_seed(1)
            f = OrientationSharedFusion(views=('vflip',), hidden_dim=8, num_levels=1, groups=2, checkpoint_views=flag).train()
            torch.nn.init.normal_(f.gates[0][-1].weight, std=.5)
            conv.zero_grad()
            out = f(images, lambda x: [conv(x)])[0]
            (out ** 2).sum().backward()
            results.append((out.detach(), conv.weight.grad.clone(), f.gates[0][0].weight.grad.clone()))
        for a, b in zip(*results):
            self.assertTrue(torch.allclose(a, b, atol=1e-5))

    def test_rejects_unknown_views(self):
        with self.assertRaises(ValueError):
            OrientationSharedFusion(views=('rot45',))


class Model(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = build(BASELINE)
        cls.fused = build(CONFIG)

    def test_every_baseline_parameter_loads_and_only_gate_parameters_are_new(self):
        matched, info = BaseSolver._matched_state(self.fused.state_dict(), self.base.state_dict())
        self.assertEqual([k for k in info['unmatched'] if k not in GENERATED], [])
        extra = [k for k in self.fused.state_dict() if k not in self.base.state_dict()]
        self.assertTrue(extra)
        self.assertTrue(all(k.startswith('orientation_fusion.gates.') for k in extra), extra)
        added = sum(v.numel() for k, v in self.fused.state_dict().items() if k in extra and 'num_batches' not in k)
        self.assertLess(added, 200_000)

    def test_with_the_gate_fixed_on_the_image_the_detector_equals_the_baseline(self):
        base, fused = self.base.eval(), self.fused.eval()
        fused.load_state_dict(base.state_dict(), strict=False)
        saved = [g[-1].bias.detach().clone() for g in fused.orientation_fusion.gates]
        groups = fused.orientation_fusion.groups
        with torch.no_grad():
            for g in fused.orientation_fusion.gates:
                g[-1].bias[:groups] = 60.
            x = torch.rand(1, 3, 640, 640)
            a, b = base(x), fused(x)
            for g, s in zip(fused.orientation_fusion.gates, saved):
                g[-1].bias.copy_(s)
        self.assertTrue(torch.allclose(a['pred_logits'], b['pred_logits'], atol=1e-3))
        self.assertTrue(torch.allclose(a['pred_boxes'], b['pred_boxes'], atol=1e-4))

    def test_default_shifts_put_the_flipped_pyramid_on_the_image_grid(self):
        """Kernels made flip-symmetric, content in the image centre: what remains is grid misalignment only."""
        torch.manual_seed(0)
        backbone = build(BASELINE).backbone.eval()
        with torch.no_grad():
            for m in backbone.modules():
                if isinstance(m, torch.nn.Conv2d):
                    m.weight.copy_((m.weight + m.weight.flip(2) + m.weight.flip(3) + m.weight.flip(2, 3)) / 4)
        yy, xx = torch.meshgrid(torch.arange(640.), torch.arange(640.), indexing='ij')
        g = torch.Generator().manual_seed(1)
        x = torch.zeros(1, 3, 640, 640)
        for _ in range(25):
            cy, cx = (torch.rand(2, generator=g) * 240 + 200).tolist()
            sigma = float(torch.rand(1, generator=g) * 8 + 3)
            x += float(torch.rand(1, generator=g)) * torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
        x = x.clamp(0, 1)

        def centre(t):
            h = t.shape[-1]
            t = t[..., h // 4: h - h // 4, h // 4: h - h // 4]
            return t - t.mean((-1, -2), keepdim=True)

        def error(fusion, view):
            out = [fusion.align(p, view) for p in backbone(fusion.view(x, view))]
            return max(float((centre(a) - centre(b)).norm() / centre(b).norm()) for a, b in zip(out, reference))
        with torch.no_grad():
            reference = backbone(x)
            for view in ('vflip', 'hflip'):
                self.assertLess(error(OrientationSharedFusion(views=(view,)), view), .05, view)
                self.assertGreater(error(OrientationSharedFusion(views=(view,), image_shift=0, cell_shift=0), view), .3, view)


if __name__ == '__main__':
    unittest.main()
