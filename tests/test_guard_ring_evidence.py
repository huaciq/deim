"""Targeted checks for GREA. Run on CPU: python -m unittest tests.test_guard_ring_evidence -v"""
import unittest
from pathlib import Path

import torch

from engine.core import YAMLConfig
from engine.deim.guard_ring_evidence import GuardRingEvidence

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/deim_dfine/deim_hgnetv2_m_sardet100k_grea_36e_b32.yml'
BASELINE = ROOT / 'configs/deim_dfine/deim_hgnetv2_m_sardet100k_36e_b32_nozoom_nocrop.yml'


class Geometry(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.module = GuardRingEvidence(16, 6, levels=2, dim=8, points=8)
        self.query = torch.randn(2, 5, 16)
        self.boxes = torch.tensor([[.5, .5, .2, .1]]).repeat(2, 5, 1)

    def test_core_stays_inside_and_ring_clears_box_plus_guard_for_any_query(self):
        guard = torch.tensor([.01, .02])
        for scale in (0., 1., 50.):
            for layer in (self.module.core_offsets, self.module.ring_radii):
                torch.nn.init.normal_(layer.weight, std=scale)
            core, ring = self.module.sample_points(self.query, self.boxes, guard)
            half = self.boxes[..., None, 2:] / 2
            center = self.boxes[..., None, :2]
            self.assertTrue(((core - center).abs() <= half + 1e-6).all())
            # Outside the box by at least the inner radius plus the guard, on some axis.
            normalized = ((ring - center).abs() / (half * 1.25 + guard)).amax(-1)
            self.assertTrue((normalized >= 1 - 1e-5).all())
            limit = ((ring - center).abs() / (half * 2 + guard)).amax(-1)
            self.assertTrue((limit <= 1 + 1e-5).all())

    def test_outside_points_are_excluded_unless_all_are_outside(self):
        level = torch.zeros(1, 1, 4, 4)
        level[..., 0, 0] = 1.
        points = torch.tensor([[[[-.5, .5], [.1, .1]]]])
        pooled = GuardRingEvidence._pool([level], points, torch.zeros(1, 1, 2))
        inside_only = GuardRingEvidence._pool([level], points[:, :, 1:], torch.zeros(1, 1, 1))
        self.assertTrue(torch.allclose(pooled, inside_only))
        all_outside = GuardRingEvidence._pool([level], torch.full((1, 1, 2, 2), -.5), torch.zeros(1, 1, 2))
        self.assertTrue(torch.isfinite(all_outside).all())

    def test_contrast_sees_a_target_against_clutter(self):
        level = torch.zeros(1, 1, 64, 64)
        level[..., 28:36, 28:36] = 1.
        boxes = torch.tensor([[[.5, .5, .125, .125]]])
        module = GuardRingEvidence(16, 6, levels=1, dim=1, points=8)
        core, ring = module.sample_points(torch.zeros(1, 1, 16), boxes, torch.tensor([1 / 64, 1 / 64]))
        target = GuardRingEvidence._pool([level], core, torch.zeros(1, 1, 8))
        clutter = GuardRingEvidence._pool([level], ring, torch.zeros(1, 1, 8))
        self.assertGreater(target.item(), .9)
        self.assertLess(clutter.item(), .1)

    def test_gate_starts_uniform_and_residual_starts_at_zero(self):
        levels = [torch.rand(2, 8, 16, 16), torch.rand(2, 8, 8, 8)]
        delta, target, gate = self.module(self.query, self.boxes, levels, torch.rand(2, 5, 6))
        self.assertEqual(delta.abs().sum().item(), 0.)
        self.assertEqual(target.shape, (2, 5, 8))
        self.assertTrue(torch.allclose(gate, torch.full_like(gate, 1 / 3)))


class FullModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = YAMLConfig(str(CONFIG))
        torch.manual_seed(0)
        cls.model = cls.cfg.model
        cls.criterion = cls.cfg.criterion

    def targets(self):
        make = lambda boxes, labels: dict(boxes=torch.tensor(boxes), labels=torch.tensor(labels), image_id=torch.tensor([1]),
                                          orig_size=torch.tensor([256, 256]), area=torch.ones(len(labels)), iscrowd=torch.zeros(len(labels)))
        return [make([[.5, .5, .1, .1], [.2, .5, .1, .2]], [4, 1]), make([[.3, .3, .1, .1]], [3])]

    def test_zero_alpha_keeps_baseline_predictions(self):
        baseline = YAMLConfig(str(BASELINE)).model.eval()
        ours = self.model.eval()
        missing, unexpected = ours.load_state_dict(baseline.state_dict(), strict=False)
        self.assertEqual(unexpected, [])
        self.assertTrue(all('grea' in name for name in missing), missing)
        images = torch.rand(1, 3, 640, 640)
        with torch.no_grad():
            first, second = baseline(images), ours(images)
        self.assertTrue(torch.allclose(first['pred_boxes'], second['pred_boxes'], atol=1e-6))
        self.assertTrue(torch.allclose(first['pred_logits'], second['pred_logits'], atol=1e-5))

    def test_every_new_parameter_learns_and_gradient_reaches_low_level_backbone(self):
        self.model.train()
        self.criterion.train()

        def set_alpha(value):
            with torch.no_grad():
                for layer in self.model.decoder.decoder.grea_layers.values():
                    layer.alpha.fill_(value)

        self.addCleanup(set_alpha, 0.)
        set_alpha(.1)
        images = torch.rand(2, 3, 320, 320)
        outputs = self.model(images, targets=self.targets())
        self.assertEqual(outputs['grea_evidence_logits'].shape, outputs['pred_logits'].shape)
        self.assertEqual(outputs['grea_gate'].shape[:2], outputs['pred_logits'].shape[:2])
        losses = self.criterion(outputs, self.targets(), epoch=0, step=0, global_step=0, epoch_step=1)
        self.assertIn('loss_grea_evidence', losses)
        self.model.zero_grad()
        sum(losses.values()).backward()
        missing = [n for n, p in self.model.named_parameters() if 'grea' in n and p.requires_grad and p.grad is None]
        self.assertEqual(missing, [])
        stage = next(p for n, p in self.model.backbone.named_parameters() if 'stages.0' in n and p.grad is not None)
        self.assertGreater(stage.grad.abs().sum().item(), 0)
        metrics = self.criterion.bccq_metrics
        self.assertEqual(metrics['grea_positive_c4'].item(), 1.)
        self.assertAlmostEqual(sum(metrics[f'grea_gate_sum_c4_{k}'].item() for k in ('target', 'clutter', 'contrast')), 1., places=4)

    def test_inference_needs_no_training_outputs(self):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(torch.rand(1, 3, 640, 640))
        self.assertEqual(outputs['pred_boxes'].shape, (1, 300, 4))
        self.assertNotIn('grea_evidence_logits', outputs)

    def test_config_differs_from_baseline_only_by_the_module(self):
        ours, theirs = YAMLConfig(str(CONFIG)).yaml_cfg, YAMLConfig(str(BASELINE)).yaml_cfg
        allowed = {'__include__', 'experiment_protocol_document', 'experiment_baseline_revision', 'output_dir', 'HGNetv2', 'DFINETransformer', 'DEIMCriterion'}
        for key in set(ours) - allowed:
            self.assertEqual(ours[key], theirs.get(key), key)
        self.assertEqual({k: v for k, v in ours['DFINETransformer'].items() if not k.startswith(('use_grea', 'grea_'))}, theirs['DFINETransformer'])
        self.assertEqual({k: v for k, v in ours['DEIMCriterion']['weight_dict'].items() if k != 'loss_grea_evidence'}, theirs['DEIMCriterion']['weight_dict'])
        self.assertEqual({k: v for k, v in ours['HGNetv2'].items() if k != 'return_idx'}, {k: v for k, v in theirs['HGNetv2'].items() if k != 'return_idx'})
        self.assertEqual(YAMLConfig(str(BASELINE)).model.decoder.grea_pyramid, None)


if __name__ == '__main__':
    unittest.main()
