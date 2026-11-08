"""Q-MAL: the edge-tolerant MAL quality target and its routing through DEIMCriterion."""

import unittest
from pathlib import Path

import torch

from engine.core.yaml_utils import load_config
from engine.deim.box_ops import box_cxcywh_to_xyxy, box_iou
from engine.deim.deim_criterion import DEIMCriterion, edge_tolerant_iou
from engine.deim.matcher import HungarianMatcher

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/deim_dfine/deim_hgnetv2_m_sardet100k_qmal_36e_b32.yml'


def pixel_box(x0, y0, x1, y1, size=(640, 640)):
    """Normalized cxcywh from pixel xyxy on an (H, W) input."""
    h, w = size
    return torch.tensor([[(x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h]], dtype=torch.float64)


def plain_iou(a, b):
    return torch.diag(box_iou(box_cxcywh_to_xyxy(a), box_cxcywh_to_xyxy(b))[0])


class EdgeTolerantIoUTest(unittest.TestCase):
    def test_zero_tolerance_is_plain_iou(self):
        torch.manual_seed(0)
        a = torch.rand(64, 4, dtype=torch.float64) * .4 + .1
        b = a + torch.randn_like(a) * .02
        torch.testing.assert_close(edge_tolerant_iou(a, b, 0., (640, 800)).double(), plain_iou(a, b).double(),
                                   rtol=1e-5, atol=1e-6)

    def test_errors_within_tolerance_are_exact(self):
        target = pixel_box(100, 100, 108, 108)
        for pred in [(101, 100, 108, 108), (99, 101, 109, 107), (100.6, 99.2, 107.5, 108.9)]:
            self.assertAlmostEqual(edge_tolerant_iou(pixel_box(*pred), target, 1., (640, 640)).item(), 1., places=5)

    def test_larger_errors_still_penalized_like_one_pixel_less(self):
        target = pixel_box(100, 100, 108, 108)
        two_px = edge_tolerant_iou(pixel_box(102, 100, 108, 108), target, 1., (640, 640)).item()
        one_px_plain = plain_iou(pixel_box(101, 100, 108, 108), target).item()
        self.assertAlmostEqual(two_px, one_px_plain, places=5)
        self.assertAlmostEqual(two_px, 7 / 8, places=5)

    def test_large_boxes_barely_change(self):
        target = pixel_box(100, 100, 228, 228)
        pred = pixel_box(103, 100, 228, 228)
        plain, tolerant = plain_iou(pred, target).item(), edge_tolerant_iou(pred, target, 1., (640, 640)).item()
        self.assertLess(tolerant - plain, .01)

    def test_tolerance_is_in_pixels_of_each_axis(self):
        # Non-square input: one pixel horizontally is 1/800, vertically 1/480.
        size = (480, 800)
        target = pixel_box(100, 100, 110, 110, size)
        self.assertAlmostEqual(edge_tolerant_iou(pixel_box(101, 101, 110, 110, size), target, 1., size).item(), 1., places=5)
        self.assertLess(edge_tolerant_iou(pixel_box(102, 100, 110, 110, size), target, 1., size).item(), 1.)

    def test_disjoint_boxes_stay_valid(self):
        iou = edge_tolerant_iou(pixel_box(0, 0, 2, 2), pixel_box(10, 10, 12, 12), 5., (640, 640))
        self.assertTrue(torch.isfinite(iou).all())
        self.assertGreaterEqual(iou.item(), 0.)


def criterion(tolerance):
    weights = dict(loss_mal=1, loss_bbox=5, loss_giou=2)
    return DEIMCriterion(HungarianMatcher(dict(cost_class=2, cost_bbox=5, cost_giou=2), use_focal_loss=True),
                         weights, ['mal'], num_classes=3, gamma=1.5, mal_edge_tolerance_px=tolerance)


class CriterionRoutingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        target = pixel_box(100, 100, 108, 108).float()
        self.targets = [dict(labels=torch.tensor([1]), boxes=target)]
        boxes = torch.rand(1, 10, 4) * .5 + .1
        boxes[0, 0] = pixel_box(101, 100, 108, 108).float()[0]  # one pixel off: matched query
        logits = torch.full((1, 10, 3), -4.)
        logits[0, 0, 1] = 2.
        self.outputs = dict(pred_logits=logits, pred_boxes=boxes)

    def test_zero_tolerance_matches_original_mal(self):
        original = DEIMCriterion(HungarianMatcher(dict(cost_class=2, cost_bbox=5, cost_giou=2), use_focal_loss=True),
                                 dict(loss_mal=1), ['mal'], num_classes=3, gamma=1.5)
        indices = original.matcher(self.outputs, self.targets)['indices']
        a = original.loss_labels_mal(self.outputs, self.targets, indices, 1.)['loss_mal']
        q = criterion(0.)
        q.input_size = (640, 640)
        b = q.loss_labels_mal(self.outputs, self.targets, indices, 1.)['loss_mal']
        torch.testing.assert_close(a, b)

    def test_tolerance_raises_target_for_one_pixel_error(self):
        indices = criterion(0.).matcher(self.outputs, self.targets)['indices']
        losses = {}
        for tol in (0., 1.):
            c = criterion(tol)
            c.input_size = (640, 640)
            losses[tol] = c.loss_labels_mal(self.outputs, self.targets, indices, 1.)['loss_mal'].item()
        # The matched logit is high (p=0.88); a target of 1 instead of ~0.69 lowers its BCE.
        self.assertLess(losses[1.], losses[0.])

    def test_forward_requires_input_size(self):
        c = criterion(1.)
        with self.assertRaises(ValueError):
            c(self.outputs, self.targets, epoch=0)
        self.assertIsNone(c.input_size)

    def test_config_enables_only_q_mal(self):
        cfg = load_config(str(CONFIG))
        base = load_config(str(ROOT / 'configs/deim_dfine/deim_hgnetv2_m_sardet100k_36e_b32_nozoom_nocrop.yml'))
        self.assertEqual(cfg['DEIMCriterion']['mal_edge_tolerance_px'], 1.0)
        self.assertNotIn('mal_edge_tolerance_px', base['DEIMCriterion'])
        self.assertIsNone(cfg['DEIMCriterion'].get('boxes_weight_format'))
        for key in ('epoches', 'flat_epoch', 'no_aug_epoch', 'seed', 'train_dataloader', 'val_dataloader', 'optimizer'):
            self.assertEqual(cfg[key], base[key], key)
        self.assertNotEqual(cfg['output_dir'], base['output_dir'])


if __name__ == '__main__':
    unittest.main()
