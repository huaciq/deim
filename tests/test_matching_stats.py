"""Training-batch matching statistics in DEIMCriterion: values on a hand-built case, and no effect on the losses."""

import unittest
from pathlib import Path

import torch

from engine.core.yaml_utils import load_config
from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.matcher import HungarianMatcher

ROOT = Path(__file__).resolve().parents[1]
A = [.3, .3, .1, .1]  # GT of class 1 (the tracked class)
B = [.7, .7, .1, .1]  # GT of class 0
Q = 6


def stage(a_query, decoy=False):
    """Query a_query sits exactly on A with a confident class-1 logit; query 1 on B. Others are far away."""
    boxes = torch.tensor([[.1 + .05 * q, .9, .05, .05] for q in range(Q)])
    logits = torch.full((Q, 3), -4.)
    boxes[a_query], logits[a_query, 1] = torch.tensor(A), 3.
    boxes[1], logits[1, 0] = torch.tensor(B), 3.
    if decoy:  # query 0 shifted off A and unconfident, so the matcher prefers a_query here
        boxes[0], logits[0, 1] = torch.tensor([.33, .3, .1, .1]), -2.
    return dict(pred_logits=logits[None], pred_boxes=boxes[None])


def outputs():
    dec1, dec2, dec3, dec4 = stage(2, decoy=True), stage(0), stage(0), stage(0)
    return dict(dec4, aux_outputs=[dec1, dec2, dec3], pre_outputs=stage(0), enc_aux_outputs=[stage(0)],
                enc_meta=dict(class_agnostic=False))


def criterion(log):
    return DEIMCriterion(HungarianMatcher(dict(cost_class=2, cost_bbox=5, cost_giou=2), use_focal_loss=True),
                         dict(loss_mal=1, loss_bbox=5, loss_giou=2), ['mal', 'boxes'], num_classes=3, gamma=1.5,
                         log_matching_stats=log, matching_stats_class=1 if log else None)


TARGETS = [dict(labels=torch.tensor([1, 0]), boxes=torch.tensor([A, B]))]


class MatchingStatsTest(unittest.TestCase):
    def test_losses_unchanged(self):
        off, on = criterion(False)(outputs(), TARGETS), criterion(True)(outputs(), TARGETS)
        self.assertEqual(off.keys(), on.keys())
        for key in off:
            torch.testing.assert_close(off[key], on[key], rtol=0, atol=0, msg=key)

    def test_disabled_records_nothing(self):
        c = criterion(False)
        c(outputs(), TARGETS)
        self.assertEqual(c.matching_metrics, {})

    def test_values(self):
        c = criterion(True)
        c(outputs(), TARGETS)
        m = c.matching_metrics
        expected = {
            'gt': 2, 'gt_tank': 1,
            # Union keeps (0, A), (2, A) and (1, B): query 2 is matched only at decoder 1.
            'union': 3, 'union_tank': 2,
            'same_dec_all': 1, 'same_dec_all_tank': 0,
            'same_dec_1_2': 1, 'same_dec_2_3': 2, 'same_dec_3_4': 2,
            # Query 0 is background only at decoder 1; query 2 at decoders 2-4.
            'boxpos_clsbg_dec1': 1, 'boxpos_clsbg_dec2': 1, 'boxpos_clsbg_dec3': 1, 'boxpos_clsbg_dec4': 1,
            'boxpos_clsbg_dec1_tank': 1, 'clsbg_run2': 1, 'clsbg_run2_tank': 1, 'clsbg_all': 0,
            'iou_enc': 2, 'iou50_dec1': 2, 'iou75_dec4_tank': 1,
        }
        for name, value in expected.items():
            self.assertAlmostEqual(m['match_' + name], value, places=5, msg=name)
        self.assertTrue(all(isinstance(v, float) for v in m.values()))

    def test_interval_samples_calls(self):
        c = DEIMCriterion(HungarianMatcher(dict(cost_class=2, cost_bbox=5, cost_giou=2), use_focal_loss=True),
                          dict(loss_mal=1, loss_bbox=5, loss_giou=2), ['mal', 'boxes'], num_classes=3, gamma=1.5,
                          log_matching_stats=True, matching_stats_class=1, matching_stats_interval=3)
        recorded = []
        for _ in range(7):
            c(outputs(), TARGETS)
            recorded.append(bool(c.matching_metrics))
        self.assertEqual(recorded, [True, False, False, True, False, False, True])

    def test_e1_config(self):
        cfg = load_config(str(ROOT / 'configs/deim_dfine/deim_hgnetv2_m_sardet100k_match_cls05_36e_b32.yml'))
        base = load_config(str(ROOT / 'configs/deim_dfine/deim_hgnetv2_m_sardet100k_36e_b32_nozoom_nocrop.yml'))
        self.assertEqual(cfg['DEIMCriterion']['matcher']['weight_dict'], dict(cost_class=.5, cost_bbox=5, cost_giou=2))
        self.assertEqual(cfg['DEIMCriterion']['weight_dict'], base['DEIMCriterion']['weight_dict'])
        self.assertTrue(cfg['DEIMCriterion']['log_matching_stats'])
        self.assertEqual(cfg['DEIMCriterion']['matching_stats_class'], 3)
        self.assertEqual(cfg['DEIMCriterion']['matching_stats_interval'], 10)


if __name__ == '__main__':
    unittest.main()
