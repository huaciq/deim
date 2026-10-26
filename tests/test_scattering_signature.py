"""Functional regression tests for SCS geometry, matching and baseline parity."""
import random
import unittest

import torch

from engine.core.yaml_utils import load_config
from engine.data.dataloader import BatchImageCollateFunction
from engine.deim.dfine_decoder import DFINETransformer
from engine.deim.scattering_signature import SCSLoss, scattering_descriptors


class ScatteringSignatureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_descriptor_shape_range_constant_and_invalid_ring(self):
        image = torch.full((1, 3, 40, 40), .2)
        target = dict(boxes=torch.tensor([[.5, .5, .25, .25]]))
        signature, valid = scattering_descriptors(image, [target])[0]
        self.assertEqual(signature.shape, (1, 28))
        self.assertTrue(valid.all())
        self.assertTrue(torch.isfinite(signature).all())
        torch.testing.assert_close(signature[0, :16], torch.zeros(16))
        torch.testing.assert_close(signature[0, 16:24].sum(), torch.tensor(1.))
        torch.testing.assert_close(signature[0, 24:], torch.zeros(4))
        _, valid = scattering_descriptors(image, [dict(boxes=torch.tensor([[.5, .5, 1., 1.]]))])[0]
        self.assertFalse(valid.any())

    def test_positive_affine_intensity_change_and_neighbor_exclusion(self):
        torch.manual_seed(1)
        image = torch.rand(1, 3, 40, 40) * .5 + .1
        target = dict(boxes=torch.tensor([[.4, .5, .3, .3]]))
        a, _ = scattering_descriptors(image, [target], epsilon=1e-8)[0]
        b, _ = scattering_descriptors(image * .7 + .08, [target], epsilon=1e-8)[0]
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
        image = torch.full((1, 3, 40, 40), .2)
        boxes = torch.tensor([[.4, .5, .3, .3], [.7, .5, .3, .3]])
        first, _ = scattering_descriptors(image, [dict(boxes=boxes)])[0]
        image[:, :, 14:26, 22:34] = .9
        second, _ = scattering_descriptors(image, [dict(boxes=boxes)])[0]
        torch.testing.assert_close(first[0], second[0])

    def test_mixup_keeps_description_box_order_and_original_augmentation(self):
        torch.manual_seed(2)
        images = torch.rand(2, 3, 40, 40)
        def items():
            return [(im.clone(), dict(boxes=torch.tensor([[.5, .5, .3, .3]]),
                                      labels=torch.tensor([i]), area=torch.tensor([144.]),
                                      image_id=torch.tensor([i]))) for i, im in enumerate(images)]
        expected = scattering_descriptors(images, [t for _, t in items()])
        baseline = BatchImageCollateFunction(mixup_prob=1, mixup_epochs=[4, 49])
        scs = BatchImageCollateFunction(mixup_prob=1, mixup_epochs=[4, 49], scs_signatures=True)
        baseline.set_epoch(4)
        scs.set_epoch(4)
        random.seed(10)
        base_images, base_targets = baseline(items())
        random.seed(10)
        scs_images, scs_targets = scs(items())
        torch.testing.assert_close(base_images, scs_images, rtol=0, atol=0)
        for i in range(2):
            torch.testing.assert_close(base_targets[i]['boxes'], scs_targets[i]['boxes'])
            torch.testing.assert_close(scs_targets[i]['scs_signature'],
                                       torch.cat([expected[i][0], expected[i - 1][0]]))
            self.assertEqual(len(scs_targets[i]['scs_valid']), len(scs_targets[i]['labels']))

    def test_matched_only_losses_margin_and_invalid_description(self):
        prediction = torch.zeros(1, 3, 28, requires_grad=True)
        logits = torch.zeros(1, 3, 7, requires_grad=True)
        targets = [dict(labels=torch.tensor([3, 6]), scs_signature=torch.ones(2, 28),
                        scs_valid=torch.tensor([True, False]))]
        indices = [(torch.tensor([2, 0]), torch.tensor([0, 1]))]
        loss = SCSLoss(7, [[3, 4, 1.]])
        result = loss(dict(pred_scs=prediction, pred_logits=logits), targets, indices)
        torch.testing.assert_close(result['loss_scs_sig'], torch.tensor(.95))
        torch.testing.assert_close(result['loss_scs_pair'], torch.tensor(.25))
        sum(result.values()).backward()
        self.assertTrue(prediction.grad[0, 2].abs().sum() > 0)
        self.assertEqual(float(prediction.grad[0, :2].abs().sum()), 0)
        self.assertEqual(float(logits.grad[0, :2].abs().sum()), 0)
        self.assertLess(float(logits.grad[0, 2, 3]), 0)
        self.assertGreater(float(logits.grad[0, 2, 4]), 0)
        empty = [dict(labels=torch.empty(0, dtype=torch.long), scs_signature=torch.empty(0, 28),
                      scs_valid=torch.empty(0, dtype=torch.bool))]
        result = loss(dict(pred_scs=prediction, pred_logits=logits), empty,
                      [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))])
        self.assertTrue(all(torch.isfinite(v) and v == 0 for v in result.values()))

    def test_decoder_final_queries_dn_alignment_and_disabled_parity(self):
        kwargs = dict(num_classes=7, hidden_dim=32, feat_channels=[32, 32, 32], nhead=4,
                      num_queries=5, num_layers=2, dim_feedforward=64, reg_max=8,
                      num_denoising=4, eval_spatial_size=None)
        torch.manual_seed(3)
        baseline = DFINETransformer(**kwargs)
        rng_baseline = torch.get_rng_state()
        torch.manual_seed(3)
        scs = DFINETransformer(**kwargs, use_scs=True, scs_hidden_dim=16)
        torch.testing.assert_close(torch.get_rng_state(), rng_baseline, rtol=0, atol=0)
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(value, scs.state_dict()[name], rtol=0, atol=0)
        features = [torch.randn(1, 32, size, size) for size in [4, 2, 1]]
        targets = [dict(labels=torch.tensor([3]), boxes=torch.tensor([[.5, .5, .3, .3]]))]
        baseline.train()
        scs.train()
        captured = []
        hook = scs.scs_head.register_forward_pre_hook(lambda module, inputs:captured.append(inputs[0]))
        torch.manual_seed(4)
        base = baseline(features, targets)
        torch.manual_seed(4)
        output = scs(features, targets)
        hook.remove()
        self.assertEqual(output['pred_scs'].shape, (1, 5, 28))
        self.assertEqual(captured[0].shape, (1, 5, 32))
        self.assertEqual(output['dn_meta']['dn_num_split'][1], 5)
        for name in ['pred_logits', 'pred_boxes', 'pred_corners']:
            torch.testing.assert_close(output[name], base[name], rtol=0, atol=0)
        output['pred_scs'].square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in scs.scs_head.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in scs.decoder.layers[-1].parameters()))
        baseline.eval()
        scs.eval()
        with torch.no_grad():
            base, output = baseline(features), scs(features)
        self.assertNotIn('pred_scs', output)
        for name in ['pred_logits', 'pred_boxes', 'pred_corners']:
            torch.testing.assert_close(output[name], base[name], rtol=0, atol=0)

    def test_only_scs_fields_differ_from_frozen_protocol(self):
        baseline = load_config('configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_102e_b28.yml', {})
        scs = load_config('configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_scs_m1_102e_b28.yml', {})
        for section in ['optimizer', 'epoches', 'warmup_iter', 'flat_epoch', 'no_aug_epoch',
                        'lr_gamma', 'eval_spatial_size', 'use_amp', 'use_ema', 'seed']:
            self.assertEqual(baseline[section], scs[section])
        for name in ['train_dataloader', 'val_dataloader']:
            for key in ['total_batch_size', 'dataset']:
                self.assertEqual(baseline[name][key], scs[name][key])
        counts = {0:3023, 3:447, 4:1755, 5:1930}
        rows = {}
        for c, k, w in scs['DEIMCriterion']['scs_pairs']:
            rows.setdefault(c, []).append((k, w))
        for c, rivals in rows.items():
            norm = sum(counts[k] ** -.5 for k, _ in rivals)
            for k, weight in rivals:
                self.assertAlmostEqual(weight, counts[k] ** -.5 / norm)
        self.assertNotIn(6, rows)


if __name__ == '__main__':
    unittest.main()
