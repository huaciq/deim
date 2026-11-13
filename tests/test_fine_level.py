"""Targeted checks for the stride-4 level. Run on CPU: python -m unittest tests.test_fine_level -v"""
import unittest
from pathlib import Path

import torch

from engine.core import YAMLConfig
from engine.deim.dfine_decoder import MSDeformableAttention
from engine.solver._solver import BaseSolver

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/deim_dfine/deim_hgnetv2_m_sardet100k_fine_level_36e_b32.yml'
BASELINE = ROOT / 'configs/deim_dfine/deim_hgnetv2_m_sardet100k_36e_b32_nozoom_nocrop.yml'
OLD_SERVER = ROOT / 'configs/deim_dfine/deim_hgnetv2_m_sardet100k_fine_level_36e_b32_old_server.yml'
OLD_BASELINE = ROOT / 'configs/deim_dfine/deim_hgnetv2_m_sardet100k_36e_b32_nozoom_nocrop_old_server.yml'
GENERATED = {'decoder.anchors', 'decoder.valid_mask'}


def build(path):
    torch.manual_seed(0)
    return YAMLConfig(str(path)).model


class Attention(unittest.TestCase):
    def test_fine_points_only_add_layers(self):
        torch.manual_seed(0)
        base = MSDeformableAttention(16, 4, 3, [3, 6, 3])
        fine = MSDeformableAttention(16, 4, 3, [3, 6, 3], fine_points=3)
        for name, value in base.state_dict().items():
            self.assertEqual(value.shape, fine.state_dict()[name].shape, name)

    def test_samples_from_the_fine_level(self):
        torch.manual_seed(0)
        layer = MSDeformableAttention(16, 4, 3, [3, 6, 3], fine_points=3)
        query = torch.randn(1, 2, 16)
        reference = torch.tensor([[[[.5, .5, .1, .1]], [[.2, .2, .1, .1]]]])
        shapes = [[8, 8], [4, 4], [2, 2], [16, 16]]
        value = [torch.zeros(1, 4, 4, h * w) for h, w in shapes[:3]] + [torch.ones(1, 4, 4, 256)]
        output = layer(query, reference, value, shapes)
        # The other levels are zero, so the result is the attention share of the fine points.
        self.assertTrue((output > 0).all())


class Model(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = build(BASELINE)
        cls.fine = build(CONFIG)

    def test_every_baseline_parameter_loads_with_the_same_shape(self):
        matched, info = BaseSolver._matched_state(self.fine.state_dict(), self.base.state_dict())
        unmatched = [k for k in info['unmatched'] if k not in GENERATED]
        self.assertEqual(unmatched, [])
        for name in self.base.state_dict():
            self.assertTrue(name in matched or name in GENERATED, name)

    def test_new_layers_do_not_shift_the_common_initial_weights(self):
        fine = self.fine.state_dict()
        for name, value in self.base.state_dict().items():
            if name not in GENERATED and 'backbone' not in name:
                self.assertTrue(torch.equal(value, fine[name]), name)

    def test_anchors_of_the_fine_level(self):
        decoder = self.fine.decoder
        self.assertEqual(decoder.feat_strides, [8, 16, 32, 4])
        anchors, _ = decoder._generate_anchors([[80, 80], [40, 40], [20, 20], [160, 160]])
        self.assertEqual(anchors.shape[1], 8400 + 25600)
        wide = decoder._generate_anchors([[80, 80]])[0]
        base_anchor = self.base.decoder._generate_anchors([[80, 80], [40, 40], [20, 20]])[0]
        self.assertTrue(torch.allclose(wide[:, 3000:3010], base_anchor[:, 3000:3010]))
        # Inverse sigmoid of 0.025 is the width of the stride-4 anchors, half of stride 8.
        self.assertAlmostEqual(float(anchors[0, -1000:, 2:].sigmoid().median()), .025, places=4)

    def test_training_and_eval_forward_reach_the_new_layers(self):
        model = self.fine
        model.train()
        images = torch.rand(2, 3, 480, 480)
        targets = [{'boxes': torch.tensor([[.5, .5, .05, .05]]), 'labels': torch.tensor([3])}, {'boxes': torch.tensor([[.3, .3, .1, .2]]), 'labels': torch.tensor([4])}]
        output = model(images, targets)
        loss = output['pred_boxes'].sum() + output['pred_logits'].sum() + output['enc_aux_outputs'][0]['pred_logits'].sum()
        loss.backward()
        self.assertIsNotNone(model.decoder.fine_neck.s2_conv.weight.grad)
        self.assertGreater(float(model.decoder.fine_neck.out.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.decoder.decoder.layers[0].cross_attn.fine_weights.bias.grad.abs().sum()), 0)
        self.assertGreater(float(model.backbone.stages[0].parameters().__next__().grad.abs().sum()), 0)
        model.zero_grad()
        model.eval()
        with torch.no_grad():
            result = model(torch.rand(1, 3, 640, 640))
        self.assertEqual(result['pred_boxes'].shape[:2], (1, 300))
        self.assertEqual(model.decoder.anchors.shape[1], 34000)

    def test_config_differs_from_e000_only_by_the_module(self):
        fine, base = YAMLConfig(str(CONFIG)).yaml_cfg, YAMLConfig(str(BASELINE)).yaml_cfg
        changed = {k for k in set(fine) | set(base) if fine.get(k) != base.get(k)} - {'__include__'}
        self.assertEqual(changed, {'HGNetv2', 'DFINETransformer', 'output_dir', 'experiment_protocol_document', 'experiment_baseline_revision'})
        self.assertEqual(fine['HGNetv2']['return_idx'], [0, 1, 2, 3])
        difference = {k for k in set(fine['DFINETransformer']) | set(base['DFINETransformer']) if fine['DFINETransformer'].get(k) != base['DFINETransformer'].get(k)}
        self.assertEqual(difference, {'use_fine_level', 'fine_s2_channels', 'fine_dim', 'fine_points', 'fine_checkpoint'})

    def test_old_server_config_differs_from_e000_old_only_by_the_module(self):
        old, base, new = YAMLConfig(str(OLD_SERVER)).yaml_cfg, YAMLConfig(str(OLD_BASELINE)).yaml_cfg, YAMLConfig(str(CONFIG)).yaml_cfg
        changed = {k for k in set(old) | set(base) if old.get(k) != base.get(k)} - {'__include__'}
        self.assertEqual(changed, {'HGNetv2', 'DFINETransformer', 'output_dir', 'experiment_protocol_document', 'experiment_baseline_revision'})
        self.assertEqual(old['DFINETransformer'], new['DFINETransformer'])
        self.assertEqual(old['HGNetv2'], new['HGNetv2'])
        for split in ('train_dataloader', 'val_dataloader'):
            self.assertTrue(old[split]['dataset']['img_folder'].startswith('/data2/zxy/sar/datasets/SARDet_100K/'))


class Checkpoint(unittest.TestCase):
    def test_checkpoint_keeps_outputs_gradients_and_one_bn_update(self):
        from engine.deim.fine_level import FineLevelNeck
        torch.manual_seed(0)
        plain = FineLevelNeck(8, 16, 8).train()
        stored = FineLevelNeck(8, 16, 8, checkpoint=True).train()
        stored.load_state_dict(plain.state_dict())
        s2, p3 = torch.randn(2, 8, 12, 12), torch.randn(2, 16, 6, 6)
        outputs = []
        for module in (plain, stored):
            a, b = s2.clone().requires_grad_(), p3.clone().requires_grad_()
            out = module(a, b)
            (out * torch.linspace(-1, 1, out.numel()).reshape(out.shape)).sum().backward()
            outputs.append((out.detach(), a.grad, b.grad, {n: p.grad for n, p in module.named_parameters()}))
        (o1, a1, b1, g1), (o2, a2, b2, g2) = outputs
        self.assertTrue(torch.allclose(o1, o2, atol=1e-6))
        self.assertTrue(torch.allclose(a1, a2, atol=1e-5) and torch.allclose(b1, b2, atol=1e-5))
        for name in g1:
            self.assertTrue(torch.allclose(g1[name], g2[name], atol=1e-5), name)
        for name, buffer in plain.state_dict().items():
            self.assertTrue(torch.equal(buffer, stored.state_dict()[name]), name)
        self.assertEqual(int(stored.s2_norm.num_batches_tracked), 1)

    def test_eval_does_not_checkpoint(self):
        from engine.deim.fine_level import FineLevelNeck
        neck = FineLevelNeck(8, 16, 8, checkpoint=True).eval()
        with torch.no_grad():
            self.assertEqual(neck(torch.randn(1, 8, 12, 12), torch.randn(1, 16, 6, 6)).shape, (1, 16, 12, 12))


if __name__ == '__main__':
    unittest.main()
