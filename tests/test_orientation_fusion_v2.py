"""E013-v2 numerical contracts: BN replay, boundaries, pretraining and encoder split."""
import copy
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from engine.core import YAMLConfig
from engine.core.yaml_utils import load_config
from engine.deim.orientation_projection_fusion import ProjectedOrientationFusion

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'configs/deim_dfine/deim_hgnetv2_m_sardet100k_36e_b32_nozoom_nocrop.yml'
V2 = ROOT / 'configs/deim_dfine/deim_hgnetv2_m_sardet100k_orientation_fusion_v2_36e_b32.yml'


class Replay(unittest.TestCase):
    def test_checkpoint_preserves_bn_outputs_gradients_and_running_state_over_updates(self):
        for momentum in (0.1, None):
            torch.manual_seed(19)
            encoder = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8, momentum=momentum),
                                    nn.ReLU(), nn.Conv2d(8, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8, momentum=momentum))
            fusion = ProjectedOrientationFusion(hidden_dim=8, num_levels=1, groups=2)
            nn.init.normal_(fusion.gates[0][-1].weight, std=0.2)
            images = torch.rand(3, 3, 24, 24) + torch.linspace(0, 2, 24)[None, None, :, None]
            results = []
            for flag in (False, True):
                enc, gate = copy.deepcopy(encoder).train(), copy.deepcopy(fusion).train()
                gate.checkpoint_views = flag
                opt = torch.optim.SGD(list(enc.parameters()) + list(gate.parameters()), lr=0.01)
                steps = []
                for step in range(2):
                    opt.zero_grad(set_to_none=True)
                    out = gate(images, lambda x: [enc(x)], checkpoint_modules=(enc,))[0]
                    out.square().mean().backward()
                    self.assertEqual(enc[1].num_batches_tracked.item(), 2 * (step + 1))
                    steps.append((out.detach().clone(), [p.grad.clone() for p in enc.parameters()],
                                  [p.grad.clone() for p in gate.parameters()], copy.deepcopy(enc.state_dict())))
                    opt.step()
                results.append(steps)
            for plain, replayed in zip(*results):
                torch.testing.assert_close(plain[0], replayed[0], atol=1e-6, rtol=1e-5)
                for a, b in zip(plain[1] + plain[2], replayed[1] + replayed[2]):
                    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
                for name in plain[3]:
                    torch.testing.assert_close(plain[3][name], replayed[3][name], atol=1e-6, rtol=1e-5)

    def test_invalid_aligned_cells_use_original_even_when_gate_prefers_flip(self):
        fusion = ProjectedOrientationFusion(hidden_dim=8, num_levels=1, groups=2, checkpoint_views=False).eval()
        with torch.no_grad():
            fusion.gates[0][-1].bias[:2] = -100
            fusion.gates[0][-1].bias[2:] = 100
        original, flipped = torch.randn(2, 8, 8, 8), torch.randn(2, 8, 8, 8)
        calls = iter((original, flipped))
        out = fusion(torch.zeros(2, 3, 64, 64), lambda x: [next(calls)])[0]
        torch.testing.assert_close(out[..., 0, :], original[..., 0, :], atol=0, rtol=0)
        torch.testing.assert_close(out[..., 1:, :], fusion.align(flipped, 'vflip')[..., 1:, :], atol=0, rtol=0)


class Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        torch.manual_seed(0)
        cls.base = YAMLConfig(str(BASE)).model.eval()
        torch.manual_seed(0)
        cls.v2 = YAMLConfig(str(V2)).model.eval()

    def test_protocol_and_pretraining_are_preserved(self):
        original, revised = load_config(str(BASE)), load_config(str(V2))
        for key in ('train_dataloader', 'val_dataloader', 'optimizer', 'DEIMCriterion', 'DFINETransformer',
                    'HGNetv2', 'HybridEncoder', 'epoches', 'seed', 'use_amp', 'sync_bn', 'ema'):
            self.assertEqual(original[key], revised[key], key)
        state = self.v2.state_dict()
        for name, value in self.base.state_dict().items():
            torch.testing.assert_close(value, state[name], atol=0, rtol=0, msg=name)
        self.assertEqual(sum(p.numel() for p in self.v2.parameters()) - sum(p.numel() for p in self.base.parameters()), 101808)
        legacy = ROOT / 'configs/deim_dfine/deim_hgnetv2_m_sardet100k_orientation_fusion_36e_b32.yml'
        self.assertNotIn('stage', load_config(str(legacy))['DEIM']['orientation_fusion'])

    def test_single_encoder_body_and_original_gate_recover_baseline(self):
        self.v2.load_state_dict(self.base.state_dict(), strict=False)
        fusion = self.v2.orientation_fusion
        saved = [g[-1].bias.detach().clone() for g in fusion.gates]
        calls = {'backbone': 0, 'aifi': 0, 'fpn0': 0}
        def count(name):
            def hook(*args):
                calls[name] += 1
            return hook
        handles = [self.v2.backbone.register_forward_hook(count('backbone')),
                   self.v2.encoder.encoder[0].register_forward_hook(count('aifi')),
                   self.v2.encoder.fpn_blocks[0].register_forward_hook(count('fpn0'))]
        try:
            with torch.no_grad():
                for g in fusion.gates:
                    g[-1].bias[:fusion.groups] = 60
                images = torch.rand(1, 3, 640, 640)
                a, b = self.base(images), self.v2(images)
            torch.testing.assert_close(a['pred_logits'], b['pred_logits'], atol=1e-3, rtol=1e-4)
            torch.testing.assert_close(a['pred_boxes'], b['pred_boxes'], atol=1e-4, rtol=1e-4)
            self.assertEqual(calls, {'backbone': 2, 'aifi': 1, 'fpn0': 1})
        finally:
            for handle in handles:
                handle.remove()
            with torch.no_grad():
                for gate, bias in zip(fusion.gates, saved):
                    gate[-1].bias.copy_(bias)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
