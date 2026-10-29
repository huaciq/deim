"""Targeted CrossGate-DBDM sampling, gradients and pretrained-path checks."""

from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

from engine.backbone.crossgate_dbdm import CrossGateDBDM
from engine.backbone.hgnetv2 import HGNetv2
from engine.core.yaml_utils import load_config
from engine.core import YAMLConfig
from engine.solver._solver import BaseSolver
from engine.solver.det_engine import _module_training_metrics


ROOT = Path(__file__).resolve().parents[1]


class CrossGateDBDMTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(84)

    def test_zero_offsets_sample_the_four_local_phases_without_channel_mixing(self):
        module = CrossGateDBDM(3, 3)
        for height, width in [(6, 8), (7, 9)]:
            with self.subTest(size=(height, width)):
                features = torch.randn(2, 3, height, width)
                offsets = torch.zeros(2, 8, (height + 1) // 2, (width + 1) // 2)
                actual = module._sample_blocks(features, offsets)
                padded = F.pad(features, (0, width % 2, 0, height % 2), mode='replicate')
                expected = torch.cat([padded[:, :, dy::2, dx::2]
                                      for dy, dx in [(0, 0), (0, 1), (1, 0), (1, 1)]], dim=1)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_offsets_are_differentiable_input_pixel_displacements(self):
        module = CrossGateDBDM(1, 1)
        features = torch.arange(8, dtype=torch.float32).reshape(1, 1, 1, 8).expand(1, 1, 6, 8)
        offsets = torch.zeros(1, 8, 3, 4, requires_grad=True)
        with torch.no_grad():
            offsets[:, 0::2].fill_(.25)
        samples = module._sample_blocks(features, offsets)
        for phase, x_shift in enumerate([0, 1, 0, 1]):
            expected = (torch.arange(4) * 2 + x_shift + .25).clamp_max(7)
            torch.testing.assert_close(samples[0, phase], expected[None].expand(3, 4))
        samples.sum().backward()
        self.assertTrue(torch.isfinite(offsets.grad).all())
        self.assertGreater(float(offsets.grad[:, 0::2].abs().sum()), 0.)

    def test_sobel_energy_is_zero_on_flat_features_and_localizes_an_edge(self):
        module = CrossGateDBDM(3, 3)
        for constant in [0., 1.]:
            energy = module._edge_energy(torch.full((2, 3, 7, 9), constant))
            torch.testing.assert_close(energy, torch.zeros_like(energy), rtol=0, atol=0)
        step = torch.zeros(2, 3, 7, 9)
        step[..., 4:] = 1.
        energy = module._edge_energy(step)
        self.assertGreater(float(energy[..., 3:5].sum()), 0.)
        torch.testing.assert_close(energy[..., :3], torch.zeros_like(energy[..., :3]), rtol=0, atol=0)
        torch.testing.assert_close(energy[..., 5:], torch.zeros_like(energy[..., 5:]), rtol=0, atol=0)

    def test_original_path_initialization_and_branch_gradients_with_autocast(self):
        for amp in [False, True]:
            with self.subTest(amp=amp):
                module = CrossGateDBDM(4, 8).eval()
                features = torch.randn(2, 4, 9, 11, requires_grad=True)
                with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
                    expected = module.lab(module.act(module.bn(module.conv(features))))
                    initial = module(features)
                self.assertEqual(initial.shape, (2, 8, 5, 6))
                self.assertEqual(initial.dtype, expected.dtype)
                torch.testing.assert_close(initial, expected, rtol=0, atol=0)
                initial.float().square().mean().backward()
                self.assertNotEqual(float(module.res_scale.grad), 0.)
                module.zero_grad(set_to_none=True)
                with torch.no_grad():
                    module.res_scale.fill_(.2)
                with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
                    result = module(features)
                    loss = result.float().square().mean()
                loss.backward()
                for branch in ['offset_conv', 'fuse', 'cross_gate']:
                    gradients = [parameter.grad for parameter in getattr(module, branch).parameters()]
                    self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all()
                                        for gradient in gradients), branch)
                    self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0., branch)

    def test_hgnet_stage2_routes_module_and_retains_original_checkpoint_keys(self):
        kwargs = dict(name='B2', use_lab=True, pretrained=False, freeze_at=-1, freeze_norm=False)
        torch.manual_seed(85)
        baseline = HGNetv2(**kwargs).eval()
        original_rng = torch.get_rng_state()
        torch.manual_seed(85)
        adapted = HGNetv2(**kwargs, use_crossgate_dbdm=True).eval()
        torch.testing.assert_close(torch.get_rng_state(), original_rng, rtol=0, atol=0)
        old_state = baseline.state_dict()
        matched, info = BaseSolver._matched_state(adapted.state_dict(), old_state)
        self.assertEqual(set(matched), set(old_state))
        self.assertEqual(info['unmatched'], [])
        self.assertTrue(info['missed'])
        self.assertTrue(all(name.startswith('stages.1.downsample.') for name in info['missed']))
        adapted.load_state_dict(matched, strict=False)
        for name, reference in old_state.items():
            torch.testing.assert_close(adapted.state_dict()[name], reference, rtol=0, atol=0)
        self.assertIsInstance(adapted.stages[1].downsample, CrossGateDBDM)
        self.assertNotIsInstance(adapted.stages[2].downsample, CrossGateDBDM)
        images = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            reference, initial = baseline(images), adapted(images)
        for actual, expected in zip(initial, reference):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        calls = []
        block = adapted.stages[1].downsample
        hook = block.register_forward_hook(lambda module, args, result: calls.append((args[0].shape, result.shape)))
        with torch.no_grad():
            block.res_scale.fill_(.2)
        outputs = adapted(images)
        hook.remove()
        self.assertEqual(calls, [(torch.Size([2, 96, 16, 16]), torch.Size([2, 96, 8, 8]))])
        self.assertEqual([tuple(value.shape) for value in outputs],
                         [(2, 384, 8, 8), (2, 768, 4, 4), (2, 1536, 2, 2)])
        self.assertGreater(float((outputs[0] - reference[0]).detach().abs().max()), 0.)
        sum(value.square().mean() for value in outputs).backward()
        self.assertTrue(torch.isfinite(block.offset_conv[-1].weight.grad).all())
        owner = torch.nn.Module()
        owner.backbone = adapted
        self.assertAlmostEqual(_module_training_metrics(owner)['crossgate_dbdm_res_scale'], .2, places=6)
        config = load_config(str(ROOT / 'configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_58e_b28_old_server.yml'), {})
        groups = YAMLConfig.get_optim_params(config['optimizer'], owner)
        bn_group = next(group for group in groups
                        if any(parameter is block.fuse.bn.weight for parameter in group['params']))
        offset_group = next(group for group in groups
                            if any(parameter is block.offset_conv[0].weight for parameter in group['params']))
        self.assertEqual(bn_group['weight_decay'], 0.)
        self.assertEqual(offset_group['lr'], .000035)

    def test_config_preserves_58e_protocol_and_enables_only_crossgate_dbdm(self):
        directory = ROOT / 'configs/deim_dfine'
        baseline = load_config(str(directory / 'deim_hgnetv2_m_sar_aircraft_58e_b28_old_server.yml'), {})
        config = load_config(str(directory / 'deim_hgnetv2_m_sar_aircraft_crossgate_dbdm_58e_b28_old_server.yml'), {})
        self.assertTrue(config['HGNetv2'].pop('use_crossgate_dbdm'))
        self.assertFalse(config['HGNetv2']['pretrained'])
        self.assertFalse(config['HybridEncoder'].get('use_rem', False))
        self.assertFalse(config['HybridEncoder'].get('use_dsg', False))
        for name in ['use_scs', 'use_slqc', 'use_mpq']:
            self.assertFalse(config['DFINETransformer'].get(name, False))
        self.assertTrue((ROOT / config.pop('experiment_protocol_document')).is_file())
        self.assertNotEqual(config['output_dir'], baseline['output_dir'])
        for candidate in [config, baseline]:
            candidate.pop('__include__', None)
            candidate.pop('output_dir')
            candidate.pop('experiment_protocol_document', None)
        self.assertEqual(config, baseline)


if __name__ == '__main__':
    unittest.main()
