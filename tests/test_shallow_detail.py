"""S2 routing, zero-residual learning, detector gradients and protocol isolation."""
import copy
import io
from pathlib import Path
import unittest

import torch
from torch import nn

from engine.backbone.hgnetv2 import HGNetv2
from engine.core import YAMLConfig
from engine.core.yaml_utils import load_config
from engine.deim.deim import DEIM
from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.dfine_decoder import DFINETransformer
from engine.deim.hybrid_encoder import HybridEncoder
from engine.deim.matcher import HungarianMatcher
from engine.deim.shallow_detail import ShallowDetailInjection
from engine.solver.det_engine import _module_training_metrics


def encoder(mode=None):
    torch.manual_seed(104)
    return HybridEncoder(in_channels=[32, 64, 128], hidden_dim=32, nhead=4,
                         dim_feedforward=64, depth_mult=.34,
                         s2_injection_mode=mode, s2_channels=16 if mode else None)


def detector(mode):
    neck = encoder(mode)
    head = DFINETransformer(num_classes=7, hidden_dim=32, feat_channels=[32] * 3,
                            num_queries=5, num_layers=2, nhead=4, num_points=2,
                            dim_feedforward=64, reg_max=8, num_denoising=4,
                            eval_spatial_size=None)
    return DEIM(nn.Identity(), neck, head)


def features():
    return [torch.randn(2, channels, side, side, requires_grad=True)
            for channels, side in [(16, 16), (32, 8), (64, 4), (128, 2)]]


class ShallowDetailTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(105)

    def test_initial_encoder_outputs_base_weights_and_rng_are_preserved(self):
        base = encoder()
        base_rng = torch.get_rng_state()
        inputs = features()
        downsample_state = None
        for mode in ShallowDetailInjection.MODES:
            adapted = encoder(mode)
            torch.testing.assert_close(torch.get_rng_state(), base_rng, rtol=0, atol=0)
            for key, value in base.state_dict().items():
                torch.testing.assert_close(adapted.state_dict()[key], value, rtol=0, atol=0)
            current = adapted.s2_injection.downsample.state_dict()
            if downsample_state is None:
                downsample_state = copy.deepcopy(current)
            for key in current:
                torch.testing.assert_close(current[key], downsample_state[key], rtol=0, atol=0)
            self.assertIsNone(adapted.p2_inject)
            self.assertIsNone(adapted.detail_preservation)
            # Recreate the base per mode: train-mode calls update BN buffers.
            reference = encoder()
            for training in (True, False):
                reference.train(training)
                adapted.train(training)
                with torch.no_grad():
                    expected = reference(inputs[1:])
                    actual = adapted(inputs)
                self.assertEqual([x.shape[-2:] for x in actual], [(8, 8), (4, 4), (2, 2)])
                for left, right in zip(expected, actual):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_backbone_exposes_s2_without_changing_existing_outputs_or_state(self):
        backbone = HGNetv2('B2', pretrained=False, use_lab=True,
                           freeze_at=-1, freeze_norm=False).eval()
        image = torch.randn(1, 3, 128, 128)
        keys = list(backbone.state_dict())
        with torch.no_grad():
            original = backbone(image)
            backbone.return_idx = [0, 1, 2, 3]
            expanded = backbone(image)
        self.assertEqual(keys, list(backbone.state_dict()))
        self.assertEqual([(x.shape[1], x.shape[-1]) for x in expanded],
                         [(96, 32), (384, 16), (768, 8), (1536, 4)])
        for left, right in zip(original, expanded[1:]):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_original_detection_objective_opens_zero_bn_and_trains_structure(self):
        targets = [dict(labels=torch.tensor([3]), boxes=torch.tensor([[.5, .5, .3, .3]]))
                   for _ in range(2)]
        for mode in ShallowDetailInjection.MODES:
            model = detector(mode).train()
            inputs = features()
            branch = model.encoder.s2_injection
            criterion = DEIMCriterion(
                HungarianMatcher(dict(cost_class=2, cost_bbox=5, cost_giou=2), use_focal_loss=True),
                dict(loss_mal=1, loss_bbox=5, loss_giou=2, loss_fgl=.15, loss_ddf=1.5),
                ['mal', 'boxes', 'local'], num_classes=7, reg_max=8)
            optim = torch.optim.SGD(model.parameters(), lr=.01)
            for step in range(2):
                optim.zero_grad(set_to_none=True)
                inputs[0].grad = None
                outputs = model(inputs, targets)
                losses = criterion(outputs, targets)
                self.assertFalse(any('mpq' in k or 'scs' in k or 's2' in k for k in losses))
                sum(losses.values()).backward()
                for name, parameter in branch.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                if step == 0:
                    self.assertGreater(branch.downsample.norm.weight.grad.abs().sum().item(), 0)
                    self.assertEqual(branch.downsample.depthwise.weight.grad.abs().sum().item(), 0)
                    self.assertEqual(branch.last_metrics['s2_residual_ratio'].item(), 0)
                else:
                    self.assertGreater(branch.downsample.depthwise.weight.grad.abs().sum().item(), 0)
                    self.assertGreater(inputs[0].grad.abs().sum().item(), 0)
                    if mode != 'plain':
                        self.assertGreater(branch.structure.context[1].weight.grad.abs().sum().item(), 0)
                    metrics = _module_training_metrics(model)
                    self.assertGreater(metrics['s2_gamma_mean_abs'], 0)
                    self.assertGreater(metrics['s2_residual_ratio'], 0)
                    self.assertFalse(branch.last_metrics['s2_residual_ratio'].requires_grad)
                optim.step()

    def test_multiscale_amp_and_nonzero_residual_are_retained_without_targets(self):
        for mode in ShallowDetailInjection.MODES:
            branch = ShallowDetailInjection(96, 256, mode).eval()
            branch.downsample.norm.weight.data.fill_(.1)
            for size in (480, 640, 800):
                detail = torch.randn(1, 96, size // 4, size // 4)
                main = torch.randn(1, 256, size // 8, size // 8)
                with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
                    result = branch(detail, main.to(torch.bfloat16))
                self.assertEqual(result.shape, main.shape)
                self.assertEqual(result.dtype, torch.bfloat16)
                self.assertTrue(torch.isfinite(result).all())
                self.assertGreater((result.float() - main.to(torch.bfloat16).float()).abs().sum().item(), 0)
            self.assertEqual(branch.last_metrics, {})

    def test_checkpoint_loading_and_deploy_keep_the_learned_detail_branch(self):
        base = encoder()
        for mode in ShallowDetailInjection.MODES:
            adapted = encoder(mode)
            result = adapted.load_state_dict(base.state_dict(), strict=False)
            self.assertFalse(result.unexpected_keys)
            self.assertTrue(result.missing_keys)
            self.assertTrue(all(k.startswith('s2_injection.') for k in result.missing_keys))
            model = detector(mode).eval()
            model.encoder.s2_injection.downsample.norm.weight.data.fill_(.1)
            inputs = features()
            with torch.no_grad():
                expected = model(inputs)
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            restored = detector(mode).eval()
            restored.load_state_dict(torch.load(buffer, weights_only=True))
            with torch.no_grad():
                actual = restored(inputs)
                deployed = copy.deepcopy(restored).deploy()(inputs)
            for key in ('pred_logits', 'pred_boxes'):
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                torch.testing.assert_close(deployed[key], expected[key], rtol=1e-4, atol=1e-5)
            self.assertIsNotNone(restored.encoder.s2_injection)

    def test_frozen_configs_build_and_optimizer_covers_only_declared_new_parameters(self):
        base = load_config('configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_102e_b28.yml', {})
        expected_counts = dict(plain=25952, strip=60224, local=59840)
        configured = {}
        for mode, parameter_count in expected_counts.items():
            path = Path('configs/deim_dfine/sar_aircraft_detail') / f's2_{mode}.yml'
            cfg = load_config(str(path), {})
            configured[mode] = cfg
            for key, value in base.items():
                if key not in ('__include__', 'output_dir', 'HGNetv2', 'HybridEncoder', 'DFINETransformer'):
                    self.assertEqual(cfg[key], value, key)
            for section in ('HGNetv2', 'HybridEncoder', 'DFINETransformer'):
                for key, value in base[section].items():
                    if (section, key) != ('HGNetv2', 'return_idx'):
                        self.assertEqual(cfg[section][key], value, (section, key))
            self.assertEqual(cfg['HGNetv2']['return_idx'], [0, 1, 2, 3])
            self.assertEqual(cfg['DEIMCriterion'], base['DEIMCriterion'])
            model = YAMLConfig(str(path)).model
            self.assertEqual(model.backbone.return_idx, [0, 1, 2, 3])
            self.assertEqual(model.encoder.out_strides, [8, 16, 32])
            self.assertEqual(model.encoder.out_channels, [256, 256, 256])
            self.assertIsNone(model.encoder.p2_inject)
            for name in ('scs_head', 'slqc', 'mpq_head'):
                self.assertIsNone(getattr(model.decoder, name, None))
            branch = model.encoder.s2_injection
            self.assertEqual(sum(p.numel() for p in branch.parameters()), parameter_count)
            groups = YAMLConfig.get_optim_params(cfg['optimizer'], model)
            owners = {id(p): group for group in groups for p in group['params']}
            for name, parameter in branch.named_parameters():
                group = owners[id(parameter)]
                self.assertEqual(group.get('lr', cfg['optimizer']['lr']), .00035)
                decay = group.get('weight_decay', cfg['optimizer']['weight_decay'])
                self.assertEqual(decay, 0 if 'norm' in name else .0001)
        normalized = []
        for mode, cfg in configured.items():
            cfg = copy.deepcopy(cfg)
            for key in ('__include__', 'output_dir'):
                cfg.pop(key, None)
            cfg['HybridEncoder']['s2_injection_mode'] = 'plain'
            normalized.append(cfg)
        self.assertEqual(normalized[0], normalized[1])
        self.assertEqual(normalized[0], normalized[2])

    def test_existing_haar_route_and_invalid_double_injection(self):
        existing = HybridEncoder(in_channels=[32, 64, 128], hidden_dim=32, nhead=4,
                                 dim_feedforward=64, depth_mult=.34, p2_channels=16)
        self.assertIsNone(existing.s2_injection)
        self.assertEqual(len(existing(features())), 3)
        with self.assertRaisesRegex(ValueError, 'alternative'):
            HybridEncoder(p2_channels=96, s2_channels=96, s2_injection_mode='plain')
        with self.assertRaisesRegex(ValueError, 's2_channels'):
            HybridEncoder(s2_injection_mode='plain')
        with self.assertRaisesRegex(ValueError, 'match projected S3'):
            ShallowDetailInjection(16, 32)(torch.randn(2, 16, 16, 16), torch.randn(2, 32, 7, 7))


if __name__ == '__main__':
    unittest.main()
