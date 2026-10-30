"""OST-DA geometry, bilinear aggregation, pretrained integration and training."""

import copy
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from engine.core import YAMLConfig
from engine.core.workspace import create
from engine.core.yaml_utils import load_config
from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.dfine_decoder import DFINETransformer, MSDeformableAttention, TransformerDecoderLayer
from engine.deim.hybrid_encoder import HybridEncoder
from engine.deim.matcher import HungarianMatcher
from engine.deim.orthogonal_topology_attention import OrthogonalTopologyDeformAttn
from engine.deim.utils import deformable_attention_core_func
from engine.solver._solver import BaseSolver
from engine.solver.det_engine import _to_float32


ROOT = Path(__file__).resolve().parents[1]
DECODER_ARGS = dict(num_classes=7, hidden_dim=32, feat_channels=[32] * 3, num_queries=5,
                    num_layers=3, eval_idx=1, nhead=8, num_points=4, dim_feedforward=64,
                    reg_max=32, num_denoising=4, eval_spatial_size=None)


def decoder(enabled=True, **extra):
    return DFINETransformer(**copy.deepcopy(DECODER_ARGS), use_ost_da=enabled, **extra)


def inputs():
    feats = [torch.randn(2, 32, h, w) for h, w in [(8, 12), (4, 6), (2, 3)]]
    targets = [dict(labels=torch.tensor([3]), boxes=torch.tensor([[.4, .5, .3, .4]])),
               dict(labels=torch.tensor([1, 6]), boxes=torch.tensor([[.3, .4, .2, .3], [.7, .6, .3, .2]]))]
    return feats, targets


class OstDaTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(367)

    def test_head_point_partitions_polar_initialization_and_parameter_count(self):
        module = OrthogonalTopologyDeformAttn()
        baseline = MSDeformableAttention(256, 8, 3, 4)
        self.assertEqual((module.n_heads, module.n_levels, module.n_points), (8, 3, 4))
        self.assertEqual((module.n_free_heads, module.n_topo_heads), (4, 4))
        for name in ['sampling_offsets', 'attention_weights']:
            for key, value in getattr(baseline, name).state_dict().items():
                torch.testing.assert_close(getattr(module, name).state_dict()[key], value, rtol=0, atol=0)
        self.assertEqual(module.angle_proj.weight.shape, (4, 256))
        self.assertEqual(module.angle_proj.weight.abs().sum().item(), 0.)
        self.assertEqual(module.angle_proj.bias.abs().sum().item(), 0.)
        self.assertIsInstance(module.topo_gate[1], nn.LayerNorm)
        self.assertIsInstance(module.topo_gate[2], nn.Sigmoid)
        self.assertEqual(sum(p.numel() for p in module.parameters()), 272932)
        self.assertEqual(sum(p.numel() for p in module.parameters()) - sum(p.numel() for p in baseline.parameters()), 198916)
        _, weights = module.sampling_geometry(torch.randn(2, 5, 256), torch.tensor([[[.5, .5, .4, .3]] * 5] * 2).unsqueeze(2))
        torch.testing.assert_close(weights, torch.full((2, 5, 8, 3, 4), 1 / 12))

    def test_free_and_axial_coordinates_match_analytic_per_head_angles_and_reference_boxes(self):
        module = OrthogonalTopologyDeformAttn(8, 2, 4, 4)
        raw = torch.arange(4 * 2 * 4 * 2).reshape(4, 2, 4, 2).float() / 20 - 1
        angles = [math.pi / 4, 3 * math.pi / 4]
        with torch.no_grad():
            module.sampling_offsets.bias.copy_(raw.flatten())
            module.angle_proj.bias.copy_(torch.tensor([-math.log(3), math.log(3)]))
        boxes = torch.tensor([[[[.3, .6, .8, .4], [.7, .2, .2, .6]]]])
        locations, weights = module.sampling_geometry(torch.zeros(1, 1, 8), boxes)
        expected = torch.empty_like(locations)
        for head in range(4):
            for level in range(2):
                for point in range(4):
                    if head < 2:
                        offset = raw[head, level, point].numpy()
                    else:
                        theta = angles[head - 2]
                        body, wing = np.array([math.cos(theta), math.sin(theta)]), np.array([-math.sin(theta), math.cos(theta)])
                        self.assertAlmostEqual(float(body @ wing), 0.)
                        r = raw[head, level, point, 0].item()
                        delta = .25 * math.tanh(raw[head, level, point, 1].item())
                        offset = r * body + delta * wing if point < 2 else r * wing + delta * body
                    xy = boxes[0, 0, level, :2].numpy() + offset / 4 * boxes[0, 0, level, 2:].numpy() * .5
                    expected[0, 0, head, level, point] = torch.tensor(xy)
        torch.testing.assert_close(locations, expected)
        torch.testing.assert_close(weights.sum((-1, -2)), torch.ones(1, 1, 4))

    def test_orthogonal_perturbation_is_bounded_and_swaps_axis_for_wing_points(self):
        module = OrthogonalTopologyDeformAttn(8, 1, 4, 4)
        raw = torch.zeros(4, 1, 4, 2)
        raw[2:, :, :, 1] = torch.tensor([-1000., 1000., -1000., 1000.])
        with torch.no_grad():
            module.sampling_offsets.bias.copy_(raw.flatten())
        boxes = torch.tensor([[[[.5, .5, 1., 1.]]]])
        locations, _ = module.sampling_geometry(torch.zeros(1, 1, 8), boxes)
        offsets = (locations - .5) * 8
        # theta=pi/2: body=(0,1), wing=(-1,0).
        expected = torch.tensor([[.25, 0.], [-.25, 0.], [0., -.25], [0., .25]])
        torch.testing.assert_close(offsets[0, 0, 2:, 0], expected.expand(2, -1, -1), atol=1e-7, rtol=0)

    def test_bilinear_core_matches_manual_pixel_interpolation_and_zero_padding(self):
        shapes = [(3, 5), (2, 4)]
        maps = [torch.randn(2, 2, 3, h, w) for h, w in shapes]
        for level, values in enumerate(maps):
            for batch in range(2):
                for head in range(2):
                    values[batch, head] += batch * 100 + head * 10 + level * 2
        value = torch.cat([level.flatten(3).permute(0, 3, 1, 2) for level in maps], dim=1)
        locations = torch.rand(2, 3, 2, 2, 2, 2) * 1.6 - .3
        locations[:, -1] = -2.
        weights = torch.randn(2, 3, 2, 4).softmax(-1).reshape(2, 3, 2, 2, 2)
        actual = OrthogonalTopologyDeformAttn._ms_deform_attn_core(value, shapes, locations, weights)
        expected = np.zeros((2, 3, 2, 3))
        for batch in range(2):
            for query in range(3):
                for head in range(2):
                    for level, (height, width) in enumerate(shapes):
                        for point in range(2):
                            x = locations[batch, query, head, level, point, 0].item() * width - .5
                            y = locations[batch, query, head, level, point, 1].item() * height - .5
                            for px in [math.floor(x), math.floor(x) + 1]:
                                for py in [math.floor(y), math.floor(y) + 1]:
                                    if 0 <= px < width and 0 <= py < height:
                                        coefficient = (1 - abs(x - px)) * (1 - abs(y - py))
                                        expected[batch, query, head] += (maps[level][batch, head, :, py, px].numpy()
                                            * coefficient * weights[batch, query, head, level, point].item())
        torch.testing.assert_close(actual, torch.tensor(expected.reshape(2, 3, 6), dtype=torch.float32), atol=3e-5, rtol=2e-6)
        torch.testing.assert_close(actual[:, -1], torch.zeros(2, 6), rtol=0, atol=0)

    def test_flat_and_native_dfine_values_and_shared_level_references_are_equivalent(self):
        model = decoder()
        module = model.decoder.layers[0].cross_attn
        memory = torch.randn(2, 126, 32)
        shapes = [(8, 12), (4, 6), (2, 3)]
        split = model.decoder.value_op(memory, None, None, None, shapes)
        query = torch.randn(2, 5, 32)
        reference = torch.tensor([[[[.5, .5, .4, .3]]] * 5] * 2)
        with torch.no_grad():
            flat = module(query, reference, memory, shapes)
            native = module(query, reference.expand(-1, -1, 3, -1), split, shapes)
        torch.testing.assert_close(native, flat, rtol=0, atol=0)

    def test_value_projection_full_channel_gate_and_output_projection_follow_given_formula(self):
        module = OrthogonalTopologyDeformAttn(8, 1, 4, 4)
        with torch.no_grad():
            module.value_proj.weight.copy_(torch.eye(8) * 3)
            module.value_proj.bias.fill_(.2)
            module.topo_gate[0].weight.zero_()
            module.topo_gate[0].bias.copy_(torch.arange(8).float())
            module.output_proj.weight.copy_(torch.eye(8) * 2)
            module.output_proj.bias.fill_(-.1)
        memory = torch.randn(1, 6, 8)
        sampled = torch.randn(1, 2, 8)
        with patch.object(module, '_ms_deform_attn_core', return_value=sampled) as core:
            actual = module(torch.randn(1, 2, 8), torch.tensor([[[[.5, .5, .3, .2]]] * 2]), memory, [(2, 3)])
        torch.testing.assert_close(core.call_args.args[0], (memory * 3 + .2).reshape(1, 6, 4, 2))
        gate = F.layer_norm(torch.arange(8).float(), (8,)).sigmoid()
        torch.testing.assert_close(actual, 2 * sampled * gate - .1)

    def test_angle_offsets_attention_values_and_gate_learn_with_fp32_sampling_under_amp(self):
        for amp, dtype in [(False, torch.float32), (True, torch.float32), (True, torch.bfloat16)]:
            with self.subTest(amp=amp, dtype=dtype):
                module = OrthogonalTopologyDeformAttn(32, 3, 8, 4)
                query = torch.randn(2, 3, 32, dtype=dtype, requires_grad=True)
                value = torch.randn(2, 64, 32, dtype=dtype, requires_grad=True)
                reference = torch.tensor([[[[.5, .5, .4, .3]]] * 3] * 2, dtype=dtype, requires_grad=True)
                with patch('engine.deim.orthogonal_topology_attention.deformable_attention_core_func', wraps=deformable_attention_core_func) as core:
                    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
                        result = module(query, reference, value, [(6, 8), (3, 4), (2, 2)])
                self.assertEqual(result.dtype, dtype)
                self.assertTrue(all(tensor.dtype == torch.float32 for tensor in [core.call_args.args[0], *core.call_args.args[2:]]))
                result.float().square().mean().backward()
                for name, parameter in module.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                for branch in [module.angle_proj, module.sampling_offsets, module.attention_weights,
                               module.value_proj, module.topo_gate, module.output_proj]:
                    self.assertGreater(sum(p.grad.abs().sum().item() for p in branch.parameters()), 0.)
                for tensor in [query, value, reference]:
                    self.assertTrue(torch.isfinite(tensor.grad).all())
                    if tensor is not query:
                        self.assertGreater(tensor.grad.abs().sum().item(), 0.)
                # All three query-conditioned projections start with zero weights.
                # Parameters learn immediately; query gradients start after their update.
                torch.testing.assert_close(query.grad, torch.zeros_like(query), rtol=0, atol=0)
                torch.optim.SGD(module.parameters(), lr=.1).step()
                module.zero_grad(set_to_none=True)
                for tensor in [query, value, reference]:
                    tensor.grad = None
                with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
                    updated = module(query, reference, value, [(6, 8), (3, 4), (2, 2)])
                updated.float().square().mean().backward()
                self.assertTrue(torch.isfinite(query.grad).all())
                self.assertGreater(query.grad.abs().sum().item(), 0.)

    def test_replacement_preserves_common_weights_rng_and_existing_checkpoint_loading(self):
        torch.manual_seed(373)
        baseline = decoder(False)
        expected_rng = torch.get_rng_state()
        torch.manual_seed(373)
        adapted = decoder()
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        for name, value in baseline.state_dict().items():
            if name.endswith('cross_attn.num_points_scale'):
                self.assertNotIn(name, adapted.state_dict())
            else:
                torch.testing.assert_close(adapted.state_dict()[name], value, rtol=0, atol=0)
        matched, info = BaseSolver._matched_state(adapted.state_dict(), baseline.state_dict())
        missing, unexpected = adapted.load_state_dict(matched, strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(missing)
        self.assertEqual(set(missing), set(info['missed']))
        self.assertFalse(info['unmatched'])
        self.assertTrue(all('.cross_attn.' in name for name in missing))
        self.assertTrue(all(isinstance(layer.cross_attn, OrthogonalTopologyDeformAttn) for layer in adapted.decoder.layers))
        self.assertTrue(all(isinstance(layer.cross_attn, MSDeformableAttention) for layer in baseline.decoder.layers))

    def test_structural_constraints_and_scaled_auxiliary_layers(self):
        for kwargs in [dict(n_heads=3), dict(n_points=3), dict(n_points=[3, 6, 3]), dict(d_model=30)]:
            with self.assertRaises(ValueError):
                OrthogonalTopologyDeformAttn(**kwargs)
        self.assertEqual(OrthogonalTopologyDeformAttn(n_points=[4, 4, 4]).n_points, 4)
        with self.assertRaisesRegex(ValueError, 'bilinear'):
            TransformerDecoderLayer(use_ost_da=True, cross_attn_method='discrete')
        model = decoder(layer_scale=2)
        self.assertEqual([layer.cross_attn.d_model for layer in model.decoder.layers], [32, 32, 64])
        feats, targets = inputs()
        result = model(feats, targets)
        result['pred_logits'].square().mean().backward()
        self.assertGreater(model.decoder.layers[-1].cross_attn.angle_proj.weight.grad.abs().sum().item(), 0.)

    def test_actual_single_and_three_method_training_include_all_decoder_dn_and_aux_paths(self):
        for combined, amp in [(False, False), (True, False), (True, True)]:
            with self.subTest(combined=combined, amp=amp):
                encoder = HybridEncoder(in_channels=[32, 64, 128], hidden_dim=32, nhead=4,
                                        dim_feedforward=64, depth_mult=.34, use_spi_module=combined)
                model = decoder()
                weights = dict(loss_mal=1, loss_bbox=5, loss_giou=2, loss_fgl=.15, loss_ddf=1.5)
                if combined:
                    weights['loss_wfdr'] = .15
                loss_fn = DEIMCriterion(
                    HungarianMatcher(dict(cost_class=2, cost_bbox=5, cost_giou=2), use_focal_loss=True),
                    weights, ['mal', 'boxes', 'local'], num_classes=7, reg_max=32, use_wfdr=combined)
                feats = [torch.randn(2, c, h, w) for c, h, w in [(32, 8, 12), (64, 4, 6), (128, 2, 3)]]
                _, targets = inputs()
                optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(model.parameters()), lr=1e-3)
                records, handles = {}, []
                for index, layer in enumerate(model.decoder.layers):
                    handles.append(layer.cross_attn.register_forward_pre_hook(
                        lambda module, args, index=index: records.__setitem__(index, args)))
                try:
                    for step in range(2):
                        optimizer.zero_grad(set_to_none=True)
                        records.clear()
                        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
                            output = model(encoder(feats), targets)
                        losses = loss_fn(_to_float32(output), targets)
                        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
                        self.assertEqual(set(records), {0, 1, 2})
                        for args in records.values():
                            self.assertEqual(args[1].shape[-2:], (1, 4))
                            self.assertGreater(args[0].shape[1], 5)  # DN queries follow the same attention path.
                            self.assertEqual(len(args[2]), 3)
                        self.assertIn('loss_fgl_aux_0', losses)
                        self.assertIn('loss_mal_dn_0', losses)
                        self.assertEqual('loss_wfdr' in losses, combined)
                        sum(losses.values()).backward()
                        for layer in model.decoder.layers:
                            for name, parameter in layer.cross_attn.named_parameters():
                                self.assertIsNotNone(parameter.grad, name)
                                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                            self.assertGreater(layer.cross_attn.angle_proj.weight.grad.abs().sum().item(), 0.)
                            self.assertGreater(layer.cross_attn.topo_gate[0].weight.grad.abs().sum().item(), 0.)
                        if combined:
                            for adapter in encoder.spi_modules:
                                self.assertGreater(adapter.alpha.grad.abs().sum().item(), 0.)
                                self.assertGreater(adapter.beta.grad.abs().sum().item(), 0.)
                        optimizer.step()
                finally:
                    for handle in handles:
                        handle.remove()

    def test_checkpoint_and_deployment_keep_all_required_attention_layers_and_outputs(self):
        model = decoder().eval()
        for layer in model.decoder.layers:
            layer.cross_attn.angle_proj.bias.data.fill_(.4)
        feats, _ = inputs()
        with torch.no_grad():
            expected = model(feats)
        restored = decoder().eval()
        restored.load_state_dict(model.state_dict())
        restored.convert_to_deploy()
        restored.decoder.convert_to_deploy()
        self.assertEqual(len(restored.decoder.layers), model.eval_idx + 1)
        self.assertTrue(all(isinstance(layer.cross_attn, OrthogonalTopologyDeformAttn) for layer in restored.decoder.layers))
        with torch.no_grad():
            actual = restored(feats)
        for name in ['pred_logits', 'pred_boxes', 'pred_corners']:
            torch.testing.assert_close(actual[name], expected[name])

    def test_controls_and_method_configs_keep_sampling_budget_and_all_other_protocol_settings(self):
        folder = ROOT / 'configs/deim_dfine'
        outputs = set()
        variants = [('uniform4', False, False), ('ost_da', False, True),
                    ('spi_module_wfdr_uniform4', True, False), ('spi_module_wfdr_ost_da', True, True)]
        for suffix, combined, enabled in variants:
            path = folder / f'deim_hgnetv2_m_sar_aircraft_{suffix}_58e_b28_old_server.yml'
            parent = 'spi_module_wfdr_58e_b28' if combined else '58e_b28'
            config = load_config(str(path), {})
            reference = load_config(str(folder / f'deim_hgnetv2_m_sar_aircraft_{parent}_old_server.yml'), {})
            self.assertEqual(config['DFINETransformer'].pop('num_points'), 4)
            self.assertEqual(reference['DFINETransformer'].pop('num_points'), [3, 6, 3])
            self.assertEqual(3 * 4, sum([3, 6, 3]))
            self.assertEqual(config['DFINETransformer'].pop('use_ost_da', False), enabled)
            actual = create('DFINETransformer', YAMLConfig(str(path)).global_cfg)
            expected_type = OrthogonalTopologyDeformAttn if enabled else MSDeformableAttention
            self.assertTrue(all(isinstance(layer.cross_attn, expected_type) for layer in actual.decoder.layers))
            if enabled:
                self.assertEqual(sum(sum(p.numel() for p in layer.cross_attn.parameters()) for layer in actual.decoder.layers), 1091728)
            self.assertTrue((ROOT / config['experiment_protocol_document']).is_file())
            self.assertNotIn(config['output_dir'], outputs)
            outputs.add(config['output_dir'])
            for candidate in [config, reference]:
                candidate.pop('__include__', None)
                candidate.pop('output_dir')
                candidate.pop('experiment_protocol_document')
            self.assertEqual(config, reference)


if __name__ == '__main__':
    unittest.main()
