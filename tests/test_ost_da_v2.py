"""OST-DA v2: mirrored geometry, per-level density, residual learning and integration."""

import copy
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from engine.core import YAMLConfig
from engine.core.workspace import create
from engine.core.yaml_utils import load_config
from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.dfine_decoder import DFINETransformer, MSDeformableAttention, TransformerDecoderLayer
from engine.deim.hybrid_encoder import HybridEncoder
from engine.deim.matcher import HungarianMatcher
from engine.deim.orthogonal_topology_attention import OrthogonalTopologyDeformAttn
from engine.deim.symmetric_topology_attention import SymmetricOrthogonalTopologyDeformAttn
from engine.deim.utils import deformable_attention_core_func_v2
from engine.solver._solver import BaseSolver
from engine.solver.det_engine import _to_float32


ROOT = Path(__file__).resolve().parents[1]
ARGS = dict(num_classes=7, hidden_dim=32, feat_channels=[32] * 3, num_queries=5,
            num_layers=3, eval_idx=1, nhead=8, num_points=[4, 6, 4], dim_feedforward=64,
            reg_max=32, num_denoising=4, eval_spatial_size=None)


def decoder(**overrides):
    args = copy.deepcopy(ARGS)
    args.update(use_ost_da=True, ost_da_version='v2')
    args.update(overrides)
    return DFINETransformer(**args)


def inputs():
    feats = [torch.randn(2, 32, h, w) for h, w in [(8, 12), (4, 6), (2, 3)]]
    targets = [dict(labels=torch.tensor([3]), boxes=torch.tensor([[.4, .5, .3, .4]])),
               dict(labels=torch.tensor([1, 6]), boxes=torch.tensor([[.3, .4, .2, .3], [.7, .6, .3, .2]]))]
    return feats, targets


class OstDaV2Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(409)

    def test_per_level_density_only_half_axis_steps_and_noncollapsed_initial_cross(self):
        for points in ([4, 6, 4], 6, [4, 8, 6]):
            with self.subTest(points=points):
                module = SymmetricOrthogonalTopologyDeformAttn(n_points=points)
                counts = points if isinstance(points, list) else [points] * 3
                self.assertEqual(module.n_points_list, counts)
                self.assertEqual(module.free_sampling_offsets.out_features, 4 * sum(counts) * 2)
                self.assertEqual(module.topo_steps.out_features, 4 * sum(k // 2 for k in counts))
                self.assertEqual(module.gate_alpha.shape, (1, 1, 256))
                query = torch.randn(2, 3, 256)
                boxes = torch.tensor([.5, .5, 1., 1.]).reshape(1, 1, 1, 4).expand(2, 3, 1, 4)
                locations, weights = module.sampling_geometry(query, boxes)
                self.assertEqual(locations.shape, (2, 3, 8, sum(counts), 2))
                torch.testing.assert_close(weights, torch.full_like(weights, 1 / sum(counts)))
                for count, level in zip(counts, locations.split(counts, dim=-2)):
                    offsets = (level[:, :, 4:] - .5) * (2 * count)
                    torch.testing.assert_close(offsets[..., 0::2, :] + offsets[..., 1::2, :],
                                               torch.zeros_like(offsets[..., 0::2, :]), atol=2e-6, rtol=0)
                    self.assertTrue((offsets.square().sum(-1) > .9).all())
                    self.assertTrue((offsets.abs().amin(-1) < 2e-6).all())
                    # All four topology heads collectively allocate equal density to both axes.
                    torch.testing.assert_close((offsets[..., 0].abs() > .5).sum(),
                                               (offsets[..., 1].abs() > .5).sum())

    def test_arbitrary_angles_steps_and_free_offsets_match_analytic_multilevel_geometry(self):
        module = SymmetricOrthogonalTopologyDeformAttn(8, 3, 4, [4, 6, 4])
        free = torch.linspace(-1, 2, 2 * 14 * 2).reshape(2, 14, 2)
        radii = torch.linspace(-2, 3, 2 * 7).reshape(2, 7)
        angles = [math.pi / 4, 3 * math.pi / 4]
        with torch.no_grad():
            module.free_sampling_offsets.bias.copy_(free.flatten())
            module.topo_steps.bias.copy_(radii.flatten())
            module.angle_proj.bias.copy_(torch.tensor([-math.log(3), math.log(3)]))
        boxes = torch.tensor([[[[.3, .6, .8, .4], [.7, .2, .2, .6], [.4, .7, .3, .5]]]])
        locations, weights = module.sampling_geometry(torch.zeros(1, 1, 8), boxes)
        expected = torch.empty_like(locations)
        point_start = pair_start = 0
        for level, count in enumerate([4, 6, 4]):
            for head in range(4):
                if head < 2:
                    offsets = free[head, point_start:point_start + count].numpy()
                else:
                    body_pairs = ((count // 2 + 1) // 2 if (head - 2) % 2 == 0 else count // 4)
                    body_axis = np.array([math.cos(angles[head - 2]), math.sin(angles[head - 2])])
                    wing_axis = np.array([-body_axis[1], body_axis[0]])
                    offsets = []
                    for pair in range(count // 2):
                        axis = body_axis if pair < body_pairs else wing_axis
                        offset = radii[head - 2, pair_start + pair].item() * axis
                        offsets.extend([offset, -offset])
                    offsets = np.array(offsets)
                center, size = boxes[0, 0, level, :2].numpy(), boxes[0, 0, level, 2:].numpy()
                expected[0, 0, head, point_start:point_start + count] = torch.tensor(center + offsets / count * size * .5)
            point_start += count
            pair_start += count // 2
        torch.testing.assert_close(locations, expected)
        torch.testing.assert_close(weights.sum(-1), torch.ones(1, 1, 4))

    def test_pair_symmetry_survives_learned_query_dependence_and_both_partners_share_gradients(self):
        module = SymmetricOrthogonalTopologyDeformAttn(8, 1, 4, 4)
        with torch.no_grad():
            module.topo_steps.weight.normal_()
            module.angle_proj.weight.normal_()
        query = torch.randn(1, 2, 8, requires_grad=True)
        boxes = torch.tensor([.5, .5, .4, .8]).reshape(1, 1, 1, 4).expand(1, 2, 1, 4)
        locations, _ = module.sampling_geometry(query, boxes)
        torch.testing.assert_close(locations[:, :, 2:, 0::2] + locations[:, :, 2:, 1::2],
                                   torch.ones(1, 2, 2, 2, 2), atol=1e-7, rtol=0)
        positive = torch.autograd.grad(locations[0, 0, 2, 0, 1], module.topo_steps.bias, retain_graph=True)[0]
        negative = torch.autograd.grad(locations[0, 0, 2, 1, 1], module.topo_steps.bias, retain_graph=True)[0]
        torch.testing.assert_close(positive, -negative, rtol=0, atol=0)
        self.assertGreater(positive.abs().sum().item(), 0.)
        locations[:, :, 2:].square().sum().backward()
        self.assertGreater(query.grad.abs().sum().item(), 0.)

    def test_residual_gate_is_exactly_unattenuated_at_initialization_and_follows_tanh_formula(self):
        module = SymmetricOrthogonalTopologyDeformAttn(8, 1, 4, 4)
        self.assertFalse(any(isinstance(layer, (nn.Sigmoid, nn.LayerNorm)) for layer in module.modules()))
        memory, sampled = torch.randn(1, 6, 8), torch.randn(1, 2, 8)
        query = torch.randn(1, 2, 8)
        reference = torch.tensor([.5, .5, .3, .2]).reshape(1, 1, 1, 4).expand(1, 2, 1, 4)
        with patch.object(module, '_ms_deform_attn_core', return_value=sampled) as core:
            initial = module(query, reference, memory, [(2, 3)])
            torch.testing.assert_close(initial, module.output_proj(sampled), rtol=0, atol=0)
            torch.testing.assert_close(core.call_args.args[0], module.value_proj(memory).reshape(1, 6, 4, 2))
            self.assertEqual(core.call_args.args[-1], [4])
            with torch.no_grad():
                module.gate_alpha.copy_(torch.linspace(-.4, .6, 8).reshape(1, 1, 8))
            actual = module(query, reference, memory, [(2, 3)])
        expected = module.output_proj(sampled * (1 + module.gate_alpha * module.topo_gate_linear(sampled).tanh()))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_packed_bilinear_core_matches_manual_nonuniform_grid_and_zero_padding(self):
        shapes, counts = [(3, 5), (2, 4), (1, 3)], [4, 6, 4]
        maps = [torch.randn(2, 2, 3, h, w) for h, w in shapes]
        value = torch.cat([level.flatten(3).permute(0, 3, 1, 2) for level in maps], dim=1)
        locations = torch.rand(2, 3, 2, sum(counts), 2) * 1.6 - .3
        locations[:, -1] = -2.
        weights = torch.randn(2, 3, 2, sum(counts)).softmax(-1)
        actual = SymmetricOrthogonalTopologyDeformAttn._ms_deform_attn_core(value, shapes, locations, weights, counts)
        expected = np.zeros((2, 3, 2, 3))
        start = 0
        for level, ((height, width), count) in enumerate(zip(shapes, counts)):
            for batch in range(2):
                for query in range(3):
                    for head in range(2):
                        for point in range(start, start + count):
                            x = locations[batch, query, head, point, 0].item() * width - .5
                            y = locations[batch, query, head, point, 1].item() * height - .5
                            for px in [math.floor(x), math.floor(x) + 1]:
                                for py in [math.floor(y), math.floor(y) + 1]:
                                    if 0 <= px < width and 0 <= py < height:
                                        expected[batch, query, head] += ((1 - abs(x - px)) * (1 - abs(y - py))
                                            * maps[level][batch, head, :, py, px].numpy()
                                            * weights[batch, query, head, point].item())
            start += count
        torch.testing.assert_close(actual, torch.tensor(expected.reshape(2, 3, 6), dtype=torch.float32), atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(actual[:, -1], torch.zeros(2, 6), rtol=0, atol=0)

    def test_native_values_and_shared_references_match_flat_values_and_per_level_references(self):
        model = decoder()
        module = model.decoder.layers[0].cross_attn
        memory, query = torch.randn(2, 126, 32), torch.randn(2, 5, 32)
        shapes = [(8, 12), (4, 6), (2, 3)]
        split = model.decoder.value_op(memory, None, None, None, shapes)
        reference = torch.tensor([.5, .5, .4, .3]).reshape(1, 1, 1, 4).expand(2, 5, 1, 4)
        with torch.no_grad():
            flat = module(query, reference, memory, shapes)
            native = module(query, reference.expand(-1, -1, 3, -1), split, shapes)
        torch.testing.assert_close(native, flat, rtol=0, atol=0)

    def test_zero_gate_scale_learns_first_then_gate_weights_learn_under_fp32_and_amp(self):
        for amp, dtype in [(False, torch.float32), (True, torch.float32), (True, torch.bfloat16)]:
            with self.subTest(amp=amp, dtype=dtype):
                module = SymmetricOrthogonalTopologyDeformAttn(32, 3, 8, [4, 6, 4])
                query = torch.randn(2, 3, 32, dtype=dtype, requires_grad=True)
                value = torch.randn(2, 64, 32, dtype=dtype, requires_grad=True)
                reference = torch.tensor([.5, .5, .4, .3], dtype=dtype).reshape(1, 1, 1, 4).expand(2, 3, 1, 4).clone().requires_grad_()
                optimizer = torch.optim.SGD(module.parameters(), lr=.1)
                for step in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    for tensor in (query, value, reference):
                        tensor.grad = None
                    with patch('engine.deim.symmetric_topology_attention.deformable_attention_core_func_v2',
                               wraps=deformable_attention_core_func_v2) as core:
                        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
                            output = module(query, reference, value, [(6, 8), (3, 4), (2, 2)])
                    self.assertEqual(output.dtype, dtype)
                    self.assertEqual(core.call_args.kwargs, dict(method='default', value_shape='reshape'))
                    self.assertEqual(core.call_args.args[-1], [4, 6, 4])
                    for tensor in [core.call_args.args[0], *core.call_args.args[2:4]]:
                        self.assertEqual(tensor.dtype, torch.float32)
                    output.float().square().mean().backward()
                    for name, parameter in module.named_parameters():
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    for branch in (module.free_sampling_offsets, module.topo_steps, module.angle_proj,
                                   module.attention_weights, module.value_proj, module.output_proj):
                        self.assertGreater(sum(p.grad.abs().sum().item() for p in branch.parameters()), 0.)
                    self.assertGreater(module.gate_alpha.grad.abs().sum().item(), 0.)
                    gate_grad = sum(p.grad.abs().sum().item() for p in module.topo_gate_linear.parameters())
                    if step == 0:
                        self.assertEqual(gate_grad, 0.)
                        torch.testing.assert_close(query.grad, torch.zeros_like(query), rtol=0, atol=0)
                    else:
                        self.assertGreater(gate_grad, 0.)
                        self.assertGreater(query.grad.abs().sum().item(), 0.)
                    for tensor in (query, value, reference):
                        self.assertTrue(torch.isfinite(tensor.grad).all())
                    optimizer.step()

    def test_common_initialization_rng_and_coco_loading_preserve_the_symmetric_start(self):
        torch.manual_seed(419)
        control = decoder(use_ost_da=False)
        expected_rng = torch.get_rng_state()
        torch.manual_seed(419)
        adapted = decoder()
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        for name, tensor in control.state_dict().items():
            if '.cross_attn.' not in name:
                torch.testing.assert_close(adapted.state_dict()[name], tensor, rtol=0, atol=0)
        original = decoder(use_ost_da=False, num_points=[3, 6, 3])
        before = copy.deepcopy(adapted.state_dict())
        matched, info = BaseSolver._matched_state(adapted.state_dict(), original.state_dict())
        missing, unexpected = adapted.load_state_dict(matched, strict=False)
        self.assertFalse(unexpected)
        self.assertEqual(set(missing), set(info['missed']) | set(info['unmatched']))
        for name, tensor in adapted.state_dict().items():
            if '.cross_attn.' in name:
                torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
        self.assertTrue(all(isinstance(layer.cross_attn, SymmetricOrthogonalTopologyDeformAttn) for layer in adapted.decoder.layers))

    def test_constraints_version_routing_and_scaled_auxiliary_layers(self):
        for kwargs in (dict(n_points=2), dict(n_points=3), dict(n_points=[4, 6]),
                       dict(n_points=[4, 5, 4]), dict(n_heads=3), dict(d_model=30), dict(n_levels=0)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SymmetricOrthogonalTopologyDeformAttn(**kwargs)
        with self.assertRaisesRegex(ValueError, 'bilinear'):
            TransformerDecoderLayer(use_ost_da=True, ost_da_version='v2', cross_attn_method='discrete')
        with self.assertRaisesRegex(ValueError, 'ost_da_version'):
            decoder(ost_da_version='invalid')
        legacy = decoder(num_points=4, ost_da_version='v1')
        self.assertTrue(all(isinstance(layer.cross_attn, OrthogonalTopologyDeformAttn) for layer in legacy.decoder.layers))
        model = decoder(layer_scale=2)
        self.assertEqual([layer.cross_attn.gate_alpha.shape[-1] for layer in model.decoder.layers], [32, 32, 64])
        feats, targets = inputs()
        model(feats, targets)['pred_logits'].square().mean().backward()
        self.assertGreater(model.decoder.layers[-1].cross_attn.angle_proj.weight.grad.abs().sum().item(), 0.)

    def test_actual_dn_aux_training_and_spi_wfdr_combination_reach_all_new_parameters(self):
        for combined, amp in [(False, False), (True, True)]:
            with self.subTest(combined=combined, amp=amp):
                encoder = HybridEncoder(in_channels=[32, 64, 128], hidden_dim=32, nhead=4,
                                        dim_feedforward=64, depth_mult=.34, use_spi_module=combined)
                model = decoder()
                weights = dict(loss_mal=1, loss_bbox=5, loss_giou=2, loss_fgl=.15, loss_ddf=1.5)
                if combined:
                    weights['loss_wfdr'] = .15
                criterion = DEIMCriterion(
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
                        losses = criterion(_to_float32(output), targets)
                        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
                        self.assertEqual(set(records), {0, 1, 2})
                        self.assertTrue(all(args[0].shape[1] > 5 for args in records.values()))
                        self.assertIn('loss_fgl_aux_0', losses)
                        self.assertIn('loss_mal_dn_0', losses)
                        self.assertEqual('loss_wfdr' in losses, combined)
                        sum(losses.values()).backward()
                        for layer in model.decoder.layers:
                            module = layer.cross_attn
                            for name, parameter in module.named_parameters():
                                self.assertIsNotNone(parameter.grad, name)
                                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                            self.assertGreater(module.topo_steps.weight.grad.abs().sum().item(), 0.)
                            self.assertGreater(module.angle_proj.weight.grad.abs().sum().item(), 0.)
                            self.assertGreater(module.gate_alpha.grad.abs().sum().item(), 0.)
                            if step:
                                self.assertGreater(module.topo_gate_linear.weight.grad.abs().sum().item(), 0.)
                        optimizer.step()
                finally:
                    for handle in handles:
                        handle.remove()

    def test_checkpoint_and_deploy_keep_learned_gate_scales_and_all_required_layers(self):
        model = decoder().eval()
        for index, layer in enumerate(model.decoder.layers):
            with torch.no_grad():
                layer.cross_attn.gate_alpha.fill_(.2 + .1 * index)
                layer.cross_attn.topo_steps.bias.mul_(1.3)
                layer.cross_attn.angle_proj.bias.fill_(.4)
        feats, _ = inputs()
        with torch.no_grad():
            expected = model(feats)
        restored = decoder().eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        restored.convert_to_deploy()
        restored.decoder.convert_to_deploy()
        self.assertEqual(len(restored.decoder.layers), model.eval_idx + 1)
        with torch.no_grad():
            actual = restored(feats)
        for key in ('pred_logits', 'pred_boxes', 'pred_corners'):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_new_method_and_density_control_configs_preserve_the_frozen_protocol(self):
        folder = ROOT / 'configs/deim_dfine'
        outputs = set()
        for suffix, enabled in [('sampling464', False), ('ost_da_v2', True)]:
            path = folder / f'deim_hgnetv2_m_sar_aircraft_{suffix}_58e_b28_old_server.yml'
            config = load_config(str(path), {})
            reference = load_config(str(folder / 'deim_hgnetv2_m_sar_aircraft_58e_b28_old_server.yml'), {})
            self.assertEqual(config['DFINETransformer'].pop('num_points'), [4, 6, 4])
            self.assertEqual(reference['DFINETransformer'].pop('num_points'), [3, 6, 3])
            self.assertEqual(config['DFINETransformer'].pop('use_ost_da', False), enabled)
            self.assertEqual(config['DFINETransformer'].pop('ost_da_version', 'v1'), 'v2' if enabled else 'v1')
            actual = create('DFINETransformer', YAMLConfig(str(path)).global_cfg)
            expected_type = SymmetricOrthogonalTopologyDeformAttn if enabled else MSDeformableAttention
            self.assertTrue(all(isinstance(layer.cross_attn, expected_type) for layer in actual.decoder.layers))
            self.assertTrue((ROOT / config['experiment_protocol_document']).is_file())
            self.assertNotIn(config['output_dir'], outputs)
            outputs.add(config['output_dir'])
            for candidate in (config, reference):
                for key in ('__include__', 'output_dir', 'experiment_protocol_document'):
                    candidate.pop(key, None)
            self.assertEqual(config, reference)


if __name__ == '__main__':
    unittest.main()
