"""Functional checks for the complete BCCQ model, objective and 102e protocol."""
import copy
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
import yaml

from engine.core import YAMLConfig
from engine.core.yaml_utils import load_config
from engine.deim.band_conditioned_queries import BandConditionedClassQueries, joint_detection_logits
from engine.deim.dfine_decoder import DFINETransformer
from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.matcher import HungarianMatcher
from engine.deim.radar_enhancement import HaarIDWT2D
from engine.optim.ema import ModelEMA


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_bccq_102e_b28_old_server.yml'
BASE = ROOT / 'configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_102e_b28_old_server.yml'


def model_args():
    return dict(num_classes=7, hidden_dim=32, feat_channels=[32] * 3, num_queries=6,
                feat_strides=[8, 16, 32], num_layers=4, nhead=4, num_points=[2, 2, 2],
                dim_feedforward=64, reg_max=8, num_denoising=4, eval_spatial_size=None,
                bccq_dim=16, bccq_heads=4, bccq_feedforward=32, bccq_roi_size=3,
                bccq_query_chunk_size=3)


def make_model(**overrides):
    args = model_args()
    args.update(overrides)
    return DFINETransformer(**args, use_bccq=True)


def make_criterion():
    matcher = HungarianMatcher(dict(cost_class=2, cost_bbox=5, cost_giou=2), use_focal_loss=True)
    return DEIMCriterion(matcher, dict(loss_mal=1, loss_bbox=5, loss_giou=2, loss_fgl=.15,
                                      loss_ddf=1.5, loss_bccq_type=1, loss_bccq_quality=1),
                         ['mal', 'boxes', 'local'], num_classes=7, reg_max=8,
                         gamma=1.5, use_bccq=True)


def fixture():
    features = [torch.randn(2, 32, h, w) for h, w in [(8, 12), (4, 6), (2, 3)]]
    targets = [dict(labels=torch.tensor([0, 6]), boxes=torch.tensor([[.3, .4, .2, .3], [.7, .6, .3, .2]])),
               dict(labels=torch.tensor([3]), boxes=torch.tensor([[.4, .5, .3, .4]]))]
    return features, targets


class BCCQTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(193)

    def test_signed_sources_odd_size_reconstruction_and_coordinate_mapping(self):
        branch = BandConditionedClassQueries(8, 7, 1, 0, dim=8, heads=2, feedforward=16, roi_size=5)
        memory = torch.randn(1, 5 * 7 + 3 * 4, 8)
        sources, scales = branch.prepare_sources(memory, [(5, 7), (3, 4)])
        restored = HaarIDWT2D()(sources[2:], (5, 7))
        torch.testing.assert_close(restored, sources[0], rtol=2e-6, atol=2e-6)
        self.assertEqual(len(sources), 6)
        self.assertTrue(any((source < 0).any() for source in sources[2:]))
        self.assertEqual(scales[2], (7 / 8, 5 / 6))
        ramp = torch.arange(3 * 4).float().reshape(1, 1, 3, 4)
        reference = torch.tensor([[[1 / 7, 1 / 5, 0., 0.]]])
        # The first 2x2 Haar cell has center (1/W,1/H) in the original P3 extent.
        pooled = branch.sample_source(ramp, reference, scales[2])
        torch.testing.assert_close(pooled, torch.zeros_like(pooled), atol=1e-6, rtol=0)

    def test_six_source_uniform_routing_has_signed_mean_and_normalized_entropy(self):
        branch = BandConditionedClassQueries(8, 7, 1, 0, dim=8, heads=2, feedforward=16)
        layer = branch.layers[0]
        with torch.no_grad():
            for projection in (layer.query, layer.gate_query):
                projection.weight.zero_()
                projection.bias.zero_()
        values = [1., -8., -3., 2., -5., 1.]
        packed = [torch.cat((torch.full((1, 8, 4, 4), value), torch.zeros(1, 8, 4, 4),
                             torch.full((1, 8, 4, 4), value)), 1) for value in values]
        captured = []
        handle = layer.output.register_forward_pre_hook(lambda _m, args: captured.append(args[0].detach()))
        _, gates, entropy = branch._chunk(0, ((1., 1.),) * 6, torch.randn(1, 2, 7, 8),
                                          torch.rand(1, 2, 4), *packed)
        handle.remove()
        torch.testing.assert_close(captured[0], torch.full((1, 2, 7, 8), sum(values) / 6), atol=2e-6, rtol=0)
        torch.testing.assert_close(gates, torch.full_like(gates, 1 / 6))
        torch.testing.assert_close(entropy, torch.full_like(entropy, math.log(6)))

    def test_probability_factorization_and_extreme_logits_are_finite(self):
        types = torch.randn(2, 6, 7, requires_grad=True)
        quality = torch.randn(2, 6, 1, requires_grad=True)
        joint = joint_detection_logits(types, quality)
        expected = types.softmax(-1) * quality.sigmoid()
        torch.testing.assert_close(joint.sigmoid(), expected)
        torch.testing.assert_close(joint.sigmoid().sum(-1), quality.sigmoid().squeeze(-1))
        joint.square().mean().backward()
        self.assertTrue(torch.isfinite(types.grad).all() and torch.isfinite(quality.grad).all())
        extremes = torch.tensor([[[1000., -1000., -1000., -1000., -1000., -1000., -1000.]]], requires_grad=True)
        result = joint_detection_logits(extremes, torch.full((1, 1, 1), 1000., requires_grad=True))
        self.assertTrue(torch.isfinite(result).all())
        result.sum().backward()
        self.assertTrue(torch.isfinite(extremes.grad).all())

    def test_projection_before_roi_pooling_matches_projection_after_pooling(self):
        branch = BandConditionedClassQueries(8, 7, 1, 0, dim=8, heads=2, feedforward=16)
        sources = tuple(torch.randn(2, 8, h, w) for h, w in [(8, 12), (4, 6)] + [(4, 6)] * 4)
        boxes = torch.rand(2, 3, 4) * 1.5 - .25
        layer = branch.layers[0]
        packed = layer.project_sources(sources)
        for source, pack in zip(sources, packed):
            raw = branch.sample_source(source, boxes, (1., 1.))
            actual_raw, actual_key, actual_value = branch.sample_source(pack, boxes, (1., 1.)).split(8, -1)
            torch.testing.assert_close(actual_raw, raw)
            torch.testing.assert_close(actual_key, layer.key(raw), atol=3e-7, rtol=2e-6)
            torch.testing.assert_close(actual_value, layer.value(raw), atol=3e-7, rtol=2e-6)

    def test_type_and_quality_objectives_match_independent_scalar_equations(self):
        criterion = make_criterion()
        types = torch.zeros(1, 3, 7, requires_grad=True)
        quality = torch.tensor([[[0.], [math.log(3)], [-math.log(3)]]], requires_grad=True)
        boxes = torch.tensor([[[.3, .4, .2, .2], [.6, .5, .2, .2], [.1, .1, .1, .1]]], requires_grad=True)
        targets = [dict(labels=torch.tensor([0, 6]), boxes=torch.tensor([[.3, .4, .2, .2], [.6, .5, .4, .2]]))]
        indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
        losses = criterion.loss_bccq(dict(bccq_type_logits=types, bccq_quality_logits=quality, pred_boxes=boxes),
                                    targets, indices, 2.)
        target = .5 ** 1.5
        expected = (math.log(2) + target * -math.log(.75) + (1 - target) * -math.log(.25)
                    + .25 ** 1.5 * -math.log(.75)) / 2
        self.assertAlmostEqual(losses['loss_bccq_type'].item(), math.log(7), places=6)
        self.assertAlmostEqual(losses['loss_bccq_quality'].item(), expected, places=5)
        sum(losses.values()).backward()
        self.assertIsNone(boxes.grad)
        torch.testing.assert_close(types.grad[0, 2], torch.zeros(7))
        self.assertGreater(quality.grad[0, 2].abs().item(), 0.)

    def test_empty_positive_type_loss_remains_ddp_connected(self):
        criterion = make_criterion()
        types = torch.randn(1, 3, 7, requires_grad=True)
        quality = torch.randn(1, 3, 1, requires_grad=True)
        empty = torch.empty(0, dtype=torch.long)
        targets = [dict(labels=empty, boxes=torch.empty(0, 4))]
        losses = criterion.loss_bccq(dict(bccq_type_logits=types, bccq_quality_logits=quality,
                                        pred_boxes=torch.rand(1, 3, 4)), targets, [(empty, empty)], 1.)
        sum(losses.values()).backward()
        torch.testing.assert_close(types.grad, torch.zeros_like(types))
        self.assertTrue(torch.isfinite(quality.grad).all())

    def test_gate_sums_and_counts_recover_instance_weighted_class_statistics(self):
        criterion = make_criterion()
        gates = torch.zeros(1, 3, 7, 6)
        gates[0, 0, 0, 0] = 1.
        gates[0, 1, 0, 1] = 1.
        gates[0, 2, 1, 2] = 1.
        entropy = torch.zeros(1, 3, 7)
        entropy[0, 1, 0] = .5
        targets = [dict(labels=torch.tensor([0, 0, 1]))]
        indices = [(torch.arange(3), torch.arange(3))]
        criterion._record_bccq_metrics(dict(bccq_source_weights=gates, bccq_source_entropy=entropy,
                                            bccq_type_logits=torch.zeros(1, 3, 7)), targets, indices)
        metrics = criterion.bccq_metrics
        self.assertEqual(metrics['bccq_positive_c0'].item(), 2)
        self.assertEqual(metrics['bccq_gate_sum_c0_p3'].item(), 1.)
        self.assertEqual(metrics['bccq_gate_sum_c0_p4'].item(), 1.)
        self.assertEqual(metrics['bccq_gate_entropy_sum_c0'].item(), .5)
        self.assertEqual(metrics['bccq_positive_c6'].item(), 0)
        self.assertEqual(metrics['bccq_gate_sum_c6_p3'].item(), 0)

    def test_query_and_box_detach_preserve_visual_and_all_branch_gradients(self):
        branch = BandConditionedClassQueries(16, 7, 2, 1, dim=16, heads=4, feedforward=32,
                                             roi_size=3, query_chunk_size=2)
        memory = torch.randn(2, 8 * 8 + 4 * 4, 16, requires_grad=True)
        instances = torch.randn(2, 5, 16, requires_grad=True)
        boxes = (torch.rand(2, 5, 4) * .4 + .2).requires_grad_()
        states = branch.initialize_queries(instances)
        sources = branch.prepare_sources(memory, [(8, 8), (4, 4)])
        for layer in range(2):
            states, gates, entropy = branch.step(layer, states, boxes, sources)
        F.cross_entropy(branch.classify(states).flatten(0, 1), torch.arange(10) % 7).backward()
        self.assertIsNone(instances.grad)
        self.assertIsNone(boxes.grad)
        self.assertGreater(memory.grad.abs().sum().item(), 0.)
        for name, parameter in branch.named_parameters():
            if not name.startswith('quality_heads.'):
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater((gates[:, :, 0] - gates[:, :, 1]).abs().sum().item(), 1e-7)
        torch.testing.assert_close(gates.sum(-1), torch.ones(2, 5, 7))
        self.assertEqual(entropy.shape, (2, 5, 7))

    def test_complete_four_layer_aux_dn_losses_and_teacher_routing(self):
        model, criterion = make_model(), make_criterion()
        features, targets = fixture()
        output = model(features, targets)
        self.assertEqual(len(output['aux_outputs']), 3)
        self.assertEqual(len(output['dn_outputs']), 4)
        for item in [output, *output['aux_outputs'], *output['dn_outputs']]:
            self.assertEqual(item['bccq_type_logits'].shape[-1], 7)
            self.assertEqual(item['bccq_quality_logits'].shape[-1], 1)
            torch.testing.assert_close(item['pred_logits'], joint_detection_logits(
                item['bccq_type_logits'], item['bccq_quality_logits']))
        self.assertNotIn('bccq_type_logits', output['pre_outputs'])
        self.assertNotIn('bccq_type_logits', output['enc_aux_outputs'][0])
        for aux in output['aux_outputs']:
            torch.testing.assert_close(aux['teacher_logits'], output['pred_logits'])
        losses = criterion(output, targets)
        for suffix in ['', '_aux_0', '_aux_1', '_aux_2', '_dn_0', '_dn_1', '_dn_2', '_dn_3']:
            self.assertIn('loss_bccq_type' + suffix, losses)
            self.assertIn('loss_bccq_quality' + suffix, losses)
            self.assertNotIn('loss_mal' + suffix, losses)
        for suffix in ['_pre', '_enc_0', '_dn_pre']:
            self.assertIn('loss_mal' + suffix, losses)
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        sum(losses.values()).backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertEqual(criterion.bccq_metrics['bccq_positive_c0'].item(), 1)
        self.assertEqual(criterion.bccq_metrics['bccq_positive_c6'].item(), 1)

    def test_chunking_and_recomputation_preserve_outputs_and_gradients(self):
        first = make_model(num_denoising=0, bccq_checkpoint=False, bccq_query_chunk_size=100)
        second = copy.deepcopy(first)
        second.decoder.bccq.query_chunk_size = 2
        second.decoder.bccq.activation_checkpoint = True
        features, targets = fixture()
        outputs = []
        for model in (first, second):
            result = model(features, targets)
            sum(make_criterion()(result, targets).values()).backward()
            outputs.append(result)
        torch.testing.assert_close(outputs[0]['pred_logits'], outputs[1]['pred_logits'], rtol=3e-5, atol=3e-6)
        reference = dict(first.named_parameters())
        for name, parameter in second.named_parameters():
            if parameter.grad is not None:
                torch.testing.assert_close(parameter.grad, reference[name].grad, rtol=2e-3, atol=3e-5, msg=name)

    def test_shared_initialization_and_rng_preserve_original_box_predictions(self):
        args = model_args()
        torch.manual_seed(81)
        baseline = DFINETransformer(**copy.deepcopy(args))
        baseline_rng = torch.get_rng_state()
        torch.manual_seed(81)
        improved = DFINETransformer(**copy.deepcopy(args), use_bccq=True)
        torch.testing.assert_close(torch.get_rng_state(), baseline_rng)
        baseline_weights = baseline.state_dict()
        for name, value in improved.state_dict().items():
            if name in baseline_weights:
                torch.testing.assert_close(value, baseline_weights[name], rtol=0, atol=0)
        features, _ = fixture()
        baseline.eval()
        improved.eval()
        torch.testing.assert_close(baseline(features)['pred_boxes'], improved(features)['pred_boxes'], rtol=0, atol=0)

    def test_ema_optimizer_restore_and_deploy_preserve_complete_classification(self):
        model = make_model(eval_idx=2)
        features, targets = fixture()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        sum(make_criterion()(model(features, targets), targets).values()).backward()
        optimizer.step()
        ema = ModelEMA(model, warmups=0)
        ema.update(model)
        stream = io.BytesIO()
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), ema=ema.state_dict()), stream)
        stream.seek(0)
        saved = torch.load(stream, weights_only=False)
        restored = make_model(eval_idx=2)
        restored.load_state_dict(saved['model'], strict=True)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4)
        restored_optimizer.load_state_dict(saved['optimizer'])
        restored_ema = ModelEMA(restored, warmups=0)
        restored_ema.load_state_dict(saved['ema'])
        model.eval()
        restored.eval()
        expected = model(features)
        torch.testing.assert_close(expected['pred_logits'], restored(features)['pred_logits'], rtol=0, atol=0)
        self.assertEqual(len(restored_optimizer.state), len(optimizer.state))
        for module in list(restored.modules()):
            if hasattr(module, 'convert_to_deploy'):
                module.convert_to_deploy()
        actual = restored(features)
        torch.testing.assert_close(actual['pred_logits'], expected['pred_logits'])
        torch.testing.assert_close(actual['pred_boxes'], expected['pred_boxes'])
        self.assertEqual(len(restored.decoder.bccq.layers), 3)

    def test_cpu_amp_preserves_finite_complete_objective_and_gradients(self):
        model, criterion = make_model(), make_criterion()
        features, targets = fixture()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            output = model(features, targets)
        losses = criterion(output, targets)
        sum(losses.values()).backward()
        self.assertEqual(output['pred_logits'].dtype, torch.float32)
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_102e_config_changes_only_complete_method_and_output_metadata(self):
        base = load_config(str(BASE), {})
        candidate = load_config(str(CONFIG), {})
        for values in (base, candidate):
            for key in ['__include__', 'output_dir', 'experiment_protocol_document']:
                values.pop(key, None)
        bccq = {key: candidate['DFINETransformer'].pop(key) for key in list(candidate['DFINETransformer'])
                if key.startswith('bccq_') or key == 'use_bccq'}
        self.assertTrue(bccq['use_bccq'])
        self.assertEqual((bccq['bccq_dim'], bccq['bccq_roi_size'], bccq['bccq_heads']), (128, 5, 4))
        self.assertTrue(candidate['DEIMCriterion'].pop('use_bccq'))
        for key in ['loss_bccq_type', 'loss_bccq_quality']:
            self.assertEqual(candidate['DEIMCriterion']['weight_dict'].pop(key), 1)
        self.assertEqual(base, candidate)
        cfg = YAMLConfig(str(CONFIG))
        model, criterion = cfg.model, cfg.criterion
        self.assertTrue(model.decoder.use_bccq and criterion.use_bccq)
        self.assertEqual(len(model.decoder.decoder.bccq.layers), 4)
        self.assertEqual(cfg.yaml_cfg['epoches'], 102)
        self.assertEqual(sum(p.numel() for p in model.decoder.decoder.bccq.parameters()), 1_341_956)
        baseline_model = YAMLConfig(str(BASE)).model
        self.assertEqual(sum(p.numel() for p in model.parameters()) -
                         sum(p.numel() for p in baseline_model.parameters()), 1_336_559)

    def test_real_config_full_detector_connects_backbone_encoder_and_bccq_losses(self):
        cfg = YAMLConfig(str(CONFIG))
        model, criterion = cfg.model, cfg.criterion
        _, targets = fixture()
        # 128x160 retains >=300 encoder positions, exercising the real M model,
        # all six sources and 5x5 pooling with the actual 300-query configuration.
        output = model(torch.randn(2, 3, 128, 160), targets)
        self.assertEqual(output['pred_logits'].shape, (2, 300, 7))
        losses = criterion(output, targets)
        sum(losses.values()).backward()
        self.assertTrue(all(torch.isfinite(loss) for loss in losses.values()))
        for name, parameter in model.decoder.decoder.bccq.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(model.decoder.decoder.bccq.input_proj[0].weight.grad.abs().sum().item(), 0.)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.backbone.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters()))

    def test_launcher_metadata_archives_the_actual_complete_102e_method(self):
        script = (ROOT / 'tools/run_sar_module_experiment.sh').read_text(encoding='utf-8')
        program = re.findall(r"python - <<'PY'\n(.*?)\nPY", script, re.S)[0]
        with tempfile.TemporaryDirectory(prefix='deim_bccq_metadata_') as directory:
            temporary = Path(directory)
            cfg = load_config(str(CONFIG), {})
            cfg.pop('__include__', None)
            cfg['train_dataloader']['dataset']['ann_file'] = str(temporary / 'train.json')
            cfg['experiment_protocol_document'] = str(ROOT / cfg['experiment_protocol_document'])
            (temporary / 'stats.json').write_text('{}', encoding='utf-8')
            source = temporary / 'config.yml'
            source.write_text(yaml.safe_dump(cfg), encoding='utf-8')
            run = temporary / 'run'
            run.mkdir()
            environment = dict(OUT=str(run), CONFIG=str(source), MODE='new', ATTEMPT='new',
                               INITIAL='/home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth')
            with patch.dict(os.environ, environment), redirect_stdout(io.StringIO()):
                exec(compile(program, 'run_sar_module_experiment.sh:metadata', 'exec'), {})
            metadata = json.loads((run / 'run_metadata.json').read_text(encoding='utf-8'))
            self.assertEqual((metadata['epochs'], metadata['global_batch'], metadata['devices']), (102, 28, [0, 1]))
            method = metadata['bccq_settings']
            self.assertEqual((method['class_queries'], method['classification_layers']), (7, 4))
            self.assertEqual(len(method['sources']), 6)
            self.assertTrue(method['settings']['bccq_checkpoint'])
            self.assertEqual((method['quality_gamma'], method['quality_negative_alpha']), (1.5, 1.))
            self.assertEqual(metadata['training_only_loss_weights']['loss_bccq_type'], 1)
            self.assertIn('use_bccq', metadata['resolved_config']['DEIMCriterion'])


if __name__ == '__main__':
    unittest.main()
