"""Check real Decoder readouts, frozen Haar interventions and GT-aligned attribution."""

import copy
import gzip
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import yaml

from engine.deim.dfine_decoder import DFINETransformer
from engine.deim.hybrid_encoder import HybridEncoder
from engine.deim.postprocessor import PostProcessor
from engine.deim.radar_enhancement import RadarEnhancementModule
from tools.compare_rem_102e import evaluate_coco
from tools.rem_mechanism_probe import analyze, package, prepare, run_jobs, summarize_evidence
from tools.rem_probe_utils import (ARRAY_KEYS, RESPONSE_NAMES, SCALES, VARIANTS, BandReadout, EvidenceReadout,
                                   anchor_evidence, make_anchors, merge_query_shards, paired_decomposition,
                                   pool_responses, postprocess_with_queries, raw_only_predictions)


class TinyDetector(nn.Module):
    def __init__(self, rem):
        super().__init__()
        self.encoder = HybridEncoder(in_channels=[32, 64, 128], hidden_dim=32, nhead=8,
                                     dim_feedforward=64, depth_mult=.34, use_rem=rem)
        self.decoder = DFINETransformer(num_classes=7, hidden_dim=32, feat_channels=[32] * 3,
            num_queries=5, num_layers=4, eval_idx=-1, nhead=8, num_points=[3, 6, 3],
            dim_feedforward=64, reg_max=32, num_denoising=4, eval_spatial_size=None)

    def forward(self, features):
        return self.decoder(self.encoder(features))


def data():
    return dict(info={}, images=[dict(id=i, width=256, height=256, file_name=f'{i}.jpg') for i in (1, 2)],
                categories=[dict(id=c, name=name) for c, name in ((0, 'A220'), (2, 'A330'), (6, 'other'))],
                annotations=[dict(id=i, image_id=1, category_id=c, bbox=b, area=b[2]*b[3], iscrowd=0)
                             for i, c, b in ((1, 0, [20, 20, 40, 40]), (2, 6, [120, 20, 40, 40]),
                                             (3, 2, [30, 100, 100, 100]))])


def arrays(rem=False):
    boxes = np.array([[40, 40, 40, 40], [140, 40, 40, 40], [80, 150, 100, 100], [230, 230, 20, 20]], np.float32) / 256
    result = dict(image_ids=np.array([1, 2]), orig_sizes=np.array([[256, 256], [256, 256]]),
                  boxes=np.stack((boxes, boxes)), raw_logits=np.full((2, 4, 7), -10., np.float32),
                  quality=np.zeros((2, 4, 1), np.float32), core_abs=np.ones((2, 4, 6), np.float32),
                  ring_abs=np.ones((2, 4, 6), np.float32), valid_fractions=np.ones((2, 4, 2), np.float32))
    result['raw_logits'][0, 0, 0] = 2.
    result['raw_logits'][0, 1, [0, 6]] = [.2, 1.]
    result['quality'][0, 1, 0] = .3
    result['raw_logits'][0, 2, 2] = 1.5
    result['raw_logits'][1, 0, 0] = .3
    if rem:
        result['raw_logits'][0, 1, 0] = .6
        result['quality'][0, 1, 0] = .4
    result['final_logits'] = result['raw_logits'] + result['quality']
    if rem:
        order = [2, 0, 3, 1]
        for key in ('boxes', 'raw_logits', 'quality', 'final_logits', 'core_abs', 'ring_abs', 'valid_fractions'):
            result[key] = result[key][:, order].copy()
    return result


def native_predictions(values, raw=False, topk=8):
    return postprocess_with_queries(PostProcessor(num_classes=7, num_top_queries=topk),
        dict(pred_logits=torch.from_numpy(values['raw_logits' if raw else 'final_logits']),
             pred_boxes=torch.from_numpy(values['boxes'])),
        torch.from_numpy(values['orig_sizes']), values['image_ids'])


class RemMechanismProbeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1007)
        self.temporary = tempfile.TemporaryDirectory(prefix='deim_rem_mechanism_probe_')
        self.folder = Path(self.temporary.name).resolve()
        self.assertEqual(self.folder.parent, Path(tempfile.gettempdir()).resolve())
        self.assertTrue(self.folder.name.startswith('deim_rem_mechanism_probe_'))

    def tearDown(self):
        self.temporary.cleanup()  # absolute cleanup target was verified within the temp root

    def prepared(self):
        parent = self.folder / 'comparison'
        parent.mkdir()
        (parent / 'val.json').write_text(json.dumps(data()))
        selections = {}
        for role in ('baseline', 'rem'):
            directory = parent / role
            directory.mkdir()
            config = dict(num_classes=7, HGNetv2=dict(pretrained=False), HybridEncoder=dict(use_rem=role == 'rem'),
                          train_dataloader=dict(total_batch_size=28),
                          val_dataloader=dict(total_batch_size=16, dataset=dict(img_folder='/old', ann_file='/old.json')))
            for name in ('resolved_config.yml', 'inference_config.yml'):
                (directory / name).write_text(yaml.safe_dump(config))
            selection = dict(checkpoint=str(directory / 'best_stg2.pth'), logical_epoch=90 if role == 'baseline' else 98)
            selections[role] = selection
            (directory / 'selection.json').write_text(json.dumps(selection))
            with gzip.open(directory / 'predictions.json.gz', 'wt') as stream:
                json.dump(native_predictions(arrays(role == 'rem')), stream)
        (parent / 'manifest.json').write_text(json.dumps(dict(data_root=str(self.folder / 'dataset'), runs=selections)))
        (parent / 'report').mkdir()
        (parent / 'report/summary.json').write_text(json.dumps(dict(evaluations={r: dict(metrics=dict(AP=0.)) for r in selections})))
        out = self.folder / 'probe'
        manifest = prepare(parent, out)
        return parent, out, manifest

    def test_native_four_layer_decoder_decomposition_and_query_indices_preserve_predictions(self):
        features = [torch.randn(2, c, h, w) for c, h, w in ((32, 8, 12), (64, 4, 6), (128, 2, 3))]
        for has_rem in (False, True):
            model = TinyDetector(has_rem).eval()
            if has_rem:
                model.encoder.rem.alpha.data.fill_(.2)
            last = model.decoder.decoder.eval_idx
            model.decoder.dec_bbox_head[last].layers[-1].weight.data.normal_(0., .03)
            model.decoder.decoder.lqe_layers[last].reg_conf.layers[-1].weight.data.normal_(0., .2)
            model.decoder.decoder.lqe_layers[last].reg_conf.layers[-1].bias.data.fill_(.25)
            state = copy.deepcopy(model.state_dict())
            with torch.inference_mode():
                expected = model(features)
                readout = EvidenceReadout(model, 'rem' if has_rem else 'baseline')
                actual = model(features)
                torch.testing.assert_close(actual['pred_logits'], readout.raw + readout.quality, rtol=0, atol=0)
                torch.testing.assert_close(actual['pred_logits'], expected['pred_logits'], rtol=0, atol=0)
                torch.testing.assert_close(actual['pred_boxes'], expected['pred_boxes'], rtol=0, atol=0)
                self.assertEqual(readout.quality.shape, (2, 5, 1))
                self.assertGreater(readout.quality.std().item(), 0.)
                torch.testing.assert_close(readout.raw[..., 0] - readout.raw[..., 1],
                                           actual['pred_logits'][..., 0] - actual['pred_logits'][..., 1])
                core, ring, valid = pool_responses(readout.bands.maps(), actual['pred_boxes'])
                self.assertEqual(core.shape, (2, 5, 6))
                self.assertTrue(torch.isfinite(core).all() and torch.isfinite(ring).all())
                self.assertTrue(((valid >= 0) & (valid <= 1)).all())
                pp = PostProcessor(num_classes=7, num_top_queries=10)
                records = postprocess_with_queries(pp, actual, torch.tensor([[256, 256], [256, 256]]), [1, 2])
                for record in records:
                    b, q, c = record['image_id'] - 1, record['query_index'], record['category_id']
                    self.assertAlmostEqual(record['score'], actual['pred_logits'][b, q, c].sigmoid().item())
                readout.close()
            for name, value in state.items():
                torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)

    def test_band_attenuation_and_response_maps_have_analytic_haar_values_without_weight_changes(self):
        module = RadarEnhancementModule(2).eval()
        for name in ('ll', 'lh', 'hl', 'hh'):
            setattr(module, name, nn.Identity())
        module.alpha.data.fill_(.25)
        checker = torch.tensor([[1., -1.], [-1., 1.]]).repeat(2, 2)
        x = torch.stack((torch.ones(4, 4), checker))[None]
        for variant, total in (('rem', .25), ('rem_half', .125), ('rem_ll_half', .1875), ('rem_hf_half', .1875)):
            readout = BandReadout(module, SCALES[variant])
            calls = []
            h1 = module.dwt.register_forward_hook(lambda *args: calls.append('dwt'))
            h2 = module.idwt.register_forward_hook(lambda *args: calls.append('idwt'))
            with torch.no_grad():
                y = module(x)
                maps = readout.maps()
            self.assertEqual(calls, ['dwt', 'idwt'])
            torch.testing.assert_close(maps[:, 1], torch.full((1, 4, 4), total), rtol=0, atol=0)
            torch.testing.assert_close(maps[:, 2], torch.full((1, 4, 4), .125 * SCALES[variant][0]), rtol=0, atol=0)
            torch.testing.assert_close(maps[:, 5], torch.full((1, 4, 4), .125 * SCALES[variant][3]), rtol=0, atol=0)
            torch.testing.assert_close((y - x).abs().mean(1), maps[:, 1], rtol=0, atol=0)
            self.assertEqual(module.alpha.item(), .25)
            h1.remove()
            h2.remove()
            readout.close()
        torch.testing.assert_close(module(x), x * 1.25, rtol=0, atol=0)

    def test_response_pool_excludes_outside_samples_instead_of_inventing_contrast(self):
        maps = torch.ones(1, 6, 8, 8) * torch.arange(1., 7.)[None, :, None, None]
        boxes = torch.tensor([[[.5, .5, .5, .5], [.05, .05, .5, .5], [-2., -2., .1, .1]]])
        core, ring, valid = pool_responses(maps, boxes)
        for q in (0, 1):
            torch.testing.assert_close(core[0, q], torch.arange(1., 7.))
            torch.testing.assert_close(ring[0, q], torch.arange(1., 7.))
        self.assertLess(valid[0, 1, 1].item(), 1.)
        torch.testing.assert_close(core[0, 2], torch.zeros(6))
        torch.testing.assert_close(ring[0, 2], torch.zeros(6))
        self.assertTrue(torch.isfinite(core).all())

    def test_shard_merge_keeps_image_order_aligned_for_all_arrays(self):
        values = arrays()
        shards = [{key: value[index:index+1] for key, value in values.items()} for index in (1, 0)]
        merged = merge_query_shards(shards)
        for key in ARRAY_KEYS:
            np.testing.assert_array_equal(merged[key], values[key])

    def test_anchor_matching_uses_gt_and_geometry_not_query_identity(self):
        b, r = arrays(), arrays(True)
        anchors = make_anchors(data(), dict(baseline=native_predictions(b), rem=native_predictions(r)))
        self.assertEqual(sum(a['group'] == 'a220_wrongclass' for a in anchors), 1)
        self.assertEqual(sum(a['group'] == 'a220_background' for a in anchors), 1)
        records = anchor_evidence(anchors, b, 'baseline') + anchor_evidence(anchors, r, 'rem')
        wrong = [row for row in records if row['group'] == 'a220_wrongclass']
        self.assertNotEqual(wrong[0]['query_index'], wrong[1]['query_index'])
        change = next(row for row in paired_decomposition(records) if row['group'] == 'a220_wrongclass')
        self.assertAlmostEqual(change['delta_raw_logit'], .4, places=6)
        self.assertAlmostEqual(change['delta_quality_logit'], .1, places=6)
        self.assertAlmostEqual(change['delta_final_logit'], .5, places=6)

    def test_missing_candidate_is_not_treated_as_zero_logit(self):
        anchors = make_anchors(data(), dict(baseline=native_predictions(arrays()), rem=native_predictions(arrays(True))))
        missing = arrays(True)
        missing['boxes'][:] = [.95, .95, .01, .01]
        records = anchor_evidence(anchors, arrays(), 'baseline') + anchor_evidence(anchors, missing, 'rem')
        self.assertFalse(any(row['present'] for row in records if row['variant'] == 'rem'))
        self.assertEqual(paired_decomposition(records), [])
        groups, pairs = summarize_evidence(records, [])
        self.assertEqual(next(r for r in groups if r['variant'] == 'rem' and r['group'] == 'a220_true')['present'], 0)
        self.assertIsNone(next(r for r in pairs if r['variant'] == 'rem' and r['group'] == 'a220_true')['mean_delta_raw_logit'])

    def test_removing_lqe_reselects_queries_and_can_fix_quality_induced_fp_ranking(self):
        annotation = dict(info={}, images=[dict(id=1, width=256, height=256)], categories=[dict(id=0, name='A220')],
                          annotations=[dict(id=1, image_id=1, category_id=0, bbox=[20, 20, 40, 40], area=1600, iscrowd=0)])
        path = self.folder / 'one.json'
        path.write_text(json.dumps(annotation))
        values = arrays()
        values = {key: value[:1, :2].copy() if value.ndim >= 3 else value[:1].copy() for key, value in values.items()}
        values['raw_logits'][:] = -20
        values['raw_logits'][0, :, 0] = [2., 1.]
        values['quality'][0, :, 0] = [0., 3.]
        values['final_logits'] = values['raw_logits'] + values['quality']
        before_boxes = values['boxes'].copy()
        full = native_predictions(values, topk=2)
        raw = raw_only_predictions(values, topk=2)
        self.assertNotEqual(full[0]['query_index'], raw[0]['query_index'])
        self.assertAlmostEqual(evaluate_coco(path, full)['metrics']['AP50'], 50.)
        self.assertAlmostEqual(evaluate_coco(path, raw)['metrics']['AP50'], 100.)
        np.testing.assert_array_equal(values['boxes'], before_boxes)

    def test_preparation_analysis_and_package_use_existing_comparison_and_no_weights(self):
        parent, out, manifest = self.prepared()
        archived = json.loads((out / 'manifest.json').read_text())
        archived['comparison'] = str(self.folder / 'unavailable_on_another_computer')
        (out / 'manifest.json').write_text(json.dumps(archived))
        for variant in VARIANTS:
            values = arrays(variant != 'baseline')
            np.savez_compressed(out / variant / 'queries.npz', **values)
            with gzip.open(out / variant / 'predictions.json.gz', 'wt') as stream:
                json.dump(native_predictions(values), stream)
            (out / variant / 'export_metadata.json').write_text(json.dumps(dict(topk=8)))
            runtime = yaml.safe_load((out / variant / 'inference_config.yml').read_text())
            self.assertEqual(runtime['val_dataloader']['dataset']['ann_file'], str(out / 'val.json'))
            self.assertEqual(runtime['HybridEncoder']['use_rem'], variant != 'baseline')
            if variant != 'baseline':
                self.assertEqual(manifest['runs'][variant]['checkpoint'], manifest['runs']['rem']['checkpoint'])
        summary = analyze(out)
        self.assertIn('rem', summary['raw_only_evaluations'])
        self.assertIsNone(next(r for r in summary['anchor_groups']
                               if r['variant'] == 'rem' and r['group'] == 'a220_background')['gt_raw_top1'])
        self.assertTrue((out / 'report/paired_decomposition.csv').is_file())
        package(out)
        with tarfile.open(out / 'mechanism_bundle.tar.gz') as archive:
            names = archive.getnames()
        self.assertIn('rem/queries.npz', names)
        self.assertIn('reference_summary.json', names)
        self.assertFalse(any(name.endswith('.pth') for name in names))
        self.assertTrue((parent / 'baseline/resolved_config.yml').is_file())

    def test_resume_only_launches_unfinished_variant_on_both_gpus(self):
        _, out, manifest = self.prepared()
        for variant in VARIANTS[:-1]:
            (out / variant / 'export.done').write_text('completed\n')
        with patch('tools.rem_mechanism_probe.subprocess.run') as execute, \
             patch('tools.rem_mechanism_probe.analyze'), patch('tools.rem_mechanism_probe.package'):
            run_jobs(out, manifest)
        execute.assert_called_once()
        self.assertIn('--nproc_per_node=2', execute.call_args.args[0])
        self.assertIn('rem_hf_half', execute.call_args.args[0])
        self.assertEqual(execute.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '0,1')
        self.assertTrue((out / 'DONE').is_file())


if __name__ == '__main__':
    unittest.main()
