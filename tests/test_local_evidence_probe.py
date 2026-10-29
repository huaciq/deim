"""Functional contracts for local sampling, frozen candidate identity and held-out fitting."""
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from tools.local_evidence_probe import (
    FeatureTap, MODES, competition_score, conditional_geometry, coco_metrics,
    create_folds, detection_arrays, feature_matrix, fit_fold, grouped_folds,
    local_descriptors, prepare, report, sampling_points, targets_for, thumbnail_groups, write_npz,
)


class LocalEvidenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_sampling_positions_orientation_and_no_box_gradient(self):
        box = torch.tensor([[[.5, .4, .4, .2]]], requires_grad=True)
        body, sides = sampling_points(box)
        self.assertEqual(tuple(body.shape), (1, 1, 9, 2))
        self.assertEqual(tuple(sides.shape), (1, 1, 4, 3, 8, 2))
        torch.testing.assert_close(body[0, 0, 4], torch.tensor([.5, .4]))
        torch.testing.assert_close(sides[0, 0, 0, :, 0, 0], torch.tensor([.32, .30, .28]))
        torch.testing.assert_close(sides[0, 0, 2, :, 0, 0], torch.tensor([.68, .70, .72]))
        torch.testing.assert_close(sides[0, 0, 1, :, 0, 1], torch.tensor([.31, .30, .29]))
        torch.testing.assert_close(sides[0, 0, 3, :, 0, 1], torch.tensor([.49, .50, .51]))
        self.assertFalse(body.requires_grad)

    def test_local_descriptors_retain_positions_and_report_outside_points(self):
        yy, xx = torch.meshgrid((torch.arange(32) + .5) / 32,
                                (torch.arange(32) + .5) / 32, indexing='ij')
        feature = torch.stack((xx, yy) * 4)[None]
        box = torch.tensor([[[.5, .5, .4, .4], [.7, .5, .4, .4], [2., 2., .4, .4]]])
        roi, edge = local_descriptors(feature, box, seed=173)
        self.assertEqual(tuple(roi.shape), (1, 3, 81))
        self.assertEqual(tuple(edge.shape), (1, 3, 204))
        self.assertGreater(float((roi[:, 0, :72] - roi[:, 1, :72]).abs().sum()), .1)
        torch.testing.assert_close(roi[0, 0, -9:], torch.ones(9))
        torch.testing.assert_close(roi[0, 2, -9:], torch.zeros(9))
        torch.testing.assert_close(edge[0, 2, -12:], torch.zeros(12))
        torch.testing.assert_close(local_descriptors(feature, box, 173)[0], roi)

    def test_candidates_match_postprocessor_and_hooks_do_not_change_predictions(self):
        from engine.backbone.hgnetv2 import HGNetv2
        from engine.deim.deim import DEIM
        from engine.deim.hybrid_encoder import HybridEncoder
        from engine.deim.dfine_decoder import DFINETransformer
        from engine.deim.dfine_utils import weighting_function
        from engine.deim.postprocessor import PostProcessor

        torch.manual_seed(7)
        backbone = HGNetv2('B2', pretrained=False)
        encoder = HybridEncoder(in_channels=[384, 768, 1536], hidden_dim=32, nhead=4,
                                dim_feedforward=64, depth_mult=.34, eval_spatial_size=None)
        decoder = DFINETransformer(num_classes=7, hidden_dim=32, feat_channels=[32] * 3,
                                   num_queries=8, num_layers=2, nhead=4, num_points=2,
                                   dim_feedforward=64, reg_max=8, num_denoising=0, eval_spatial_size=None)
        model = DEIM(backbone, encoder, decoder).eval().requires_grad_(False)
        # Non-square original dimensions expose accidental (height,width) reversal.
        sizes = torch.tensor([[900, 1200], [1500, 800]])
        inputs = torch.rand(2, 3, 64, 64)
        with torch.inference_mode():
            original = model(inputs)
            tap = FeatureTap(model)
            observed = model(inputs)
            for key in ('pred_logits', 'pred_boxes', 'pred_corners'):
                torch.testing.assert_close(original[key], observed[key], rtol=0, atol=0)
            project = weighting_function(decoder.reg_max, decoder.up, decoder.reg_scale)
            data = detection_arrays(observed, sizes, decoder.decoder.lqe_layers[-1], project, tap.values, topk=20)
            reference = PostProcessor(num_classes=7, num_top_queries=20)(observed, sizes)
            for i in range(2):
                torch.testing.assert_close(data['box'][i], reference[i]['boxes'], rtol=0, atol=0)
                torch.testing.assert_close(data['score'][i], reference[i]['scores'], rtol=0, atol=0)
                torch.testing.assert_close(data['label'][i], reference[i]['labels'], rtol=0, atol=0)
            self.assertEqual(data['roi'].shape[-1], 162)
            self.assertEqual(data['border'].shape[-1], 408)
            tap.close()
            self.assertEqual(len(model.backbone.stages[0]._forward_hooks), 0)

    def test_grouping_never_splits_duplicate_scene_and_is_reproducible(self):
        rng = np.random.default_rng(4)
        base = rng.normal(size=(5, 1024))
        thumbnails = np.concatenate((base, base * 1.3 + .2))
        groups = thumbnail_groups(thumbnails)
        np.testing.assert_array_equal(groups[:5], groups[5:])
        image_ids = np.arange(10)
        folds = grouped_folds(image_ids, groups, 1)
        for i in range(5):
            self.assertEqual(folds[i], folds[i + 5])
        self.assertEqual(folds, grouped_folds(image_ids, groups, 1))
        self.assertEqual(set(folds.values()), {0, 1})
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / 'groups.json'
            file.write_text(json.dumps({str(i): str(i % 3) for i in image_ids}), encoding='utf-8')
            args = argparse.Namespace(groups=str(file), group_threshold=.95, seed=0, repeats=3)
            result = create_folds(args, image_ids, thumbnails)
            self.assertEqual(result['source'], 'provided_scene_groups')
            for assignment in result['repetitions']:
                self.assertEqual(assignment['0'], assignment['3'])

    def test_targets_separate_class_confusion_geometry_and_ignore_regions(self):
        ann = dict(annotations=[
            dict(image_id=1, category_id=0, bbox=[0, 0, 10, 10]),
            dict(image_id=1, category_id=1, bbox=[20, 20, 10, 10], iscrowd=1),
        ])
        d = dict(image_id=np.array([1, 1, 1, 2]), label=np.array([0, 1, 1, 0]),
                 box=np.array([[0, 0, 10, 10], [0, 0, 10, 10], [20, 20, 30, 30], [0, 0, 10, 10]]),
                 score=np.ones(4, np.float32))
        y, correct, eligible = targets_for(d, ann)
        np.testing.assert_array_equal(y, [[1, 1], [0, 1], [0, 0], [0, 0]])
        np.testing.assert_array_equal(correct, [True, False, False, False])
        np.testing.assert_array_equal(eligible, [True, True, False, True])

    def test_reranking_changes_only_scores_and_coco_measures_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            ann = dict(images=[dict(id=1, width=100, height=80)], categories=[dict(id=0, name='aircraft')],
                       annotations=[dict(id=1, image_id=1, category_id=0, bbox=[0, 0, 10, 10], area=100, iscrowd=0)])
            path = Path(tmp) / 'ann.json'
            path.write_text(json.dumps(ann), encoding='utf-8')
            d = dict(image_id=np.array([1, 1]), label=np.array([0, 0]),
                     box=np.array([[30, 30, 40, 40], [0, 0, 10, 10]], np.float32),
                     score=np.array([.9, .8], np.float32), probs=np.array([[.9, .9], [.8, .01]]))
            frozen = copy.deepcopy(d)
            original = coco_metrics(path, d, d['score'])
            ranked = coco_metrics(path, d, competition_score(d))
            self.assertGreater(ranked['AP'], original['AP'] + 40)
            for key in frozen:
                np.testing.assert_array_equal(d[key], frozen[key])

    def test_fit_uses_train_statistics_and_resume_replays_next_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('models', 'oof'):
                (root / name).mkdir()
            rng = np.random.default_rng(19)
            x = rng.normal(size=(60, 5)).astype(np.float32)
            x[40:] += 100  # held-out statistics must not enter normalizer
            y = rng.uniform(size=(60, 2)).astype(np.float32)
            train = np.arange(60) < 40
            args = argparse.Namespace(output=tmp, fit_seed=3, lr=.001, weight_decay=.001,
                                      fit_batch_size=13, epochs=2)
            score, eligible = np.ones(60, np.float32), np.ones(60, bool)
            fit_fold(x, y, score, eligible, train, ~train, args, 'continuous', torch.device('cpu'))
            args.epochs = 1
            fit_fold(x, y, score, eligible, train, ~train, args, 'resumed', torch.device('cpu'))
            args.epochs = 2
            fit_fold(x, y, score, eligible, train, ~train, args, 'resumed', torch.device('cpu'))
            a = torch.load(root / 'models/continuous.pth', weights_only=False)
            b = torch.load(root / 'models/resumed.pth', weights_only=False)
            np.testing.assert_allclose(a['mean'], x[train].mean(0))
            for key in a['model']:
                torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
            with np.load(root / 'oof/resumed.npz') as saved:
                np.testing.assert_array_equal(saved['rows'], np.flatnonzero(~train))

    def test_geometry_diagnostic_conditions_on_class_correctness(self):
        d = dict(score=np.full(40, .65), label=np.zeros(40, int))
        quality = np.linspace(.5, 1, 40)
        y = np.stack((quality, quality), -1)
        correct = np.arange(40) < 20
        prediction = np.concatenate((quality[:20], -quality[20:]))
        result = conditional_geometry(d, y, correct, prediction)
        self.assertAlmostEqual(result['spearman'], 1.)
        self.assertEqual(result['candidates'], 20)

    def test_protocol_saves_global_batch_and_does_not_edit_baseline_config(self):
        from engine.core.yaml_utils import load_config
        with tempfile.TemporaryDirectory() as tmp:
            source = Path('configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_102e_b28.yml')
            original = source.read_text(encoding='utf-8')
            args = argparse.Namespace(output=tmp, config=str(source), checkpoint=str(Path(tmp) / 'baseline.pth'),
                                      ann=str(Path(tmp) / 'val.json'), img_folder=tmp, groups=None,
                                      export_batch_size=16, workers=0, fit_batch_size=4096)
            prepare(args, world=2)
            saved = load_config(str(Path(tmp) / 'resolved_config.yml'))
            manifest = json.loads((Path(tmp) / 'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['train_dataloader']['total_batch_size'], 28)
            self.assertEqual(saved['val_dataloader']['total_batch_size'], 16)
            self.assertEqual(saved['val_dataloader']['dataset']['ann_file'], args.ann)
            self.assertEqual(manifest['probe']['topk'], 300)
            self.assertEqual(manifest['world_size'], 2)
            self.assertEqual(source.read_text(encoding='utf-8'), original)

    def test_report_reassembles_held_out_predictions_and_rejects_wrong_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('features', 'oof', 'ranks', 'summary'):
                (root / name).mkdir()
            ann = dict(images=[dict(id=i, width=100, height=80) for i in (1, 2)],
                       categories=[dict(id=0, name='aircraft'), dict(id=1, name='other')],
                       annotations=[dict(id=i, image_id=i, category_id=0, bbox=[0, 0, 10, 10],
                                         area=100, iscrowd=0) for i in (1, 2)])
            ann_path = root / 'val.json'
            ann_path.write_text(json.dumps(ann), encoding='utf-8')
            for image_id in (1, 2):
                write_npz(root / 'features' / f'{image_id}.npz',
                    image_id=np.full(2, image_id), query_id=np.array([0, 1]), label=np.array([0, 0]),
                    score=np.array([.9, .8], np.float32), probs=np.array([[.9, .05], [.8, .05]], np.float32),
                    box=np.array([[30, 30, 40, 40], [0, 0, 10, 10]], np.float32),
                    box_normalized=np.ones((2, 4), np.float32), orig_size=np.tile([100, 80], (2, 1)),
                    quality=np.zeros((2, 1), np.float32), fdr=np.zeros((2, 16), np.float32),
                    roi=np.zeros((2, 162), np.float16), border=np.zeros((2, 408), np.float16),
                    thumbnail=np.arange(1024, dtype=np.float32))
            (root / 'folds.json').write_text(json.dumps(dict(source='provided_scene_groups', n_groups=2,
                groups={'1': 'a', '2': 'b'}, repetitions=[{'1': 0, '2': 1}])), encoding='utf-8')
            for mode in MODES:
                for fold in (0, 1):
                    q = np.array([[.1, .1], [.9, .9]], np.float32) if mode == 'cls_fdr_local' else np.ones((2, 2), np.float32)
                    write_npz(root / 'oof' / f'{mode}_rep0_fold{fold}.npz',
                              rows=np.arange(2) + fold * 2, quality=q)
            args = argparse.Namespace(output=tmp, ann=str(ann_path), repeats=1)
            with contextlib.redirect_stdout(io.StringIO()):
                report(args)
            result = json.loads((root / 'results.json').read_text(encoding='utf-8'))
            self.assertGreater(result['local_comparison']['delta_vs_best_existing'], 40)
            self.assertTrue(result['local_comparison']['preliminary_candidate'])
            with np.load(root / 'ranks/cls_fdr_local_rep0.npz') as ranks:
                np.testing.assert_array_equal(ranks['image_id'], [1, 1, 2, 2])
                np.testing.assert_allclose(ranks['original_score'], [.9, .8, .9, .8])
                np.testing.assert_allclose(ranks['box'][0], [30, 30, 40, 40])
            write_npz(root / 'oof/score_rep0_fold0.npz', rows=np.array([2, 3]), quality=np.ones((2, 2)))
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'saved fold'):
                report(args)


if __name__ == '__main__':
    unittest.main()
