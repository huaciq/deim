"""Functional checks for cross-server config adaptation and paired diagnosis."""

import gzip
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import yaml

from tools.compare_rem_102e import (analyze, class_agnostic, evaluate_coco, package, pair_detections,
                                   prepare, run_exports, select_checkpoint, summarize_training)


def dataset():
    return dict(info={}, images=[dict(id=1, width=256, height=256, file_name='one.png'),
                                 dict(id=2, width=256, height=256, file_name='empty.png')],
                categories=[dict(id=0, name='A'), dict(id=1, name='B')],
                annotations=[dict(id=i, image_id=1, category_id=c, bbox=b, area=b[2]*b[3], iscrowd=0)
                             for i, c, b in ((1, 0, [10, 10, 40, 40]), (2, 1, [100, 10, 40, 40]),
                                             (3, 0, [200, 10, 40, 40]), (4, 1, [10, 100, 20, 20]))])


def detection(category, box, score=.9, image=1):
    return dict(image_id=image, category_id=category, bbox=box, score=score)


def predictions():
    return dict(baseline=[detection(0, [10, 10, 40, 40]), detection(0, [200, 10, 40, 40]),
                          detection(1, [10, 100, 20, 20]), detection(0, [0, 0, 10, 10], image=2)],
                rem=[detection(0, [10, 10, 50, 40], .8), detection(1, [100, 10, 40, 40]),
                     detection(1, [200, 10, 40, 40]), detection(1, [10, 100, 20, 20], .2),
                     detection(0, [0, 0, 10, 10], image=2)])


class Rem102DiagnosisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='deim_rem102_diagnosis_')
        self.folder = Path(self.temporary.name).resolve()
        self.assertEqual(self.folder.parent, Path(tempfile.gettempdir()).resolve())
        self.assertTrue(self.folder.name.startswith('deim_rem102_diagnosis_'))

    def tearDown(self):
        # The resolved absolute cleanup target was checked within the temp root.
        self.temporary.cleanup()

    def source(self, role):
        run = self.folder / f'source_{role}'
        run.mkdir()
        config = dict(num_classes=2, epoches=102, HGNetv2=dict(pretrained=True),
                      HybridEncoder=dict(use_rem=role == 'rem'), output_dir='/root/original',
                      train_dataloader=dict(collate_fn=dict(stop_epoch=90), dataset=dict(img_folder='/root/train')),
                      val_dataloader=dict(total_batch_size=64, dataset=dict(img_folder='/root/val', ann_file='/root/val.json')))
        (run / 'resolved_config.yml').write_text(yaml.safe_dump(config), encoding='utf-8')
        rows = [dict(epoch=e, test_coco_eval_bbox=[ap, .88, .73, .5, .64, .58])
                for e, ap in ((49, .60), (90, .645), (101, .63))]
        (run / 'log.txt').write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
        (run / 'best_stg2.pth').write_bytes(b'test checkpoint path; not loaded')
        return run, config, rows

    def prepared(self):
        baseline, _, _ = self.source('baseline')
        rem, _, _ = self.source('rem')
        data = self.folder / 'old_server_data'
        (data / 'COCO').mkdir(parents=True)
        (data / 'COCO/val.json').write_text(json.dumps(dataset()), encoding='utf-8')
        out = self.folder / 'output'
        manifest = prepare(baseline, rem, data, out)
        return out, manifest

    def test_best_validation_ap_selects_stage_two_not_last_epoch_or_available_stage_one(self):
        run, config, rows = self.source('baseline')
        (run / 'best_stg1.pth').write_bytes(b'other stage')
        path, best = select_checkpoint(run, config, rows)
        self.assertEqual(path.name, 'best_stg2.pth')
        self.assertEqual(best['epoch'], 90)
        (run / 'best_stg2.pth').unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'epoch 90'):
            select_checkpoint(run, config, rows)

    def test_prepare_uses_old_server_annotations_and_preserves_original_configuration(self):
        out, manifest = self.prepared()
        self.assertEqual(json.loads((out / 'val.json').read_text()), dataset())
        self.assertEqual(manifest['export']['gpus'], [0, 1])
        for role in ('baseline', 'rem'):
            source = Path(manifest['runs'][role]['source_run'])
            original_bytes = (source / 'resolved_config.yml').read_bytes()
            self.assertEqual((out / role / 'resolved_config.yml').read_bytes(), original_bytes)
            original = yaml.safe_load(original_bytes)
            runtime = yaml.safe_load((out / role / 'inference_config.yml').read_text())
            self.assertEqual(runtime['HybridEncoder'], original['HybridEncoder'])
            self.assertEqual(runtime['train_dataloader'], original['train_dataloader'])
            self.assertEqual(runtime['val_dataloader']['total_batch_size'], 16)
            self.assertEqual(runtime['val_dataloader']['dataset']['img_folder'], manifest['data_root'])
            self.assertEqual(runtime['val_dataloader']['dataset']['ann_file'], manifest['annotation_file'])
            self.assertFalse(runtime['HGNetv2']['pretrained'])
            self.assertEqual(yaml.safe_load((source / 'resolved_config.yml').read_text()), original)

    def test_pairs_distinguish_gains_class_losses_geometry_losses_and_low_score_candidates(self):
        result = pair_detections(dataset(), predictions())
        groups = {(r['iou_threshold'], r['category'], r['size']): r for r in result['groups']}
        g75 = groups[(.75, 'all', 'all')]
        self.assertEqual([g75[k] for k in ('gt', 'both', 'gained', 'lost', 'neither')], [4, 1, 1, 2, 0])
        g90 = groups[(.90, 'all', 'all')]
        self.assertEqual([g90[k] for k in ('both', 'gained', 'lost', 'neither')], [0, 1, 3, 0])
        self.assertEqual(groups[(.75, 'all', 'medium')]['net_gained'], 0)
        candidate = {r['gt_id']: r for r in result['candidates']}
        self.assertEqual(candidate[3]['rem_best_any_class_iou'], 1.)
        self.assertEqual(candidate[3]['rem_best_same_class_iou'], 0.)
        self.assertEqual(candidate[4]['rem_best_same_class_iou'], 1.)
        self.assertEqual(candidate[4]['rem_max_correct_score_iou75'], .2)
        common = result['common_geometry']
        self.assertEqual(len(common), 1)
        self.assertEqual(common[0]['gt_id'], 1)
        self.assertAlmostEqual(common[0]['rem_width_error_pct'], 25.)
        self.assertAlmostEqual(common[0]['rem_right640'], 25.)
        self.assertAlmostEqual(common[0]['delta_iou'], -.2)

    def test_coco_metrics_keep_full_scores_and_mark_absent_area_as_unavailable(self):
        data = dataset()
        data['annotations'] = [data['annotations'][0]]
        ann = self.folder / 'val.json'
        ann.write_text(json.dumps(data), encoding='utf-8')
        result = evaluate_coco(ann, [detection(0, [10, 10, 40, 40], .01)])
        self.assertAlmostEqual(result['metrics']['AP'], 100.)
        self.assertAlmostEqual(result['metrics']['AP75'], 100.)
        self.assertAlmostEqual(result['metrics']['APm'], 100.)
        self.assertIsNone(result['metrics']['APs'])
        self.assertIsNone(result['metrics']['APl'])

    def test_analysis_and_bundle_produce_paired_reports_without_copying_checkpoints(self):
        out, _ = self.prepared()
        for role, detections in predictions().items():
            with gzip.open(out / role / 'predictions.json.gz', 'wt', encoding='utf-8') as stream:
                json.dump(detections, stream)
        result = analyze(out)
        self.assertEqual(result['errors_at_score03']['baseline']['gt'], 4)
        self.assertEqual(result['errors_at_score03']['baseline']['fp_types']['background']['count'], 1)
        self.assertEqual(result['errors_at_score03']['rem']['fp_types']['classification']['count'], 1)
        self.assertIsNone(result['delta_metrics']['APl'])
        for name in ('summary.json', 'report.md', 'gt_pairs.csv', 'paired_groups.csv',
                     'common_gt_geometry.csv', 'candidate_quality.csv', 'class_metrics.csv', 'epoch_metrics.csv'):
            self.assertTrue((out / 'report' / name).is_file(), name)
        package(out)
        with tarfile.open(out / 'diagnosis_bundle.tar.gz') as archive:
            names = archive.getnames()
        self.assertIn('report/summary.json', names)
        self.assertIn('baseline/predictions.json.gz', names)
        self.assertIn('rem/predictions.json.gz', names)
        self.assertFalse(any(name.endswith('.pth') for name in names))

    def test_resume_skips_finished_export_and_launches_remaining_role_on_two_gpus(self):
        out, manifest = self.prepared()
        (out / 'baseline/export.done').write_text('completed\n')
        with patch('tools.compare_rem_102e.subprocess.run') as execute, \
             patch('tools.compare_rem_102e.analyze') as analysis, \
             patch('tools.compare_rem_102e.package') as packaging:
            run_exports(out, manifest)
        execute.assert_called_once()
        command = execute.call_args.args[0]
        self.assertIn('--nproc_per_node=2', command)
        self.assertIn(manifest['runs']['rem']['checkpoint'], command)
        self.assertEqual(execute.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '0,1')
        self.assertTrue((out / 'rem/export.done').is_file())
        self.assertTrue((out / 'DONE').is_file())
        analysis.assert_called_once_with(out)
        packaging.assert_called_once_with(out)

    def test_test_split_reports_class_agnostic_gap_and_type_confusion(self):
        baseline, _, _ = self.source('baseline')
        rem, _, _ = self.source('rem')
        data = self.folder / 'old_server_data'
        (data / 'COCO').mkdir(parents=True)
        (data / 'COCO/test.json').write_text(json.dumps(dataset()), encoding='utf-8')
        out = self.folder / 'output_test'
        manifest = prepare(baseline, rem, data, out, split='test')
        self.assertEqual((manifest['split'], Path(manifest['annotation_file']).name), ('test', 'test.json'))
        self.assertFalse((out / 'val.json').exists())
        for role, detections in predictions().items():
            with gzip.open(out / role / 'predictions.json.gz', 'wt', encoding='utf-8') as stream:
                json.dump(detections, stream)
        result = analyze(out)
        self.assertEqual(result['split'], 'test')
        # REM puts GT 3 (class A) in class B: class-aware AP drops, class-agnostic AP does not.
        self.assertGreater(result['classification_gap']['rem']['AP'], 0)
        self.assertGreaterEqual(result['class_agnostic']['rem']['AP'], result['evaluations']['rem']['metrics']['AP'])
        rows = {r['gt_category']: r for r in result['confusion_at_score03']['rem']}
        self.assertEqual((rows['A']['gt'], rows['A']['located'], rows['A']['correct'], rows['A']['pred_B']), (2, 2, 1, 1))
        self.assertEqual((rows['B']['gt'], rows['B']['missed'], rows['B']['correct']), (2, 1, 1))
        self.assertIn('类别无关 AP', (out / 'report/report.md').read_text(encoding='utf-8'))
        package(out)
        with tarfile.open(out / 'diagnosis_bundle.tar.gz') as archive:
            self.assertIn('test.json', archive.getnames())

    def test_class_agnostic_keeps_one_copy_of_each_query_box(self):
        data, detections = class_agnostic(dataset(), [detection(0, [10, 10, 40, 40], .2),
                                                      detection(1, [10, 10, 40, 40], .7),
                                                      detection(1, [100, 10, 40, 40], .5)])
        self.assertEqual([c['id'] for c in data['categories']], [0])
        self.assertEqual({a['category_id'] for a in data['annotations']}, {0})
        self.assertEqual(sorted(d['score'] for d in detections), [.5, .7])
        self.assertEqual({d['category_id'] for d in detections}, {0})

    def test_training_summary_separates_pre_and_post_stop_periods(self):
        _, config, rows = self.source('baseline')
        summary = summarize_training(rows, config)
        self.assertEqual(summary['best_epoch'], 90)
        self.assertIsNone(summary['before_stop_last12_means']['AP'])
        self.assertAlmostEqual(summary['after_stop_means']['AP'], 63.75)
        self.assertAlmostEqual(summary['last12_means']['AP'], 62.5)


if __name__ == '__main__':
    unittest.main()
