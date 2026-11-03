"""Functional checks for the CPU confusion-cue follow-up on a paired diagnosis."""

import gzip
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from tools.analyze_confusion_cues import CROP_WINDOW, analyze, auc
from tools.compare_rem_102e import prepare
import tests.test_rem_102e_diagnosis as diagnosis_tests
from tests.test_rem_102e_diagnosis import dataset, detection


def predictions():
    # GT 3 (class A) is typed B with A as the runner-up of the same box.
    baseline = [detection(0, [10, 10, 40, 40]), detection(1, [100, 10, 40, 40]),
                detection(1, [200, 10, 40, 40]), detection(0, [200, 10, 40, 40], .5),
                detection(1, [10, 100, 20, 20])]
    return dict(baseline=baseline, rem=baseline[:2] + [detection(0, [200, 10, 40, 40])] + baseline[4:])


class ConfusionCueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='deim_confusion_cues_')
        self.folder = Path(self.temporary.name).resolve()

    def tearDown(self):
        self.temporary.cleanup()

    def diagnosis(self):
        helper = diagnosis_tests.Rem102DiagnosisTests()
        helper.folder = self.folder
        baseline, _, _ = helper.source('baseline')
        rem, _, _ = helper.source('rem')
        data = self.folder / 'data'
        (data / 'COCO').mkdir(parents=True)
        (data / 'COCO/test.json').write_text(json.dumps(dataset()), encoding='utf-8')
        for name in ('one.png', 'empty.png'):
            Image.new('L', (256, 256), 80).save(data / name)
        out = self.folder / 'diag'
        prepare(baseline, rem, data, out, split='test')
        for role, detections in predictions().items():
            with gzip.open(out / role / 'predictions.json.gz', 'wt', encoding='utf-8') as stream:
                json.dump(detections, stream)
        return out

    def test_ranking_geometry_clustering_and_crops(self):
        out = self.diagnosis()
        summary = analyze(out)
        self.assertEqual(summary['focus_pairs'], [['A', 'B']])
        ranking = {r['category']: r for r in summary['ranking']['baseline']}
        self.assertEqual((ranking['A']['located'], ranking['A']['wrong']), (2, 1))
        self.assertEqual((ranking['A']['top1'], ranking['A']['top2']), (.5, 1.))
        self.assertEqual(ranking['A']['wrong_true_second'], 1.)
        self.assertAlmostEqual(ranking['A']['wrong_median_margin'], .4)
        self.assertEqual({r['category']: r['wrong'] for r in summary['ranking']['rem']}, dict(A=0, B=0))
        geometry = summary['geometry'][0]
        self.assertEqual(geometry['size_auc'], .75)
        self.assertEqual(geometry['b_correct_relative_gt'], (1. + .5) / 2)
        self.assertEqual((geometry['a_as_b_n'], geometry['a_as_b_relative_gt']), (1, 1.))
        clustering = {r['category']: r for r in summary['clustering']['baseline']}
        self.assertEqual((clustering['A']['rate'], clustering['A']['dispersion']), (.5, 0.))
        self.assertEqual(clustering['A']['errors_in_mostly_wrong_images'], 0.)
        with Image.open(out / 'confusion_cues/crops/A_as_B.png') as sheet:
            self.assertEqual(sheet.width, 8 * CROP_WINDOW)
        self.assertTrue((out / 'confusion_cues/report.md').is_file())

    def test_auc_counts_ties_half(self):
        self.assertEqual(auc([2, 3], [1, 2]), (1 + .5 + 1 + 1) / 4)
        self.assertEqual(auc([1], [2]), 0.)
        self.assertIsNone(auc([], [1]))


if __name__ == '__main__':
    unittest.main()
