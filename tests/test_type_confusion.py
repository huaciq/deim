"""Functional checks for the CPU type-confusion follow-up on a paired diagnosis."""

import gzip
import json
from pathlib import Path
import tempfile
import unittest

from tools.analyze_type_confusion import analyze, size_table
from tools.compare_rem_102e import prepare
import tests.test_rem_102e_diagnosis as diagnosis_tests
from tests.test_rem_102e_diagnosis import dataset, predictions


class TypeConfusionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='deim_type_confusion_')
        self.folder = Path(self.temporary.name).resolve()

    def tearDown(self):
        self.temporary.cleanup()

    def test_groups_by_source_size_and_reports_confusions(self):
        helper = diagnosis_tests.Rem102DiagnosisTests()
        helper.folder = self.folder
        baseline, _, _ = helper.source('baseline')
        rem, _, _ = helper.source('rem')
        data = dataset()
        data['images'][1].update(width=512, height=512)  # the empty image forms its own size group
        (self.folder / 'data/COCO').mkdir(parents=True)
        (self.folder / 'data/COCO/test.json').write_text(json.dumps(data), encoding='utf-8')
        out = self.folder / 'diag'
        prepare(baseline, rem, self.folder / 'data', out, split='test')
        for role, detections in predictions().items():
            with gzip.open(out / role / 'predictions.json.gz', 'wt', encoding='utf-8') as stream:
                json.dump(detections, stream)
        summary = analyze(out)
        groups = {r['image_size']: r for r in summary['by_image_size']}
        self.assertEqual(sorted(groups), ['256x256', '512x512'])
        self.assertEqual((groups['256x256']['gt'], groups['512x512']['gt']), (4, 0))
        self.assertAlmostEqual(groups['256x256']['rem_correct_given_located'], 2 / 3)
        top = summary['top_confusions']['rem']
        self.assertEqual((top[0]['gt'], top[0]['predicted'], top[0]['count']), ('A', 'B', 1))
        self.assertTrue((out / 'type_confusion/report.md').is_file())
        self.assertFalse((out / 'type_confusion/subset_gt.json').exists())

    def test_box_size_in_source_and_input_pixels(self):
        data = dataset()
        data['images'][0].update(width=1280, height=1280)
        rows = {r['category']: r for r in size_table(data)}
        self.assertEqual(rows['A']['source_p50'], 40.)
        self.assertEqual(rows['A']['input_p50'], 20.)
        self.assertEqual(rows['A']['source_iqr_over_median'], 0.)


if __name__ == '__main__':
    unittest.main()
