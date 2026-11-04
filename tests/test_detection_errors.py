"""Functional check of the AP50 decomposition and seen/unseen split on a synthetic scene."""

import gzip
import json
from pathlib import Path
import tempfile
import unittest

from tools.diagnose_detection_errors import analyze
from tests.test_sar_characteristics import write_split


SEEN_BOX = [4., 4., 10., 10.]
UNSEEN_BOX = [30., 30., 12., 12.]
PREDICTIONS = [dict(image_id=1, category_id=0, bbox=[45., 45., 8., 8.], score=.95),  # background false positive
               dict(image_id=1, category_id=0, bbox=SEEN_BOX, score=.9),             # seen, correct type
               dict(image_id=1, category_id=1, bbox=SEEN_BOX, score=.05),            # lower copy of the same box
               dict(image_id=1, category_id=0, bbox=UNSEEN_BOX, score=.8)]           # unseen, wrong type


class DetectionErrorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='deim_detection_errors_')
        folder = Path(self.temporary.name).resolve()
        data = folder / 'data'
        data.mkdir()
        write_split(data, 'trainval', [(1, [(0, 4, 4, 10, 10, 200, 7)])])
        write_split(data, 'test', [(3, [(0, 4, 4, 10, 10, 200, 7), (1, 30, 30, 12, 12, 200, 11)])])
        predictions = folder / 'predictions.json.gz'
        with gzip.open(predictions, 'wt', encoding='utf-8') as stream:
            json.dump(PREDICTIONS, stream)
        self.out = folder / 'out'
        self.summary = analyze(predictions, data, self.out)

    def tearDown(self):
        self.temporary.cleanup()

    def test_splits_seen_and_unseen(self):
        self.assertEqual((self.summary['gt'], self.summary['seen'], self.summary['unseen']), (2, 1, 1))

    def test_oracles_separate_type_confusion_from_background(self):
        everything = self.summary['subsets']['all']
        self.assertAlmostEqual(everything['official']['AP50'], 25, places=1)
        self.assertAlmostEqual(everything['type_oracle']['AP50'], 75, places=1)
        self.assertAlmostEqual(everything['both_oracles']['AP50'], 100, places=1)
        self.assertAlmostEqual(everything['class_agnostic']['AP50'], 200 / 3, places=1)

    def test_subset_ignores_the_other_targets(self):
        seen, unseen = self.summary['subsets']['seen'], self.summary['subsets']['unseen']
        self.assertAlmostEqual(seen['official']['classes']['Boeing737'], 50, places=1)
        self.assertIsNone(seen['official']['classes']['Boeing787'])
        self.assertIsNone(unseen['official']['classes']['Boeing737'])
        self.assertAlmostEqual(unseen['official']['classes']['Boeing787'], 0, places=1)
        self.assertAlmostEqual(unseen['type_oracle']['AP50'], 100, places=1)

    def test_rates_show_located_but_wrong_type(self):
        rates = self.summary['subsets']['unseen']['rates']
        self.assertEqual((rates['gt'], rates['located_rate'], rates['correct_given_located']), (1, 1., 0.))
        self.assertEqual(self.summary['top_confusions']['unseen'][0]['predicted'], 'Boeing737')

    def test_writes_report_and_refuses_existing_output(self):
        for name in ('report.md', 'summary.json', 'targets.csv'):
            self.assertTrue((self.out / name).is_file())
        self.assertFalse((self.out / '_gt.json').exists())
        with self.assertRaises(FileExistsError):
            analyze(self.out.parent / 'predictions.json.gz', self.out.parent / 'data', self.out)


if __name__ == '__main__':
    unittest.main()
