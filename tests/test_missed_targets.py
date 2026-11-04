"""Functional check of the missed-target statuses, groups and false positives on a synthetic scene."""

import gzip
import json
from pathlib import Path
import tempfile
import unittest

from tools.diagnose_missed_targets import analyze, status_of
from tests.test_sar_characteristics import write_split


class MissedTargetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='deim_missed_targets_')
        folder = Path(self.temporary.name).resolve()
        data = folder / 'data'
        data.mkdir()
        write_split(data, 'trainval', [(1, [(0, 4, 16, 10, 10, 200, 7)])])
        write_split(data, 'test', [(3, [(0, 4, 16, 10, 10, 200, 7), (1, 30, 30, 12, 12, 200, 11), (1, 0, 50, 10, 10, 200, 13)]),
                                   (4, [(1, 40, 8, 10, 10, 200, 17), (1, 8, 40, 10, 10, 200, 19)])])
        predictions = [dict(image_id=1, category_id=0, bbox=[4., 16., 10., 10.], score=.9),    # seen, located
                       dict(image_id=1, category_id=0, bbox=[30., 30., 12., 12.], score=.1),   # low score, wrong class is fine
                       dict(image_id=2, category_id=1, bbox=[45., 8., 12., 10.], score=.8),    # poor box on (40, 8), IoU 0.29
                       dict(image_id=2, category_id=1, bbox=[50., 50., 8., 8.], score=.7)]     # background false positive
        path = folder / 'predictions.json.gz'
        with gzip.open(path, 'wt', encoding='utf-8') as stream:
            json.dump(predictions, stream)
        self.out = folder / 'out'
        self.summary = analyze(path, data, self.out)

    def tearDown(self):
        self.temporary.cleanup()

    def test_status_of_each_case(self):
        candidates = [dict(bbox=[0., 0., 10., 10.], score=.9), dict(bbox=[20., 0., 10., 10.], score=.1), dict(bbox=[46., 0., 10., 10.], score=.9)]
        self.assertEqual(status_of([0, 0, 10, 10], candidates)[0], 'located')
        self.assertEqual(status_of([20, 0, 10, 10], candidates)[0], 'low_score')
        self.assertEqual(status_of([40, 0, 10, 10], candidates)[0], 'poor_box')
        self.assertEqual(status_of([70, 70, 10, 10], candidates), ('no_candidate', 0., None))
        self.assertEqual(status_of([0, 0, 10, 10], []), ('no_candidate', 0., None))

    def test_counts_statuses_by_subset(self):
        status = {r['group']: r for r in self.summary['status']}
        self.assertEqual((status['见过']['gt'], status['见过']['located']), (1, 1))
        self.assertEqual({k: status['没见过'][k] for k in ('gt', 'located', 'low_score', 'poor_box', 'no_candidate')},
                         dict(gt=4, located=0, low_score=1, poor_box=1, no_candidate=2))
        self.assertEqual(status['没见过 Boeing787']['gt'], 4)

    def test_groups_edge_and_image_kind(self):
        edge = {r['group']: r for r in self.summary['edge']}
        self.assertEqual(edge['切片边缘']['unseen_gt'], 1)  # the box at x=0
        kinds = {r['group']: r for r in self.summary['image_kind']}
        self.assertEqual((kinds['图中全部没见过']['unseen_gt'], kinds['图中有见过的目标']['unseen_gt']), (2, 2))
        self.assertEqual(self.summary['images'], dict(total=2, all_unseen=1))

    def test_false_positive_and_concentration(self):
        fp = self.summary['false_positives']
        self.assertEqual((fp['total'], fp['by_class'], fp['in_all_unseen_images']), (1, {'Boeing787': 1}, 1))
        self.assertEqual(self.summary['unseen_misses_by_image']['total'], 4)

    def test_writes_outputs_and_refuses_existing_directory(self):
        for name in ('report.md', 'summary.json', 'targets.csv', 'sheets/unseen_missed_Boeing787.png', 'sheets/false_positives.png'):
            self.assertTrue((self.out / name).is_file(), name)
        with self.assertRaises(FileExistsError):
            analyze(self.out.parent / 'predictions.json.gz', self.out.parent / 'data', self.out)


if __name__ == '__main__':
    unittest.main()
