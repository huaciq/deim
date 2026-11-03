"""Functional check of the CPU split-overlap count on synthetic tiles."""

from pathlib import Path
import tempfile
import unittest

from tools.check_split_overlap import analyze
from tests.test_sar_characteristics import write_split


class SplitOverlapTests(unittest.TestCase):
    def test_counts_test_targets_seen_only_in_val(self):
        with tempfile.TemporaryDirectory(prefix='deim_split_overlap_') as folder:
            data = Path(folder) / 'data'
            data.mkdir()
            # Pattern 7 is in train and test, pattern 9 in val and test, pattern 11 only in test.
            write_split(data, 'train', [(1, [(0, 4, 4, 10, 10, 200, 7)])])
            write_split(data, 'val', [(2, [(1, 30, 30, 12, 12, 200, 9)])])
            write_split(data, 'test', [(3, [(0, 4, 4, 10, 10, 200, 7), (1, 30, 30, 12, 12, 200, 9),
                                            (1, 4, 40, 10, 10, 200, 11)])])
            summary = analyze(data, Path(folder) / 'out')
            row = next(r for r in summary['pairs'] if r['reference'] == 'train+val' and r['threshold'] == .9)
            self.assertEqual((row['gt'], row['matched'], row['val_only']), (3, 2, 1))
            classes = {r['category']: r for r in summary['test_by_class']}
            self.assertEqual((classes['Boeing787']['in_train'], classes['Boeing787']['val_only']), (0, 1))
            self.assertTrue((Path(folder) / 'out/report.md').is_file())


if __name__ == '__main__':
    unittest.main()
