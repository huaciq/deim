"""Query groups, separation, rescoring and false-positive grouping on a synthetic scene; edge statistics."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from tools.diagnose_score_evidence import analyze, edge_statistics, group_queries, query_rows, rescored
from tests.test_sar_characteristics import write_split


def evidence_rows(entries):
    """entries: image_id, label, box, score, concentration; other evidence follows concentration."""
    rows = dict(image_id=np.array([e[0] for e in entries]), label=np.array([e[1] for e in entries]),
                box=np.array([e[2] for e in entries], float), score=np.array([e[3] for e in entries], float),
                concentration=np.array([e[4] for e in entries], float))
    rows.update(raw_score=rows['score'], lqe_delta=np.zeros(len(entries)), concentration_min=rows['concentration'],
                concentration_std=np.zeros(len(entries)), entropy=1 - rows['concentration'])
    return rows


SEEN_BOX, UNSEEN_BOX = [4., 16., 10., 10.], [30., 30., 12., 12.]
ENTRIES = [(1, 0, SEEN_BOX, .9, .9),                 # seen target, located
           (1, 0, [4.5, 16., 10., 10.], .05, .4),    # its duplicate: covered, not the top box, uncertain edges
           (1, 1, UNSEEN_BOX, .01, .85),             # unseen target, accurate low score, certain edges
           (1, 1, [46., 2., 10., 10.], .02, .3),     # background
           (1, 0, [46., 46., 8., 8.], .8, .35),      # false positive
           (2, 0, [46., 46., 8., 8.], .7, .35)]      # the same object repeated in another tile


class ScoreEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='deim_score_evidence_')
        folder = Path(self.temporary.name).resolve()
        self.data = folder / 'data'
        self.data.mkdir()
        write_split(self.data, 'trainval', [(1, [(0, 4, 16, 10, 10, 200, 7)])])
        test = write_split(self.data, 'test', [(3, [(0, 4, 16, 10, 10, 200, 7), (1, 30, 30, 12, 12, 200, 11)]), (4, [(1, 4, 40, 10, 10, 200, 13)])])
        self.dataset = test
        rows = evidence_rows(ENTRIES)
        self.path = folder / 'evidence.npz'
        np.savez_compressed(self.path, checkpoint='final.pth', **rows)
        self.rows, self.out = rows, folder / 'out'

    def tearDown(self):
        self.temporary.cleanup()

    def test_groups_each_query(self):
        seen = {1: True, 2: False, 3: False}
        masks, outrank = group_queries(self.dataset, self.rows, seen)
        self.assertEqual({k: np.nonzero(v)[0].tolist() for k, v in masks.items() if v.any()},
                         dict(seen_located=[0], duplicate=[1], unseen_low=[2], background=[3], false_positive=[4, 5]))
        self.assertEqual(outrank, [2])  # the background query and the false positive outrank the target

    def test_rescoring_lifts_certain_boxes(self):
        lifted = rescored(self.rows, 'concentration', 2.)
        self.assertGreater(lifted[2], self.rows['score'][2])
        self.assertLess(lifted[3], self.rows['score'][3])
        self.assertTrue((rescored(self.rows, 'entropy', 2.) > self.rows['score'])[2])

    def test_analyze_reports_separation_groups_and_unlabelled(self):
        summary = analyze(self.path, self.data, self.out, unlabelled=[1])
        pairs = {(r['positive'], r['negative']): r for r in summary['separation']}
        self.assertEqual(pairs[('unseen_low', 'background')]['concentration'], 1.)
        self.assertEqual(pairs[('unseen_low', 'duplicate')]['concentration'], 1.)
        fp = summary['false_positives']
        self.assertEqual(fp['boxes'], 2)
        self.assertEqual(sum(g['boxes'] for g in fp['groups']), 2)
        self.assertEqual(summary['unlabelled']['groups'], [1])
        self.assertGreaterEqual(summary['unlabelled']['boxes'], 1)
        self.assertGreater(summary['unlabelled']['AP50'], summary['rescoring']['base']['all'] - 1e-6)
        for name in ('report.md', 'summary.json', 'false_positive_groups.csv', 'false_positive_groups.png'):
            self.assertTrue((self.out / name).is_file(), name)
        self.assertFalse((self.out / '_gt.json').exists())
        with self.assertRaises(FileExistsError):
            analyze(self.path, self.data, self.out)

    def test_edge_statistics_separate_sharp_from_flat(self):
        sharp = torch.zeros(1, 1, 4 * 33)
        sharp[0, 0, ::33] = 20.
        out = edge_statistics(torch.cat((sharp, torch.zeros(1, 1, 4 * 33)), 1))
        self.assertAlmostEqual(float(out['concentration'][0, 0]), .25, places=3)  # one bin holds all: top-4 mean is 1/4
        self.assertAlmostEqual(float(out['concentration'][0, 1]), 1 / 33, places=3)
        self.assertLess(float(out['entropy'][0, 0]), .01)
        self.assertAlmostEqual(float(out['entropy'][0, 1]), 1., places=3)

    def test_query_rows_use_the_top_class_and_canvas_pixels(self):
        raw = torch.tensor([[[0., 2., -1.]]])
        calibrated = raw + 1.
        boxes = torch.tensor([[[.5, .5, .2, .1]]])
        row = query_rows(raw, calibrated, torch.zeros(1, 1, 4 * 33), boxes, torch.tensor([[1500, 1500]]))
        self.assertEqual(row['label'].tolist(), [[1]])
        self.assertAlmostEqual(float(row['score'][0, 0]), float(torch.tensor(3.).sigmoid()), places=5)
        self.assertAlmostEqual(float(row['raw_score'][0, 0]), float(torch.tensor(2.).sigmoid()), places=5)
        self.assertAlmostEqual(float(row['lqe_delta'][0, 0]), 1., places=5)
        self.assertEqual(row['box'][0, 0].tolist(), [600., 675., 300., 150.])


if __name__ == '__main__':
    unittest.main()
