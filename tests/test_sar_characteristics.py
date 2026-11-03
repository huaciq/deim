"""Functional checks for the CPU SAR imaging-characteristics analysis."""

import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from tools.analyze_sar_characteristics import analyze, orientation


def scene(seed, targets, background=20):
    """64x64 speckle-like clutter with bright rectangular targets (x, y, w, h, level, pattern seed)."""
    rng = np.random.default_rng(seed)
    values = rng.gamma(4, background / 4, (64, 64))
    for x, y, w, h, level, pattern in targets:
        values[y:y + h, x:x + w] = level * (0.5 + np.random.default_rng(pattern).random((h, w)))
    return Image.fromarray(np.clip(values, 0, 255).astype(np.uint8))


def write_split(root, split, images):
    data = dict(images=[], annotations=[], categories=[dict(id=0, name='Boeing737'), dict(id=1, name='Boeing787')])
    for image_id, (seed, targets) in enumerate(images, 1):
        name = f'{split}_{image_id}.png'
        scene(seed, [t[1:] for t in targets]).save(root / name)
        data['images'].append(dict(id=image_id, file_name=name, width=64, height=64))
        for category, x, y, w, h, _, _ in targets:
            data['annotations'].append(dict(id=len(data['annotations']) + 1, image_id=image_id, category_id=category,
                                            bbox=[x, y, w, h], area=w * h, iscrowd=0))
    (root / 'COCO').mkdir(exist_ok=True)
    (root / f'COCO/{split}.json').write_text(json.dumps(data), encoding='utf-8')
    return data


class SarCharacteristicsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='deim_sar_characteristics_')
        self.folder = Path(self.temporary.name).resolve()

    def tearDown(self):
        self.temporary.cleanup()

    def test_size_ratio_tcr_duplicates_and_outcomes(self):
        data = self.folder / 'data'
        data.mkdir()
        # 787 is twice the size of 737 where they co-occur; pattern 7 repeats across images (overlapping tiles).
        write_split(data, 'train', [(1, [(0, 4, 4, 10, 10, 200, 7), (1, 30, 30, 20, 20, 150, 8)]),
                                    (2, [(0, 40, 4, 10, 10, 180, 9)])])
        write_split(data, 'val', [(3, [(0, 4, 40, 10, 10, 200, 10)])])
        test = write_split(data, 'test', [(4, [(0, 4, 4, 10, 10, 200, 7), (1, 30, 30, 20, 20, 60, 11)]),
                                          (5, [(0, 20, 20, 10, 10, 200, 7)])])
        diag = self.folder / 'diag'
        for role in ('baseline', 'rem'):
            (diag / role).mkdir(parents=True)
            detections = [dict(image_id=1, category_id=0, bbox=[4, 4, 10, 10], score=.9),
                          dict(image_id=1, category_id=0, bbox=[30, 30, 20, 20], score=.8),  # 787 typed 737
                          dict(image_id=2, category_id=0, bbox=[20, 20, 10, 10], score=.9)]
            with gzip.open(diag / role / 'predictions.json.gz', 'wt', encoding='utf-8') as stream:
                json.dump(detections, stream)
        (diag / 'manifest.json').write_text(json.dumps(dict(split='test')), encoding='utf-8')
        (diag / 'test.json').write_text(json.dumps(test), encoding='utf-8')
        # The diagnosis archives the split annotation with image ids 1..n of that split.
        summary = analyze(data, self.folder / 'out', [diag])
        ratio = {(r['split'], r['a'], r['b']): r for r in summary['size_ratio']}
        self.assertEqual(ratio[('train', 'A220', 'ARJ21')]['images'], 0)
        self.assertEqual((ratio[('test', 'Boeing787', 'Boeing737')]['images'], ratio[('test', 'Boeing787', 'Boeing737')]['ratio'][1]), (1, 2.))
        tcr = {(r['split'], r['category']): r['tcr_db'][1] for r in summary['tcr']}
        self.assertGreater(tcr[('test', 'Boeing737')], tcr[('test', 'Boeing787')])  # 787 is the weak scatterer in test
        outcome = {(r['category'], r['role'], r['outcome']): r['gt']
                   for r in summary['diagnosed_tables']['test']['tcr_by_outcome']}
        self.assertEqual((outcome[('Boeing737', 'baseline', 'correct')], outcome[('Boeing787', 'baseline', 'wrong')]), (2, 1))
        test_dup = {r['threshold']: r for r in summary['duplicates'] if r['split'] == 'test'}
        self.assertEqual((test_dup[.9]['gt'], test_dup[.9]['unique']), (3, 2))
        leak = {r['threshold']: r['matched_in_train'] for r in summary['train_leak'] if r['split'] == 'test'}
        self.assertEqual(leak[.9], 2)
        errors = {r['category']: r for r in summary['unique_errors']}
        self.assertEqual((errors['Boeing787']['wrong'], errors['Boeing737']['wrong']), (1, 0))
        for name in ('report.md', 'summary.json', 'targets.csv', 'images.csv'):
            self.assertTrue((self.folder / 'out' / name).is_file())
        with self.assertRaises(FileExistsError):
            analyze(data, self.folder / 'out', [diag])

    def test_orientation_of_elongated_scatterers(self):
        patch = np.zeros((20, 20), np.float32)
        patch[9:11, 2:18] = 100  # horizontal bar
        angle, ratio = orientation(patch)
        self.assertAlmostEqual(angle % 180, 0, places=3)
        self.assertGreater(ratio, 3)
        angle, _ = orientation(patch.T)
        self.assertAlmostEqual(angle, 90, places=3)


if __name__ == '__main__':
    unittest.main()
