import unittest

from tools.sar_analysis_core import classify_image, error_analysis, object_size


def gt(i, category, box):
    return dict(id=i, image_id=1, category_id=category, bbox=box)


def dt(category, box, score=.9):
    return dict(image_id=1, category_id=category, bbox=box, score=score)


class AnalysisAccountingTests(unittest.TestCase):
    def test_exclusive_fp_types_and_separate_missing_gt(self):
        targets = [gt(1, 0, [0, 0, 10, 10]), gt(2, 1, [30, 0, 10, 10]),
                   gt(3, 2, [60, 0, 10, 10]), gt(4, 3, [90, 0, 10, 10])]
        predictions = [dt(0, [0, 0, 10, 10], .99), dt(0, [0, 0, 10, 10], .98),
                       dt(5, [30, 0, 10, 10]), dt(2, [65, 0, 10, 10]),
                       dt(6, [95, 0, 10, 10]), dt(0, [200, 0, 10, 10])]
        dataset = dict(images=[dict(id=1, width=640, height=640)], annotations=targets)
        summary, records, misses, _ = error_analysis(dataset, predictions)
        self.assertEqual([r['kind'] for r in records], ['tp', 'duplicate', 'classification',
                         'localization', 'classification_localization', 'background'])
        self.assertEqual((summary['tp'], summary['fp'], summary['fn']), (1, 5, 3))
        self.assertEqual(len(misses), 3)

    def test_matching_uses_best_unmatched_neighbor(self):
        targets = [gt(1, 0, [0, 0, 10, 10]), gt(2, 0, [2, 0, 10, 10])]
        records, misses = classify_image(targets, [dt(0, [0, 0, 10, 10], .99),
                                                  dt(0, [.5, 0, 10, 10], .9)])
        self.assertEqual([r['kind'] for r in records], ['tp', 'tp'])
        self.assertEqual(misses, [])

    def test_threshold_negative_image_and_anisotropic_resize(self):
        records, misses = classify_image([], [dt(0, [0, 0, 2, 2], .29)])
        self.assertEqual(records, [])
        records, _ = classify_image([], [dt(0, [0, 0, 2, 2], .3)])
        self.assertEqual(records[0]['kind'], 'background')
        size = object_size(gt(1, 0, [0, 0, 100, 50]), dict(width=1000, height=500))
        self.assertEqual((size['input_width'], size['input_height'], size['input_side']), (64, 64, 64))


if __name__ == '__main__':
    unittest.main()
