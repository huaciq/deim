import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from tools.dataset.prepare_sar_aircraft import convert_box, prepare_dataset


class PrepareAircraftTests(unittest.TestCase):
    def test_source_edge_coordinates_are_preserved(self):
        self.assertEqual(convert_box([459, 1433, 536, 1499], 1500, 1500), [459, 1433, 77, 66])
        self.assertEqual(convert_box([0, 0, 10, 10], 10, 10), [0, 0, 10, 10])

    def test_invalid_source_box_is_not_silently_clipped(self):
        with self.assertRaises(ValueError):
            convert_box([2, 2, 11, 9], 10, 10)

    def test_original_splits_stable_ids_classes_and_negative_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['Annotations', 'JPEGImages', 'ImageSets/Main']:
                (root / name).mkdir(parents=True)
            for i, name in enumerate(['Boeing737', 'A220', None], 1):
                stem = f'{i:07d}'
                Image.new('RGB', (32, 32)).save(root / 'JPEGImages' / f'{stem}.jpg')
                obj = '' if name is None else f'<object><name>{name}</name><bndbox><xmin>2</xmin><ymin>3</ymin><xmax>12</xmax><ymax>14</ymax></bndbox></object>'
                xml = f'<annotation><size><width>32</width><height>32</height></size>{obj}</annotation>'
                (root / 'Annotations' / f'{stem}.xml').write_text(xml)
            for split, ids in dict(train=[1], val=[2], trainval=[1, 2], test=[3]).items():
                (root / 'ImageSets/Main' / f'{split}.txt').write_text(''.join(f'{i:07d}\n' for i in ids))
            summary = prepare_dataset(root, root / 'COCO')
            train = json.loads((root / 'COCO/train.json').read_text())
            trainval = json.loads((root / 'COCO/trainval.json').read_text())
            self.assertEqual([c['name'] for c in train['categories']], ['A220', 'Boeing737'])
            self.assertEqual(train['annotations'][0]['bbox'], [2, 3, 10, 11])
            self.assertEqual(train['annotations'][0], trainval['annotations'][0])
            self.assertEqual(train['images'][0]['file_name'], 'JPEGImages/0000001.jpg')
            self.assertEqual(summary['splits']['test']['empty_images'], 1)
            self.assertEqual(summary['max_box_roundtrip_error'], 0)


if __name__ == '__main__':
    unittest.main()
