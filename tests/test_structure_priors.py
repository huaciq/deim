"""Functional checks for the CPU structure-prior analysis (symmetry axis, look direction, working ENL)."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from tools.check_structure_priors import analyze, axial_difference, symmetry_curve, symmetry_window, axis_from_curve


def plane_mask(size, angle):
    """Swept-wing aircraft silhouette, mirror symmetric about the fuselage at `angle` (degrees from x, y down)."""
    c = (size - 1) / 2
    y, x = np.mgrid[:size, :size] - c
    t = np.radians(angle)
    u, v = x * np.cos(t) + y * np.sin(t), -x * np.sin(t) + y * np.cos(t)
    fuselage = (np.abs(v) <= 2) & (np.abs(u) <= 20)
    wing = (np.abs(v) <= 18) & (u <= 4 - .5 * np.abs(v)) & (u >= -2 - .5 * np.abs(v))
    tail = (np.abs(v) <= 7) & (u >= -20) & (u <= -16)
    return fuselage | wing | tail


def scene(seed, planes, size=160, shadow=12):
    """Speckled clutter, aircraft at (x, y, angle); the strip left of each box is a dark shadow."""
    rng = np.random.default_rng(seed)
    values = rng.gamma(4, 40 / 4, (size, size))
    boxes = []
    for x, y, angle in planes:
        mask = plane_mask(48, angle)
        patch = values[y:y + 48, x:x + 48]
        patch[mask] = rng.gamma(4, 200 / 4, mask.sum())
        ys, xs = np.nonzero(mask)
        box = [x + xs.min(), y + ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1]
        values[box[1]:box[1] + box[3], max(box[0] - shadow, 0):box[0]] *= .2
        boxes.append([float(v) for v in box])
    return Image.fromarray(np.clip(values, 0, 255).astype(np.uint8)), boxes


class StructurePriorTests(unittest.TestCase):
    def test_symmetry_axis_recovers_fuselage(self):
        for angle in (0, 30, 75, 120, 160):
            image, boxes = scene(angle, [(56, 56, angle)])
            values = np.log1p(np.asarray(image, np.float32))
            box = boxes[0]
            window = symmetry_window(values, box[0] + box[2] / 2, box[1] + box[3] / 2, max(box[2:]) * 1.2)
            axis, peak, sharpness, margin = axis_from_curve(symmetry_curve(window))
            self.assertLessEqual(axial_difference(axis, angle), 6, (angle, axis))
            self.assertGreater(margin, 0)

    def test_report_look_direction_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix='deim_structure_priors_') as tmp:
            root = Path(tmp)
            (root / 'COCO').mkdir()
            data = dict(images=[], annotations=[], categories=[dict(id=0, name='Boeing737')])
            for image_id, planes in enumerate([[(10, 20, 10), (90, 90, 100)], [(60, 30, 45)], [(20, 100, 150), (100, 10, 80)]], 1):
                image, boxes = scene(image_id, planes)
                image.save(root / f'{image_id}.png')
                data['images'].append(dict(id=image_id, file_name=f'{image_id}.png', width=160, height=160))
                for box in boxes:
                    data['annotations'].append(dict(id=len(data['annotations']) + 1, image_id=image_id, category_id=0,
                                                    bbox=box, area=box[2] * box[3], iscrowd=0))
            (root / 'COCO/train.json').write_text(json.dumps(data), encoding='utf-8')
            out = root / 'out'
            summary = analyze(root, out, splits=('train',), workers=2)
            look = summary['look']['train']
            self.assertEqual(look['gt']['left_darker'], 1.0)
            self.assertLess(look['gt']['lr'][2], -.5)
            self.assertGreater(look['control']['lr'][1], -.5)
            self.assertIsNotNone(look['enl_work'])
            sym = summary['symmetry']['train']
            self.assertEqual(sym['targets'], 5)
            self.assertTrue((out / 'report.md').exists() and (out / 'sheets/train_random.png').exists())
            with self.assertRaises(FileExistsError):
                analyze(root, out, splits=('train',))


if __name__ == '__main__':
    unittest.main()
