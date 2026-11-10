import copy
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from engine.core.yaml_utils import load_config
from engine.data._misc import convert_to_tv_tensor
from engine.data.region_bank import build_bank, RegionBank, context_boxes, visible_and_preserved
from engine.data.transforms import Compose, RegionDense


class TinyDataset:
    def __init__(self, annotation):
        self.annotation = annotation
        self.ids = sorted(x['id'] for x in annotation['images'])
        self.epoch = 8

    def load_item(self, index):
        image_id = self.ids[index]
        grid = np.arange(80 * 80 * 3).reshape(80, 80, 3)
        image = Image.fromarray(((grid + 13 * image_id) % 256).astype(np.uint8))
        items = [x for x in self.annotation['annotations'] if x['image_id'] == image_id]
        boxes = torch.tensor([a['bbox'] for a in items], dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        target = dict(boxes=convert_to_tv_tensor(boxes, 'boxes', spatial_size=(80, 80)),
                      labels=torch.tensor([a['category_id'] for a in items]),
                      iscrowd=torch.zeros(len(items), dtype=torch.int64),
                      area=(boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
                      image_id=torch.tensor([image_id]), idx=torch.tensor([index]),
                      orig_size=torch.tensor([80, 80]))
        return image, target


class RegionDenseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name) / 'bank.npz'
        annotation = dict(images=[dict(id=i, width=80, height=80) for i in range(10)],
                          annotations=[], categories=[dict(id=0, name='a'), dict(id=1, name='b')])
        for i in range(6):
            annotation['annotations'].append(dict(image_id=i, category_id=i % 2, bbox=[10 + i, 11, 4, 3]))
        for box in ([5, 5, 4, 4], [60, 5, 4, 4]):
            annotation['annotations'].append(dict(image_id=6, category_id=0, bbox=box))
        for box in ([5, 5, 4, 4], [5, 60, 4, 4]):
            annotation['annotations'].append(dict(image_id=7, category_id=1, bbox=box))
        annotation['annotations'].append(dict(image_id=8, category_id=0, bbox=[10, 10, 60, 60]))
        cls.dataset = TinyDataset(annotation)
        cls.arrays, cls.report = build_bank(annotation)
        np.savez_compressed(cls.path, **cls.arrays)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def transform(self):
        transform = RegionDense(str(self.path), probability=1)
        transform.load_bank(self.dataset)
        return transform

    def test_class_exposure_weights_match_training_labels(self):
        for pool in self.report['pools']:
            np.testing.assert_allclose(pool['expected_class_share'], self.report['original_class_share'], atol=1e-12)

    def test_four_two_whole_and_empty_routes(self):
        bank = RegionBank(self.path)
        self.assertEqual(len(bank.plan(0, random.Random(1))), 4)
        self.assertEqual(len(bank.plan(6, random.Random(1))), 2)
        self.assertEqual(len(bank.plan(7, random.Random(1))), 2)
        self.assertIsNone(bank.plan(8))
        self.assertIsNone(bank.plan(9))
        self.assertIsNone(bank.plan(0, random.Random(1), max_targets=1))

    def test_pixels_sizes_context_and_all_primary_targets_survive(self):
        transform = self.transform()
        for primary_id in (0, 6, 7):
            image, target = self.dataset.load_item(primary_id)
            original_boxes = target['boxes'].clone()
            plan = transform.bank.plan(primary_id, random.Random(primary_id + 1))
            result, labels, details = transform.apply_plan(image, target, self.dataset, plan, trace=True)
            self.assertEqual(len({s['image_id'] for s in plan}), len(plan))
            self.assertEqual(sum(len(x['labels']) for x in details), len(labels['boxes']))
            for sample, detail in zip(plan, details):
                source, original = self.dataset.load_item(sample['image_id'])
                ordinary = source.resize((640, 640), Image.Resampling.BILINEAR)
                expected = ordinary.crop(sample['crop'])
                x, y = sample['offset']
                actual = result.crop((x, y, x + expected.width, y + expected.height))
                np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
                before, after = np.array(detail['input_boxes']), np.array(detail['output_boxes'])
                np.testing.assert_allclose(after[:, 2:] - after[:, :2], before[:, 2:] - before[:, :2])
                all_boxes = original['boxes'].as_subclass(torch.Tensor).numpy() * 8
                visible, protected = visible_and_preserved(all_boxes, context_boxes(all_boxes, 640, 1), sample['crop'])
                self.assertTrue(protected[visible].all())
                if sample['primary']:
                    self.assertEqual(detail['gt_indices'], list(range(len(target['boxes']))))
            torch.testing.assert_close(target['boxes'], original_boxes)

    def test_inactive_stages_and_probability_leave_sample_unchanged(self):
        transform = self.transform()
        image, target = self.dataset.load_item(0)
        for epoch in (0, 3, 20, 32, 35):
            self.dataset.epoch = epoch
            output = transform(image, target, self.dataset)
            self.assertIs(output[0], image)
            self.assertIs(output[1], target)
        self.dataset.epoch = 8
        transform.probability = 0
        self.assertIs(transform(image, target, self.dataset)[0], image)

    def test_full_image_pipeline_and_protocol_control(self):
        base = load_config('configs/deim_dfine/deim_hgnetv2_m_sardet100k_no_dense_o2o_36e_b32.yml', {})
        new = load_config('configs/deim_dfine/deim_hgnetv2_m_sardet100k_region_dense_36e_b32.yml', {})
        base.pop('__include__', None)
        new.pop('__include__', None)
        baseline_ops = base['train_dataloader']['dataset']['transforms']['ops']
        new_ops = new['train_dataloader']['dataset']['transforms']['ops']
        self.assertEqual(baseline_ops[1:], new_ops[1:])
        edited = copy.deepcopy(new)
        for key in ('output_dir', 'experiment_protocol_document'):
            edited[key] = base[key]
        edited['train_dataloader']['dataset']['transforms']['ops'] = baseline_ops
        edited['train_dataloader']['dataset']['transforms']['policy']['ops'] = ['Mosaic', 'RandomPhotometricDistort']
        self.assertEqual(edited, base)
        params = copy.deepcopy(new['train_dataloader']['dataset']['transforms'])
        params.pop('type', None)
        params['ops'][0].update(bank_path=str(self.path), probability=1)
        pipeline = Compose(**params)
        self.dataset.epoch = 8
        image, target = self.dataset.load_item(6)
        image, target, _ = pipeline(image, target, self.dataset)
        self.assertEqual(tuple(image.shape), (3, 640, 640))
        self.assertTrue(torch.isfinite(target['boxes']).all())
        self.assertTrue((target['boxes'] >= 0).all())
        self.assertTrue((target['boxes'] <= 1).all())
        self.assertGreater(len(target['labels']), 2)


if __name__ == '__main__':
    unittest.main()
