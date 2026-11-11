import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from engine.core.yaml_utils import load_config
from engine.data.dataloader import batch_image_collate_fn
from engine.data.dataset.region_dense_sar_dataset import RegionDenseSARCocoDetection
from engine.data.region_dense_sar import SourceQueuePlanner, extract_image_features
from engine.data.transforms import Compose
from tools.build_region_dense_sar import build_geometry, write_plans


class RegionDenseSARTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cfg = load_config('configs/deim_dfine/deim_hgnetv2_m_sardet100k_region_dense_sar_36e_b32.yml', {})
        cls.settings = copy.deepcopy(cfg['train_dataloader']['dataset']['plan_settings'])
        cls.settings['epochs'] = [4, 6]
        cls.settings['queue_capacity'] = 12
        annotation = dict(images=[], annotations=[], categories=[dict(id=0, name='a'), dict(id=1, name='b')])
        for i in range(24):
            filename = f'{i}.png'
            annotation['images'].append(dict(id=i, width=80, height=80, file_name=filename))
            pixels = ((np.arange(80*80*3).reshape(80, 80, 3) + 13*i) % 256).astype(np.uint8)
            Image.fromarray(pixels).save(cls.root / filename)
            boxes = [] if i == 23 else ([[10, 10, 60, 60]] if i >= 20 else [[12+i % 5, 13, 4, 3]])
            for box in boxes:
                annotation['annotations'].append(dict(id=i, image_id=i, category_id=i % 2,
                                                       bbox=box, area=box[2]*box[3], iscrowd=0))
        cls.ann = cls.root / 'annotations.json'
        cls.ann.write_text(json.dumps(annotation))
        cls.bank, boxes, images, _ = build_geometry(annotation, cls.settings)
        features = np.zeros((24, 3, 5)); valid = np.zeros((24, 3), bool)
        for i, info in enumerate(images):
            _, features[i], valid[i] = extract_image_features(
                (i, str(cls.root / info['file_name']), 640, boxes[i], cls.bank['crops'][i], 2., 256))
        scale = np.maximum(np.quantile(features[valid], .75, axis=0) - np.quantile(features[valid], .25, axis=0), 1.)
        cls.bank.update(features=features / scale, feature_valid=valid)
        cls.plan_dir = cls.root / 'plans'
        cls.plan_dir.mkdir()
        np.savez_compressed(cls.plan_dir / 'bank.npz', **cls.bank)
        write_plans(cls.bank, cls.settings, 8, 2, cls.plan_dir, {})

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def dataset(self, transforms=None):
        return RegionDenseSARCocoDetection(str(self.root), str(self.ann), transforms,
                                           str(self.plan_dir), self.settings)

    def test_complete_source_passes_preserve_all_instances_including_whole_and_empty(self):
        planner = SourceQueuePlanner(self.bank, queue_capacity=12, seed=0)
        for _ in range(3):
            expected_passes = len(planner.completed_passes) + 1
            used = []
            while len(planner.completed_passes) < expected_passes:
                row = planner.next()
                used.extend(i for i in row[1:] if i >= 0)
                if row[0] >= 0:
                    self.assertTrue(all(self.bank['crops'][i, row[0], 0] >= 0 for i in row[1:] if i >= 0))
            self.assertEqual(sorted(used), list(range(24)))
            np.testing.assert_array_equal(self.bank['counts'][used].sum(0), self.bank['counts'].sum(0))
        limited = SourceQueuePlanner(self.bank, probability=1., max_targets=1)
        self.assertTrue(all(limited.next()[0] == -1 for _ in range(24)))

    def test_background_choice_and_missing_background_fallback(self):
        bank = dict(self.bank)
        bank['features'] = np.zeros_like(bank['features'])
        bank['features'][1] = 1.
        bank['features'][2] = 20.
        bank['feature_valid'] = np.ones_like(bank['feature_valid'])
        planner = SourceQueuePlanner(bank, nearest_candidates=1)
        self.assertEqual(planner.select(0, [1, 2], 0), 1)
        planner.valid[0] = False
        self.assertIn(planner.select(0, [1, 2], 0), [1, 2])
        self.assertEqual(planner.feature_fallbacks, 1)

    def test_ddp_partition_preserves_steps_and_ordinary_sampler(self):
        ds = self.dataset()
        for epoch in (0, 4, 5, 20, 35):
            ds.set_epoch(epoch)
            parts = []
            for rank in range(2):
                sampler = ds.make_sampler(4, 2, rank, True, True)
                sampler.set_epoch(epoch)
                part = list(sampler)
                self.assertEqual(len(part) // 4, 3)
                if not ds.plan_active:
                    baseline = DistributedSampler(ds, num_replicas=2, rank=rank, shuffle=True)
                    baseline.set_epoch(epoch)
                    self.assertEqual(part, list(baseline))
                parts.append(part)
            if ds.plan_active:
                self.assertFalse(set(parts[0]) & set(parts[1]))
                self.assertEqual(sorted(parts[0] + parts[1]), list(range(24)))

    def test_composition_preserves_pixels_sizes_and_all_sources(self):
        ds = self.dataset(); ds.set_epoch(4)
        tested = 0
        for index in range(ds.output_canvases):
            image, target, plan, trace = ds.load_planned_item(index, trace=True)
            if not plan:
                continue
            self.assertEqual(len(target['labels']), sum(int(p['counts'].sum()) for p in plan))
            for sample, detail in zip(plan, trace):
                source, original = ds.load_item(ds.dataset_indices[sample['image_id']])
                expected = source.resize((640, 640), Image.Resampling.BILINEAR).crop(sample['crop'])
                x, y = sample['offset']
                np.testing.assert_array_equal(np.asarray(expected), np.asarray(image.crop((x, y, x+expected.width, y+expected.height))))
                self.assertEqual(len(detail['gt_indices']), len(original['labels']))
                before, after = np.array(detail['input_boxes']), np.array(detail['output_boxes'])
                np.testing.assert_allclose(before[:, 2:]-before[:, :2], after[:, 2:]-after[:, :2])
            tested += 1
        self.assertGreater(tested, 0)

    def test_resume_epoch_and_worker_count_read_same_plan(self):
        ops = [dict(type='Resize', size=[640, 640]),
               dict(type='ConvertPILImage', dtype='float32', scale=True),
               dict(type='ConvertBoxes', fmt='cxcywh', normalize=True)]
        running = self.dataset(Compose(copy.deepcopy(ops)))
        running.set_epoch(4)
        running[0]  # Materialize the previous epoch's mmap before advancing.
        running.set_epoch(5)
        resumed = self.dataset(Compose(copy.deepcopy(ops)))
        resumed.set_epoch(5)
        left = list(DataLoader(running, batch_size=2, sampler=[0, 1, 2, 3], num_workers=0,
                               collate_fn=batch_image_collate_fn))
        right = list(DataLoader(resumed, batch_size=2, sampler=[0, 1, 2, 3], num_workers=2,
                                collate_fn=batch_image_collate_fn))
        for (a, ta), (b, tb) in zip(left, right):
            torch.testing.assert_close(a, b)
            for x, y in zip(ta, tb):
                for key in ('image_id', 'labels', 'boxes', 'area'):
                    torch.testing.assert_close(x[key], y[key])

    def test_only_declared_data_construction_changes_from_baseline(self):
        base = load_config('configs/deim_dfine/deim_hgnetv2_m_sardet100k_no_dense_o2o_36e_b32.yml', {})
        new = load_config('configs/deim_dfine/deim_hgnetv2_m_sardet100k_region_dense_sar_36e_b32.yml', {})
        for cfg in (base, new):
            cfg.pop('__include__', None)
        ds = new['train_dataloader']['dataset']
        self.assertEqual(ds['transforms']['ops'], base['train_dataloader']['dataset']['transforms']['ops'][1:])
        for key in ('plan_dir', 'plan_settings'):
            ds.pop(key)
        ds['type'] = 'CocoDetection'
        ds['transforms'] = copy.deepcopy(base['train_dataloader']['dataset']['transforms'])
        for key in ('output_dir', 'experiment_protocol_document'):
            if key in base:
                new[key] = base[key]
            else:
                new.pop(key, None)
        self.assertEqual(new, base)


if __name__ == '__main__':
    unittest.main()
