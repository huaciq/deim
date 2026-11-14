import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from engine.core.yaml_utils import load_config
from engine.data.dataloader import batch_image_collate_fn
from engine.data.dataset.region_dense_adaptive_dataset import RegionDenseAdaptiveCocoDetection
from engine.data.region_dense_adaptive import (
    AdaptiveSourceQueuePlanner, centered_crop, protected_geometry, strip_background_features)
from engine.data.region_dense_sar import SourceQueuePlanner, background_features, extract_image_features
from engine.data.transforms import Compose
from tools.build_region_dense_sar import build_geometry
from tools.build_region_dense_adaptive import write_adaptive_plans


class AdaptiveRegionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cfg = load_config('configs/deim_dfine/deim_hgnetv2_m_sardet100k_region_dense_adaptive_36e_b32.yml', {})
        cls.settings = copy.deepcopy(cfg['train_dataloader']['dataset']['plan_settings'])
        cls.settings.update(size=64, epochs=[4, 6], queue_capacity=16)
        annotation = dict(images=[], annotations=[], categories=[dict(id=0, name='a'), dict(id=1, name='b')])
        for i in range(32):
            filename = f'{i}.png'
            annotation['images'].append(dict(id=i, width=80, height=80, file_name=filename))
            pixels = np.random.default_rng(i).integers(0, 256, (80, 80, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(cls.root / filename)
            boxes = ([[20, 20, 3, 4]] if i < 8 else [[20+i % 3*6, 20+i % 4*4, 18, 18]]
                     if i < 24 else [[10, 10, 60, 60]] if i < 28 else [])
            for b in boxes:
                annotation['annotations'].append(dict(id=i+1, image_id=i, category_id=i % 2,
                                                       bbox=b, area=b[2]*b[3], iscrowd=0))
        cls.ann = cls.root / 'annotations.json'
        cls.ann.write_text(json.dumps(annotation))
        cls.bank, boxes, images, _ = build_geometry(annotation, cls.settings)
        cls.bank['regions'], cls.bank['required_wh'] = protected_geometry(boxes, 64, 1.)
        features = np.zeros((32, 3, 5)); valid = np.zeros((32, 3), bool)
        strips = np.zeros((32, 2, 65, 5)); strip_valid = np.zeros((32, 2, 65), bool)
        for i, info in enumerate(images):
            filename = str(cls.root / info['file_name'])
            _, features[i], valid[i] = extract_image_features(
                (i, filename, 64, boxes[i], cls.bank['crops'][i], 2., 256))
            im = Image.open(filename).convert('RGB').resize((64, 64), Image.Resampling.BILINEAR)
            strips[i], strip_valid[i] = strip_background_features(im, boxes[i], cls.bank['regions'][i])
        center = np.median(features[valid], axis=0)
        scale = np.maximum(np.quantile(features[valid], .75, axis=0)-np.quantile(features[valid], .25, axis=0), 1.)
        cls.bank.update(features=(features-center)/scale, feature_valid=valid,
                        feature_center=center, feature_scale=scale)
        strips[strip_valid] = (strips[strip_valid]-center)/scale
        cls.strips, cls.strip_valid = strips, strip_valid
        cls.plan_dir = cls.root / 'plans'; cls.plan_dir.mkdir()
        np.savez_compressed(cls.plan_dir / 'bank.npz', **cls.bank)
        np.save(cls.plan_dir / 'strip_features.npy', strips)
        np.save(cls.plan_dir / 'strip_valid.npy', strip_valid)
        write_adaptive_plans(cls.bank, strips, strip_valid, cls.settings, 8, 2, cls.plan_dir, {})

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def dataset(self, transforms=None):
        return RegionDenseAdaptiveCocoDetection(str(self.root), str(self.ann), transforms,
                                                str(self.plan_dir), self.settings)

    def planner(self, **kwargs):
        return AdaptiveSourceQueuePlanner(self.bank, self.strips, self.strip_valid, size=64, **kwargs)

    def test_prefix_descriptors_equal_actual_pixels_for_every_legal_crop(self):
        image = Image.fromarray(np.random.default_rng(82).integers(0, 256, (32, 32, 3), dtype=np.uint8))
        boxes = np.array([[1.2, 3.4, 4.9, 7.8], [12.5, 14.1, 14.6, 16.5]])
        region = np.array([0, 0, 17., 20.])
        for im in (image, Image.new('RGB', (32, 32), (31, 31, 31))):
            f, v = strip_background_features(im, boxes, region, min_pixels=8)
            for axis in (0, 1):
                for length in range(int(region[axis+2]), 33):
                    tile = [32, 32]; tile[axis] = length
                    crop = centered_crop(region, *tile, 32)
                    actual, valid = background_features(im, boxes, [crop], min_pixels=8)
                    self.assertEqual(v[axis, length], valid[0])
                    np.testing.assert_allclose(f[axis, length], actual[0], atol=1e-12, rtol=0)
        _, v = strip_background_features(image, np.array([[0., 0., 32., 32.]]), [0, 0, 32, 32])
        self.assertFalse(v.any())

    def test_disabled_adaptive_path_exactly_preserves_e010_planner(self):
        old = SourceQueuePlanner(self.bank, queue_capacity=16, seed=3)
        new = self.planner(queue_capacity=16, seed=3, adaptive=False)
        for _ in range(100):
            a, b = old.next(), new.next()
            self.assertEqual(a, b[:5])
            self.assertEqual(b[5], -1)

    def test_complete_passes_keep_every_source_and_protected_extent(self):
        planner = self.planner(queue_capacity=32, probability=1., seed=0)
        adaptive_count = 0
        for _ in range(3):
            wanted = len(planner.completed_passes)+1
            used = []
            while len(planner.completed_passes) < wanted:
                row = planner.next(); group = [i for i in row[1:5] if i >= 0]
                used.extend(group)
                if row[0] >= 3:
                    adaptive_count += 1
                    axis = row[0]-3
                    self.assertGreaterEqual(row[5], self.bank['required_wh'][group[0], axis])
                    self.assertGreaterEqual(64-row[5], self.bank['required_wh'][group[1], axis])
                    self.assertEqual(len(group), 2)
                if any(i >= 24 for i in group):
                    self.assertEqual(row[0], -1)
            self.assertEqual(sorted(used), list(range(32)))
        self.assertGreater(adaptive_count, 0)

    def test_unequal_background_choice_query_limit_and_missing_features(self):
        planner = self.planner(probability=1., nearest_candidates=1)
        planner.strips = self.strips.copy(); planner.strip_valid = np.ones_like(self.strip_valid)
        planner.strips[8] = 0.; planner.strips[0] = 1.; planner.strips[1] = 20.
        planner.pending = [0, 1]
        self.assertEqual(planner.select_adaptive(8)[1], 0)
        planner.strip_valid[8] = False
        self.assertIn(planner.select_adaptive(8)[1], [0, 1])
        self.assertEqual(planner.adaptive_feature_fallbacks, 1)
        planner.max_targets = 1
        self.assertIsNone(planner.select_adaptive(8))

    def test_real_composition_preserves_both_axes_and_swapped_slots(self):
        ds = self.dataset(); ds.set_epoch(4)
        forced = []
        for axis in (0, 1):
            cut = int((self.bank['required_wh'][8, axis]+64-self.bank['required_wh'][0, axis])//2)
            forced.extend([[axis+3, 8, 0, -1, -1, cut], [axis+3, 0, 8, -1, -1, 64-cut]])
        ds._plan_rows = np.array(forced, dtype=np.int32); ds._loaded_plan_epoch = 4
        for i in range(4):
            image, target, plan, details = ds.load_planned_item(i, trace=True)
            self.assertEqual(len(target['labels']), 2)
            for sample, detail in zip(plan, details):
                source, original = ds.load_item(ds.dataset_indices[sample['image_id']])
                expected = source.resize((64,64), Image.Resampling.BILINEAR).crop(sample['crop'])
                x,y = sample['offset']
                np.testing.assert_array_equal(np.asarray(expected), np.asarray(image.crop((x,y,x+expected.width,y+expected.height))))
                before, after = np.array(detail['input_boxes']), np.array(detail['output_boxes'])
                np.testing.assert_allclose(before[:,2:]-before[:,:2], after[:,2:]-after[:,:2], atol=1e-12, rtol=0)
                self.assertEqual(len(detail['gt_indices']), len(original['labels']))

    def test_ddp_steps_epoch_restore_and_worker_independence(self):
        ds = self.dataset()
        for epoch in (0,4,5,20,35):
            ds.set_epoch(epoch); parts=[]
            for rank in (0,1):
                sampler=ds.make_sampler(4,2,rank,True,True); sampler.set_epoch(epoch)
                part=list(sampler); parts.append(part)
                self.assertEqual(len(part)//4,4)
                if not ds.plan_active:
                    baseline=DistributedSampler(ds,num_replicas=2,rank=rank,shuffle=True); baseline.set_epoch(epoch)
                    self.assertEqual(part,list(baseline))
            if ds.plan_active:
                self.assertEqual(sorted(parts[0]+parts[1]),list(range(32)))
                self.assertFalse(set(parts[0]) & set(parts[1]))
        ops=[dict(type='Resize',size=[64,64]),dict(type='ConvertPILImage',dtype='float32',scale=True),
             dict(type='ConvertBoxes',fmt='cxcywh',normalize=True)]
        running=self.dataset(Compose(copy.deepcopy(ops))); running.set_epoch(4); running[0]; running.set_epoch(5)
        resumed=self.dataset(Compose(copy.deepcopy(ops))); resumed.set_epoch(5)
        a=list(DataLoader(running,batch_size=2,sampler=[0,1,2,3],num_workers=0,collate_fn=batch_image_collate_fn))
        b=list(DataLoader(resumed,batch_size=2,sampler=[0,1,2,3],num_workers=2,collate_fn=batch_image_collate_fn))
        for (x,tx),(y,ty) in zip(a,b):
            torch.testing.assert_close(x,y)
            for u,v in zip(tx,ty):
                for k in ('image_id','boxes','labels','area'):torch.testing.assert_close(u[k],v[k])

    def test_only_v3_data_layout_changes_from_e010_configuration(self):
        base=load_config('configs/deim_dfine/deim_hgnetv2_m_sardet100k_region_dense_sar_36e_b32.yml',{})
        new=load_config('configs/deim_dfine/deim_hgnetv2_m_sardet100k_region_dense_adaptive_36e_b32.yml',{})
        for cfg in (base,new):cfg.pop('__include__',None)
        for k in ('output_dir','experiment_protocol_document'):new[k]=base[k]
        for k in ('type','plan_dir'):new['train_dataloader']['dataset'][k]=base['train_dataloader']['dataset'][k]
        self.assertTrue(new['train_dataloader']['dataset']['plan_settings'].pop('adaptive_partition'))
        self.assertEqual(new,base)


if __name__ == '__main__':
    unittest.main()
