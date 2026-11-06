"""Targeted checks for M1. Run on CPU: python -m unittest tests.test_imaging_geometry -v"""
import unittest
from pathlib import Path

import torch
import torchvision
from PIL import Image

from engine.core import YAMLConfig
from engine.data.dataset.coco_dataset import ConvertCocoPolysToMask
from engine.data._misc import convert_to_tv_tensor, SanitizeBoundingBoxes
from engine.data.transforms._transforms import ConvertBoxes, GeometryHorizontalFlip, GeometryTargets
from engine.deim.box_ops import box_cxcywh_to_xyxy
from engine.deim.imaging_geometry import ImagingGeometryHead, radiometric_statistics

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/deim_dfine/deim_hgnetv2_m_sardet100k_m1_geometry_36e_b32.yml'
BASELINE = ROOT / 'configs/deim_dfine/deim_hgnetv2_m_sardet100k_36e_b32_nozoom_nocrop.yml'


def annotation(x, y, w, h, category=1):
    return dict(bbox=[x, y, w, h], category_id=category, area=w * h, iscrowd=0)


def sample(annotations):
    image = Image.new('RGB', (100, 100))
    _, target = ConvertCocoPolysToMask(keep_overflow=True)(image, dict(image_id=1, annotations=annotations))
    target['idx'] = torch.tensor([0])
    target['boxes'] = convert_to_tv_tensor(target['boxes'], key='boxes', spatial_size=(100, 100))
    return image, target


class DataPipeline(unittest.TestCase):
    def test_overflow_is_recorded_before_clipping(self):
        _, target = sample([annotation(-10, 20, 30, 10), annotation(90, 30, 20, 10)])
        self.assertTrue(torch.allclose(target['boxes'], torch.tensor([[0., 20, 20, 30], [90, 30, 100, 40]])))
        self.assertTrue(torch.allclose(target['labels'][:, 1:], torch.tensor([[.1, 0, 0, 0], [0, 0, .1, 0]])))

    def test_flip_mirrors_overflow_and_records_state(self):
        image, target = sample([annotation(-10, 20, 30, 10)])
        flip = GeometryHorizontalFlip(p=1.)
        _, flipped, _ = flip(image, target, None)
        self.assertEqual(float(flipped['flip_state']), -1.)
        self.assertTrue(torch.allclose(flipped['boxes'].as_subclass(torch.Tensor), torch.tensor([[80., 20, 100, 30]])))
        self.assertTrue(torch.allclose(flipped['labels'][:, 1:], torch.tensor([[0., 0, .1, 0]])))
        image, target = sample([annotation(-10, 20, 30, 10)])
        _, same, _ = GeometryHorizontalFlip(p=0.)(image, target, None)
        self.assertEqual(float(same['flip_state']), 1.)
        self.assertTrue(torch.allclose(same['labels'][:, 1:], torch.tensor([[.1, 0, 0, 0]])))

    def test_sanitize_keeps_overflow_aligned_and_split_restores_labels(self):
        image, target = sample([annotation(-10, 20, 30, 10, 2), annotation(50, 50, .5, .5, 3), annotation(90, 30, 20, 10, 4)])
        image, target = SanitizeBoundingBoxes(min_size=1)(image, target)
        self.assertEqual(len(target['labels']), 2)
        image, target, _ = GeometryHorizontalFlip(p=0.)(image, target, None)
        image, target = ConvertBoxes(fmt='cxcywh', normalize=True)(image, target)
        image, target, _ = GeometryTargets()(image, target, None)
        self.assertEqual(target['labels'].tolist(), [2, 4])
        self.assertEqual(target['labels'].dtype, torch.int64)
        self.assertTrue(torch.allclose(target['official_overflow'], torch.tensor([[.1, 0, 0, 0], [0, 0, .1, 0]])))
        self.assertTrue(bool(target['geometry_clean']))

    def test_mosaic_like_sample_is_not_clean(self):
        image, target = sample([annotation(10, 20, 30, 10)])
        target['idx'] = torch.tensor([0, 1, 2, 3])
        _, target, _ = GeometryTargets()(image, target, None)
        self.assertFalse(bool(target['geometry_clean']))


class Head(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.head = ImagingGeometryHead(8, 1, num_domains=2, embed_dim=8, amodal_dim=8)

    def test_statistics_are_flip_invariant(self):
        images = torch.rand(2, 3, 64, 64)
        self.assertTrue(torch.allclose(radiometric_statistics(images), radiometric_statistics(images.flip(-1)), atol=1e-5))

    def test_translation_sign_follows_flip_and_official_box_is_kept(self):
        boxes = torch.tensor([[.5, .5, .2, .2]])
        common = dict(official_overflow=torch.zeros(1, 4), geometry_clean=torch.tensor(True))
        targets = [dict(boxes=boxes, flip_state=torch.tensor(1.), **common), dict(boxes=boxes, flip_state=torch.tensor(-1.), **common)]
        translation = torch.tensor([[25.6, 0.], [25.6, 0.]])
        result = self.head.response_targets(targets, translation)
        self.assertAlmostEqual(result[0]['boxes'][0, 0].item(), .4, places=5)
        self.assertAlmostEqual(result[1]['boxes'][0, 0].item(), .6, places=5)
        self.assertTrue(torch.allclose(result[0]['official_xyxy'], box_cxcywh_to_xyxy(boxes)))

    def test_mixup_and_mosaic_targets_are_untouched(self):
        boxes = torch.tensor([[.5, .5, .2, .2]])
        targets = [dict(boxes=boxes, flip_state=torch.tensor(1.), official_overflow=torch.zeros(1, 4), geometry_clean=torch.tensor(True), mixup=torch.ones(1)),
                   dict(boxes=boxes, flip_state=torch.tensor(1.), official_overflow=torch.zeros(1, 4), geometry_clean=torch.tensor(False))]
        for item in self.head.response_targets(targets, torch.full((2, 2), 25.6)):
            self.assertFalse(item['geometry_clean'])
            self.assertTrue(torch.equal(item['boxes'], boxes))

    def test_extension_only_moves_border_edges(self):
        boxes = torch.tensor([[[.1, .5, .2, .2], [.5, .5, .2, .2]]])
        memory = torch.rand(1, 16, 8)
        values, gate = self.head.extension(torch.rand(1, 2, 8), boxes, memory, [[4, 4]])
        self.assertEqual(gate.tolist(), [[[True, False, False, False], [False, False, False, False]]])
        self.assertEqual(values[0, 1].abs().sum().item(), 0.)
        self.assertLess(values.max().item(), 1e-2)
        moved = self.head.compose(boxes, torch.zeros(1, 2), torch.tensor([[[.5, 0., 0., 0.], [0., 0., 0., 0.]]]))
        xyxy = box_cxcywh_to_xyxy(moved)
        self.assertAlmostEqual(xyxy[0, 0, 0].item(), -.1, places=5)
        self.assertAlmostEqual(xyxy[0, 0, 2].item(), .2, places=5)
        self.assertTrue(torch.allclose(moved[0, 1], boxes[0, 1]))

    def test_domains_recover_a_translation_from_flipped_noisy_residuals(self):
        """Domain A has labels at +0.8 and domain B at 0. Flips make image-frame residuals change sign."""
        torch.manual_seed(1)
        head = ImagingGeometryHead(4, 1, num_domains=4, embed_dim=16, amodal_dim=8, use_amodal=False)
        optimizer = torch.optim.AdamW(head.parameters(), 2e-2, weight_decay=0.)
        for _ in range(1500):
            domain = torch.randint(0, 2, (32,))
            memory = (domain[:, None].float() * 2 - 1).repeat(1, 4).unsqueeze(1) + .1 * torch.randn(32, 1, 4)
            images = torch.full((32, 3, 16, 16), .5) + domain[:, None, None, None] * .3
            translation, _ = head.translation(memory, [[1, 1]], images)
            flip = torch.randint(0, 2, (32,)).float() * 2 - 1
            # Canonical residual: the sign has been undone, as in the criterion. Heavy-tailed noise.
            residual_image = (domain.float() * .8 * flip + .5 * torch.distributions.Laplace(0., 1.).sample((32,)))
            residual = residual_image * flip
            loss = torch.nn.functional.smooth_l1_loss(translation[:, 0], residual, beta=.5)
            loss = loss + torch.nn.functional.smooth_l1_loss(translation[:, 1], .5 * torch.distributions.Laplace(0., 1.).sample((32,)), beta=.5)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            for value, expected in ((1, .8), (0, 0.)):
                memory = (torch.full((64, 1), float(value)) * 2 - 1).repeat(1, 4).unsqueeze(1)
                images = torch.full((64, 3, 16, 16), .5 + value * .3)
                translation, _ = head.translation(memory, [[1, 1]], images)
                self.assertAlmostEqual(translation[:, 0].mean().item(), expected, delta=.15)
                self.assertAlmostEqual(translation[:, 1].mean().item(), 0., delta=.15)


class FullModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = YAMLConfig(str(CONFIG))
        cls.cfg = cfg
        torch.manual_seed(0)
        cls.model = cfg.model
        cls.criterion = cfg.criterion

    def targets(self):
        make = lambda boxes, labels, overflow, flip: dict(
            boxes=torch.tensor(boxes), labels=torch.tensor(labels), official_overflow=torch.tensor(overflow),
            flip_state=torch.tensor(flip), geometry_clean=torch.tensor(True), image_id=torch.tensor([1]),
            orig_size=torch.tensor([256, 256]), area=torch.ones(len(labels)), iscrowd=torch.zeros(len(labels)))
        return [make([[.5, .5, .1, .1], [.05, .5, .1, .2]], [4, 1], [[0., 0, 0, 0], [.05, 0, 0, 0]], 1.),
                make([[.3, .3, .1, .1]], [3], [[0., 0, 0, 0]], -1.)]

    def test_training_losses_reach_the_head_and_response_path_is_detached(self):
        self.model.train()
        self.criterion.train()
        outputs = self.model(torch.rand(2, 3, 320, 320), targets=self.targets())
        self.assertTrue(all(k in outputs for k in ('geometry_translation', 'geometry_extension', 'geometry_response_targets')))
        losses = self.criterion(outputs, self.targets(), epoch=0, step=0, global_step=0, epoch_step=1)
        self.assertIn('loss_geo_translation', losses)
        self.assertIn('loss_geo_extension', losses)
        self.model.zero_grad()
        (losses['loss_geo_translation'] + losses['loss_geo_extension']).backward(retain_graph=True)
        outside = [n for n, p in self.model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0 and 'imaging_geometry' not in n]
        self.assertEqual(outside, [])
        self.model.zero_grad()
        sum(losses.values()).backward()
        missing = [n for n, p in self.model.named_parameters() if p.requires_grad and p.grad is None and 'imaging_geometry' in n]
        self.assertEqual(missing, [])

    def test_inference_starts_at_the_response_box(self):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(torch.rand(1, 3, 640, 640))
        difference = (outputs['pred_boxes'] - outputs['response_boxes']).abs().max().item()
        self.assertLess(difference, 1e-2)
        self.assertNotIn('geometry_translation', outputs)

    def test_baseline_config_has_no_geometry(self):
        cfg = YAMLConfig(str(BASELINE))
        self.assertIsNone(cfg.model.decoder.imaging_geometry)
        self.assertFalse(cfg.yaml_cfg['train_dataloader']['dataset'].get('keep_overflow', False))
        different = {'__include__', 'experiment_protocol_document', 'experiment_baseline_revision', 'output_dir', 'DFINETransformer', 'DEIMCriterion', 'train_dataloader'}
        other = YAMLConfig(str(CONFIG)).yaml_cfg
        for key in set(other) - different:
            self.assertEqual(other[key], cfg.yaml_cfg.get(key), key)
        for key in ('epoches', 'flat_epoch', 'no_aug_epoch', 'optimizer', 'lr_gamma', 'warmup_iter'):
            self.assertEqual(other[key], cfg.yaml_cfg[key])
        decoder = {k: v for k, v in other['DFINETransformer'].items() if not k.startswith(('geometry_', 'use_imaging'))}
        self.assertEqual(decoder, cfg.yaml_cfg['DFINETransformer'])
        weights = {k: v for k, v in other['DEIMCriterion']['weight_dict'].items() if not k.startswith('loss_geo')}
        self.assertEqual(weights, cfg.yaml_cfg['DEIMCriterion']['weight_dict'])
        ours = other['train_dataloader']['dataset']['transforms']['ops']
        theirs = cfg.yaml_cfg['train_dataloader']['dataset']['transforms']['ops']
        rename = {'GeometryHorizontalFlip': 'RandomHorizontalFlip'}
        self.assertEqual([rename.get(op['type'], op['type']) for op in ours if op['type'] != 'GeometryTargets'], [op['type'] for op in theirs])


@unittest.skipUnless(Path('/root/zxy/TCRA-SAR/datasets/SARDet_100K/Annotations/train.json').is_file(), 'needs the server dataset')
class RealPipeline(unittest.TestCase):
    def test_items_keep_geometry_fields_through_every_stage(self):
        dataset = YAMLConfig(str(CONFIG)).train_dataloader.dataset
        for epoch, expect_mosaic in ((0, False), (10, None), (33, False)):
            dataset.set_epoch(epoch)
            seen = set()
            for index in range(0, 6000, 200):
                image, target = dataset[index]
                self.assertEqual(target['labels'].ndim, 1)
                self.assertEqual(target['labels'].dtype, torch.int64)
                self.assertEqual(target['official_overflow'].shape, (len(target['labels']), 4))
                self.assertEqual(target['boxes'].shape[0], len(target['labels']))
                self.assertIn(float(target['flip_state']), (-1., 1.))
                self.assertTrue((target['official_overflow'] >= 0).all())
                seen.add(bool(target['geometry_clean']))
            if expect_mosaic is False:
                self.assertEqual(seen, {True})
            else:
                self.assertEqual(seen, {True, False})

    def test_flipped_overflow_matches_official_annotation(self):
        import json
        dataset = YAMLConfig(str(CONFIG)).train_dataloader.dataset
        dataset.set_epoch(33)
        data = json.load(open(dataset.ann_file))
        images = {i['id']: i for i in data['images']}
        wanted = {}
        for a in data['annotations']:
            im = images[a['image_id']]
            x, y, w, h = a['bbox']
            if x < -2 and w > 2 and h > 2:
                wanted.setdefault(a['image_id'], []).append((-x / im['width'], (x + w) / im['width']))
        found = 0
        for index, image_id in enumerate(dataset.ids):
            if image_id not in wanted:
                continue
            _, target = dataset[index]
            column = 0 if float(target['flip_state']) > 0 else 2
            values = target['official_overflow'][:, column]
            for overflow, _ in wanted[image_id]:
                self.assertLess((values - overflow).abs().min().item(), 1e-4)
                found += 1
            if found > 50:
                break
        self.assertGreater(found, 50)


if __name__ == '__main__':
    unittest.main()
