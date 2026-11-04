"""Scheme-B main protocol: config, uniform-scale canvas transform and COCO-airplane head initialization."""

from pathlib import Path
import unittest

import PIL.Image
import torch

from engine.core.yaml_utils import load_config
from engine.data._misc import convert_to_tv_tensor
from engine.data.transforms._transforms import ConvertBoxes, PadToCanvas, Resize
from tools.init_head_from_coco_class import adapt


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_b_canvas1024_132e_b12_old_server.yml'


class ProtocolBTests(unittest.TestCase):
    def test_config(self):
        cfg = load_config(str(CONFIG), {})
        self.assertTrue((ROOT / cfg['experiment_protocol_document']).is_file())
        self.assertIs(cfg['fixed_schedule'], True)
        train, val = cfg['train_dataloader'], cfg['val_dataloader']
        self.assertTrue(train['dataset']['ann_file'].endswith('/COCO/trainval.json'))
        self.assertTrue(val['dataset']['ann_file'].endswith('/COCO/val.json'))
        self.assertEqual([op['type'] for op in train['dataset']['transforms']['ops']],
                         ['RandomHorizontalFlip', 'PadToCanvas', 'Resize', 'SanitizeBoundingBoxes',
                          'ConvertPILImage', 'ConvertBoxes'])
        self.assertEqual([op['type'] for op in val['dataset']['transforms']['ops']],
                         ['PadToCanvas', 'Resize', 'ConvertPILImage'])
        self.assertEqual(cfg['eval_spatial_size'], [1024, 1024])
        self.assertEqual((cfg['epoches'], cfg['flat_epoch'], cfg['no_aug_epoch'], cfg['warmup_iter']), (132, 70, 12, 600))
        self.assertEqual(train['dataset']['transforms']['policy']['epoch'], [4, 70, 120])
        self.assertEqual((train['collate_fn']['stop_epoch'], train['collate_fn']['mixup_epochs']), (120, [4, 70]))
        self.assertEqual(cfg['epoches'] - cfg['no_aug_epoch'], train['collate_fn']['stop_epoch'])
        self.assertEqual(train['total_batch_size'], 12)

    def test_rem_adds_only_rem(self):
        skip = ('__include__', 'output_dir')
        base = {k: v for k, v in load_config(str(CONFIG), {}).items() if k not in skip}
        rem = load_config(str(CONFIG.with_name(CONFIG.name.replace('_132e', '_rem_132e'))), {})
        rem = {k: v for k, v in rem.items() if k not in skip}
        self.assertTrue(rem['HybridEncoder'].pop('use_rem'))
        base['HybridEncoder'].pop('use_rem', None)
        self.assertEqual(rem, base)

    def test_baseline_new_server_changes_only_paths(self):
        old = load_config(str(CONFIG), {})
        new = load_config(str(CONFIG.with_name(CONFIG.name.replace('old_server', 'new_server'))), {})
        for cfg in (old, new):
            cfg.pop('__include__', None)
            cfg.pop('output_dir')
        for loader, name in (('train_dataloader', 'trainval.json'), ('val_dataloader', 'val.json')):
            self.assertEqual(new[loader]['dataset'].pop('img_folder'), '/root/zxy/datasets/SAR-AIRcraft-1.0')
            self.assertEqual(new[loader]['dataset'].pop('ann_file'), f'/root/zxy/datasets/SAR-AIRcraft-1.0/COCO/{name}')
            old[loader]['dataset'].pop('img_folder')
            old[loader]['dataset'].pop('ann_file')
        self.assertEqual(new, old)

    def test_rem_new_server_changes_only_paths(self):
        old = load_config(str(CONFIG.with_name(CONFIG.name.replace('_132e', '_rem_132e'))), {})
        new = load_config(str(CONFIG.with_name(CONFIG.name.replace('_132e', '_rem_132e').replace('old_server', 'new_server'))), {})
        for cfg in (old, new):
            cfg.pop('__include__', None)
            cfg.pop('output_dir')
        for loader, name in (('train_dataloader', 'trainval.json'), ('val_dataloader', 'val.json')):
            self.assertEqual(new[loader]['dataset'].pop('img_folder'), '/root/zxy/datasets/SAR-AIRcraft-1.0')
            self.assertEqual(new[loader]['dataset'].pop('ann_file'), f'/root/zxy/datasets/SAR-AIRcraft-1.0/COCO/{name}')
            old[loader]['dataset'].pop('img_folder')
            old[loader]['dataset'].pop('ann_file')
        self.assertEqual(new, old)

    def test_speckle_adds_only_speckle(self):
        skip = ('__include__', 'output_dir')
        base = {k: v for k, v in load_config(str(CONFIG), {}).items() if k not in skip}
        speckle = load_config(str(CONFIG.with_name(CONFIG.name.replace('_132e', '_speckle_132e'))), {})
        speckle = {k: v for k, v in speckle.items() if k not in skip}
        transforms = speckle['train_dataloader']['dataset']['transforms']
        op = transforms['ops'].pop(1)
        self.assertEqual(op, dict(type='SpeckleResample', p=.5, window=7, noise_sigma=.8, structure_gain=2., bright_ratio=3.))
        self.assertEqual(transforms['policy']['ops'].pop(), 'SpeckleResample')
        self.assertEqual(speckle, base)
        new = load_config(str(CONFIG.with_name(CONFIG.name.replace('_132e', '_speckle_132e').replace('old_server', 'new_server'))), {})
        self.assertTrue(new['train_dataloader']['dataset']['ann_file'].startswith('/root/zxy/datasets/'))
        self.assertTrue(new['val_dataloader']['dataset']['ann_file'].startswith('/root/zxy/datasets/'))
        self.assertEqual(new['train_dataloader']['dataset']['transforms']['ops'][1]['type'], 'SpeckleResample')

    def test_rot90_adds_only_rot90(self):
        skip = ('__include__', 'output_dir')
        base = {k: v for k, v in load_config(str(CONFIG), {}).items() if k not in skip}
        rotated = load_config(str(CONFIG.with_name(CONFIG.name.replace('_132e', '_rot90_132e'))), {})
        rotated = {k: v for k, v in rotated.items() if k not in skip}
        transforms = rotated['train_dataloader']['dataset']['transforms']
        self.assertEqual(transforms['ops'].pop(1), dict(type='RandomRot90', p=.5, angles=[90, 180, 270]))
        self.assertEqual(transforms['policy']['ops'].pop(), 'RandomRot90')
        self.assertEqual(rotated, base)
        new = load_config(str(CONFIG.with_name(CONFIG.name.replace('_132e', '_rot90_132e').replace('old_server', 'new_server'))), {})
        self.assertTrue(new['train_dataloader']['dataset']['ann_file'].startswith('/root/zxy/datasets/'))
        self.assertTrue(new['val_dataloader']['dataset']['ann_file'].startswith('/root/zxy/datasets/'))
        self.assertEqual(new['train_dataloader']['dataset']['transforms']['ops'][1]['type'], 'RandomRot90')

    def test_canvas_gives_every_image_the_same_scale(self):
        for side in (800, 1500):
            image = PIL.Image.new('RGB', (side, side), 30)
            boxes = convert_to_tv_tensor(torch.tensor([[100., 200., 220., 290.]]), key='boxes', spatial_size=(side, side))
            target = dict(boxes=boxes, orig_size=torch.tensor([side, side]))
            image, target, dataset = PadToCanvas(1500)(image, target, 'dataset')
            self.assertEqual((image.size, dataset), ((1500, 1500), 'dataset'))
            self.assertEqual(target['orig_size'].tolist(), [1500, 1500])
            self.assertEqual(target['boxes'].tolist(), [[100., 200., 220., 290.]])
            image, target = Resize([1024, 1024])(image, target)
            self.assertTrue(torch.allclose(target['boxes'], torch.tensor([[100., 200., 220., 290.]]) * 1024 / 1500, atol=1e-3))
            normalized = ConvertBoxes(fmt='cxcywh', normalize=True)(target['boxes'])
            # postprocessor: normalized x orig_size (canvas) -> original pixels
            restored = torch.tensor([160., 245., 120., 90.])
            self.assertTrue(torch.allclose(normalized[0] * 1500, restored, atol=1e-2))
        with self.assertRaises(ValueError):
            PadToCanvas(1500)(PIL.Image.new('RGB', (1600, 800)), dict(orig_size=torch.tensor([1600, 800])), None)

    def test_head_init_copies_airplane_row(self):
        state = {'decoder.enc_score_head.weight': torch.randn(80, 8), 'decoder.enc_score_head.bias': torch.randn(80),
                 'decoder.dec_score_head.0.weight': torch.randn(80, 8), 'decoder.dec_score_head.0.bias': torch.randn(80),
                 'decoder.denoising_class_embed.weight': torch.randn(81, 8), 'decoder.other.weight': torch.randn(80, 8)}
        new, changed = adapt(state, 7)
        self.assertEqual(len(changed), 5)
        self.assertEqual(new['decoder.enc_score_head.weight'].shape, (7, 8))
        self.assertTrue(torch.equal(new['decoder.dec_score_head.0.bias'], state['decoder.dec_score_head.0.bias'][4].repeat(7)))
        embed = new['decoder.denoising_class_embed.weight']
        self.assertEqual(embed.shape, (8, 8))
        self.assertTrue(torch.equal(embed[:7], state['decoder.denoising_class_embed.weight'][4].repeat(7, 1)))
        self.assertTrue(torch.equal(embed[7], state['decoder.denoising_class_embed.weight'][80]))
        self.assertIs(new['decoder.other.weight'], state['decoder.other.weight'])


if __name__ == '__main__':
    unittest.main()
