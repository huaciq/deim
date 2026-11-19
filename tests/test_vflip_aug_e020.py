"""E020 = E000-old + RandomVerticalFlip, and the run-comparison tool used to read it.

Run on CPU: python -m unittest tests.test_vflip_aug_e020 -v
"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torchvision import tv_tensors

from engine.core.workspace import GLOBAL_CONFIG
from engine.core.yaml_utils import load_config
from engine.data.transforms._transforms import RandomVerticalFlip
from tools.sardet100k import compare_runs

FOLDER = Path(__file__).resolve().parents[1] / 'configs/deim_dfine'
OLD = FOLDER / 'deim_hgnetv2_m_sardet100k_36e_b32_nozoom_nocrop_old_server.yml'
E020 = FOLDER / 'deim_hgnetv2_m_sardet100k_vflip_aug_36e_b32_old_server.yml'
META = {'__include__', 'output_dir', 'experiment_protocol_document', 'experiment_baseline_revision'}


def flat(cfg, prefix=''):
    out = {}
    for key, value in cfg.items():
        out.update(flat(value, f'{prefix}{key}.') if isinstance(value, dict) else {f'{prefix}{key}': value})
    return out


class Config(unittest.TestCase):
    def test_only_a_vertical_flip_is_added_after_the_horizontal_flip(self):
        a, b = load_config(str(OLD)), load_config(str(E020))
        fa, fb = flat(a), flat(b)
        diff = {k for k in set(fa) | set(fb) if fa.get(k, '<none>') != fb.get(k, '<none>')} - META
        self.assertEqual(diff, {'train_dataloader.dataset.transforms.ops'})
        ops_a = a['train_dataloader']['dataset']['transforms']['ops']
        ops_b = b['train_dataloader']['dataset']['transforms']['ops']
        at = [o['type'] for o in ops_a].index('RandomHorizontalFlip') + 1
        self.assertEqual(ops_b, ops_a[:at] + [{'type': 'RandomVerticalFlip'}] + ops_a[at:])
        self.assertTrue(b['output_dir'].startswith('/data2/zxy/sar/experiments/'))
        self.assertTrue(b['train_dataloader']['dataset']['img_folder'].startswith('/data2/zxy/sar/datasets/SARDet_100K/'))
        self.assertNotIn('RandomVerticalFlip', b['train_dataloader']['dataset']['transforms']['policy']['ops'])


class Transform(unittest.TestCase):
    def test_registered_and_flips_image_and_boxes(self):
        self.assertIn('RandomVerticalFlip', GLOBAL_CONFIG)
        image = tv_tensors.Image(torch.arange(80.).view(1, 10, 8))
        boxes = tv_tensors.BoundingBoxes([[1., 2., 3., 5.]], format='XYXY', canvas_size=(10, 8))
        out_image, out = RandomVerticalFlip(p=1.0)(image, {'boxes': boxes, 'labels': torch.tensor([2])})
        torch.testing.assert_close(out_image.as_subclass(torch.Tensor), image.flip(-2).as_subclass(torch.Tensor))
        torch.testing.assert_close(out['boxes'].as_subclass(torch.Tensor), torch.tensor([[1., 5., 3., 8.]]))
        self.assertEqual(out['labels'].tolist(), [2])


class CompareRuns(unittest.TestCase):
    def _run(self, folder, name, scale, best_last):
        run = Path(folder) / name
        (run / 'eval').mkdir(parents=True)
        (run / 'test_eval').mkdir()
        rows = [{'epoch': e, 'test_coco_eval_bbox': [scale * (0.5 + 0.01 * e)] * 12} for e in range(3)]
        if not best_last:
            rows[-1]['test_coco_eval_bbox'] = [0.0] * 12
        (run / 'log.txt').write_text('\n'.join(json.dumps(r) for r in rows))
        params = SimpleNamespace(areaRngLbl=['all', 'small', 'medium', 'large'], maxDets=[1, 10, 100],
                                 iouThrs=np.linspace(.5, .95, 10), catIds=[0, 1], imgIds=list(range(5)))
        precision = np.ones((10, 101, 2, 4, 3)) * scale
        precision[:, :, 1] *= 0.5
        precision[:, :, 0, 1] = -1          # no small objects in class 0
        recall = np.ones((10, 2, 4, 3)) * scale
        for target in (run / 'eval/latest.pth', run / 'test_eval/eval.pth'):
            torch.save({'params': params, 'precision': precision, 'recall': recall}, target)
        return run

    def test_reads_best_val_epoch_per_class_and_test(self):
        with tempfile.TemporaryDirectory() as folder:
            ann = Path(folder) / 'test.json'
            ann.write_text(json.dumps({'categories': [{'id': 0, 'name': 'ship'}, {'id': 1, 'name': 'tank'}]}))
            a = self._run(folder, 'a', 0.8, True)
            b = self._run(folder, 'b', 0.6, False)
            records = compare_runs.main(['--test-ann', str(ann), '--run', f'A={a}:test_eval',
                                         '--run', f'B={b}:test_eval'])
        self.assertAlmostEqual(records[0]['val']['AP'], 80 * 0.52)
        self.assertEqual(records[0]['best_epoch_one_based'], 3)
        self.assertEqual(records[1]['best_epoch_one_based'], 2)
        self.assertIn('val_per_class', records[0])
        self.assertNotIn('val_per_class', records[1])     # best epoch is not the last one
        test = records[0]['test']
        self.assertAlmostEqual(test['per_class']['ship'], 80)
        self.assertAlmostEqual(test['per_class']['tank'], 40)
        self.assertAlmostEqual(test['AP'], 60)
        self.assertAlmostEqual(test['APs'], 40)            # only class 1 has small objects
        self.assertEqual(test['images'], 5)


if __name__ == '__main__':
    unittest.main()
