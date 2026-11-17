"""E019 must equal E000 except that HGNetv2 starts from local ImageNet stage1 weights (no -t).

Run on CPU: python -m unittest tests.test_imagenet_init_e019 -v
"""
import tempfile
import unittest
from pathlib import Path

import torch

from engine.backbone.hgnetv2 import HGNetv2
from engine.core.yaml_utils import load_config

FOLDER = Path(__file__).resolve().parents[1] / 'configs/deim_dfine'
E000 = FOLDER / 'deim_hgnetv2_m_sardet100k_36e_b32_nozoom_nocrop.yml'
E019 = FOLDER / 'deim_hgnetv2_m_sardet100k_imagenet_init_36e_b32.yml'
META = {'__include__', 'output_dir', 'experiment_protocol_document', 'experiment_baseline_revision'}


def flat(cfg, prefix=''):
    out = {}
    for key, value in cfg.items():
        out.update(flat(value, f'{prefix}{key}.') if isinstance(value, dict) else {f'{prefix}{key}': value})
    return out


class Config(unittest.TestCase):
    def test_only_backbone_initialisation_differs_from_e000(self):
        a, b = flat(load_config(str(E000))), flat(load_config(str(E019)))
        diff = {k for k in set(a) | set(b) if a.get(k, '<none>') != b.get(k, '<none>')} - META
        self.assertEqual(diff, {'HGNetv2.pretrained', 'HGNetv2.local_model_dir'})
        self.assertFalse(a['HGNetv2.pretrained'])
        self.assertTrue(b['HGNetv2.pretrained'])
        self.assertEqual(b['HGNetv2.local_model_dir'], '/root/zxy/pretrain/hgnetv2/')
        self.assertEqual(b['HGNetv2.name'], 'B2')
        self.assertEqual((b['HGNetv2.freeze_at'], b['HGNetv2.freeze_norm']), (-1, False))
        self.assertTrue(b['output_dir'].startswith('/root/zxy/experiments/'))


class Loading(unittest.TestCase):
    def test_pretrained_backbone_is_read_from_local_model_dir(self):
        kwargs = dict(name='B2', use_lab=True, return_idx=[1, 2, 3], freeze_at=-1, freeze_norm=False)
        torch.manual_seed(1)
        source = HGNetv2(pretrained=False, **kwargs)
        with tempfile.TemporaryDirectory() as folder:
            torch.save(source.state_dict(), Path(folder) / 'PPHGNetV2_B2_stage1.pth')
            torch.manual_seed(2)
            loaded = HGNetv2(pretrained=True, local_model_dir=folder + '/', **kwargs)
        torch.manual_seed(2)
        fresh = HGNetv2(pretrained=False, **kwargs)
        for (name, a), b, c in zip(source.state_dict().items(), loaded.state_dict().values(), fresh.state_dict().values()):
            torch.testing.assert_close(a, b, atol=0, rtol=0, msg=name)
        self.assertTrue(any(not torch.equal(a, c) for a, c in zip(source.state_dict().values(), fresh.state_dict().values())))


if __name__ == '__main__':
    unittest.main()
