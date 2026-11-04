"""SpeckleResample: statistics of the new speckle, kept structure and epoch policy; its check tool."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import PIL.Image
import torch

from engine.data.transforms._transforms import SpeckleResample
from engine.data.transforms.container import Compose
from tests.test_sar_characteristics import write_split
from tools.check_speckle_resampling import analyze


def scene(seed=0, size=256, level=80., looks=6.):
    """Homogeneous gamma clutter, a black strip and one strong point target."""
    rng = np.random.default_rng(seed)
    values = level * rng.gamma(looks, 1 / looks, (size, size))
    values[:, -40:] = 0
    values[100:104, 100:104] = 250
    return np.clip(values, 0, 255).astype(np.float32)


class Dataset:
    def __init__(self, epoch):
        self.epoch = epoch


class SpeckleResampleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.values = scene()
        self.op = SpeckleResample(p=1., noise_sigma=0.)
        self.new = self.op.resample(torch.from_numpy(self.values)).numpy()

    def test_looks_match_the_scene(self):
        self.assertAlmostEqual(self.op.looks(torch.from_numpy(self.values)), 6., delta=1.)

    def test_keeps_mean_and_speckle_strength(self):
        old, new = self.values[10:90, 10:200], self.new[10:90, 10:200]
        self.assertAlmostEqual(new.mean() / old.mean(), 1., delta=.03)
        self.assertAlmostEqual(new.std() / old.std(), 1., delta=.25)

    def test_changes_clutter_but_keeps_black_and_point_target(self):
        old, new = self.values[10:90, 10:200].ravel(), self.new[10:90, 10:200].ravel()
        self.assertLess(np.corrcoef(old, new)[0, 1], .75)
        self.assertEqual(float(self.new[:, -30:].max()), 0.)
        self.assertGreater(self.new[100:104, 100:104].mean(), 200.)

    def test_noise_correlation_follows_sigma_with_unit_variance(self):
        for sigma, low, high in ((0., -.05, .05), (.8, .4, .85)):
            field = SpeckleResample(noise_sigma=sigma).noise(400, 400)
            self.assertAlmostEqual(float(field.std()), 1., delta=.05)
            lag = float(torch.corrcoef(torch.stack([field[:, :-1].ravel(), field[:, 1:].ravel()]))[0, 1])
            self.assertTrue(low < lag < high, (sigma, lag))

    def test_nakagami_field_reproduces_the_measured_looks(self):
        self.assertAlmostEqual(SpeckleResample.nakagami_shape(3.66), 1., delta=.03)  # Rayleigh amplitude
        for looks in (2.5, 4.6, 7., 20.):
            field = SpeckleResample(noise_sigma=.8).speckle_field(looks, 500, 500)
            self.assertAlmostEqual(float(field.mean()), 1., delta=.03)
            self.assertAlmostEqual(float((field.mean() / field.std()) ** 2), looks, delta=looks * .15)

    def test_pil_sample_keeps_mode_size_and_target(self):
        for mode in ('L', 'RGB'):
            image = PIL.Image.fromarray(self.values.astype(np.uint8)).convert(mode)
            target = dict(boxes=torch.tensor([[1., 2., 3., 4.]]))
            result, out, dataset = SpeckleResample(p=1.)(image, target, 'dataset')
            self.assertEqual((result.mode, result.size, dataset), (mode, image.size, 'dataset'))
            self.assertIs(out, target)
            self.assertFalse(np.array_equal(np.asarray(result), np.asarray(image)))
        same, _, _ = SpeckleResample(p=0.)(image, target, 'dataset')
        self.assertIs(same, image)

    def test_black_image_is_returned_unchanged(self):
        image = PIL.Image.new('L', (64, 64))
        self.assertIs(SpeckleResample(p=1.)(image, {}, 'd')[0], image)

    def test_policy_limits_it_to_the_augmented_epochs(self):
        compose = Compose([{'type': 'SpeckleResample', 'p': 1.}],
                          policy=dict(name='stop_epoch', epoch=[4, 70, 120], ops=['Mosaic', 'SpeckleResample']))
        image = PIL.Image.fromarray(self.values.astype(np.uint8))
        for epoch, changed in ((3, False), (4, True), (100, True), (119, True), (120, False)):
            result = compose(image, {}, Dataset(epoch))[0]
            self.assertEqual(result is not image, changed, epoch)

    def test_check_tool_reports_every_variant(self):
        with tempfile.TemporaryDirectory(prefix='deim_speckle_') as folder:
            data = Path(folder) / 'data'
            data.mkdir()
            write_split(data, 'trainval', [(1, [(0, 4, 4, 10, 10, 200, 7)]), (2, [(1, 30, 30, 12, 12, 200, 9)])])
            summary = analyze(data, Path(folder) / 'out', images=2, sigmas=(0.5, 1.), workers=1)
            self.assertEqual(summary['images'], 2)
            self.assertEqual(set(summary['variants']), {'stored', 'sigma=0.5', 'sigma=1'})
            for name in ('report.md', 'summary.json', 'sheet.png'):
                self.assertTrue((Path(folder) / 'out' / name).is_file())
            with self.assertRaises(FileExistsError):
                analyze(data, Path(folder) / 'out', images=2, sigmas=(0.5,))


if __name__ == '__main__':
    unittest.main()
