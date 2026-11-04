"""RandomRot90: boxes stay on the object, orientations covered, epoch policy, protocol pipeline."""

import unittest

import numpy as np
import PIL.Image
import torch

from engine.data._misc import BoundingBoxes, convert_to_tv_tensor
from engine.data.transforms._transforms import ConvertBoxes, PadToCanvas, RandomRot90, Resize
from engine.data.transforms.container import Compose

WIDTH, HEIGHT = 40, 30
BOX = [5., 3., 17., 9.]  # asymmetric: every rotation and flip gives a different image


def sample():
    pixels = np.zeros((HEIGHT, WIDTH), np.uint8)
    pixels[3:9, 5:17] = 200
    pixels[3:5, 5:8] = 255  # marks one corner of the object
    boxes = convert_to_tv_tensor(torch.tensor([BOX]), key='boxes', spatial_size=(HEIGHT, WIDTH))
    return PIL.Image.fromarray(pixels), dict(boxes=boxes, orig_size=torch.tensor([WIDTH, HEIGHT]), area=torch.tensor([72.]))


def pixel_box(image):
    ys, xs = np.nonzero(np.asarray(image))
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


class Dataset:
    def __init__(self, epoch):
        self.epoch = epoch


class RandomRot90Tests(unittest.TestCase):
    def test_boxes_stay_tight_on_the_object(self):
        for angle in (90, 180, 270):
            image, target = sample()
            result, out, dataset = RandomRot90(p=1., angles=(angle,))(image, target, 'dataset')
            self.assertEqual(dataset, 'dataset')
            self.assertEqual(out['boxes'][0].tolist(), pixel_box(result), angle)
            self.assertIsInstance(out['boxes'], BoundingBoxes)
            self.assertEqual(out['orig_size'].tolist(), list(result.size))
            self.assertEqual(result.size, (WIDTH, HEIGHT) if angle == 180 else (HEIGHT, WIDTH))
            self.assertEqual(out['area'].tolist(), [72.])

    def test_empty_boxes_and_probability(self):
        image, target = sample()
        target['boxes'] = convert_to_tv_tensor(torch.zeros((0, 4)), key='boxes', spatial_size=(HEIGHT, WIDTH))
        _, out, _ = RandomRot90(p=1., angles=(90,))(image, target, 'dataset')
        self.assertEqual(out['boxes'].shape, (0, 4))
        same = RandomRot90(p=0.)(image, target, 'dataset')
        self.assertIs(same[0], image)

    def test_flip_and_rotation_reach_all_eight_orientations(self):
        torch.manual_seed(0)
        seen = set()
        for mirrored in (False, True):
            for angle in (None, 90, 180, 270):
                image, target = sample()
                if mirrored:
                    image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                if angle is not None:
                    image = RandomRot90(p=1., angles=(angle,))(image, target, 'dataset')[0]
                seen.add((image.size, np.asarray(image).tobytes()))
        self.assertEqual(len(seen), 8)

    def test_draws_every_configured_angle(self):
        torch.manual_seed(0)
        op = RandomRot90(p=1.)
        results = set()
        for _ in range(60):
            image, target = sample()
            results.add(op(image, target, 'dataset')[0].tobytes())
        self.assertEqual(len(results), 3)

    def test_policy_limits_it_to_the_augmented_epochs(self):
        compose = Compose([{'type': 'RandomRot90', 'p': 1.}],
                          policy=dict(name='stop_epoch', epoch=[4, 70, 120], ops=['Mosaic', 'RandomRot90']))
        for epoch, changed in ((3, False), (4, True), (119, True), (120, False)):
            image, target = sample()
            result = compose(image, target, Dataset(epoch))[0]
            self.assertEqual(result is not image, changed, epoch)

    def test_protocol_pipeline_keeps_the_common_scale(self):
        image = PIL.Image.fromarray(np.pad(np.asarray(sample()[0]), ((0, 770), (0, 760))))  # an 800 x 800 tile
        boxes = convert_to_tv_tensor(torch.tensor([BOX]), key='boxes', spatial_size=(800, 800))
        target = dict(boxes=boxes, orig_size=torch.tensor([800, 800]))
        image, target, _ = RandomRot90(p=1., angles=(270,))(image, target, 'dataset')
        expected = torch.tensor([pixel_box(image)]) * 1024 / 1500
        image, target, _ = PadToCanvas(1500)(image, target, 'dataset')
        image, target = Resize([1024, 1024])(image, target)
        self.assertEqual(target['orig_size'].tolist(), [1500, 1500])
        self.assertTrue(torch.allclose(target['boxes'].as_subclass(torch.Tensor), expected, atol=1e-3))
        normalized = ConvertBoxes(fmt='cxcywh', normalize=True)(target['boxes'])
        self.assertTrue(((normalized >= 0) & (normalized <= 1)).all())


if __name__ == '__main__':
    unittest.main()
