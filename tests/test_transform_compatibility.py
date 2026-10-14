import unittest

import torch
from PIL import Image

from engine.data._misc import convert_to_tv_tensor
from engine.data.transforms._transforms import ConvertBoxes, ConvertPILImage


class TransformCompatibilityTests(unittest.TestCase):
    def test_image_conversion_via_both_hooks_and_forward(self):
        image = Image.new("RGB", (20, 10), color=(255, 128, 0))
        transform = ConvertPILImage(dtype="float32", scale=True)
        expected = torch.tensor([1.0, 128 / 255, 0.0])
        for output in (transform(image), transform.transform(image, {}),
                       transform._transform(image, {})):
            self.assertEqual(tuple(output.shape), (3, 10, 20))
            self.assertEqual(output.dtype, torch.float32)
            torch.testing.assert_close(output[:, 0, 0], expected)

    def test_box_conversion_via_both_hooks_and_forward(self):
        boxes = convert_to_tv_tensor(torch.tensor([[2., 1., 10., 5.]]),
                                     key="boxes", box_format="XYXY",
                                     spatial_size=(10, 20))
        transform = ConvertBoxes(fmt="cxcywh", normalize=True)
        expected = torch.tensor([[0.3, 0.3, 0.4, 0.4]])
        for output in (transform(boxes), transform.transform(boxes, {}),
                       transform._transform(boxes, {})):
            torch.testing.assert_close(output, expected)


if __name__ == "__main__":
    unittest.main()
