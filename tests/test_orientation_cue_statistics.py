"""Regression for the unequal-area halves in the D18 brightness statistic."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/sardet100k'))
from orientation_cue_check import work


class Brightness(unittest.TestCase):
    def test_uniform_odd_even_and_one_pixel_boxes_have_zero_asymmetry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'uniform.png'
            Image.fromarray(np.full((24, 24), 100, np.uint8)).save(path)
            _, _, stats = work((str(path), [[5, 5, s, s] for s in (1, 5, 6, 7)]))
            np.testing.assert_array_equal(stats[:, :2], np.zeros((4, 2)))


if __name__ == '__main__':
    unittest.main()
