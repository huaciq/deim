import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "prepare_ogsod", Path(__file__).resolve().parents[1] /
    "tools/dataset/prepare_ogsod_baseline.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        (self.images / "a.jpg").touch()
        self.source = self.root / "train.json"
        self.mapping = {1: (0, "a"), 2: (1, "b"), 3: (2, "c")}
        self.data = {
            "categories": [{"id": key, "name": value[1]}
                           for key, value in self.mapping.items()],
            "images": [{"id": 1, "file_name": "train/a.jpg"}],
            "annotations": [{"id": key, "image_id": 1, "category_id": key}
                            for key in self.mapping],
        }

    def save(self):
        self.source.write_text(json.dumps(self.data), encoding="utf-8")

    def test_remaps_and_resolves_without_modifying_source(self):
        self.save()
        original = self.source.read_bytes()
        data, paths = module.prepare(self.source, self.images, self.mapping)
        self.assertEqual([ann["category_id"] for ann in data["annotations"]], [0, 1, 2])
        self.assertEqual(data["images"][0]["file_name"], "a.jpg")
        self.assertEqual(paths, {(self.images / "a.jpg").resolve()})
        self.assertEqual(self.source.read_bytes(), original)

    def test_missing_image_rejected(self):
        self.data["images"][0]["file_name"] = "missing.jpg"
        self.save()
        with self.assertRaises(FileNotFoundError):
            module.prepare(self.source, self.images, self.mapping)

    def test_inconsistent_categories_rejected(self):
        self.data["categories"][0]["name"] = "different"
        self.save()
        with self.assertRaises(ValueError):
            module.prepare(self.source, self.images, self.mapping)


if __name__ == "__main__":
    unittest.main()
