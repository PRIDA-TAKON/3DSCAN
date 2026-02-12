
import unittest
import os
import shutil
import json
from pathlib import Path
import sys

# Add repo root to path
sys.path.append(os.getcwd())

from scripts.step2_colmap_sfm import convert_colmap_to_transforms

class TestEmptyInput(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("tests/temp_colmap_empty")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.colmap_dir = self.test_dir / "colmap"
        self.colmap_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.test_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.test_dir / "transforms.json"

        # Create dummy cameras.txt
        with open(self.colmap_dir / "cameras.txt", "w") as f:
            f.write("# Camera list\n")
            f.write("1 PINHOLE 800 600 800 800 400 300\n")

        # Create EMPTY images.txt
        with open(self.colmap_dir / "images.txt", "w") as f:
            f.write("# Image list\n")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_empty_conversion(self):
        success = convert_colmap_to_transforms(self.colmap_dir, self.images_dir, self.output_path)
        self.assertTrue(success)
        self.assertTrue(self.output_path.exists())

        with open(self.output_path, "r") as f:
            data = json.load(f)

        self.assertIn("frames", data)
        self.assertEqual(len(data["frames"]), 0)

if __name__ == "__main__":
    unittest.main()
