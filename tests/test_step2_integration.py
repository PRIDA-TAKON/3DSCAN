
import unittest
import numpy as np
import os
import shutil
import json
from pathlib import Path
import sys

# Add repo root to path
sys.path.append(os.getcwd())

from scripts.step2_colmap_sfm import convert_colmap_to_transforms, qvec2rotmat

class TestColmapConversion(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("tests/temp_colmap_data")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.colmap_dir = self.test_dir / "colmap"
        self.colmap_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.test_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.test_dir / "transforms.json"

        # Create dummy cameras.txt
        with open(self.colmap_dir / "cameras.txt", "w") as f:
            f.write("# Camera list with one line of data per camera\n")
            f.write("1 PINHOLE 800 600 800 800 400 300\n")

        # Create dummy images.txt
        with open(self.colmap_dir / "images.txt", "w") as f:
            f.write("# Image list with two lines of data per image\n")
            # Image 1
            # Qvec: identity (1, 0, 0, 0) -> R = I
            # Tvec: (1, 2, 3)
            # R.T = I
            # -R.T @ t = -(1, 2, 3) = (-1, -2, -3)
            # c2w = [[1, 0, 0, -1], [0, 1, 0, -2], [0, 0, 1, -3], [0, 0, 0, 1]]
            # Apply flip_mat (diag(1, -1, -1, 1))
            # c2w @ flip = [[1, 0, 0, -1], [0, -1, 0, -2], [0, 0, -1, -3], [0, 0, 0, 1]]
            f.write("1 1 0 0 0 1 2 3 1 image1.jpg\n")
            f.write("0 0 0\n")

            # Image 2
            # Qvec: 180 deg rotation around X axis -> (0, 1, 0, 0)
            # R = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
            # Tvec: (0, 0, 0)
            # R.T = R
            # -R.T @ t = 0
            # c2w = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
            # Apply flip_mat
            # c2w @ flip = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            f.write("2 0 1 0 0 0 0 0 1 image2.jpg\n")
            f.write("0 0 0\n")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_conversion(self):
        success = convert_colmap_to_transforms(self.colmap_dir, self.images_dir, self.output_path)
        self.assertTrue(success)
        self.assertTrue(self.output_path.exists())

        with open(self.output_path, "r") as f:
            data = json.load(f)

        frames = data["frames"]
        self.assertEqual(len(frames), 2)

        # Check Image 1
        f1 = frames[0]
        self.assertEqual(f1["file_path"], "images/image1.jpg")
        m1 = np.array(f1["transform_matrix"])
        expected_m1 = np.array([
            [1, 0, 0, -1],
            [0, -1, 0, -2],
            [0, 0, -1, -3],
            [0, 0, 0, 1]
        ])
        np.testing.assert_allclose(m1, expected_m1, atol=1e-5)

        # Check Image 2
        f2 = frames[1]
        self.assertEqual(f2["file_path"], "images/image2.jpg")
        m2 = np.array(f2["transform_matrix"])
        expected_m2 = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        np.testing.assert_allclose(m2, expected_m2, atol=1e-5)

if __name__ == "__main__":
    unittest.main()
