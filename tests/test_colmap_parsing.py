
import unittest
import numpy as np
import tempfile
import os
import sys
from pathlib import Path

# Add scripts to path to import the module
sys.path.append(os.path.join(os.path.dirname(__file__), "../scripts"))
import step2_colmap_sfm

class TestColmapParsing(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.images_txt = Path(self.test_dir.name) / "images.txt"

    def tearDown(self):
        self.test_dir.cleanup()

    def create_dummy_images_txt(self, content):
        with open(self.images_txt, "w") as f:
            f.write(content)

    def test_read_images_text_basic(self):
        content = """# Header
# Header 2
1 0.9 0.1 0.1 0.1 1.0 2.0 3.0 1 image01.jpg
100 200 -1
2 0.8 0.2 0.2 0.2 1.5 2.5 3.5 1 image02.jpg
120 220 5
"""
        self.create_dummy_images_txt(content)
        images = step2_colmap_sfm.read_images_text(self.images_txt)

        self.assertEqual(len(images), 2)
        self.assertIn(1, images)
        self.assertIn(2, images)

        img1 = images[1]
        np.testing.assert_allclose(img1["qvec"], [0.9, 0.1, 0.1, 0.1])
        np.testing.assert_allclose(img1["tvec"], [1.0, 2.0, 3.0])
        self.assertEqual(img1["camera_id"], 1)
        self.assertEqual(img1["name"], "image01.jpg")

    def test_read_images_text_single(self):
        content = """1 0.9 0.1 0.1 0.1 1.0 2.0 3.0 1 image01.jpg
100 200 -1
"""
        self.create_dummy_images_txt(content)
        images = step2_colmap_sfm.read_images_text(self.images_txt)
        self.assertEqual(len(images), 1)

    def test_read_images_text_empty(self):
        content = """# Header only
"""
        self.create_dummy_images_txt(content)
        images = step2_colmap_sfm.read_images_text(self.images_txt)
        self.assertEqual(len(images), 0)

    def test_read_images_text_empty_points(self):
        # Case where points line is just a newline (no points)
        content = """# Header
1 0.9 0.1 0.1 0.1 1.0 2.0 3.0 1 image01.jpg

2 0.8 0.2 0.2 0.2 1.5 2.5 3.5 1 image02.jpg
120 220 5
"""
        self.create_dummy_images_txt(content)
        images = step2_colmap_sfm.read_images_text(self.images_txt)
        self.assertEqual(len(images), 2)
        self.assertIn(1, images)
        self.assertIn(2, images)

if __name__ == "__main__":
    unittest.main()
