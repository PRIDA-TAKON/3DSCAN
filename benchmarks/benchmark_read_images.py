import time
import numpy as np
import os
import sys

# Add scripts to path to import the module
sys.path.append(os.path.join(os.path.dirname(__file__), "../scripts"))
import step2_colmap_sfm

def read_images_text_baseline(path):
    images = {}
    with open(path, "r") as f:
        while True:
            line = f.readline()
            if not line: break
            if line.startswith("#") or not line.strip(): continue

            # Line 1: Image ID, Qvec, Tvec, Camera ID, Name
            els = line.split()
            image_id = int(els[0])
            qvec = np.array([float(x) for x in els[1:5]])
            tvec = np.array([float(x) for x in els[5:8]])
            camera_id = int(els[8])
            image_name = els[9]

            # Line 2: Points 2D (discard)
            f.readline()

            images[image_id] = {
                "qvec": qvec, "tvec": tvec, "camera_id": camera_id, "name": image_name
            }
    return images

def generate_large_images_txt(path, num_images=10000):
    with open(path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {num_images}, mean observations per image: 2\n")

        for i in range(1, num_images + 1):
            # Image line
            f.write(f"{i} 0.9 0.1 0.1 0.1 1.0 2.0 3.0 1 image_{i}.jpg\n")
            # Points line (dummy data, but long enough to matter)
            points = " ".join(["100 200 -1"] * 100)
            f.write(f"{points}\n")

if __name__ == "__main__":
    test_file = "benchmarks/images_large.txt"
    if not os.path.exists(test_file):
        print("Generating test file...")
        generate_large_images_txt(test_file, num_images=50000)

    print("Running baseline...")
    start = time.time()
    res_base = read_images_text_baseline(test_file)
    end = time.time()
    print(f"Baseline: {end - start:.4f} seconds")

    print("Running step2_colmap_sfm.read_images_text (Optimized)...")
    start = time.time()
    res_opt = step2_colmap_sfm.read_images_text(test_file)
    end = time.time()
    print(f"Optimized: {end - start:.4f} seconds")

    def read_images_text_manual_opt(path):
        images = {}
        with open(path, "r") as f:
            while True:
                line = f.readline()
                if not line: break
                if line.startswith("#") or not line.strip(): continue

                els = line.split()
                image_id = int(els[0])

                # Direct conversion from strings to numpy array
                # This should be faster than list comprehension with float()
                qvec = np.array(els[1:5], dtype=np.float64)
                tvec = np.array(els[5:8], dtype=np.float64)

                camera_id = int(els[8])
                image_name = els[9]

                f.readline()

                images[image_id] = {
                    "qvec": qvec, "tvec": tvec, "camera_id": camera_id, "name": image_name
                }
        return images

    print("Running Manual Opt...")
    start = time.time()
    res_manual = read_images_text_manual_opt(test_file)
    end = time.time()
    print(f"Manual Opt: {end - start:.4f} seconds")

    def read_images_text_splitlines(path):
        with open(path, "r") as f:
            lines = f.read().splitlines()

        lines = [l for l in lines if not l.startswith("#") and l.strip()]
        image_lines = lines[0::2]

        images = {}
        for line in image_lines:
            els = line.split()
            image_id = int(els[0])
            qvec = np.array([float(x) for x in els[1:5]])
            tvec = np.array([float(x) for x in els[5:8]])
            cam_id = int(els[8])
            name = els[9]
            images[image_id] = {'qvec': qvec, 'tvec': tvec, 'camera_id': cam_id, 'name': name}

        return images

    print("Running Splitlines...")
    start = time.time()
    res_split = read_images_text_splitlines(test_file)
    end = time.time()
    print(f"Splitlines: {end - start:.4f} seconds")

    # Verify correctness
    assert len(res_base) == len(res_opt)
    k = list(res_base.keys())[0]
    np.testing.assert_allclose(res_base[k]["qvec"], res_opt[k]["qvec"])
    np.testing.assert_allclose(res_base[k]["tvec"], res_opt[k]["tvec"])
    assert res_base[k]["name"] == res_opt[k]["name"]
    print("Verification passed!")
