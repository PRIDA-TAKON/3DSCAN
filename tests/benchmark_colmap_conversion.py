
import sys
import os
import time
import numpy as np

# Add repo root to path
sys.path.append(os.getcwd())

# We import the modified module to test the new implementation later,
# but for baseline we use a local copy of the old function.
try:
    import scripts.step2_colmap_sfm as step2
except ImportError:
    step2 = None

def old_qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def generate_dummy_data(n_images=10000):
    images = {}
    for i in range(n_images):
        q = np.random.rand(4)
        q /= np.linalg.norm(q)
        t = np.random.rand(3)
        images[i] = {
            "qvec": q,
            "tvec": t,
            "camera_id": 1,
            "name": f"image_{i}.jpg"
        }
    return images

def original_implementation(images):
    sorted_image_ids = sorted(images.keys())
    flip_mat = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    frames = []
    start_time = time.time()
    for img_id in sorted_image_ids:
        img = images[img_id]
        R = old_qvec2rotmat(img["qvec"])
        t = img["tvec"]
        c2w = np.eye(4)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ t
        c2w = c2w @ flip_mat
        frame = {
            "file_path": f"images/{img['name']}",
            "transform_matrix": c2w.tolist()
        }
        frames.append(frame)
    end_time = time.time()
    return end_time - start_time, frames

def test_new_implementation(images):
    # This function will call the actual function in scripts/step2_colmap_sfm.py
    # We need to mock the inputs to convert_colmap_to_transforms,
    # but convert_colmap_to_transforms reads from file.
    # So instead we should probably extract the logic we are changing into a new function
    # in step2_colmap_sfm.py or just copy the logic here to verify my proposed changes work.

    # Since I can't easily call convert_colmap_to_transforms without files,
    # I will replicate the PROPOSED new logic here.

    sorted_image_ids = sorted(images.keys())
    flip_mat = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])

    start_time = time.time()

    n_images = len(sorted_image_ids)
    qvecs = np.empty((n_images, 4))
    tvecs = np.empty((n_images, 3))
    names = []

    for i, img_id in enumerate(sorted_image_ids):
        img = images[img_id]
        qvecs[i] = img["qvec"]
        tvecs[i] = img["tvec"]
        names.append(img["name"])

    # Proposed vectorized qvec2rotmat logic
    # We can use the one I implemented in previous turn

    q0, q1, q2, q3 = qvecs[:, 0], qvecs[:, 1], qvecs[:, 2], qvecs[:, 3]
    Rs = np.zeros((n_images, 3, 3))
    Rs[:, 0, 0] = 1 - 2 * q2**2 - 2 * q3**2
    Rs[:, 0, 1] = 2 * q1 * q2 - 2 * q0 * q3
    Rs[:, 0, 2] = 2 * q3 * q1 + 2 * q0 * q2
    Rs[:, 1, 0] = 2 * q1 * q2 + 2 * q0 * q3
    Rs[:, 1, 1] = 1 - 2 * q1**2 - 2 * q3**2
    Rs[:, 1, 2] = 2 * q2 * q3 - 2 * q0 * q1
    Rs[:, 2, 0] = 2 * q3 * q1 - 2 * q0 * q2
    Rs[:, 2, 1] = 2 * q2 * q3 + 2 * q0 * q1
    Rs[:, 2, 2] = 1 - 2 * q1**2 - 2 * q2**2

    Rs_T = Rs.transpose(0, 2, 1)

    c2w = np.eye(4)[None, :, :].repeat(n_images, axis=0)
    c2w[:, :3, :3] = Rs_T
    t_transformed = -Rs_T @ tvecs[:, :, None]
    c2w[:, :3, 3] = t_transformed[:, :, 0]

    c2w = c2w @ flip_mat[None, :, :]

    frames = []
    c2w_list = c2w.tolist()
    for i in range(n_images):
        frame = {
            "file_path": f"images/{names[i]}",
            "transform_matrix": c2w_list[i]
        }
        frames.append(frame)

    end_time = time.time()
    return end_time - start_time, frames

if __name__ == "__main__":
    n_images = 10000
    print(f"Generating {n_images} dummy images...")
    images = generate_dummy_data(n_images)

    print("Running original implementation...")
    t_orig, frames_orig = original_implementation(images)
    print(f"Original implementation took {t_orig:.4f} seconds")

    print("Running new implementation (proposed logic)...")
    t_vec, frames_vec = test_new_implementation(images)
    print(f"New implementation took {t_vec:.4f} seconds")

    print(f"Speedup: {t_orig / t_vec:.2f}x")

    # Verify correctness
    print("Verifying correctness...")
    assert len(frames_orig) == len(frames_vec)
    for i in range(len(frames_orig)):
        f1 = frames_orig[i]
        f2 = frames_vec[i]
        assert f1["file_path"] == f2["file_path"]
        m1 = np.array(f1["transform_matrix"])
        m2 = np.array(f2["transform_matrix"])
        if not np.allclose(m1, m2, atol=1e-6):
            print(f"Mismatch at index {i}")
            print("Original:", m1)
            print("Vectorized:", m2)
            sys.exit(1)

    print("✅ Correctness verified!")
