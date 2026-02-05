import os
import re
import shutil
import sys
import glob
import subprocess
import argparse
from pathlib import Path
import json
import importlib.util
import concurrent.futures
import math
import numpy as np

print("✅ Imports complete")

# ================= CONFIGURATION =================
PROJECT_NAME = "3d_scan"
# Function to find input video dynamically
def find_input_video():
    print("🔍 Searching for input video...")
    search_paths = [Path("/kaggle/input"), Path("input")]
    
    for search_path in search_paths:
        if search_path.exists():
            # Find first mp4 file recursively
            video = next(search_path.rglob("*.mp4"), None)
            if video:
                print(f"✅ Found video: {video}")
                return video
    
    print("❌ No .mp4 video found in /kaggle/input or local input/")
    return None

# Initial placeholder, allows override
VIDEO_INPUT_PATH = find_input_video()
WORKING_DIR = Path("/kaggle/working")
PROJECT_DIR = WORKING_DIR / PROJECT_NAME
DATABASE_PATH = PROJECT_DIR / "database.db"
IMAGES_DIR = PROJECT_DIR / "images"
SPARSE_PATH = PROJECT_DIR / "sparse"
OUTPUTS_DIR = Path("outputs") / PROJECT_NAME / "taichi_splatting"

# Environment tweaks
os.environ['MAX_JOBS'] = '1' # Prevent freezing on Kaggle

def run_command(cmd, shell=False):
    """Runs a shell command and raises an exception if it fails."""
    print(f"🚀 Running: {cmd}")
    try:
        if shell:
            subprocess.run(cmd, shell=True, check=True)
        else:
            if isinstance(cmd, str) and not shell:
                cmd = cmd.split()
            subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {cmd}")
        raise e

def check_gpu():
    print("🔍 Checking GPU availability...")
    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠️" * 20)
            print("⚠️ WARNING: GPU Not Detected!")
            print("⚠️ This script requires a GPU (P100 or T4) to run effectively.")
            print("⚠️ Please enable GPU Accelerator in your Kaggle Notebook settings.")
            print("⚠️" * 20)
            return False
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
        return True
    except ImportError:
        print("⚠️ torch module not found. Cannot check GPU availability.")
        return False

def check_numpy_integrity():
    """
    Checks if numpy is corrupted (e.g. from mixed environment files) by importing specific attributes.
    Returns True if healthy, False if corrupted.
    """
    try:
        # This specific import often fails if numpy is corrupted/mixed
        from numpy.lib.stride_tricks import broadcast_to
        return True
    except (ImportError, AttributeError, RuntimeError):
        return False

def install_dependencies():
    print("⏳ Installing dependencies (Manual Pipeline Mode)...")

    # 1. Upgrade pip first
    run_command("pip install --upgrade pip", shell=True)

    # 2. Force Upgrade Numpy to 2.x and essential libs to compatible versions
    print("🚀 Force upgrading core libraries for Numpy 2.0 compatibility...")
    libs_to_upgrade = [
        "numpy>=2.0",
        "numba",
        "scipy",
        "pandas",
        "scikit-learn",
        "opencv-python",
        "opencv-python-headless", 
        "opencv-contrib-python",
        "matplotlib",
        "pillow",
        "plyfile",
        "tqdm"
    ]
    run_command(f"pip install --upgrade {' '.join(libs_to_upgrade)}", shell=True)

    # 3. Install Taichi (Stable)
    run_command("pip install taichi", shell=True)

    # 4. Install Taichi Splatting from source (Patched for stable taichi)
    print("⏳ Installing taichi-splatting from source...")
    if os.path.exists("taichi-splatting"):
        shutil.rmtree("taichi-splatting")
    
    # Use shallow clone to avoid timeout (User requested fix)
    run_command("git clone --depth 1 https://github.com/taichi-dev/taichi-splatting.git", shell=True)
    
    # Patch requirements to use stable taichi
    run_command("find taichi-splatting -type f \\( -name 'pyproject.toml' -o -name 'setup.py' -o -name 'requirements.txt' \\) -exec sed -i 's/taichi-nightly/taichi/g' {} +", shell=True)
    
    # Install from source
    run_command("pip install ./taichi-splatting", shell=True)

    # Install COLMAP & ffmpeg
    print("⏳ Installing COLMAP & ffmpeg...")
    run_command("apt-get update", shell=True)
    
    try:
        run_command("colmap help", shell=True)
        print("   COLMAP already installed.")
    except:
        run_command("apt-get install -y colmap", shell=True)

    try:
        run_command("ffmpeg -version", shell=True)
        print("   ffmpeg already installed.")
    except:
        run_command("apt-get install -y ffmpeg", shell=True)
    
    try:
        run_command("which xvfb-run", shell=True)
    except:
        run_command("apt-get install -y xvfb", shell=True)


def patch_numpy_compatibility():
    """
    Patches numpy to ensure compatibility with legacy code (e.g. np.round_ removed in Numpy 2.0).
    """
    print("🔧 Checking Numpy compatibility...")
    try:
        import numpy
        if not hasattr(numpy, "round_"):
            numpy.round_ = numpy.round
            print("   🩹 Patched numpy.round_ -> numpy.round (Legacy Support Enabled)")
        else:
            print("   ✅ numpy.round_ exists. No patch needed.")
    except ImportError:
        print("   ⚠️ Numpy not installed. Skipping patch.")
    except Exception as e:
        print(f"   ⚠️ Numpy patch failed: {e}")

# --- COLMAP UTILS (Replaces Nerfstudio dependency) ---
def qvec2rotmat(qvec):
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

def read_cameras_text(path):
    cameras = {}
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            els = line.split()
            camera_id = int(els[0])
            model = els[1]
            width = int(els[2])
            height = int(els[3])
            params = np.array([float(x) for x in els[4:]])
            cameras[camera_id] = {"model": model, "width": width, "height": height, "params": params}
    return cameras

def read_images_text(path):
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

def convert_colmap_to_transforms(colmap_dir, images_dir, output_path):
    print(f"🔄 Converting COLMAP text output to {output_path}...")
    
    cameras_file = colmap_dir / "cameras.txt"
    images_file = colmap_dir / "images.txt"
    
    if not cameras_file.exists() or not images_file.exists():
        print("❌ COLMAP output files not found (cameras.txt/images.txt).")
        return False
        
    cameras = read_cameras_text(cameras_file)
    images = read_images_text(images_file)
    
    # Sort images by name
    sorted_image_ids = sorted(images.keys(), key=lambda k: images[k]["name"])
    
    frames = []
    if not cameras: 
        print("❌ No cameras found.")
        return False
        
    cam_id = list(cameras.keys())[0]
    cam = cameras[cam_id]
    
    w, h = cam["width"], cam["height"]
    fl_x = cam["params"][0]
    fl_y = cam["params"][1]
    k1 = cam["params"][2] if len(cam["params"]) > 2 else 0
    k2 = cam["params"][3] if len(cam["params"]) > 3 else 0
    p1 = cam["params"][4] if len(cam["params"]) > 4 else 0
    p2 = cam["params"][5] if len(cam["params"]) > 5 else 0
    cx = cam["params"][2] if len(cam["params"]) == 3 else w / 2.0
    cy = cam["params"][3] if len(cam["params"]) == 3 else h / 2.0
    
    angle_x = 2 * math.atan(w / (2 * fl_x))
    angle_y = 2 * math.atan(h / (2 * fl_y))
    
    json_data = {
        "camera_angle_x": angle_x,
        "camera_angle_y": angle_y,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "k1": k1, "k2": k2, "p1": p1, "p2": p2,
        "cx": cx, "cy": cy,
        "w": w, "h": h,
        "aabb_scale": 16,
        "frames": []
    }
    
    # Transformation matrix to align COLMAP (Right-Down-Forward) to Nerfstudio/OpenGL (Right-Up-Back)
    flip_mat = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    
    for img_id in sorted_image_ids:
        img = images[img_id]
        
        R = qvec2rotmat(img["qvec"])
        t = img["tvec"]
        
        # World-to-Camera to Camera-to-World
        # W2C = [R | t]
        # C2W = [R' | -R't]
        
        c2w = np.eye(4)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ t
        
        # Flip axes
        c2w = c2w @ flip_mat
        
        frame = {
            "file_path": f"images/{img['name']}",
            "transform_matrix": c2w.tolist()
        }
        frames.append(frame)
        
    json_data["frames"] = frames
    
    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=4)
        
    print(f"✅ Saved {len(frames)} frames to {output_path}")
    return True

def process_data(resume_path=None):
    """
    Manual processing pipeline:
    1. Video -> Images (ffmpeg)
    2. Feature Extraction (colmap)
    3. Matching (colmap)
    4. Reconstruction (colmap)
    5. Convert to transforms.json
    """
    if resume_path:
        print(f"🔄 RESUME MODE ENABLED. Loading data from: {resume_path}")
        resume_source = Path(resume_path)
        
        if not resume_source.exists():
            print(f"❌ Error: Resume path not found at {resume_source}")
            return False

        # Create project directory if it doesn't exist
        PROJECT_DIR.mkdir(parents=True, exist_ok=True)

        items_to_copy = ["transforms.json", "images", "sparse", "database.db"]
        
        def copy_item(item):
            src = resume_source / item
            dst = PROJECT_DIR / item
            if src.exists():
                if dst.exists():
                    if dst.is_dir(): shutil.rmtree(dst)
                    else: dst.unlink()
                if src.is_dir(): shutil.copytree(src, dst)
                else: shutil.copy2(src, dst)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            list(executor.map(copy_item, items_to_copy))

        if (PROJECT_DIR / "transforms.json").exists():
            print("✅ Data restored successfully via Resume.")
            return True
        else:
             print("❌ Failed to restore 'transforms.json'. Resume invalid.")
             return False

    # --- NORMAL PROCESSING START ---
    if VIDEO_INPUT_PATH is None or not VIDEO_INPUT_PATH.exists():
        print(f"❌ Error: Video file not found at {VIDEO_INPUT_PATH}")
        print("Please upload your video and update VIDEO_INPUT_PATH in the script.")
        return False

    print("--- 1. Clean & Setup ---")
    if PROJECT_DIR.exists():
        shutil.rmtree(PROJECT_DIR)
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    COLMAP_DIR = PROJECT_DIR / "colmap"
    COLMAP_DIR.mkdir(parents=True, exist_ok=True)

    print("--- 2. Downscale Video & Extract Frames ---")
    # Extract frames at 2 FPS (or just extract all and downsample? 2 FPS is safe for Taichi training speed)
    run_command(f"ffmpeg -i \"{VIDEO_INPUT_PATH}\" -qscale:v 1 -r 2 \"{IMAGES_DIR}/%04d.jpg\" -hide_banner", shell=True)
    
    num_images = len(list(IMAGES_DIR.glob("*.jpg")))
    print(f"✅ Extracted {num_images} images.")
    
    print("--- 3. Running COLMAP Structure-from-Motion ---")
    colmap_binary = "colmap" 
    try:
        subprocess.run(["which", "xvfb-run"], check=True, stdout=subprocess.DEVNULL)
        colmap_binary = "xvfb-run -a colmap"
        print("   Using xvfb-run for headless COLMAP")
    except:
        pass
        
    db_path = COLMAP_DIR / "database.db"
    
    print("   Feature Extraction...")
    run_command(f"{colmap_binary} feature_extractor --database_path {db_path} --image_path {IMAGES_DIR} --ImageReader.camera_model OPENCV", shell=True)
    
    print("   Matching...")
    run_command(f"{colmap_binary} sequential_matcher --database_path {db_path}", shell=True)
    
    print("   Reconstruction (Mapper)...")
    sparse_dir = COLMAP_DIR / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    run_command(f"{colmap_binary} mapper --database_path {db_path} --image_path {IMAGES_DIR} --output_path {sparse_dir}", shell=True)
    
    print("--- 4. Converting to transforms.json ---")
    # Convert binary to text for our python parser
    text_dir = COLMAP_DIR / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if sparse reconstruction succeeded (folder 0 usually exists)
    if not (sparse_dir / "0").exists():
        print("❌ COLMAP reconstruction failed (no model found).")
        return False
        
    run_command(f"{colmap_binary} model_converter --input_path {sparse_dir}/0 --output_path {text_dir} --output_type TXT", shell=True)
    
    return convert_colmap_to_transforms(text_dir, IMAGES_DIR, PROJECT_DIR / "transforms.json")

def train_model():
    print("--- Training with Taichi Splatting ---")
    # Using our standalone script
    cmd_train = f"python train_taichi.py --project_path \"{PROJECT_DIR}\" --output_path \"{OUTPUTS_DIR}\""
    run_command(cmd_train, shell=True)

def convert_ply_to_splat(ply_file: Path, output_file: Path):
    """
    Converts a PLY file to a .splat file.
    """
    print(f"⏳ Converting {ply_file.name} to .splat format...")
    # Import plyfile locally to ensure it is available (installed in deps)
    try:
        from plyfile import PlyData
        import numpy as np
    except ImportError:
        print("❌ Error: plyfile or numpy not found. Cannot convert.")
        return

    try:
        plydata = PlyData.read(str(ply_file))
        vert = plydata["vertex"]

        # Sort by scale/opacity importance approximation
        sorted_indices = np.argsort(
            -np.exp(vert["scale_0"] + vert["scale_1"] + vert["scale_2"])
            / (1 / (1 + np.exp(-vert["opacity"])))
        )

        n = len(sorted_indices)
        x = vert["x"][sorted_indices]
        y = vert["y"][sorted_indices]
        z = vert["z"][sorted_indices]
        position = np.stack([x, y, z], axis=1).astype(np.float32)

        s0 = vert["scale_0"][sorted_indices]
        s1 = vert["scale_1"][sorted_indices]
        s2 = vert["scale_2"][sorted_indices]
        scales = np.stack([s0, s1, s2], axis=1).astype(np.float32)
        scales = np.exp(scales)

        r0 = vert["rot_0"][sorted_indices]
        r1 = vert["rot_1"][sorted_indices]
        r2 = vert["rot_2"][sorted_indices]
        r3 = vert["rot_3"][sorted_indices]
        rot = np.stack([r0, r1, r2, r3], axis=1).astype(np.float32)
        length = np.sqrt(np.sum(rot ** 2, axis=1, keepdims=True))
        rot /= length
        rot_int = ((rot * 128 + 128).clip(0, 255)).astype(np.uint8)

        SH_C0 = 0.28209479177387814
        dc0 = vert["f_dc_0"][sorted_indices]
        dc1 = vert["f_dc_1"][sorted_indices]
        dc2 = vert["f_dc_2"][sorted_indices]
        R = (0.5 + SH_C0 * dc0) * 255
        G = (0.5 + SH_C0 * dc1) * 255
        B = (0.5 + SH_C0 * dc2) * 255
        R = np.clip(R, 0, 255).astype(np.uint8)
        G = np.clip(G, 0, 255).astype(np.uint8)
        B = np.clip(B, 0, 255).astype(np.uint8)
        A = np.full_like(R, 255, dtype=np.uint8)
        color = np.stack([R, G, B, A], axis=1)

        dtype_output = np.dtype([
            ('position', np.float32, 3),
            ('scale', np.float32, 3),
            ('color', np.uint8, 4),
            ('rot', np.uint8, 4)
        ])

        structured_data = np.empty(n, dtype=dtype_output)
        structured_data['position'] = position
        structured_data['scale'] = scales
        structured_data['color'] = color
        structured_data['rot'] = rot_int

        with open(output_file, "wb") as f:
            f.write(structured_data.tobytes())

        print(f"✅ Successfully converted to {output_file}")

    except Exception as e:
        print(f"❌ Conversion failed: {e}")

def export_model():
    print("--- Verifying Export ---")
    if not OUTPUTS_DIR.exists():
        print(f"❌ Error: Training output directory not found at {OUTPUTS_DIR}")
        return

    generated_splats = list(OUTPUTS_DIR.glob("*.ply"))
    
    if generated_splats:
        print(f"🎉 SUCCESS! Exported file: {generated_splats[0]}")
        splat_output = OUTPUTS_DIR / "model.splat"
        convert_ply_to_splat(generated_splats[0], splat_output)
    else:
        print(f"❌ No .ply file found in {OUTPUTS_DIR}")
        print("📂 Directory content:")
        if OUTPUTS_DIR.exists():
            for f in OUTPUTS_DIR.iterdir():
                print(f" - {f.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 3D Scan Pipeline")
    parser.add_argument("--resume_path", type=str, help="Path to existing project folder (containing transforms.json) to resume from", default=None)
    args = parser.parse_args()

    # 1. GPU Check
    if not check_gpu():
        print("WARNING: Proceeding without GPU might fail or be extremely slow.")

    # 2. Install Deps
    install_dependencies()

    # 2.1 Apply Numpy Compatibility Patch (Just in case, though we force np2.0)
    patch_numpy_compatibility()

    # 4. Process Data (or Resume)
    if process_data(resume_path=args.resume_path):
        print("✅ Data ready.")

        # 5. Train
        if (PROJECT_DIR / "transforms.json").exists():
            train_model()
            # 6. Export
            if OUTPUTS_DIR.exists():
                export_model()
            else:
                print("❌ Skipping export because output directory not found.")
        else:
            print("❌ Skipping training because transforms.json was not found.")
    else:
        print("❌ Data processing failed.")
