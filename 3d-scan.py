import os
import shutil
import sys
import glob
import subprocess
from pathlib import Path
import json
import torch

print("✅ Imports complete")

# ================= CONFIGURATION =================
PROJECT_NAME = "car_scan"
# IMPORTANT: Update this path to match your uploaded video in Kaggle
VIDEO_INPUT_PATH = '/kaggle/input/car-video/video_car.mp4'
WORKING_DIR = "/kaggle/working"
PROJECT_DIR = f"{WORKING_DIR}/{PROJECT_NAME}"
DATABASE_PATH = f"{PROJECT_DIR}/database.db"
IMAGES_DIR = f"{PROJECT_DIR}/images"
SPARSE_PATH = f"{PROJECT_DIR}/sparse"
OUTPUTS_DIR = f"{WORKING_DIR}/outputs/{PROJECT_NAME}/splatfacto"

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
    if not torch.cuda.is_available():
        print("⚠️" * 20)
        print("⚠️ WARNING: GPU Not Detected!")
        print("⚠️ This script requires a GPU (P100 or T4) to run effectively.")
        print("⚠️ Please enable GPU Accelerator in your Kaggle Notebook settings.")
        print("⚠️" * 20)
        return False
    print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    return True

def install_dependencies():
    print("⏳ Installing dependencies...")
    run_command("pip install --upgrade pip", shell=True)
    run_command("pip install torch torchvision", shell=True)
    run_command("pip install nerfstudio", shell=True)

    print("⏳ Installing COLMAP & ffmpeg...")
    run_command("apt-get update", shell=True)

    # --- FIX 1: Use mamba for COLMAP (More stable on Kaggle) ---
    try:
        print("   Attempting to install COLMAP via mamba...")
        run_command("mamba install -y -c conda-forge colmap", shell=True)
    except Exception as e:
        print("⚠️ Mamba install failed, falling back to apt-get...")
        run_command("apt-get install -y colmap", shell=True)

    run_command("apt-get install -y ffmpeg xvfb", shell=True)

    try:
        run_command("colmap help", shell=True)
        print("✅ COLMAP installed successfully.")
    except:
        print("❌ COLMAP installation failed.")

def patch_nerfstudio():
    """
    Patches nerfstudio installed in the system to fix PyTorch 2.6+ compatibility issues.
    """
    print("🔧 Patching nerfstudio for PyTorch 2.6+ compatibility...")
    try:
        potential_paths = glob.glob("/usr/local/lib/python*/dist-packages/nerfstudio/utils/eval_utils.py")
        if not potential_paths:
            potential_paths = glob.glob("/opt/conda/lib/python*/site-packages/nerfstudio/utils/eval_utils.py")

        if potential_paths:
            target_file = Path(potential_paths[0])
            print(f"   Found file: {target_file}")

            with open(target_file, "r") as f:
                content = f.read()

            old_code = 'loaded_state = torch.load(load_path, map_location="cpu")'
            new_code = 'loaded_state = torch.load(load_path, map_location="cpu", weights_only=False)'

            if old_code in content:
                new_content = content.replace(old_code, new_code)
                with open(target_file, "w") as f:
                    f.write(new_content)
                print("✅ Patch applied successfully!")
            elif 'weights_only=False' in content:
                 print("✅ Patch was already applied.")
            else:
                print(f"⚠️ Target code not found in {target_file}. The library version might be different.")
        else:
            print("⚠️ Could not locate nerfstudio/utils/eval_utils.py to patch.")
    except Exception as e:
        print(f"❌ Failed to patch nerfstudio: {e}")

def process_data():
    if not os.path.exists(VIDEO_INPUT_PATH):
        print(f"❌ Error: Video file not found at {VIDEO_INPUT_PATH}")
        print("Please upload your video and update VIDEO_INPUT_PATH in the script.")
        return False

    print("--- 1. Clean & Setup ---")
    if os.path.exists(f"{PROJECT_DIR}"):
        shutil.rmtree(f"{PROJECT_DIR}")
    os.makedirs(f"{PROJECT_DIR}", exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    print("--- 2. Downscale Video ---")
    downscaled_video = f"{WORKING_DIR}/{PROJECT_NAME}_downscaled.mp4"
    # Added -pix_fmt yuv420p for better compatibility
    run_command(f"ffmpeg -y -i \"{VIDEO_INPUT_PATH}\" -vf scale='iw/2:ih/2' -c:v libx264 -preset ultrafast -crf 23 -c:a copy \"{downscaled_video}\"", shell=True)

    print("--- 3. Extract Frames (2 FPS) ---")
    run_command(f"ffmpeg -y -i \"{downscaled_video}\" -vf \"fps=2\" \"{IMAGES_DIR}/frame_%05d.png\" -hide_banner -loglevel error", shell=True)

    num_images = sum(1 for _ in os.scandir(IMAGES_DIR))
    print(f"✅ Extracted {num_images} images.")

    print("--- 4. Feature Extraction ---")
    # Using CPU for feature extraction as per original notebook config
    cmd_extract = [
        "colmap", "feature_extractor",
        "--database_path", DATABASE_PATH,
        "--image_path", IMAGES_DIR,
        "--ImageReader.camera_model", "OPENCV",
        "--SiftExtraction.use_gpu", "0",
        "--SiftExtraction.num_threads", "16",
        "--SiftExtraction.peak_threshold", "0.004",
    ]
    run_command(cmd_extract, shell=False)

    print("--- 5. Matching (Sequential) ---")
    # --- FIX 2: Disable loop_detection to avoid crash due to missing vocab tree ---
    cmd_match = [
        "colmap", "sequential_matcher",
        "--database_path", DATABASE_PATH,
        "--SiftMatching.use_gpu", "0",
        "--SequentialMatching.loop_detection", "0", # Changed from 1 to 0
        "--SequentialMatching.overlap", "10"
    ]
    run_command(cmd_match, shell=False)

    print("--- 6. Mapper (Relaxed) ---")
    os.makedirs(SPARSE_PATH, exist_ok=True)
    cmd_mapper = [
        "colmap", "mapper",
        "--database_path", DATABASE_PATH,
        "--image_path", IMAGES_DIR,
        "--output_path", SPARSE_PATH,
        "--Mapper.min_num_matches", "10",
        "--Mapper.init_min_tri_angle", "2",
        "--Mapper.multiple_models", "0"
    ]
    run_command(cmd_mapper, shell=False)

    print("--- 7. Converting to transforms.json ---")
    recon_dir = Path(f"{SPARSE_PATH}/0")
    if not recon_dir.exists():
        print("❌ FAILED: Sparse reconstruction failed. No model found.")
        return False

    from nerfstudio.process_data.colmap_utils import colmap_to_json
    colmap_to_json(
        recon_dir=recon_dir,
        output_dir=Path(PROJECT_DIR),
    )

    if os.path.exists(f"{PROJECT_DIR}/transforms.json"):
        print("✅ transforms.json created.")
        return True
    else:
        print("❌ Failed to create transforms.json")
        return False

def train_model():
    print("--- Training Splatfacto Model ---")
    # ns-train splatfacto --data {PROJECT_DIR} --viewer.quit-on-train-completion True
    cmd_train = f"ns-train splatfacto --data \"{PROJECT_DIR}\" --viewer.quit-on-train-completion True"
    run_command(cmd_train, shell=True)

def export_model():
    print("--- Exporting .splat ---")
    training_output_path = Path(OUTPUTS_DIR)

    if not training_output_path.exists():
        print(f"❌ Error: Training output directory not found at {training_output_path}")
        return

    subfolders = [f for f in training_output_path.iterdir() if f.is_dir()]
    if not subfolders:
        print("❌ Error: No training run folders found.")
        return

    latest_run = max(subfolders, key=os.path.getmtime)
    config_path = latest_run / "config.yml"

    if not config_path.exists():
        print(f"❌ Error: Config file not found in {latest_run}")
        return

    print(f"✅ Found latest config: {config_path}")

    # Run export
    cmd_export = f"ns-export gaussian-splat --load-config \"{config_path}\" --output-dir \"{latest_run}\""
    run_command(cmd_export, shell=True)

    # Verify result
    generated_splats = list(latest_run.glob("*.splat"))
    if generated_splats:
        print(f"🎉 SUCCESS! Exported file: {generated_splats[0]}")
    else:
        print("❌ Export command finished but no .splat file was found.")

def main():
    # 1. GPU Check
    if not check_gpu():
        print("WARNING: Proceeding without GPU might fail or be extremely slow.")

    # 2. Install Deps
    install_dependencies()

    # 3. Apply Patch (Critical Fix)
    patch_nerfstudio()

    # 4. Process Data
    if process_data():
        print("✅ Data processing complete.")

        # 5. Train
        # Only run if transforms.json exists
        if os.path.exists(f"{PROJECT_DIR}/transforms.json"):
            train_model()
        else:
            print("❌ Skipping training because transforms.json was not found.")

        # 6. Export
        if os.path.exists(OUTPUTS_DIR):
            export_model()
        else:
            print("❌ Skipping export because output directory not found.")

    else:
        print("❌ Data processing failed.")

if __name__ == "__main__":
    main()
