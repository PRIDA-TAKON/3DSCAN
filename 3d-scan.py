import os
import shutil
import sys
import glob
import subprocess
import argparse
from pathlib import Path
import json
import importlib.util

print("✅ Imports complete")

# ================= CONFIGURATION =================
PROJECT_NAME = "3d_scan"
# Function to find input video dynamically
def find_input_video():
    print("🔍 Searching for input video...")
    search_paths = [Path("/kaggle/input"), Path("input")]
    
    for search_path in search_paths:
        if search_path.exists():
            # Find all mp4 files recursively
            videos = list(search_path.rglob("*.mp4"))
            if videos:
                print(f"✅ Found video: {videos[0]}")
                return videos[0]
    
    print("❌ No .mp4 video found in /kaggle/input or local input/")
    return None

# Initial placeholder, allows override
VIDEO_INPUT_PATH = None
WORKING_DIR = Path("/kaggle/working")
PROJECT_DIR = WORKING_DIR / PROJECT_NAME
DATABASE_PATH = PROJECT_DIR / "database.db"
IMAGES_DIR = PROJECT_DIR / "images"
SPARSE_PATH = PROJECT_DIR / "sparse"
OUTPUTS_DIR = Path("outputs") / PROJECT_NAME / "splatfacto"

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

def install_dependencies():
    print("⏳ Installing dependencies...")

    # Check if nerfstudio is installed
    if importlib.util.find_spec("nerfstudio") is None:
        run_command("pip install --upgrade pip", shell=True)
        # Force numpy < 2.0 to avoid compatibility issues with recent library updates
        # "Factory Reset" numpy: force reinstall to fix potential file corruption from previous patching attempts
        run_command("pip install \"numpy<2.0\" --force-reinstall", shell=True)
        run_command("pip install torch torchvision", shell=True)
        run_command("pip install nerfstudio", shell=True)
    else:
        print("   nerfstudio already installed.")

    print("⏳ Installing COLMAP & ffmpeg...")
    run_command("apt-get update", shell=True)

    # Check if colmap is installed
    try:
        run_command("colmap help", shell=True)
        print("   COLMAP already installed.")
    except:
        print("⏳ Installing COLMAP via apt-get...")
        run_command("apt-get install -y colmap", shell=True)

    # Check if ffmpeg is installed
    try:
        run_command("ffmpeg -version", shell=True)
        print("   ffmpeg already installed.")
    except:
        run_command("apt-get install -y ffmpeg", shell=True)

    # Check if xvfb is installed (required for COLMAP with GPU)
    try:
        run_command("which xvfb-run", shell=True)
        print("   xvfb already installed.")
    except:
        print("⏳ Installing xvfb...")
        run_command("apt-get install -y xvfb", shell=True)

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

def process_data(resume_path=None):
    """
    Processes video into images and run COLMAP, OR resumes from existing data.
    """
    if resume_path:
        print(f"🔄 RESUME MODE ENABLED. Loading data from: {resume_path}")
        resume_source = Path(resume_path)
        
        if not resume_source.exists():
            print(f"❌ Error: Resume path not found at {resume_source}")
            return False

        # Create project directory if it doesn't exist
        PROJECT_DIR.mkdir(parents=True, exist_ok=True)

        # List of critical items to copy
        items_to_copy = ["transforms.json", "images", "sparse", "database.db", "sparse_pc.ply"]
        
        for item in items_to_copy:
            src = resume_source / item
            dst = PROJECT_DIR / item
            
            if src.exists():
                if dst.exists():
                    print(f"   Removing existing {dst}...")
                    if dst.is_dir():
                        shutil.rmtree(dst)
                    else:
                        dst.unlink()
                
                print(f"   Copying {item}...")
                if src.is_dir():
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
            else:
                 print(f"⚠️ Warning: '{item}' not found in resume source. Proceeding cautiously.")

        if (PROJECT_DIR / "transforms.json").exists():
            print("✅ Data restored successfully via Resume.")
            return True
        else:
             print("❌ Failed to restore 'transforms.json'. Resume invalid.")
             return False

    # --- NORMAL PROCESSING START ---
    if not VIDEO_INPUT_PATH.exists():
        print(f"❌ Error: Video file not found at {VIDEO_INPUT_PATH}")
        print("Please upload your video and update VIDEO_INPUT_PATH in the script.")
        return False

    print("--- 1. Clean & Setup ---")
    if PROJECT_DIR.exists():
        shutil.rmtree(PROJECT_DIR)
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Determine COLMAP binary command (use xvfb-run if available)
    colmap_binary = "colmap"
    try:
        run_command("which xvfb-run", shell=True)
        colmap_binary = "xvfb-run -a colmap"
        print(f"✅ xvfb-run detected. Using: {colmap_binary}")
    except:
        print("⚠️ xvfb-run not found. Using raw colmap command.")

    print("--- 2. Downscale Video ---")
    downscaled_video = WORKING_DIR / f"{PROJECT_NAME}_downscaled.mp4"
    # Added -pix_fmt yuv420p for better compatibility
    run_command(f"ffmpeg -y -i \"{VIDEO_INPUT_PATH}\" -vf scale='iw/2:ih/2' -c:v libx264 -preset veryfast -crf 23 -an \"{downscaled_video}\"", shell=True)

    print("--- 3. Extract Frames (2 FPS) ---")
    run_command(f"ffmpeg -y -i \"{downscaled_video}\" -vf \"fps=2\" \"{IMAGES_DIR}/frame_%05d.png\" -hide_banner -loglevel error", shell=True)

    num_images = sum(1 for _ in os.scandir(IMAGES_DIR))
    print(f"✅ Extracted {num_images} images.")

    print("--- 4. Feature Extraction ---")
    # Using CPU for feature extraction as per original notebook config, but memory says SiftMatching should use GPU.
    # Feature extraction is separate from Matching. Memory specifically says SiftMatching.
    # However, usually if one uses GPU, the other can too.
    # The notebook says: --SiftExtraction.use_gpu 0
    # Memory says: "COLMAP SIFT matching and extraction commands in the project should have GPU acceleration enabled (`use_gpu 1`) to maximize processing speed."
    # So I should enable GPU for extraction too.

    cmd_extract = [
        colmap_binary, "feature_extractor",
        "--database_path", str(DATABASE_PATH),
        "--image_path", str(IMAGES_DIR),
        "--ImageReader.camera_model", "OPENCV",
        "--SiftExtraction.use_gpu", "0", # Disable GPU for extraction to avoid OpenGL crashes in headless mode
        "--SiftExtraction.num_threads", "16",
        "--SiftExtraction.peak_threshold", "0.004",
    ]
    run_command(" ".join(cmd_extract), shell=True)

    print("--- 5. Matching (Sequential) ---")
    # --- FIX 2: Disable loop_detection to avoid crash due to missing vocab tree ---
    cmd_match = [
        colmap_binary, "sequential_matcher",
        "--database_path", str(DATABASE_PATH),
        "--SiftMatching.use_gpu", "0",
        "--SequentialMatching.loop_detection", "0",
        "--SequentialMatching.overlap", "10"
    ]
    run_command(" ".join(cmd_match), shell=True)

    print("--- 6. Mapper (Relaxed) ---")
    SPARSE_PATH.mkdir(parents=True, exist_ok=True)
    cmd_mapper = [
        colmap_binary, "mapper",
        "--database_path", str(DATABASE_PATH),
        "--image_path", str(IMAGES_DIR),
        "--output_path", str(SPARSE_PATH),
        "--Mapper.min_num_matches", "10",
        "--Mapper.init_min_tri_angle", "2",
        "--Mapper.multiple_models", "0"
    ]
    run_command(" ".join(cmd_mapper), shell=True)

    print("--- 7. Converting to transforms.json ---")
    recon_dir = SPARSE_PATH / "0"
    if not recon_dir.exists():
        print("❌ FAILED: Sparse reconstruction failed. No model found.")
        return False

    from nerfstudio.process_data.colmap_utils import colmap_to_json
    colmap_to_json(
        recon_dir=recon_dir,
        output_dir=PROJECT_DIR,
    )

    if (PROJECT_DIR / "transforms.json").exists():
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
    training_output_path = OUTPUTS_DIR

    if not training_output_path.exists():
        print(f"❌ Error: Training output directory not found at {training_output_path}")
        return

    latest_run = None
    latest_mtime = -1

    with os.scandir(training_output_path) as it:
        for entry in it:
            if entry.is_dir():
                if entry.stat().st_mtime > latest_mtime:
                    latest_mtime = entry.stat().st_mtime
                    latest_run = Path(entry.path)

    if latest_run is None:
         print("❌ Error: No training run folders found.")
         return

    config_path = latest_run / "config.yml"

    if not config_path.exists():
        print(f"❌ Error: Config file not found in {latest_run}")
        return

    print(f"✅ Found latest config: {config_path}")

    # Run export
    cmd_export = f"ns-export gaussian-splat --load-config \"{config_path}\" --output-dir \"{latest_run}\""
    run_command(cmd_export, shell=True)

    # Verify result
    generated_splats = list(latest_run.glob("*.splat")) + list(latest_run.glob("*.ply"))
    if generated_splats:
        print(f"🎉 SUCCESS! Exported file: {generated_splats[0]}")
    else:
        print(f"❌ Export command finished but no .splat file was found in {latest_run}")
        print("📂 Directory content:")
        for f in latest_run.iterdir():
            print(f" - {f.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 3D Scan Pipeline")
    parser.add_argument("--resume_path", type=str, help="Path to existing project folder (containing transforms.json) to resume from", default=None)
    parser.add_argument("--input_video", type=str, default=None, help="Override input video path")
    args = parser.parse_args()

    # Initialize Video Path
    if args.input_video:
        VIDEO_INPUT_PATH = Path(args.input_video)
    else:
        detected_video = find_input_video()
        if detected_video:
            VIDEO_INPUT_PATH = detected_video
        else:
             print("⚠️ No video found dynamically, using default fallback path.")
             VIDEO_INPUT_PATH = Path('/kaggle/input/car-video/video_car.mp4')
             
    if not VIDEO_INPUT_PATH or not VIDEO_INPUT_PATH.exists():
         print(f"❌ Error: Video file not found at {VIDEO_INPUT_PATH}")
         print("Please upload your video to Kaggle input or specify --input_video")
         # We continue to allow checking GPU etc, but process_data will fail.

    # 1. GPU Check
    if not check_gpu():
        print("WARNING: Proceeding without GPU might fail or be extremely slow.")

    # 2. Install Deps
    install_dependencies()

    # 3. Apply Patch (Critical Fix)
    patch_nerfstudio()

    # 4. Process Data (or Resume)
    if process_data(resume_path=args.resume_path):
        print("✅ Data ready.")

        # 5. Train
        # Only run if transforms.json exists
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
