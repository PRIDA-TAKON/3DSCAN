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

print("✅ Imports complete")

# ================= CONFIGURATION =================
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
    print("⏳ Installing dependencies...")

    # Check for numpy corruption
    numpy_ok = check_numpy_integrity()
    if not numpy_ok:
        print("⚠️ Numpy corruption detected! Forcing reinstall...")

    # Check if nerfstudio is installed
    if importlib.util.find_spec("nerfstudio") is None or not numpy_ok:
        run_command("pip install --upgrade pip", shell=True)
        # Reinstall numpy if corrupted or fresh install. We remove the <2.0 constraint as we have the patch.
        if not numpy_ok:
             run_command("pip install numpy --force-reinstall", shell=True)

        run_command("pip install torch torchvision", shell=True)
        run_command("pip install nerfstudio", shell=True)
        # Install plyfile for custom .splat export
        run_command("pip install plyfile", shell=True)
    else:
        print("   nerfstudio already installed.")
        # Ensure plyfile is installed even if nerfstudio is present
        if importlib.util.find_spec("plyfile") is None:
             run_command("pip install plyfile", shell=True)

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

            # Robust regex search for the target line
            pattern = r'(loaded_state\s*=\s*torch\.load\s*\(\s*load_path\s*,\s*map_location\s*=\s*["\']cpu["\'])\s*,?\s*\)'

            if re.search(pattern, content):
                new_content = re.sub(pattern, r'\1, weights_only=False)', content)
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
        
        def copy_item(item):
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

        # Parallelize copying to speed up data transfer
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # We convert to list to ensure any exceptions are raised during execution
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

    # Determine COLMAP binary command (use xvfb-run if available)
    colmap_cmd = ["colmap"]
    try:
        subprocess.run(["which", "xvfb-run"], check=True, stdout=subprocess.DEVNULL)
        colmap_cmd = ["xvfb-run", "-a", "colmap"]
        print(f"✅ xvfb-run detected. Using: {colmap_cmd}")
    except:
        print("⚠️ xvfb-run not found. Using raw colmap command.")

    print("--- 2. Downscale Video ---")
    # Check for h264_nvenc
    use_nvenc = False
    try:
        result = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True)
        if "h264_nvenc" in result.stdout:
            use_nvenc = True
            print("✅ h264_nvenc detected. Using GPU acceleration for video processing.")
        else:
            print("⚠️ h264_nvenc not found. Using CPU encoding.")
    except Exception:
         print("⚠️ Failed to check ffmpeg encoders. Defaulting to CPU.")

    downscaled_video = WORKING_DIR / f"{PROJECT_NAME}_downscaled.mp4"

    if use_nvenc:
         # -preset fast -cq 23
         cmd_downscale = ["ffmpeg", "-y", "-i", str(VIDEO_INPUT_PATH), "-vf", "scale=trunc(iw/4)*2:trunc(ih/4)*2", "-c:v", "h264_nvenc", "-preset", "fast", "-cq", "23", "-an", str(downscaled_video)]
    else:
         # -preset veryfast -crf 23
         cmd_downscale = ["ffmpeg", "-y", "-i", str(VIDEO_INPUT_PATH), "-vf", "scale=trunc(iw/4)*2:trunc(ih/4)*2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-an", str(downscaled_video)]

    run_command(cmd_downscale, shell=False)

    print("--- 3. Extract Frames (2 FPS) ---")
    cmd_frames = ["ffmpeg", "-y", "-i", str(downscaled_video), "-vf", "fps=2", str(IMAGES_DIR / "frame_%05d.png"), "-hide_banner"]
    run_command(cmd_frames, shell=False)

    num_images = sum(1 for _ in os.scandir(IMAGES_DIR))
    print(f"✅ Extracted {num_images} images.")

    print("--- 4. Feature Extraction ---")
    # Using CPU (use_gpu 0) for feature extraction/matching to avoid OpenGL crashes in headless environments

    cmd_extract = colmap_cmd + [
        "feature_extractor",
        "--database_path", str(DATABASE_PATH),
        "--image_path", str(IMAGES_DIR),
        "--ImageReader.camera_model", "OPENCV",
        "--SiftExtraction.use_gpu", "0",
        "--SiftExtraction.num_threads", "16",
        "--SiftExtraction.peak_threshold", "0.004",
    ]
    run_command(cmd_extract, shell=False)

    print("--- 5. Matching (Sequential) ---")
    # Loop detection disabled to prevent crashes
    cmd_match = colmap_cmd + [
        "sequential_matcher",
        "--database_path", str(DATABASE_PATH),
        "--SiftMatching.use_gpu", "0",
        "--SequentialMatching.loop_detection", "0",
        "--SequentialMatching.overlap", "10"
    ]
    run_command(cmd_match, shell=False)

    print("--- 6. Mapper (Relaxed) ---")
    SPARSE_PATH.mkdir(parents=True, exist_ok=True)
    cmd_mapper = colmap_cmd + [
        "mapper",
        "--database_path", str(DATABASE_PATH),
        "--image_path", str(IMAGES_DIR),
        "--output_path", str(SPARSE_PATH),
        "--Mapper.min_num_matches", "10",
        "--Mapper.init_min_tri_angle", "2",
        "--Mapper.multiple_models", "0"
    ]
    run_command(cmd_mapper, shell=False)

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

        buffer = bytearray()
        for idx in sorted_indices:
            position = np.array([vert["x"][idx], vert["y"][idx], vert["z"][idx]], dtype=np.float32)
            scales = np.array([vert["scale_0"][idx], vert["scale_1"][idx], vert["scale_2"][idx]], dtype=np.float32)
            rot = np.array([vert["rot_0"][idx], vert["rot_1"][idx], vert["rot_2"][idx], vert["rot_3"][idx]], dtype=np.float32)

            # Color (Spherical Harmonics DC term)
            SH_C0 = 0.28209479177387814
            r = max(0, min(255, int((0.5 + SH_C0 * vert["f_dc_0"][idx]) * 255)))
            g = max(0, min(255, int((0.5 + SH_C0 * vert["f_dc_1"][idx]) * 255)))
            b = max(0, min(255, int((0.5 + SH_C0 * vert["f_dc_2"][idx]) * 255)))
            color = np.array([r, g, b, 255], dtype=np.uint8)

            # Normalize Rotation
            length = np.sqrt(np.sum(rot ** 2))
            rot /= length

            # Exp scales to get linear scale
            scales = np.exp(scales)

            # Pack into buffer
            # Format: position(3f), scale(3f), color(4b), rotation(4b)
            buffer.extend(position.tobytes())
            buffer.extend(scales.tobytes())
            buffer.extend(color.tobytes())

            # Quantize Rotation to 8-bit
            rot_int = ((rot * 128 + 128).clip(0, 255)).astype(np.uint8)
            buffer.extend(rot_int.tobytes())

        with open(output_file, "wb") as f:
            f.write(buffer)

        print(f"✅ Successfully converted to {output_file}")

    except Exception as e:
        print(f"❌ Conversion failed: {e}")

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
    generated_splats = list(latest_run.glob("*.splat"))
    if generated_splats:
        print(f"🎉 SUCCESS! Exported file: {generated_splats[0]}")
    else:
        # Check for PLY if SPLAT not found
        generated_plys = list(latest_run.glob("*.ply"))
        if generated_plys:
             print(f"⚠️ .splat not found, but found .ply: {generated_plys[0]}")
             splat_file = latest_run / "model.splat"
             convert_ply_to_splat(generated_plys[0], splat_file)
        else:
            print(f"❌ Export command finished but no .splat or .ply file was found in {latest_run}")
            print("📂 Directory content:")
            for f in latest_run.iterdir():
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

    # 2.1 Apply Numpy Compatibility Patch
    patch_numpy_compatibility()

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
