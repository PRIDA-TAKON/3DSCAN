
import os
import argparse
import subprocess
from pathlib import Path
import sys

# Constants
PROJECT_NAME = "3d_scan"
WORKING_DIR = Path("/kaggle/working") if os.path.exists("/kaggle") else Path("working_data")
PROJECT_DIR = WORKING_DIR / PROJECT_NAME
IMAGES_DIR = PROJECT_DIR / "images"
OUTPUTS_DIR = Path("outputs") / PROJECT_NAME / "taichi_splatting"
SCRIPTS_DIR = Path("scripts")

def check_gpu():
    print("🔍 Checking GPU availability...")
    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠️" * 20)
            print("⚠️ WARNING: GPU Not Detected!")
            print("⚠️ Step 3 (Training) will likely FAIL or requires CPU fallback (very slow).")
            print("⚠️" * 20)
            return False
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
        return True
    except ImportError:
        print("⚠️ torch module not found. Cannot check GPU availability.")
        return False

def run_script(script_name, args):
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        print(f"❌ Error: Script {script_name} not found in {SCRIPTS_DIR}")
        return False
        
    cmd = [sys.executable, str(script_path)] + args
    print(f"🚀 Running {script_name}...")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        print(f"❌ {script_name} failed.")
        return False

def main():
    parser = argparse.ArgumentParser(description="3D Scan Orchestrator")
    parser.add_argument("--video", help="Path to input video file")
    parser.add_argument("--resume", action="store_true", help="Resume from existing data")
    parser.add_argument("--skip_train", action="store_true", help="Skip training step")
    
    args = parser.parse_args()

    # Setup directories
    WORKING_DIR.mkdir(exist_ok=True)
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Find video if not provided
    video_path = args.video
    if not video_path:
        search_paths = [Path("/kaggle/input"), Path("input")]
        for p in search_paths:
            found = next(p.rglob("*.mp4"), None)
            if found:
                video_path = str(found)
                break
    
    if not video_path and not args.resume:
        print("❌ No input video found. Please provide --video path.")
        return

    # --- Step 0: Check & Install Dependencies ---
    install_dependencies()

    # --- Step 1: Extract Frames ---
    if not args.resume:
        print("\n=== STEP 1: Extract Frames ===")
        if not run_script("step1_extract_frames.py", ["--input_video", video_path, "--output_dir", str(IMAGES_DIR)]):
            return

    # --- Step 2: COLMAP SfM ---
    if not args.resume: # If resuming, assume transforms.json exists
        print("\n=== STEP 2: COLMAP SfM ===")
        if not run_script("step2_colmap_sfm.py", ["--images_dir", str(IMAGES_DIR), "--output_dir", str(PROJECT_DIR)]):
            return

    # --- Step 3: Train ---
    if not args.skip_train:
        print("\n=== STEP 3: Train Taichi Splatting ===")
        check_gpu()
        if not run_script("step3_train_splatting.py", ["--project_path", str(PROJECT_DIR), "--output_path", str(OUTPUTS_DIR)]):
            return

        # --- Step 4: Export ---
        print("\n=== STEP 4: Export ===")
        model_parquet = OUTPUTS_DIR / "model.parquet"
        model_ply = OUTPUTS_DIR / "model.ply"
        
        output_splat = OUTPUTS_DIR / "model.splat"
        
        if model_parquet.exists():
            run_script("step4_export.py", ["--input_parquet", str(model_parquet), "--output_splat", str(output_splat)])
        elif model_ply.exists():
            run_script("step4_export.py", ["--input_ply", str(model_ply), "--output_splat", str(output_splat)])
        else:
             print("❌ Training finished but no .parquet or .ply file found for export.")

def install_dependencies():
    print("📦 Checking dependencies...")
    try:
        import taichi_3d_gaussian_splatting
        print("✅ taichi_3d_gaussian_splatting already installed.")
    except ImportError:
        print("🚀 Installing taichi_3d_gaussian_splatting (Wanmeihuali Version)...")
        if not os.path.exists("taichi_3d_gaussian_splatting"):
            subprocess.run(["git", "clone", "--depth", "1", "https://github.com/wanmeihuali/taichi_3d_gaussian_splatting.git"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "taichi_3d_gaussian_splatting/requirements.txt"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "./taichi_3d_gaussian_splatting"], check=True)

if __name__ == "__main__":
    main()
