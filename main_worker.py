import os
import subprocess
import sys
import argparse
import time
import shutil
import json
from pathlib import Path

# --- Configuration & Environment Setup ---

def run_command(cmd, env=None, cwd=None):
    """Run a shell command and handle errors."""
    print(f"🚀 Running: {cmd}")
    try:
        process = subprocess.run(
            cmd, 
            shell=True, 
            check=True, 
            env={**os.environ, **(env or {})}, 
            cwd=cwd,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {cmd}")
        print(f"   Error: {e}")
        return False

def setup_stage1_env():
    """Setup environment for COLMAP/FFmpeg."""
    print("🛠️ Setting up Stage 1 Environment (Preprocessing)...")
    # In Kaggle, colmap and ffmpeg are usually pre-installed.
    # We might need some specific python libs for COLMAP to Transforms conversion.
    run_command("pip install numpy opencv-python")

def setup_stage2_env():
    """Setup environment for 3DGS Training."""
    print("🛠️ Setting up Stage 2 Environment (3DGS Training)...")
    # This often needs specific Taichi or Nerfstudio versions
    # For Taichi Splatting:
    run_command("pip install taichi torch torchvision plyfile pyyaml pandas")
    # If the user's fork is used:
    repo_path = Path("taichi-splatting-kaggle")
    if repo_path.exists():
        run_command("pip install -e .", cwd=str(repo_path))

# --- Data Pipeline Functions ---

def get_supabase_client():
    """Initialize Supabase client using environment variables or Kaggle Secrets."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    # Try Kaggle Secrets if env vars are missing
    if not url or not key:
        try:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            url = url or user_secrets.get_secret("SUPABASE_URL")
            key = key or user_secrets.get_secret("SUPABASE_KEY")
        except:
            pass

    try:
        from supabase import create_client
        if not url or not key:
            print("⚠️ Missing SUPABASE_URL or SUPABASE_KEY. Using mock mode.")
            return None
        return create_client(url, key)
    except ImportError:
        print("⚠️ supabase-py not installed. Using mock mode.")
        return None

def download_input(source, target_path):
    """Download video or image zip from a URL or copy from local path."""
    source_path = Path(source)
    if source_path.exists():
        print(f"📦 Source {source} exists locally. Copying to {target_path}...")
        shutil.copy(source, target_path)
        return

    print(f"📥 Downloading input from {source}...")
    import requests
    try:
        response = requests.get(source, stream=True)
        response.raise_for_status()
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("✅ Download complete.")
    except Exception as e:
        print(f"❌ Download/Copy failed: {e}")
        raise

def upload_output(file_path, destination_name):
    """Upload result files to Supabase Storage."""
    print(f"📤 Uploading results {file_path} to Supabase...")
    supabase = get_supabase_client()
    if not supabase:
        print("⚠️ Skipping upload (No Supabase client).")
        return

    bucket = os.environ.get("SUPABASE_BUCKET_NAME", "3d-scans")
    try:
        with open(file_path, 'rb') as f:
            supabase.storage.from_(bucket).upload(destination_name, f)
        print(f"✅ Uploaded to {bucket}/{destination_name}")
    except Exception as e:
        print(f"❌ Upload failed: {e}")

def update_status(job_id, status, message=""):
    """Update job status in Supabase Database."""
    print(f"🔔 Status Update [{job_id}]: {status} - {message}")
    supabase = get_supabase_client()
    if not supabase:
        return

    try:
        supabase.table("jobs").update({
            "status": status,
            "message": message,
            "updated_at": "now()"
        }).eq("id", job_id).execute()
    except Exception as e:
        print(f"⚠️ Failed to update DB status: {e}")

# --- Pipeline Stages ---

def run_preprocessing(video_path, project_dir):
    """Stage 1: Extract frames and run COLMAP."""
    print("🟢 Starting Stage 1: Pre-processing...")
    images_dir = project_dir / "images"
    
    # Step 1: Extract Frames
    cmd_extract = f"python scripts/step1_extract_frames.py --input_video {video_path} --output_dir {images_dir} --fps 2"
    if not run_command(cmd_extract):
        return False
    
    # Step 2: COLMAP SfM
    # Added error logging and timeout if necessary
    cmd_colmap = f"python scripts/step2_colmap_sfm.py --images_dir {images_dir} --output_dir {project_dir}"
    if not run_command(cmd_colmap):
        return False
        
    return True

def run_training(project_dir, output_dir):
    """Stage 2: Train 3DGS with OOM protection."""
    print("🟢 Starting Stage 2: 3DGS Training...")
    
    try:
        # Step 3: Train
        # Reduce iterations or resolution if OOM occurs? 
        # For now, we try to run and catch the failure.
        cmd_train = f"python scripts/step3_train_splatting.py --project_path {project_dir} --output_path {output_dir} --iterations 7000"
        if not run_command(cmd_train):
            return False
        
        # Step 4: Export to .splat
        cmd_export = f"python scripts/step4_export.py --input_config {output_dir} --output_splat {output_dir}/model.splat"
        run_command(cmd_export) # Export is less likely to OOM
        
        return True
    except Exception as e:
        if "Out of Memory" in str(e) or "CUDA error: out of memory" in str(e):
            print("🚨 FATAL: CUDA Out of Memory detected!")
            # Potential mitigation: clean cache or restart with lower settings
        raise e

# --- Main Logic ---

def main():
    parser = argparse.ArgumentParser(description="Medical 3DGS Main Worker")
    parser.add_argument("--job_id", required=True)
    parser.add_argument("--video_url", required=True)
    parser.add_argument("--output_dest", required=True)
    args = parser.parse_args()

    project_dir = Path("work_dir")
    output_dir = Path("output_dir")
    video_file = project_dir / "input_video.mp4"
    
    project_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        update_status(args.job_id, "RUNNING", "Initializing pipeline...")
        
        # 1. Download Data
        download_input(args.video_url, video_file)
        
        # 2. Stage 1: Pre-processing
        setup_stage1_env()
        if not run_preprocessing(video_file, project_dir):
            raise Exception("Stage 1 Pre-processing failed.")
            
        update_status(args.job_id, "RUNNING", "Stage 1 Complete. Starting Training...")
        
        # 3. Stage 2: Training
        setup_stage2_env()
        if not run_training(project_dir, output_dir):
            raise Exception("Stage 2 Training failed.")
            
        # 4. Finalize
        result_zip = f"result_{args.job_id}.zip"
        shutil.make_archive(f"result_{args.job_id}", 'zip', output_dir)
        
        upload_output(f"{result_zip}", args.output_dest)
        update_status(args.job_id, "COMPLETED", "3DGS Model generated successfully.")
        
    except Exception as e:
        print(f"💥 Pipeline Error: {e}")
        update_status(args.job_id, "FAILED", str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
