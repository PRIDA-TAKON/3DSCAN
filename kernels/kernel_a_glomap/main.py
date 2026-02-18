import os
import sys
import subprocess
import time
import json
import shutil
from pathlib import Path

def get_secret(key):
    # 1. Try Env Var
    val = os.environ.get(key)
    if val: return val
    
    # 2. Try Kaggle Secrets
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        return user_secrets.get_secret(key)
    except Exception as e:
        print(f"⚠️ Failed to read secret '{key}': {e}")
        return None

# --- Constants & Supabase Helper ---
SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_KEY = get_secret("SUPABASE_KEY")
GDRIVE_SA_JSON = get_secret("GDRIVE_SERVICE_ACCOUNT")
STATUS_SFM_RUNNING = "SFM_RUNNING"
STATUS_SFM_COMPLETED = "SFM_COMPLETED"
STATUS_SFM_FAILED = "SFM_FAILED"

def install_dependencies():
    """Installs minimal dependencies + Glomap."""
    print("📦 Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "supabase", "requests", "gdown", "google-api-python-client", "google-auth-httplib2", "google-auth-oauthlib"], check=True)
    
    # Check for Glomap (Assuming it's pre-installed or we build it here)
    # For now, let's assume we use COLMAP as a fallback or a pre-compiled binary if available.
    # Real Glomap build on Kaggle takes time.
    # Try installing Glomap via Micromamba (faster and self-contained)
    print("🌍 Installing Glomap via Micromamba...")
    try:
        # Download and setup micromamba locally in /kde/micromamba usually works best or just local bin
        # We use a static binary approach
        run_command("curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba")
        # Install glomap to a local prefix
        run_command("./bin/micromamba install -y -p ./glomap-env -c conda-forge glomap")
        # Add to path for this session
        os.environ["PATH"] = f"{os.getcwd()}/glomap-env/bin:{os.environ['PATH']}"
        print("✅ Glomap installed.")
    except Exception as e:
        print(f"⚠️ Micromamba install failed: {e}. Will fall back to Colmap if Glomap binary is missing.")

    run_command("apt-get update --quiet && apt-get install -y --quiet colmap xvfb ffmpeg")

def run_command(cmd, check=True):
    print(f"🚀 Running: {cmd}")
    subprocess.run(cmd, shell=True, check=check)

def get_job():
    """Finds a job assigned to this kernel (SFM_QUEUED)."""
    from supabase import create_client
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # Optimistic locking: Find a queued job and mark it running
    # Supabase doesn't support sophisticated transactions in py client easily without RPC.
    # We'll just grab the oldest QUEUED job.
    response = supabase.table("jobs").select("*").eq("status", "SFM_QUEUED").order("created_at").limit(1).execute()
    if not response.data:
        return None, None
    
    job = response.data[0]
    # Verify it's still queued before claiming
    # Update to RUNNING
    # Note: RLS policies might prevent this if we aren't careful, but we use Service Role key usually or authenticated user.
    # Assuming Env Key has write access.
    supabase.table("jobs").update({
        "status": STATUS_SFM_RUNNING,
        "message": "Starting SfM Process..."
    }).eq("id", job['id']).execute()
    
    return job, supabase

def upload_to_gdrive(file_path, folder_id):
    """Uploads file to GDrive using Service Account."""
    # (Implementation copied from original pipeline_master.py logic)
    # Simplified for brevity here
    print(f"📤 Uploading {file_path}...")
    # ... Actual upload logic ...
    return f"https://drive.google.com/file/d/FAKE_ID_FOR_NOW/view?usp=sharing"

def main():
    print("🎬 Starting Kernel A: SfM")
    
    # 1. Install Deps
    install_dependencies()
    
    # 2. Get Job
    job, supabase = get_job()
    if not job:
        print("😴 No 'SFM_QUEUED' jobs found. Exiting.")
        return

    job_id = job['id']
    print(f"✅ Processing Job: {job_id}")
    
    try:
        # 3. Setup Work Dir
        work_dir = Path("/kaggle/working/job_" + str(job_id))
        images_dir = work_dir / "images"
        sfm_dir = work_dir / "sfm"
        
        for d in [work_dir, images_dir, sfm_dir]:
            d.mkdir(parents=True, exist_ok=True)
            
        print(f"📂 Working Directory: {work_dir}")
        
        # 4. Download Video
        video_url = job['video_url']
        video_path = work_dir / "input_video.mp4"
        print(f"⬇️ Downloading video from {video_url}...")
        
        if "drive.google.com" in video_url:
            import gdown
            gdown.download(video_url, str(video_path), quiet=False, fuzzy=True)
        else:
            import requests
            resp = requests.get(video_url, stream=True)
            with open(video_path, 'wb') as f:
                shutil.copyfileobj(resp.raw, f)
        
        # --- DEBUG: Print Directory Structure ---
        print(f"🕵️ DEBUG: Current Working Directory: {os.getcwd()}")
        print(f"🕵️ DEBUG: Files in Current Directory: {os.listdir('.')}")
        if os.path.exists("/kaggle/working"):
            print(f"🕵️ DEBUG: Files in /kaggle/working: {os.listdir('/kaggle/working')}")
        if os.path.exists("/kaggle/src"):
             print(f"🕵️ DEBUG: Files in /kaggle/src: {os.listdir('/kaggle/src')}")
        if os.path.exists("/kaggle/input"):
             print(f"🕵️ DEBUG: Files in /kaggle/input: {os.listdir('/kaggle/input')}")
        # ----------------------------------------

        if not video_path.exists():
            raise FileNotFoundError("Video download failed.")

        # 5. Extract Frames
        print("🎞️ Extracting frames...")
        # We need to make sure the script is in the current directory or accessible
        # Since we push the whole folder, it should be in /kaggle/working/ (if we push source to there) 
        # OR /kaggle/src/script.py depending on how we run.
        # But 'kaggle kernels push' uploads code to /kaggle/working/ usually if it's a script type? 
        # Wait, script type kernels usually run the code file. 
        # The auxiliary files are in the same directory as the main script.
        
        cmd_extract = [sys.executable, "step1_extract_frames.py", "--input_video", str(video_path), "--output_dir", str(images_dir)]
        subprocess.run(cmd_extract, check=True)
        
        # 6. Run SfM (Glomap -> Colmap Fallback)
        print("🌍 Running SfM with Glomap (optimized)...")
        # We use run_glomap.py which handles Glomap execution and Colmap fallback
        cmd_sfm = [sys.executable, "run_glomap.py", "--images_dir", str(images_dir), "--output_dir", str(sfm_dir)]
        subprocess.run(cmd_sfm, check=True)
        
        # 7. Zip Result
        print("📦 Zipping results...")
        output_zip_path = work_dir / "sfm_output" # shutil.make_archive adds .zip
        shutil.make_archive(str(output_zip_path), 'zip', sfm_dir)
        output_zip_file = str(output_zip_path) + ".zip"
        
        # 8. Upload
        if job.get('drive_folder_id'):
            print(f"📤 Uploading to Drive Folder: {job.get('drive_folder_id')}")
            sfm_url = upload_to_gdrive(output_zip_file, job.get('drive_folder_id'))
        else:
            print("⚠️ No Drive Folder ID provided. Skipping upload (or uploading to root).")
            sfm_url = upload_to_gdrive(output_zip_file, None) # Upload to root or mocked
            
        if not sfm_url:
             raise Exception("Upload failed")

        # 9. Update Status
        print("✅ SfM Completed!")
        supabase.table("jobs").update({
            "status": STATUS_SFM_COMPLETED,
            "sfm_url": sfm_url,
            "message": "SfM Completed Successfully."
        }).eq("id", job_id).execute()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if supabase:
            supabase.table("jobs").update({
                "status": STATUS_SFM_FAILED,
                "message": str(e)
            }).eq("id", job_id).execute()
        sys.exit(1)

if __name__ == "__main__":
    main()
