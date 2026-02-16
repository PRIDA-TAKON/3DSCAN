
import os
import subprocess
import sys
import argparse
import time
import shutil
import json
from pathlib import Path

# Fix PATH and Display for Kaggle environments
os.environ["PATH"] = f"/opt/conda/bin:/usr/local/bin:/usr/bin:/bin:{os.environ.get('PATH', '')}"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# --- Configuration & Environment Setup ---

def run_command(cmd, env=None, cwd=None, capture_output=False):
    """Run a shell command and handle errors."""
    print(f"🚀 Running: {cmd}")
    try:
        if capture_output:
            process = subprocess.run(
                cmd, 
                shell=True, 
                check=True, 
                env={**os.environ, **(env or {})}, 
                cwd=cwd,
                text=True,
                capture_output=True
            )
            return process.stdout
        else:
            subprocess.run(
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
        if capture_output:
            return e.stdout or e.stderr
        return False

def setup_base_deps():
    """Install minimal dependencies needed for auto-fetching and status updates."""
    print("📦 Installing base dependencies (Supabase, Requests)...")
    # Using --quiet to keep the log clean during auto-fetch checks
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "supabase", "requests", "gdown", "google-api-python-client", "google-auth-httplib2", "google-auth-oauthlib"], check=True)

def setup_environment():
    """Setup everything needed for the pipeline in Kaggle."""
    print("🛠️ Setting up End-to-End Pipeline Environment...")
    
    # 1. Install Heavy ML/CV Utils
    # dataclass-wizard and pytorch-msssim are required by the Taichi implementation
    run_command("pip install --quiet numpy pandas opencv-python plyfile pyyaml torch torchvision taichi dataclass-wizard pytorch-msssim")
    
    # 2. Install COLMAP if not present
    print("📦 Checking COLMAP...")
    if not run_command("colmap --help"):
        print("📥 Installing COLMAP via apt-get...")
        run_command("apt-get update --quiet && apt-get install -y --quiet colmap")

    # 3. Install Glomap
    print("📦 Installing Glomap (Primary Mapper)...")
    if not run_command("glomap --help"):
        # List of possible conda/mamba locations in Kaggle
        # Prioritize absolute paths to avoid "fake" managers (e.g. python-mamba)
        package_managers = [
            "/opt/conda/bin/conda",
            "/opt/conda/bin/mamba",
            "conda"
        ]
        
        glomap_installed = False
        for pm in package_managers:
            # Check if manager exists
            if pm.startswith("/") and not os.path.exists(pm):
                continue
                
            print(f"📥 Attempting Glomap installation via {pm}...")
            # Use 'install' command carefully
            if run_command(f"{pm} install -c conda-forge glomap -y"):
                glomap_installed = True
                break
        
        if not glomap_installed:
            print("⚠️ Glomap installation failed via all managers. Falling back to COLMAP mapper.")
            print("💡 Note: Glomap is much faster, but COLMAP will still produce a valid result.")
        
    # 4. Setup Taichi 3DGS local package
    print("📦 Setting up Taichi 3DGS local package...")
    repo_path = Path("taichi-splatting-kaggle")
    if repo_path.exists():
        # Install from its requirements.txt manually since its setup.py ignores them
        req_file = repo_path / "requirements.txt"
        if req_file.exists():
            run_command(f"pip install --quiet -r {req_file}")
        run_command("pip install -e .", cwd=str(repo_path))
    else:
        print("⚠️ taichi-splatting-kaggle not found locally.")

# --- Data Handling & Supabase ---

def get_supabase_client():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        return None
    try:
        from supabase import create_client
        # Add a small delay and retry logic for schema cache issues
        for _ in range(3):
            client = create_client(url, key)
            if client: return client
            time.sleep(2)
        return None
    except Exception as e:
        print(f"⚠️ Failed to init Supabase client: {e}")
        return None

def update_status(job_id, status, message=""):
    print(f"🔔 [{job_id}] {status}: {message}")
    supabase = get_supabase_client()
    if not supabase: return
    try:
        supabase.table("jobs").update({
            "status": status,
            "message": message,
            "updated_at": "now()"
        }).eq("id", job_id).execute()
    except Exception as e:
        print(f"⚠️ Supabase update failed: {e}")

def upload_to_gdrive(file_path, folder_id=None):
    """Upload file to Google Drive using Service Account."""
    print(f"📤 Uploading {file_path} to Google Drive...")
    try:
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
        from google.oauth2 import service_account
        
        # Load Service Account from Kaggle Secrets
        creds_json = os.environ.get("GDRIVE_SERVICE_ACCOUNT")
        if not creds_json:
            # Try to read from file if not in env
            secrets_path = Path("/kaggle/input/secrets/gdrive_service_account.json")
            if secrets_path.exists():
                creds_json = secrets_path.read_text()
            else:
                print("⚠️ GDRIVE_SERVICE_ACCOUNT secret not found.")
                return None
                
        import json
        info = json.loads(creds_json)
        creds = service_account.Credentials.from_service_account_info(info)
        service = build('drive', 'v3', credentials=creds)
        
        file_metadata = {'name': Path(file_path).name}
        if folder_id:
            file_metadata['parents'] = [folder_id]
            
        media = MediaFileUpload(file_path, resumable=True)
        file = service.files().create(body=file_metadata, media_body=media, fields='id, webViewLink').execute()
        
        print(f"✅ Upload successful. File ID: {file.get('id')}")
        return file.get('webViewLink')
    except Exception as e:
        print(f"❌ Google Drive upload failed: {e}")
        return None

def fetch_pending_job():
    """Fetch the oldest pending job from Supabase."""
    supabase = get_supabase_client()
    if not supabase:
        print("⚠️ Supabase client not available.")
        return None
    try:
        response = supabase.table("jobs").select("*").eq("status", "PENDING").order("created_at").limit(1).execute()
        if response.data and len(response.data) > 0:
            return response.data[0]
        return None
    except Exception as e:
        print(f"⚠️ Failed to fetch pending job: {e}")
        return None

# --- Pipeline Execution ---

def main():
    parser = argparse.ArgumentParser(description="Kaggle 3DGS Master Pipeline")
    parser.add_argument("--job_id", help="UUID of the job")
    parser.add_argument("--video_url", help="Direct URL or Drive Link")
    parser.add_argument("--auto", action="store_true", help="Automatically fetch pending job from Supabase")
    parser.add_argument("--output_folder_id", help="Optional GDrive folder ID for results")
    parser.add_argument("--output_name", default="result.zip")
    args = parser.parse_args()

    # 0. Setup Base Dependencies (needed for searching Supabase)
    setup_base_deps()

    # Auto-fetch logic
    if args.auto:
        print("🔍 Searching for pending jobs in Supabase...")
        job = fetch_pending_job()
        if job:
            args.job_id = job['id']
            args.video_url = job['video_url']
            print(f"✅ Found job: {args.job_id}")
        else:
            print("😴 No pending jobs found. Exiting.")
            sys.exit(0)

    if not args.job_id or not args.video_url:
        print("❌ Error: --job_id and --video_url are required unless --auto is used.")
        sys.exit(1)

    work_dir = Path("work_dir")
    images_dir = work_dir / "images"
    sfm_dir = work_dir / "sfm"
    train_dir = work_dir / "train_output"
    video_path = work_dir / "input_video.mp4"
    
    for d in [work_dir, images_dir, sfm_dir, train_dir]:
        d.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Setup Full Environment (only if we have a job to do)
        if args.job_id == "ดึง-UUID-จาก-Frontend" or "UUID" in args.job_id:
            raise Exception("❌ คุณลืมเปลี่ยน job_id! กรุณาใส่ UUID จริงจาก Frontend หรือ Supabase ครับ")
            
        update_status(args.job_id, "RUNNING", "Setting up full environment (Glomap, Taichi)...")
        setup_environment()
        
        # 1. Download Video
        if args.video_url == "ลิงก์วิดีโอ" or "ลิงก์" in args.video_url:
            raise Exception("❌ คุณลืมเปลี่ยน video_url! กรุณาใส่ลิงก์ลิงก์วิดีโอ (Direct Link หรือ GDrive) ครับ")
            
        update_status(args.job_id, "RUNNING", "Downloading video...")
        if "drive.google.com" in args.video_url:
            import gdown
            gdown.download(args.video_url, str(video_path), quiet=False, fuzzy=True)
        else:
            import requests
            resp = requests.get(args.video_url, stream=True)
            with open(video_path, 'wb') as f:
                shutil.copyfileobj(resp.raw, f)
            
        # 2. Extract Frames
        update_status(args.job_id, "RUNNING", "Extracting frames...")
        if not run_command(f"python scripts/step1_extract_frames.py --input_video {video_path} --output_dir {images_dir}"):
            raise Exception("Frame extraction failed.")
            
        # 3. Glomap SfM
        update_status(args.job_id, "RUNNING", "Estimating camera poses (Glomap)...")
        if not run_command(f"python scripts/run_glomap.py --images_dir {images_dir} --output_dir {sfm_dir}"):
            raise Exception("SfM failed.")
            
        # 4. Training (Taichi 3DGS)
        update_status(args.job_id, "RUNNING", "Training 3DGS model...")
        config_path = work_dir / "train_config.yaml"
        config_content = f"""
train_dataset_json_path: {str(sfm_dir / 'train.json')}
val_dataset_json_path: {str(sfm_dir / 'val.json')}
pointcloud_parquet_path: {str(sfm_dir / 'point_cloud.parquet')}
summary_writer_log_dir: {str(train_dir / 'logs')}
output_model_dir: {str(train_dir / 'models')}
iterations: 7000
"""
        with open(config_path, "w") as f:
            f.write(config_content)
            
        if not run_command(f"python taichi-splatting-kaggle/gaussian_point_train.py --train_config {config_path}"):
            raise Exception("Training failed.")
            
        # 5. Export & Pack
        update_status(args.job_id, "RUNNING", "Finalizing results...")
        zip_path = f"job_{args.job_id}_{args.output_name}"
        if not zip_path.endswith(".zip"): zip_path += ".zip"
        shutil.make_archive(zip_path.replace(".zip", ""), 'zip', train_dir)
        
        # 6. Upload to Google Drive
        gdrive_link = upload_to_gdrive(zip_path, args.output_folder_id)
        
        if gdrive_link:
            # Update Supabase with the result link
            supabase = get_supabase_client()
            if supabase:
                supabase.table("jobs").update({
                    "result_url": gdrive_link,
                    "status": "COMPLETED",
                    "message": "Model generated and uploaded to Google Drive."
                }).eq("id", args.job_id).execute()
        else:
            raise Exception("Failed to upload results to Google Drive.")
            
    except Exception as e:
        print(f"💥 Pipeline Error: {e}")
        if args.job_id and args.job_id != "ดึง-UUID-จาก-Frontend":
            update_status(args.job_id, "FAILED", str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
