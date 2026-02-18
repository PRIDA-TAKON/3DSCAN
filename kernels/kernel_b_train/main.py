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
    except Exception:
        return None

# --- Constants & Supabase Helper ---
SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_KEY = get_secret("SUPABASE_KEY")
GDRIVE_SA_JSON = get_secret("GDRIVE_SERVICE_ACCOUNT")
STATUS_TRAINING_RUNNING = "TRAINING_RUNNING"
STATUS_TRAINING_COMPLETED = "TRAINING_COMPLETED"
STATUS_TRAINING_FAILED = "TRAINING_FAILED"

def install_dependencies():
    """Installs Nerfstudio & Deps."""
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "numpy<2.0.0", "supabase", "requests", "gdown", "google-api-python-client", "nerfstudio", "torch", "torchvision"], check=True)
    
def upload_to_gdrive(file_path, folder_id):
    """Uploads file to GDrive using Service Account."""
    print(f"📤 Uploading {file_path}...")
    try:
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
        from google.oauth2 import service_account
        
        creds_json = os.environ.get("GDRIVE_SERVICE_ACCOUNT")
        if not creds_json:
            print("⚠️ GDRIVE_SERVICE_ACCOUNT secret not found.")
            return None
            
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

def run_command(cmd, check=True):
    print(f"🚀 Running: {cmd}")
    subprocess.run(cmd, shell=True, check=check)

def get_job():
    """Finds a job assigned to this kernel (TRAINING_QUEUED)."""
    from supabase import create_client
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    response = supabase.table("jobs").select("*").eq("status", "TRAINING_QUEUED").order("created_at").limit(1).execute()
    if not response.data:
        return None, None
    
    job = response.data[0]
    supabase.table("jobs").update({
        "status": STATUS_TRAINING_RUNNING,
        "message": "Starting Training..."
    }).eq("id", job['id']).execute()
    
    return job, supabase

def main():
    print("🎬 Starting Kernel B: Training")
    
    # 1. Install Deps
    install_dependencies()
    
    # 2. Get Job or Use Debug SfM Outcome
    debug_sfm_url = get_secret("DEBUG_SFM_URL")
    if debug_sfm_url:
        print(f"🛠️ DEBUG MODE: Using SfM result URL: {debug_sfm_url}")
        job = {
            'id': "debug_job_" + str(int(time.time())),
            'sfm_url': debug_sfm_url,
            'drive_folder_id': get_secret("DEBUG_DRIVE_FOLDER")
        }
        supabase = None
        if SUPABASE_URL and SUPABASE_KEY:
            from supabase import create_client
            try:
                supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            except: pass
    else:
        job, supabase = get_job()
        if not job:
            print("😴 No 'TRAINING_QUEUED' jobs found. Exiting.")
            return

    job_id = job['id']
    print(f"✅ Processing Job: {job_id}")
    
    try:
        # 3. Setup Work Dir
        work_dir = Path("/kaggle/working/job_" + str(job_id))
        project_dir = work_dir / "project" # Where we unzip sfm_output
        output_dir = work_dir / "train_output"
        
        for d in [work_dir, project_dir, output_dir]:
            d.mkdir(parents=True, exist_ok=True)
            
        print(f"📂 Working Directory: {work_dir}")
        
        # 4. Download SfM Output (sfm_url from previous step)
        sfm_url = job.get('sfm_url')
        if not sfm_url:
            raise ValueError("No sfm_url found in job data.")
            
        print(f"⬇️ Downloading SfM result from {sfm_url}...")
        sfm_zip_path = work_dir / "sfm_output.zip"
        
        if "drive.google.com" in sfm_url:
            import gdown
            gdown.download(sfm_url, str(sfm_zip_path), quiet=False, fuzzy=True)
        else:
            import requests
            resp = requests.get(sfm_url, stream=True)
            with open(sfm_zip_path, 'wb') as f:
                shutil.copyfileobj(resp.raw, f)
        
        if not sfm_zip_path.exists():
            raise FileNotFoundError("SfM Zip download failed.")
            
        # 5. Unzip SfM
        print("📦 Unzipping SfM result...")
        shutil.unpack_archive(str(sfm_zip_path), str(project_dir))
        
        # 6. Run Training (Nerfstudio)
        # Assuming Nerfstudio is installed in the env (via setup_environment or base image)
        # If not, install_dependencies should handle it (commented out for now as heavy)
        print("🔥 Training Nerfstudio Splatfacto...")
        
        # step3_train_splatting.py handles ns-train command
        cmd_train = [sys.executable, "step3_train_splatting.py", "--project_path", str(project_dir), "--output_path", str(output_dir), "--iterations", "30000"]
        
        # Check if GPU is available (Kaggle should have it)
        try:
            subprocess.run(cmd_train, check=True)
        except subprocess.CalledProcessError as e:
            # Maybe try to debug output
            print(f"❌ Training failed: {e}")
            raise e
        
        # 7. Zip Result
        print("📦 Zipping training results...")
        output_zip_path = work_dir / "train_output" # shutil.make_archive adds .zip
        shutil.make_archive(str(output_zip_path), 'zip', output_dir)
        output_zip_file = str(output_zip_path) + ".zip"
        
        # 8. Upload
        if job.get('drive_folder_id'):
            print(f"📤 Uploading to Drive Folder: {job.get('drive_folder_id')}")
            train_url = upload_to_gdrive(output_zip_file, job.get('drive_folder_id'))
        else:
            print("⚠️ No Drive Folder ID provided. Skipping upload (or uploading to root).")
            train_url = upload_to_gdrive(output_zip_file, None) 
            
        if not train_url:
             raise Exception("Upload failed")

        # 9. Update Status
        print("✅ Training Completed!")
        supabase.table("jobs").update({
            "status": STATUS_TRAINING_COMPLETED,
            "train_url": train_url,
            "message": "Training Completed Successfully."
        }).eq("id", job_id).execute()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if supabase:
            supabase.table("jobs").update({
                "status": STATUS_TRAINING_FAILED,
                "message": str(e)
            }).eq("id", job_id).execute()
        sys.exit(1)

if __name__ == "__main__":
    main()
