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
STATUS_CONVERSION_RUNNING = "CONVERSION_RUNNING"
STATUS_COMPLETED = "COMPLETED"
STATUS_FAILED = "FAILED"

def install_dependencies():
    """Installs Nerfstudio & Plyfile."""
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "numpy<2.0.0", "supabase", "requests", "gdown", "google-api-python-client", "nerfstudio", "plyfile", "torch", "torchvision"], check=True)

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

def get_job():
    """Finds a job assigned to this kernel (CONVERSION_QUEUED)."""
    from supabase import create_client
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    response = supabase.table("jobs").select("*").eq("status", "CONVERSION_QUEUED").order("created_at").limit(1).execute()
    if not response.data:
        return None, None
    
    job = response.data[0]
    supabase.table("jobs").update({
        "status": STATUS_CONVERSION_RUNNING,
        "message": "Starting Conversion..."
    }).eq("id", job['id']).execute()
    
    return job, supabase

def convert_ply_to_splat(ply_path, splat_path):
    """
    Converts a Gaussian Splatting PLY file to the web-optimized .splat format.
    Ref: https://github.com/antimatter15/splat
    """
    print(f"🔄 Converting {ply_path} to {splat_path}...")
    # This is a placeholder for the actual conversion logic.
    # In a real scenario, we'd paste the conversion code here or import it.
    # For now, we'll just copy it to pretend (logic needs to be added).
    shutil.copy(ply_path, splat_path) 
    # TODO: Implement actual PLY -> SPLAT binary conversion here.

def main():
    print("🎬 Starting Kernel C: Conversion")
    
    # 1. Install Deps
    install_dependencies()
    
    # 2. Get Job
    job, supabase = get_job()
    if not job:
        print("😴 No 'CONVERSION_QUEUED' jobs found. Exiting.")
        return

    job_id = job['id']
    print(f"✅ Processing Job: {job_id}")
    
    try:
        # 3. Setup Work Dir
        work_dir = Path("/kaggle/working/job_" + str(job_id))
        train_output_dir = work_dir / "train_output" # Unzip target
        
        for d in [work_dir, train_output_dir]:
            d.mkdir(parents=True, exist_ok=True)
            
        print(f"📂 Working Directory: {work_dir}")
        
        # 4. Download Training Output (Zip) - Use train_url
        train_url = job.get('train_url')
        if not train_url:
            raise ValueError("No train_url found in job data.")
            
        print(f"⬇️ Downloading Training result from {train_url}...")
        train_zip_path = work_dir / "train_output.zip"
        
        if "drive.google.com" in train_url:
            import gdown
            gdown.download(train_url, str(train_zip_path), quiet=False, fuzzy=True)
        else:
            import requests
            resp = requests.get(train_url, stream=True)
            with open(train_zip_path, 'wb') as f:
                shutil.copyfileobj(resp.raw, f)
        
        if not train_zip_path.exists():
            raise FileNotFoundError("Training Zip download failed.")
            
        # 5. Unzip Training Output
        print("📦 Unzipping Training result...")
        shutil.unpack_archive(str(train_zip_path), str(train_output_dir))
        
        # 6. Run Conversion (Nerfstudio Export -> Splat)
        print("🔄 Converting to SPLAT format...")
        splat_output_path = work_dir / "output.splat"
        
        # step4_export.py searches for config.yml in input_config directory if it's a dir
        cmd_convert = [sys.executable, "step4_export.py", "--input_config", str(train_output_dir), "--output_splat", str(splat_output_path)]
        
        try:
            subprocess.run(cmd_convert, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Conversion failed: {e}")
            raise e
            
        if not splat_output_path.exists():
            raise FileNotFoundError("Splat OUTPUT file not created.")
        
        # 7. Upload
        folder_id = job.get('drive_folder_id')
        if folder_id:
            print(f"📤 Uploading to Drive Folder: {folder_id}")
            result_url = upload_to_gdrive(splat_output_path, folder_id)
        else:
            print("⚠️ No Drive Folder ID provided. Skipping upload (or uploading to root).")
            result_url = upload_to_gdrive(splat_output_path, None) 
            
        if not result_url:
             raise Exception("Upload failed")

        # 8. Update Status
        print("✅ Conversion Completed!")
        supabase.table("jobs").update({
            "status": STATUS_COMPLETED,
            "result_url": result_url,
            "message": "All Stages Completed Successfully."
        }).eq("id", job_id).execute()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if supabase:
            supabase.table("jobs").update({
                "status": STATUS_FAILED,
                "message": f"Conversion Failed: {str(e)}"
            }).eq("id", job_id).execute()
        sys.exit(1)

if __name__ == "__main__":
    main()
