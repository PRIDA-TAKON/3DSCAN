import os
import sys
import shutil
import time
import argparse
import subprocess
from pathlib import Path
from supabase import create_client

def get_supabase_client():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        return None
    return create_client(url, key)

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

def run_command(cmd, cwd=None):
    """Run a shell command and handle errors."""
    print(f"🚀 Running: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True, text=True, cwd=cwd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {cmd}\\n   Error: {e}")
        return False

def upload_to_gdrive(file_path, folder_id=None):
    """Upload file to Google Drive using Service Account."""
    print(f"📤 Uploading {file_path} to Google Drive...")
    try:
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
        from google.oauth2 import service_account
        
        # Load Service Account
        creds_json = os.environ.get("GDRIVE_SERVICE_ACCOUNT")
        if not creds_json:
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

def main():
    parser = argparse.ArgumentParser(description="Google Cloud Run 3DGS Worker")
    parser.add_argument("--job_id", help="UUID of the job", default=os.environ.get("JOB_ID"))
    args = parser.parse_args()

    job_id = args.job_id
    if not job_id:
        print("❌ Error: JOB_ID environment variable or --job_id argument is required.")
        sys.exit(1)

    print(f"🤖 Booting Cloud Run Worker for Job: {job_id}")
    supabase = get_supabase_client()
    if not supabase:
        print("❌ Error: Supabase credentials missing.")
        sys.exit(1)

    # Fetch Job Information
    try:
        response = supabase.table("jobs").select("*").eq("id", job_id).execute()
        if not response.data:
            print("❌ Error: Job not found in Supabase.")
            sys.exit(1)
        job = response.data[0]
        video_url = job.get("video_url")
    except Exception as e:
        print(f"❌ Error fetching job from Supabase: {e}")
        sys.exit(1)

    if not video_url:
        print("❌ Error: video_url is missing for this job.")
        update_status(job_id, "FAILED", "video_url missing.")
        sys.exit(1)

    # Prepare directories
    work_dir = Path("/tmp/work_dir")
    images_dir = work_dir / "images"
    sfm_dir = work_dir / "sfm"
    train_dir = work_dir / "train_output"
    video_path = work_dir / "input_video.mp4"
    
    for d in [work_dir, images_dir, sfm_dir, train_dir]:
        d.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Download Video
        update_status(job_id, "RUNNING", "Downloading video...")
        if "drive.google.com" in video_url:
            import gdown
            gdown.download(video_url, str(video_path), quiet=False, fuzzy=True)
        else:
            import requests
            resp = requests.get(video_url, stream=True)
            with open(video_path, 'wb') as f:
                shutil.copyfileobj(resp.raw, f)
            
        # 2. Extract Frames
        update_status(job_id, "RUNNING", "Extracting frames...")
        if not run_command(f"python scripts/step1_extract_frames.py --input_video {video_path} --output_dir {images_dir}"):
            raise Exception("Frame extraction failed.")
            
        # 3. SfM (Glomap / Colmap)
        update_status(job_id, "RUNNING", "Estimating camera poses (SfM)...")
        # Ensure scripts are available; if run_glomap.py is set up, use it
        if not run_command(f"python scripts/run_glomap.py --images_dir {images_dir} --output_dir {sfm_dir}"):
            raise Exception("SfM failed.")
            
        # 4. Training (Taichi 3DGS)
        update_status(job_id, "RUNNING", "Training 3DGS model...")
        config_path = work_dir / "train_config.yaml"
        config_content = f\"\"\"
train_dataset_json_path: {str(sfm_dir / 'train.json')}
val_dataset_json_path: {str(sfm_dir / 'val.json')}
pointcloud_parquet_path: {str(sfm_dir / 'point_cloud.parquet')}
summary_writer_log_dir: {str(train_dir / 'logs')}
output_model_dir: {str(train_dir / 'models')}
num_iterations: 30000
densify_until_iter: 10000
position_lr_final: 1e-6
\"\"\"
        with open(config_path, "w") as f:
            f.write(config_content)

        if not run_command(f"python /app/taichi-splatting-kaggle/gaussian_point_train.py --train_config {config_path}"):
            raise Exception("Training failed.")
            
        # 5. Export & Pack
        update_status(job_id, "RUNNING", "Finalizing results...")
        zip_path = f"/tmp/job_{job_id}_result"
        shutil.make_archive(zip_path, 'zip', train_dir)
        zip_file = zip_path + ".zip"
        
        # 6. Upload to Google Drive
        gdrive_folder_id = os.environ.get("GDRIVE_OUTPUT_FOLDER_ID")
        gdrive_link = None
        for attempt in range(3):
            gdrive_link = upload_to_gdrive(zip_file, gdrive_folder_id)
            if gdrive_link:
                break
            print(f"⚠️ Upload attempt {attempt+1} failed. Retrying in 5 seconds...")
            time.sleep(5)
        
        if gdrive_link:
            update_status(job_id, "COMPLETED", "Model generated successfully.")
            # Also update result_url directly
            supabase.table("jobs").update({
                "result_url": gdrive_link
            }).eq("id", job_id).execute()
        else:
            raise Exception("Failed to upload results to Google Drive.")
            
    except Exception as e:
        print(f"💥 Worker Error: {e}")
        update_status(job_id, "FAILED", str(e))
        sys.exit(1)
    finally:
        # Cleanup
        try:
            shutil.rmtree(work_dir)
        except Exception:
            pass

if __name__ == "__main__":
    main()
