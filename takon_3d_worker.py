import os
import shutil
import time
import subprocess
import zipfile
import requests
from pathlib import Path
import runpod
from supabase import create_client
import boto3
from botocore.config import Config

# --- Configuration ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
WORKER_MODE = os.environ.get("WORKER_MODE", "PROCESS")

S3_ACCESS_KEY = os.environ.get("RUNPOD_S3_ACCESS_KEY") or os.environ.get("S3_ACCESS_KEY") or os.environ.get("ACCESS_KEY")
S3_SECRET_KEY = os.environ.get("RUNPOD_S3_SECRET_KEY") or os.environ.get("S3_SECRET_KEY") or os.environ.get("SECRET_KEY")
S3_ENDPOINT = os.environ.get("RUNPOD_S3_ENDPOINT") or "https://s3api-us-il-1.runpod.io"
S3_BUCKET = os.environ.get("RUNPOD_BUCKET_NAME") or os.environ.get("S3_BUCKET_NAME") or os.environ.get("BUCKET_NAME")

def get_supabase_client():
    if not SUPABASE_URL or not SUPABASE_KEY: return None
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def get_s3_client():
    if not S3_ACCESS_KEY or not S3_SECRET_KEY: return None
    s3_config = Config(signature_version='s3v4', retries={'max_attempts': 3})
    return boto3.client('s3', endpoint_url=S3_ENDPOINT, aws_access_key_id=S3_ACCESS_KEY, aws_secret_access_key=S3_SECRET_KEY, config=s3_config, region_name='us-il-1')

def update_status(job_id, status, message="", result_url=None):
    print(f"🔔 [{job_id}] {status}: {message}", flush=True)
    supabase = get_supabase_client()
    if not supabase: return
    try:
        data = {"status": status, "message": message, "updated_at": "now()"}
        if result_url: data["result_url"] = result_url
        supabase.table("jobs").update(data).eq("id", job_id).execute()
    except Exception as e:
        print(f"⚠️ Supabase update failed: {e}", flush=True)

def run_command(cmd, cwd=None):
    print(f"🚀 Running: {cmd}", flush=True)
    full_output = []
    try:
        with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=cwd) as sp:
            for line in sp.stdout:
                print(line, end="", flush=True)
                full_output.append(line)
        
        if sp.returncode != 0:
            return False, "".join(full_output[-20:]) # ส่ง 20 บรรทัดสุดท้ายกลับไปดู
        return True, ""
    except Exception as e:
        return False, str(e)

def list_files(startpath):
    print(f"📁 Listing files in {startpath}:")
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f'{subindent}{f}')

# --- Sub-Task: PROCESS (SfM) ---
def run_process_mode(job_id, video_url, work_dir):
    update_status(job_id, "processing", "Step 1: SfM Started")
    video_path = work_dir / "input_video.mp4"
    images_dir = work_dir / "images"
    output_dir = work_dir / "processed_data"
    images_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download
    resp = requests.get(video_url, stream=True)
    with open(video_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192): f.write(chunk)

    # 2. Extract & SfM
    run_command(f"ffmpeg -i {video_path} -q:v 2 -vf \"fps=4,scale=-1:720\" -frames:v 60 {images_dir}/frame_%04d.jpg")
    cmd = f"python3 scripts/run_glomap.py --images_dir {images_dir} --output_dir {output_dir}"
    success, err = run_command(cmd)
    
    # 3. Upload
    zip_path = work_dir / "processed.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files: zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), output_dir))

    s3 = get_s3_client()
    remote_path = f"temp/{job_id}/processed.zip"
    s3.upload_file(str(zip_path), S3_BUCKET, remote_path)
    
    update_status(job_id, "ready_to_train", f"S3_PATH:{remote_path}")
    return {"status": "success"}

# --- Sub-Task: TRAIN ---
def run_train_mode(job_id, work_dir):
    supabase = get_supabase_client()
    job_data = supabase.table("jobs").select("message").eq("id", job_id).single().execute()
    msg = job_data.data.get("message", "")
    
    if "S3_PATH:" not in msg: raise Exception(f"S3 path missing in DB")
    remote_temp_path = msg.split("S3_PATH:")[1]
    
    update_status(job_id, "training", "Step 2: Training Started")
    
    zip_path = work_dir / "processed.zip"
    data_dir = work_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Download & Extract
    s3 = get_s3_client()
    s3.download_file(S3_BUCKET, remote_temp_path, str(zip_path))
    with zipfile.ZipFile(zip_path, 'r') as zip_ref: zip_ref.extractall(data_dir)
    
    # [DIAGNOSTIC] เช็คโครงสร้างไฟล์
    list_files(str(data_dir))

    # 2. Train
    # เพิ่ม --viewer.launch-viewer False และเช็คพาธ data_dir
    cmd = f"ns-train splatfacto --data {data_dir} --vis none --viewer.launch-viewer False --max-num-iterations 2000 colmap"
    success, err = run_command(cmd)
    
    if not success:
        raise Exception(f"Training Failed (Code 2). Last Output: {err}")

    update_status(job_id, "completed", "Training Finished")
    return {"status": "success"}

def handler(job):
    job_input = job["input"]
    job_id = job_input.get("id")
    video_url = job_input.get("video_url")
    work_dir = Path(f"/tmp/job_{job_id}")
    if work_dir.exists(): shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        if WORKER_MODE == "PROCESS": return run_process_mode(job_id, video_url, work_dir)
        else: return run_train_mode(job_id, work_dir)
    except Exception as e:
        update_status(job_id, "failed", str(e))
        return {"status": "error", "message": str(e)}
    finally:
        if work_dir.exists(): shutil.rmtree(work_dir)

runpod.serverless.start({"handler": handler})
