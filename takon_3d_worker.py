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
            return False, "".join(full_output[-20:])
        return True, ""
    except Exception as e:
        return False, str(e)

def get_dir_structure(path):
    structure = []
    for root, dirs, files in os.walk(path):
        rel_path = os.path.relpath(root, path)
        structure.append(f"{rel_path}/: {files}")
    return "\n".join(structure)

# --- Sub-Task: TRAIN ---
def run_train_mode(job_id, work_dir):
    supabase = get_supabase_client()
    job_data = supabase.table("jobs").select("message").eq("id", job_id).single().execute()
    msg = job_data.data.get("message", "")
    
    if "S3_PATH:" not in msg: raise Exception(f"S3 path missing in DB")
    remote_temp_path = msg.split("S3_PATH:")[1]
    
    update_status(job_id, "training", "Step 2: Training Started")
    
    zip_path = work_dir / "processed.zip"
    raw_data_dir = work_dir / "raw_data"
    final_data_dir = work_dir / "data"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    final_data_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Download & Extract
    s3 = get_s3_client()
    s3.download_file(S3_BUCKET, remote_temp_path, str(zip_path))
    with zipfile.ZipFile(zip_path, 'r') as zip_ref: zip_ref.extractall(raw_data_dir)
    
    # 🛠️ Force Restructure
    final_images_dir = final_data_dir / "images"
    final_images_dir.mkdir(parents=True, exist_ok=True)
    colmap_sparse_dir = final_data_dir / "colmap" / "sparse" / "0"
    colmap_sparse_dir.mkdir(parents=True, exist_ok=True)

    # ค้นหาและย้ายรูปภาพ
    for img_path in Path(raw_data_dir).rglob("*.jpg"):
        shutil.copy(img_path, final_images_dir / img_path.name)

    # ค้นหาและย้ายไฟล์ COLMAP .bin
    found_bins = 0
    for bin_path in Path(raw_data_dir).rglob("*.bin"):
        shutil.copy(bin_path, colmap_sparse_dir / bin_path.name)
        found_bins += 1

    # ตรวจสอบความถูกต้อง
    if found_bins == 0:
        struct = get_dir_structure(raw_data_dir)
        raise Exception(f"CRITICAL: No .bin files found in zip! Structure:\n{struct}")

    # 2. Train
    # รันใน final_data_dir เพื่อความชัวร์
    cmd = f"ns-train splatfacto --data . --vis tensorboard --max-num-iterations 2000 colmap"
    success, err = run_command(cmd, cwd=str(final_data_dir))
    
    if not success:
        struct = get_dir_structure(final_data_dir)
        raise Exception(f"Training Failed. Structure:\n{struct}\nLog: {err}")

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
        if WORKER_MODE == "PROCESS": return {"status": "success"}
        else: return run_train_mode(job_id, work_dir)
    except Exception as e:
        update_status(job_id, "failed", str(e))
        return {"status": "error", "message": str(e)}
    finally:
        if work_dir.exists(): shutil.rmtree(work_dir)

runpod.serverless.start({"handler": handler})
