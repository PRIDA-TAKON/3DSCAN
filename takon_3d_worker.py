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
    try:
        with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=cwd) as sp:
            for line in sp.stdout:
                print(line, end="", flush=True)
        return sp.returncode == 0, ""
    except Exception as e:
        return False, str(e)

# --- Sub-Task: PROCESS (SfM) ---
def run_process_mode(job_id, video_url, work_dir):
    update_status(job_id, "processing", "Step 1: Extracting Frames & SfM")
    video_path = work_dir / "input_video.mp4"
    images_dir = work_dir / "images"
    output_dir = work_dir / "processed_data"
    images_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download Video
    resp = requests.get(video_url, stream=True)
    with open(video_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192): f.write(chunk)

    # 2. Extract Frames (4fps, 720p)
    run_command(f"ffmpeg -i {video_path} -q:v 2 -vf \"fps=4,scale=-1:720\" -frames:v 100 {images_dir}/frame_%04d.jpg")

    # 3. SfM (GLOMAP)
    print("🎬 Running GLOMAP SfM...", flush=True)
    cmd = f"python3 scripts/run_glomap.py --images_dir {images_dir} --output_dir {output_dir}"
    success, err = run_command(cmd)
    if not success:
        print("⚠️ GLOMAP Failed, falling back to ns-process-data...")
        run_command(f"ns-process-data images --data {images_dir} --output-dir {output_dir}")

    # 4. Packaging (Fixed: Include images folder)
    print("📦 Packaging data for trainer...", flush=True)
    zip_path = work_dir / "processed.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # ใส่ไฟล์จาก output_dir
        for root, dirs, files in os.walk(output_dir):
            for file in files: zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), output_dir))
        # ใส่รูปภาพจาก images_dir (สำคัญ!)
        for root, dirs, files in os.walk(images_dir):
            for file in files: zipf.write(os.path.join(root, file), os.path.join("images", os.path.relpath(os.path.join(root, file), images_dir)))

    # 5. Upload to S3
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
    
    update_status(job_id, "training", "Step 2: Training Gaussian Splatting")
    
    zip_path = work_dir / "processed.zip"
    raw_data_dir = work_dir / "raw_data"
    final_data_dir = work_dir / "data"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    final_data_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Download & Extract
    s3 = get_s3_client()
    s3.download_file(S3_BUCKET, remote_temp_path, str(zip_path))
    with zipfile.ZipFile(zip_path, 'r') as zip_ref: zip_ref.extractall(raw_data_dir)
    
    # 2. Restructure for Nerfstudio
    print("🛠️ Restructuring data for Nerfstudio...", flush=True)
    # Move images
    shutil.copytree(raw_data_dir / "images", final_data_dir / "images", dirs_exist_ok=True)
    # Move colmap sparse data
    colmap_dest = final_data_dir / "colmap" / "sparse" / "0"
    colmap_dest.mkdir(parents=True, exist_ok=True)
    
    # ค้นหาไฟล์ .bin (รองรับทั้งที่อยู่ใน sparse/0/ หรือรูท)
    for bin_file in raw_data_dir.rglob("*.bin"):
        shutil.copy(bin_file, colmap_dest / bin_file.name)

    # 3. Train
    print("🧠 Starting ns-train...", flush=True)
    cmd = f"ns-train splatfacto --data . --vis tensorboard --max-num-iterations 2000 colmap"
    success, _ = run_command(cmd, cwd=str(final_data_dir))
    if not success: raise Exception("Training failed (Command error)")

    # 4. Export & Finish
    update_status(job_id, "exporting", "Exporting model...")
    train_out = final_data_dir / "outputs"
    config_file = list(train_out.rglob("config.yml"))[0]
    ply_path = work_dir / "result.ply"
    run_command(f"ns-export gaussian-splat --load-config {config_file} --output-path {ply_path}")
    
    final_path = f"results/{job_id}/model.ply"
    s3.upload_file(str(ply_path), S3_BUCKET, final_path)
    res_url = f"{S3_ENDPOINT}/{S3_BUCKET}/{final_path}"
    
    update_status(job_id, "completed", "Job Finished!", result_url=res_url)
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
