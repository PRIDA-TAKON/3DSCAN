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

# --- Configuration (Verified Sync) ---
# Last Sync: 2026-04-15 10:15 (v1.0.3)
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
    access = S3_ACCESS_KEY.strip()
    secret = S3_SECRET_KEY.strip()
    s3_config = Config(
        signature_version='s3v4',
        retries={'max_attempts': 3},
        s3={'addressing_style': 'path'}
    )
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        config=s3_config,
        region_name='us-il-1'
    )

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
    print(f"📥 Downloading video: {video_url}", flush=True)
    try:
        resp = requests.get(video_url, stream=True, timeout=30)
        if resp.status_code != 200: raise Exception(f"Video download failed: {resp.status_code}")
        with open(video_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192): f.write(chunk)
    except Exception as e:
        raise Exception(f"Failed to download video: {str(e)}")

    # 2. Extract Frames
    print("🎞️ Extracting frames (Optimized)...", flush=True)
    success, err = run_command(f"ffmpeg -i {video_path} -q:v 2 -vf \"fps=2\" -frames:v 300 {images_dir}/frame_%04d.jpg")
    if not success:
        run_command(f"ffmpeg -i {video_path} -q:v 4 -vf \"fps=2,scale=-1:720\" -frames:v 200 {images_dir}/frame_%04d.jpg")
    
    extracted_imgs = list(images_dir.glob("*.jpg"))
    print(f"📸 Extracted {len(extracted_imgs)} images.", flush=True)
    if len(extracted_imgs) < 10: raise Exception(f"Insufficient frames extracted ({len(extracted_imgs)}).")

    # 3. SfM
    print("🎬 Running SfM Pipeline...", flush=True)
    success, err = run_command(f"python3 scripts/run_glomap.py --images_dir {images_dir} --output_dir {output_dir}")
    if not success:
        print(f"⚠️ SfM Pipeline failed: {err}. Retrying with basic COLMAP...", flush=True)
        success, err = run_command(f"python3 step2_colmap_sfm.py --images_dir {images_dir} --output_dir {output_dir}")
        if not success: raise Exception(f"SfM failed after retries: {err}")

    # 4. Packaging
    print("📦 Packaging data...", flush=True)
    zip_path = work_dir / "processed.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, output_dir)
                zipf.write(abs_path, rel_path)
        
        img_count = 0
        for root, dirs, files in os.walk(images_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.join("images", os.path.relpath(abs_path, images_dir))
                    zipf.write(abs_path, rel_path)
                    img_count += 1
        
    print(f"✅ Packaged {img_count} images.", flush=True)
    
    # 5. Upload
    s3 = get_s3_client()
    if not s3: raise Exception("S3 Client configuration missing!")
    
    remote_path = f"temp/{job_id}/processed.zip"
    print(f"📤 Uploading to S3: {remote_path}...", flush=True)
    s3.upload_file(str(zip_path), S3_BUCKET, remote_path)
    
    update_status(job_id, "ready_to_train", f"S3_PATH:{remote_path}")
    return {"status": "success"}

# --- Sub-Task: TRAIN ---
def run_train_mode(job_id, work_dir):
    print(f"🧠 [TRAIN] v1.0.3-no-prompt | Starting Job: {job_id}", flush=True)
    supabase = get_supabase_client()
    job_data = supabase.table("jobs").select("message").eq("id", job_id).single().execute()
    msg = job_data.data.get("message", "")
    
    if "S3_PATH:" not in msg:
        raise Exception(f"S3 path missing in DB. Current msg: {msg}")
    
    remote_temp_path = msg.split("S3_PATH:")[1].strip()
    print(f"🔗 [TRAIN] Remote Path: {remote_temp_path}", flush=True)
    
    update_status(job_id, "training", f"Step 2: Training (v1.0.3) | S3_PATH:{remote_temp_path}")
    
    zip_path = work_dir / "processed.zip"
    raw_data_dir = work_dir / "raw_data"
    final_data_dir = work_dir / "data"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    final_data_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Download
    s3 = get_s3_client()
    print(f"📥 [TRAIN] Downloading from S3...", flush=True)
    s3.download_file(S3_BUCKET, remote_temp_path, str(zip_path))
    
    # 2. Extract
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(raw_data_dir)
    
    # 3. Restructure
    img_dest = final_data_dir / "images"
    img_dest.mkdir(parents=True, exist_ok=True)
    colmap_dest = final_data_dir / "colmap" / "sparse" / "0"
    colmap_dest.mkdir(parents=True, exist_ok=True)

    found_imgs = 0
    for img in Path(raw_data_dir).rglob("*.jpg"):
        shutil.copy(img, img_dest / img.name)
        found_imgs += 1
    
    found_bins = 0
    for bin_f in Path(raw_data_dir).rglob("*.bin"):
        shutil.copy(bin_f, colmap_dest / bin_f.name)
        found_bins += 1

    print(f"🔍 [TRAIN] Data Check: Images={found_imgs}, Bins={found_bins}", flush=True)
    if found_imgs == 0 or found_bins == 0:
        raise Exception(f"Missing critical data: Images={found_imgs}, Bins={found_bins}")

    # 4. Run Training
    print("🔥 [TRAIN] Starting ns-train (splatfacto)...", flush=True)
    # 📝 v1.0.3: เพิ่ม downscale-factor 1 และจัดลำดับ Flag ใหม่
    train_cmd = (
        f"ns-train splatfacto --data . --vis tensorboard --max-num-iterations 2000 "
        f"--pipeline.datamanager.dataparser.downscale-factor 1 "
        f"colmap --colmap-path colmap/sparse/0 --images-path images"
    )
    
    success, err = run_command(train_cmd, cwd=str(final_data_dir))
    
    if not success:
        update_status(job_id, "failed", f"Training failed: {err} | S3_PATH:{remote_temp_path}")
        raise Exception(f"Training failed: {err}")

    # 5. Export
    print("📤 [TRAIN] Exporting model to PLY...", flush=True)
    update_status(job_id, "exporting", f"Exporting model... | S3_PATH:{remote_temp_path}")
    train_out = final_data_dir / "outputs"
    config_yml = list(train_out.rglob("config.yml"))
    if not config_yml:
        raise Exception("Training finished but config.yml not found!")
        
    config_file = config_yml[0]
    ply_path = work_dir / "result.ply"
    success, err = run_command(f"ns-export gaussian-splat --load-config {config_file} --output-path {ply_path}")
    if not success: raise Exception(f"Export failed: {err}")
    
    # 6. Upload Result
    final_path = f"results/{job_id}/model.ply"
    s3.upload_file(str(ply_path), S3_BUCKET, final_path)
    res_url = f"{S3_ENDPOINT}/{S3_BUCKET}/{final_path}"
    
    update_status(job_id, "completed", "Job Finished!", result_url=res_url)
    return {"status": "success", "result_url": res_url}

def handler(job):
    job_input = job["input"]
    job_id = job_input.get("id")
    video_url = job_input.get("video_url")
    current_mode = job_input.get("mode", WORKER_MODE) 
    
    print(f"🚀 [WORKER] Job ID: {job_id} | Mode: {current_mode}", flush=True)
    
    work_dir = Path(f"/tmp/job_{job_id}")
    if work_dir.exists(): shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if current_mode == "PROCESS": return run_process_mode(job_id, video_url, work_dir)
        elif current_mode == "TRAIN": return run_train_mode(job_id, work_dir)
        else: raise Exception(f"Unknown mode: {current_mode}")
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if work_dir.exists(): shutil.rmtree(work_dir)

runpod.serverless.start({"handler": handler})
