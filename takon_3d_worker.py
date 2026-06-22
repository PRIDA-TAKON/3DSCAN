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
# Version: v1.2.1-quit-on-completion
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
WORKER_MODE = os.environ.get("WORKER_MODE", "PROCESS")

S3_ACCESS_KEY = os.environ.get("RUNPOD_S3_ACCESS_KEY") or os.environ.get("S3_ACCESS_KEY") or os.environ.get("ACCESS_KEY")
S3_SECRET_KEY = os.environ.get("RUNPOD_S3_SECRET_KEY") or os.environ.get("S3_SECRET_KEY") or os.environ.get("SECRET_KEY")
S3_ENDPOINT = os.environ.get("RUNPOD_S3_ENDPOINT") or "https://s3api-us-il-1.runpod.io"
S3_BUCKET = os.environ.get("RUNPOD_BUCKET_NAME") or os.environ.get("S3_BUCKET_NAME") or os.environ.get("BUCKET_NAME")

print(f"🔍 [S3 CONFIG DEBUG] Endpoint: {S3_ENDPOINT} | Bucket: {S3_BUCKET}", flush=True)
if S3_ACCESS_KEY:
    print(f"🔍 [S3 CONFIG DEBUG] Access Key Prefix: {S3_ACCESS_KEY[:5]}...", flush=True)
if S3_SECRET_KEY:
    print(f"🔍 [S3 CONFIG DEBUG] Secret Key Prefix: {S3_SECRET_KEY[:5]}... Length: {len(S3_SECRET_KEY)}", flush=True)

def get_supabase_client():
    if not SUPABASE_URL or not SUPABASE_KEY: return None
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def get_s3_client():
    if not S3_ACCESS_KEY or not S3_SECRET_KEY: return None
    s3_config = Config(signature_version='s3v4', retries={'max_attempts': 3}, s3={'addressing_style': 'path'})
    return boto3.client('s3', endpoint_url=S3_ENDPOINT, aws_access_key_id=S3_ACCESS_KEY.strip(), aws_secret_access_key=S3_SECRET_KEY.strip(), config=s3_config, region_name='us-il-1')

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

# --- Sub-Task: PROCESS ---
def run_process_mode(job_id, video_url, work_dir):
    update_status(job_id, "SFM_RUNNING", "Step 1: Extracting Frames & SfM")
    video_path = work_dir / "input_video.mp4"
    images_dir = work_dir / "images"
    output_dir = work_dir / "processed_data"
    images_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📥 Downloading video: {video_url}", flush=True)
    resp = requests.get(video_url, stream=True, timeout=30)
    with open(video_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192): f.write(chunk)

    print("🎞️ Extracting frames...", flush=True)
    run_command(f"ffmpeg -i {video_path} -q:v 2 -vf \"fps=2\" -frames:v 300 {images_dir}/frame_%04d.jpg")
    
    print("🎬 Running SfM Pipeline...", flush=True)
    success, err = run_command(f"python3 scripts/run_glomap.py --images_dir {images_dir} --output_dir {output_dir}")
    if not success:
        success, err = run_command(f"python3 step2_colmap_sfm.py --images_dir {images_dir} --output_dir {output_dir}")
        if not success:
            update_status(job_id, "SFM_FAILED", f"SfM Error: {err}")
            return {"status": "error", "message": err}

    print("📦 Packaging data...", flush=True)
    zip_path = work_dir / "processed.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                zipf.write(abs_path, os.path.relpath(abs_path, output_dir))
        for img in images_dir.glob("*.jpg"):
            zipf.write(str(img), f"images/{img.name}")
    
    remote_path = f"temp/{job_id}/processed.zip"
    print(f"📤 Uploading processed.zip to Supabase Storage: {remote_path}...", flush=True)
    try:
        supabase = get_supabase_client()
        with open(str(zip_path), 'rb') as f:
            supabase.storage.from_("3d-scans").upload(
                path=remote_path,
                file=f,
                file_options={"content-type": "application/zip", "x-upsert": "true"}
            )
        print("✅ Uploaded processed.zip to Supabase Storage successfully!", flush=True)
    except Exception as e:
        print(f"❌ Supabase Storage upload failed: {e}", flush=True)
        raise Exception(f"Supabase Storage upload failed: {e}")
        
    update_status(job_id, "SFM_COMPLETED", f"S3_PATH:{remote_path}")
    return {"status": "success"}

# --- Sub-Task: FULL (End-to-End) ---
def run_full_mode(job_id, video_url, work_dir):
    print(f"🚀 [FULL] End-to-End Processing | Job: {job_id}", flush=True)
    update_status(job_id, "SFM_RUNNING", "Step 1/2: Extracting Frames & SfM")
    
    video_path = work_dir / "input_video.mp4"
    images_dir = work_dir / "images"
    sfm_output_dir = work_dir / "processed_data"
    images_dir.mkdir(parents=True, exist_ok=True)
    sfm_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📥 Downloading video: {video_url}", flush=True)
    resp = requests.get(video_url, stream=True, timeout=30)
    with open(video_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192): f.write(chunk)

    print("🎞️ Extracting frames...", flush=True)
    run_command(f"ffmpeg -i {video_path} -q:v 2 -vf \"fps=2\" -frames:v 300 {images_dir}/frame_%04d.jpg")
    
    print("🎬 Running SfM Pipeline...", flush=True)
    success, err = run_command(f"python3 scripts/run_glomap.py --images_dir {images_dir} --output_dir {sfm_output_dir}")
    if not success:
        success, err = run_command(f"python3 step2_colmap_sfm.py --images_dir {images_dir} --output_dir {sfm_output_dir}")
        if not success:
            update_status(job_id, "FAILED", f"SfM Error: {err}")
            return {"status": "error", "message": err}

    update_status(job_id, "TRAINING_RUNNING", "Step 2/2: Training 3DGS Model (2000 Iterations)")

    # Restructure for Nerfstudio directly
    final_data_dir = work_dir / "data"
    final_data_dir.mkdir(parents=True, exist_ok=True)
    img_dest = final_data_dir / "images"
    img_dest.mkdir(parents=True, exist_ok=True)
    colmap_dest = final_data_dir / "colmap" / "sparse" / "0"
    colmap_dest.mkdir(parents=True, exist_ok=True)

    # Copy files
    for img in images_dir.glob("*.jpg"): 
        shutil.copy(img, img_dest / img.name)
        
    bin_files_copied = 0
    for bin_f in sfm_output_dir.rglob("*.bin"):
        shutil.copy(bin_f, colmap_dest / bin_f.name)
        bin_files_copied += 1

    if bin_files_copied == 0:
        # Fallback to sparse root
        for bin_f in (sfm_output_dir / "sparse").glob("*.bin"):
            shutil.copy(bin_f, colmap_dest / bin_f.name)
            bin_files_copied += 1

    if bin_files_copied == 0:
        err_msg = "SfM finished but no camera sparse binaries (.bin) found!"
        update_status(job_id, "FAILED", err_msg)
        return {"status": "error", "message": err_msg}

    print("🔥 Starting ns-train...", flush=True)
    train_cmd = (
        f"ns-train splatfacto --max-num-iterations 2000 --vis tensorboard --viewer.quit-on-train-completion True "
        f"colmap --data . --colmap-path colmap/sparse/0 --images-path images --downscale-factor 1"
    )
    success, err = run_command(train_cmd, cwd=str(final_data_dir))
    
    if not success:
        update_status(job_id, "FAILED", f"Train error: {err}")
        return {"status": "error", "message": f"ns-train failed: {err}"}

    print("📤 Exporting PLY...", flush=True)
    config_yml = list((final_data_dir / "outputs").rglob("config.yml"))[0]
    export_dir = work_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    
    success, err = run_command(f"ns-export gaussian-splat --load-config {config_yml} --output-dir {export_dir}", cwd=str(final_data_dir))
    if not success:
        update_status(job_id, "FAILED", f"Export failed: {err}")
        return {"status": "error", "message": f"Export failed: {err}"}
    
    ply_files = list(export_dir.glob("*.ply"))
    if not ply_files:
        update_status(job_id, "FAILED", "Export finished but no .ply file found!")
        return {"status": "error", "message": "Export finished but no .ply file found!"}
    
    final_path = f"results/{job_id}/model.ply"
    print(f"📤 Uploading final PLY to Supabase Storage: {final_path}...", flush=True)
    try:
        supabase = get_supabase_client()
        with open(str(ply_files[0]), 'rb') as f:
            supabase.storage.from_("3d-scans").upload(
                path=final_path,
                file=f,
                file_options={"content-type": "application/octet-stream", "x-upsert": "true"}
            )
        res_url = supabase.storage.from_("3d-scans").get_public_url(final_path)
        print(f"✅ Uploaded to Supabase! Public URL: {res_url}", flush=True)
    except Exception as e:
        print(f"❌ Supabase Storage upload failed: {e}", flush=True)
        raise Exception(f"Supabase Storage upload failed: {e}")
    
    update_status(job_id, "COMPLETED", "Job Finished!", result_url=res_url)
    return {"status": "success", "result_url": res_url}

# --- Sub-Task: TRAIN ---
def run_train_mode(job_id, work_dir):
    print(f"🧠 [TRAIN] v1.2.0-supabase-storage | Job: {job_id}", flush=True)
    supabase = get_supabase_client()
    res = supabase.table("jobs").select("message").eq("id", job_id).single().execute()
    msg = res.data.get("message") if res.data else ""
    
    if not msg or "S3_PATH:" not in msg:
        update_status(job_id, "TRAINING_FAILED", "Invalid message in DB: S3_PATH missing.")
        raise Exception(f"Invalid message in DB: {msg}. Need S3_PATH.")
    
    remote_temp_path = msg.split("S3_PATH:")[1].strip()
    update_status(job_id, "TRAINING_RUNNING", f"Step 2: Training (v1.0.9) | S3_PATH:{remote_temp_path}")
    
    zip_path = work_dir / "processed.zip"
    final_data_dir = work_dir / "data"
    final_data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 [TRAIN] Downloading data from Supabase Storage: {remote_temp_path}...", flush=True)
    try:
        supabase = get_supabase_client()
        with open(str(zip_path), "wb") as f:
            res = supabase.storage.from_("3d-scans").download(remote_temp_path)
            f.write(res)
        print("✅ [TRAIN] Downloaded data from Supabase Storage successfully!", flush=True)
    except Exception as e:
        print(f"❌ [TRAIN] Supabase Storage download failed: {e}", flush=True)
        raise Exception(f"Supabase Storage download failed: {e}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(work_dir / "raw")
    
    # Restructure for Nerfstudio
    img_dest = final_data_dir / "images"
    img_dest.mkdir(parents=True, exist_ok=True)
    colmap_dest = final_data_dir / "colmap" / "sparse" / "0"
    colmap_dest.mkdir(parents=True, exist_ok=True)

    for img in (work_dir / "raw").rglob("*.jpg"): shutil.copy(img, img_dest / img.name)
    for bin_f in (work_dir / "raw").rglob("*.bin"): shutil.copy(bin_f, colmap_dest / bin_f.name)

    print("🔥 [TRAIN] Starting ns-train...", flush=True)
    train_cmd = (
        f"ns-train splatfacto --max-num-iterations 2000 --vis tensorboard --viewer.quit-on-train-completion True "
        f"colmap --data . --colmap-path colmap/sparse/0 --images-path images --downscale-factor 1"
    )
    success, err = run_command(train_cmd, cwd=str(final_data_dir))
    
    if not success:
        update_status(job_id, "TRAINING_FAILED", f"Train error: {err} | S3_PATH:{remote_temp_path}")
        raise Exception(f"ns-train failed: {err}")

    print("📤 [TRAIN] Exporting PLY...", flush=True)
    config_yml = list((final_data_dir / "outputs").rglob("config.yml"))[0]
    export_dir = work_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # 📝 v1.0.9: สำคัญ! ต้องรันใน cwd เดิม (final_data_dir) เพื่อให้มันหาไฟล์ COLMAP เจอ
    success, err = run_command(f"ns-export gaussian-splat --load-config {config_yml} --output-dir {export_dir}", cwd=str(final_data_dir))
    if not success:
        update_status(job_id, "TRAINING_FAILED", f"Export failed: {err}")
        raise Exception(f"Export failed: {err}")
    
    ply_files = list(export_dir.glob("*.ply"))
    if not ply_files:
        update_status(job_id, "TRAINING_FAILED", "Export finished but no .ply file found!")
        raise Exception("Export finished but no .ply file found!")
    
    final_path = f"results/{job_id}/model.ply"
    print(f"📤 Uploading final PLY to Supabase Storage: {final_path}...", flush=True)
    try:
        supabase = get_supabase_client()
        with open(str(ply_files[0]), 'rb') as f:
            supabase.storage.from_("3d-scans").upload(
                path=final_path,
                file=f,
                file_options={"content-type": "application/octet-stream", "x-upsert": "true"}
            )
        res_url = supabase.storage.from_("3d-scans").get_public_url(final_path)
        print(f"✅ Uploaded to Supabase! Public URL: {res_url}", flush=True)
    except Exception as e:
        print(f"❌ Supabase Storage upload failed: {e}", flush=True)
        raise Exception(f"Supabase Storage upload failed: {e}")
    
    update_status(job_id, "COMPLETED", "Job Finished!", result_url=res_url)
    return {"status": "success", "result_url": res_url}


def handler(job):
    try:
        job_input = job["input"]
        job_id = job_input.get("id")
        mode = job_input.get("mode", WORKER_MODE)
        vdo = job_input.get("video_url")
        work_dir = Path(f"/tmp/job_{job_id}")
        
        print(f"🚀 [HANDLER] ID: {job_id} | Mode: {mode}", flush=True)
        if work_dir.exists(): shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        
        if mode == "PROCESS": return run_process_mode(job_id, vdo, work_dir)
        elif mode == "TRAIN": return run_train_mode(job_id, work_dir)
        elif mode in ["FULL", "ALL"]: return run_full_mode(job_id, vdo, work_dir)
        else: raise Exception(f"Unknown mode: {mode}")
    except Exception as e:
        err_msg = f"❌ [CRITICAL ERROR]: {str(e)}"
        print(err_msg, flush=True)
        if mode == "TRAIN":
            update_status(job_id, "TRAINING_FAILED", err_msg)
        elif mode == "PROCESS":
            update_status(job_id, "SFM_FAILED", err_msg)
        elif mode in ["FULL", "ALL"]:
            update_status(job_id, "FAILED", err_msg)
        return {"status": "error", "message": str(e)}

runpod.serverless.start({"handler": handler})
