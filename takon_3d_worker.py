import os
import shutil
import time
import subprocess
import zipfile
import requests
from pathlib import Path
import runpod
from supabase import create_client

# --- Configuration ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
WORKER_MODE = os.environ.get("WORKER_MODE", "PROCESS") # PROCESS or TRAIN
BUCKET_NAME = "3d-scans"

def get_supabase_client():
    if not SUPABASE_URL or not SUPABASE_KEY: return None
    return create_client(SUPABASE_URL, SUPABASE_KEY)

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
        # Stream output in real-time instead of capturing at the end
        with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=cwd) as sp:
            for line in sp.stdout:
                print(line, end="", flush=True)
        
        if sp.returncode != 0:
            return False, f"Command exited with code {sp.returncode}"
        return True, ""
    except Exception as e:
        error_detail = f"Execution failed: {str(e)}"
        print(f"❌ {error_detail}", flush=True)
        return False, error_detail

def zip_folder(folder_path, output_path):
    print(f"📦 Zipping {folder_path}...", flush=True)
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                zipf.write(os.path.join(root, file), 
                           os.path.relpath(os.path.join(root, file), folder_path))

def download_file(url, dest_path):
    print(f"📥 Downloading: {url}", flush=True)
    response = requests.get(url, stream=True, timeout=60)
    if response.status_code != 200:
        raise Exception(f"Download failed: {response.status_code}")
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
    print(f"✅ Downloaded to {dest_path}", flush=True)

# --- Sub-Task: PROCESS (COLMAP) ---
def run_process_mode(job_id, video_url, work_dir):
    update_status(job_id, "processing", "Step 1: Extracting frames & COLMAP (Processor Image)")
    
    video_path = work_dir / "input_video.mp4"
    output_dir = work_dir / "processed_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download
    download_file(video_url, video_path)

    # 2. Extract & COLMAP (If nerfstudio is not here, we use raw colmap)
    # Note: In the colmap image, we might need to use ns-process-data if we installed it, 
    # or manual colmap commands. Let's assume we use ns-process-data for consistency.
    cmd = f"ns-process-data video --data {video_path} --output-dir {output_dir} --num-frames-target 200 --verbose"
    success, err = run_command(cmd)
    if not success: raise Exception(f"COLMAP Failed: {err}")

    # 3. Zip and Upload to Temp
    update_status(job_id, "uploading_temp", "Uploading processed data to Supabase (Temp)...")
    zip_path = work_dir / "temp_data.zip"
    zip_folder(output_dir, zip_path)
    
    supabase = get_supabase_client()
    remote_path = f"temp/{job_id}/processed.zip"
    with open(zip_path, 'rb') as f:
        supabase.storage.from_(BUCKET_NAME).upload(path=remote_path, file=f, file_options={"x-upsert": "true"})
    
    update_status(job_id, "ready_to_train", "Processor finished. Ready for Training.")
    return {"status": "success", "step": "process"}

# --- Sub-Task: TRAIN (Nerfstudio) ---
def run_train_mode(job_id, work_dir):
    update_status(job_id, "training", "Step 2: Training Gaussian Splatting (Trainer Image)")
    
    supabase = get_supabase_client()
    temp_remote_path = f"temp/{job_id}/processed.zip"
    temp_zip_url = supabase.storage.from_(BUCKET_NAME).get_public_url(temp_remote_path)
    
    zip_path = work_dir / "processed.zip"
    data_dir = work_dir / "data"
    train_out = work_dir / "train_output"

    # 1. Download Temp Data
    download_file(temp_zip_url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    # 2. Train
    cmd = f"ns-train splatfacto --data {data_dir} --output-dir {train_out} --max-num-iterations 7000 --vis none --viewer.launch-viewer False colmap"
    success, err = run_command(cmd)
    if not success: raise Exception(f"Training Failed: {err}")

    # 3. Export & Upload Result
    config_file = list(train_out.glob("**/config.yml"))[0]
    ply_path = work_dir / "result.ply"
    run_command(f"ns-export gaussian-splat --load-config {config_file} --output-path {ply_path}")
    
    final_path = f"results/{job_id}/model.ply"
    with open(ply_path, 'rb') as f:
        supabase.storage.from_(BUCKET_NAME).upload(path=final_path, file=f, file_options={"x-upsert": "true"})
    
    res_url = supabase.storage.from_(BUCKET_NAME).get_public_url(final_path)
    
    # 4. Cleanup Supabase Temp
    print(f"🧹 Cleaning up temp data for job {job_id}...")
    try:
        supabase.storage.from_(BUCKET_NAME).remove([temp_remote_path])
    except: pass

    update_status(job_id, "completed", "Job Finished! Model ready.", result_url=res_url)
    return {"status": "success", "result_url": res_url}

# --- Main Handler ---
def handler(job):
    job_input = job["input"]
    job_id = job_input.get("id")
    video_url = job_input.get("video_url")
    
    work_dir = Path(f"/tmp/job_{job_id}")
    if work_dir.exists(): shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        if WORKER_MODE == "PROCESS":
            return run_process_mode(job_id, video_url, work_dir)
        else:
            return run_train_mode(job_id, work_dir)
    except Exception as e:
        error_msg = str(e)
        update_status(job_id, "failed", error_msg)
        return {"status": "error", "message": error_msg}
    finally:
        if work_dir.exists(): shutil.rmtree(work_dir)

runpod.serverless.start({"handler": handler})
