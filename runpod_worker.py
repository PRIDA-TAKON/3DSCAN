import os
import shutil
import time
import subprocess
import zipfile
from pathlib import Path
import runpod
from supabase import create_client

# --- Configuration ---
def get_supabase_client():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key: return None
    return create_client(url, key)

def update_status(job_id, status, message="", result_url=None):
    print(f"🔔 [{job_id}] {status}: {message}")
    supabase = get_supabase_client()
    if not supabase: return
    try:
        data = {"status": status, "message": message, "updated_at": "now()"}
        if result_url: data["result_url"] = result_url
        supabase.table("jobs").update(data).eq("id", job_id).execute()
    except Exception as e:
        print(f"⚠️ Supabase update failed: {e}")

def run_command(cmd, cwd=None):
    print(f"🚀 Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, text=True, capture_output=True, cwd=cwd)
        if result.stdout: print(result.stdout)
        return True, ""
    except subprocess.CalledProcessError as e:
        error_detail = f"Error: {e.stderr if e.stderr else e.stdout}"
        print(f"❌ {error_detail}")
        return False, error_detail

def zip_folder(folder_path, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                zipf.write(os.path.join(root, file), 
                           os.path.relpath(os.path.join(root, file), folder_path))

# --- Main Handler for RunPod ---
def handler(job):
    job_input = job["input"]
    job_id = job_input.get("id")
    video_url = job_input.get("video_url")
    
    if not job_id or not video_url:
        return {"error": "Missing job_id or video_url"}

    print(f"📦 Starting Nerfstudio Job: {job_id}")
    update_status(job_id, "processing", "Worker started on RunPod (Nerfstudio v1.1.1)")

    # 1. Setup Working Directory
    work_dir = Path(f"/tmp/job_{job_id}")
    if work_dir.exists(): shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    video_path = work_dir / "input_video.mp4"
    frames_dir = work_dir / "frames"
    colmap_dir = work_dir / "colmap"
    output_dir = work_dir / "output"
    
    try:
        # 2. Download Video
        print(f"📥 Downloading video...")
        import requests
        response = requests.get(video_url, stream=True)
        if response.status_code != 200:
            raise Exception(f"Download failed: {response.status_code}")
        with open(video_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192): f.write(chunk)

        # 3. Step 1: Extract Frames
        update_status(job_id, "extracting_frames")
        success, err = run_command(f"python3 step1_extract_frames.py --input_video {video_path} --output_dir {frames_dir}")
        if not success: raise Exception(f"Step 1 Failed: {err}")

        # 4. Step 2: COLMAP SFM
        update_status(job_id, "running_sfm")
        # Nerfstudio ชอบโครงสร้าง COLMAP แบบมาตรฐาน
        success, err = run_command(f"python3 step2_colmap_sfm.py --image_path {frames_dir} --output_path {colmap_dir}")
        if not success: raise Exception(f"Step 2 Failed: {err}")

        # 5. Step 3: Train Nerfstudio (Splatfacto)
        # ข้ามขั้นตอน Prepare Data ไปได้เลย เพราะ Nerfstudio อ่านโฟลเดอร์ COLMAP ได้ตรงๆ
        update_status(job_id, "training_splatting", "Training with Nerfstudio Splatfacto (7k iterations)...")
        
        # คำสั่งเทรนแบบ Headless และรันบน GPU
        # หมายเหตุ: --vis none ปิด UI และ Logger ทั้งหมดเพื่อความเสถียร
        train_cmd = (
            f"ns-train splatfacto "
            f"--data {colmap_dir} "
            f"--output-dir {output_dir} "
            f"--max-num-iterations 7000 "
            f"--vis none "
            f"--viewer.launch-viewer False "
            f"colmap"
        )
        
        success, err = run_command(train_cmd)
        if not success: raise Exception(f"Step 3 Training Failed: {err}")

        # 6. Step 4: Zip & Upload
        update_status(job_id, "exporting", "Zipping results...")
        zip_output = work_dir / f"result_{job_id}.zip"
        zip_folder(output_dir, zip_output)

        print(f"📤 Uploading result to Supabase...")
        supabase = get_supabase_client()
        bucket_name = "3d-scans"
        remote_path = f"results/{job_id}/result.zip"
        
        with open(zip_output, 'rb') as f:
            supabase.storage.from_(bucket_name).upload(
                path=remote_path,
                file=f,
                file_options={"content-type": "application/zip", "x-upsert": "true"}
            )
        
        res_url = supabase.storage.from_(bucket_name).get_public_url(remote_path)
        update_status(job_id, "completed", "Success! 3D model generated with Nerfstudio", result_url=res_url)
        return {"status": "success", "job_id": job_id, "result_url": res_url}

    except Exception as e:
        error_msg = str(e)
        update_status(job_id, "failed", error_msg)
        return {"status": "error", "message": error_msg}
    finally:
        if work_dir.exists(): shutil.rmtree(work_dir)

runpod.serverless.start({"handler": handler})
