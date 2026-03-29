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
    # แนะนำให้ใช้ Service Role Key ใน Backend เพื่อให้มีสิทธิ์เขียน Storage
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        return None
    return create_client(url, key)

def update_status(job_id, status, message="", result_url=None):
    print(f"🔔 [{job_id}] {status}: {message}")
    supabase = get_supabase_client()
    if not supabase: return
    try:
        data = {
            "status": status,
            "message": message,
            "updated_at": "now()"
        }
        if result_url:
            data["result_url"] = result_url
            
        supabase.table("jobs").update(data).eq("id", job_id).execute()
    except Exception as e:
        print(f"⚠️ Supabase update failed: {e}")

def run_command(cmd, cwd=None):
    print(f"🚀 Running: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True, text=True, cwd=cwd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {cmd}\n   Error: {e}")
        return False

def zip_folder(folder_path, output_path):
    """บีบอัดโฟลเดอร์เป็นไฟล์ ZIP"""
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

    print(f"📦 Starting Job: {job_id}")
    update_status(job_id, "processing", "Worker started on RunPod Serverless")

    # 1. Setup Working Directory
    work_dir = Path(f"/tmp/job_{job_id}")
    work_dir.mkdir(parents=True, exist_ok=True)
    video_path = work_dir / "input_video.mp4"
    frames_dir = work_dir / "frames"
    colmap_dir = work_dir / "colmap"
    output_dir = work_dir / "output"
    zip_output = work_dir / f"result_{job_id}.zip"

    try:
        # 2. Download Video
        print(f"📥 Downloading video from Supabase...")
        import requests
        response = requests.get(video_url, stream=True)
        with open(video_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # 3. Step 1: Extract Frames
        update_status(job_id, "extracting_frames")
        if not run_command(f"python step1_extract_frames.py --input_video {video_path} --output_dir {frames_dir}"):
            raise Exception("Frame extraction failed")

        # 4. Step 2: COLMAP SFM
        update_status(job_id, "running_sfm")
        if not run_command(f"python step2_colmap_sfm.py --image_path {frames_dir} --output_path {colmap_dir}"):
            raise Exception("COLMAP SFM failed")

        # 5. Step 3: Train Gaussian Splatting (Taichi)
        update_status(job_id, "training_splatting")
        # ส่งค่า path ของข้อมูล colmap และ output ที่เราเตรียมไว้ใน /tmp
        if not run_command(f"python scripts/step3_train_splatting.py --data_dir {colmap_dir} --output_dir {output_dir} --iterations 7000"):
            raise Exception("Training failed")

        # 6. Step 4: Zip & Upload to Supabase Storage
        update_status(job_id, "exporting", "Zipping results...")
        zip_folder(output_dir, zip_output)

        print(f"📤 Uploading result to Supabase Storage...")
        supabase = get_supabase_client()
        bucket_name = "scans"
        remote_path = f"results/{job_id}/result.zip"
        
        with open(zip_output, 'rb') as f:
            supabase.storage.from_(bucket_name).upload(
                path=remote_path,
                file=f,
                file_options={"content-type": "application/zip", "x-upsert": "true"}
            )
        
        # สร้าง Public URL (ถ้าตั้งค่า Bucket เป็น Public)
        res_url = supabase.storage.from_(bucket_name).get_public_url(remote_path)

        update_status(job_id, "completed", "Job finished successfully", result_url=res_url)
        return {"status": "success", "job_id": job_id, "result_url": res_url}

    except Exception as e:
        error_msg = str(e)
        update_status(job_id, "failed", error_msg)
        return {"status": "error", "message": error_msg}
    
    finally:
        # Cleanup
        if work_dir.exists():
            shutil.rmtree(work_dir)

# Register the handler with RunPod
runpod.serverless.start({"handler": handler})
