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
    access = S3_ACCESS_KEY.strip()
    secret = S3_SECRET_KEY.strip()
    # ระบุ region_name ให้ชัดเจนเป็น us-il-1 ตามที่ RunPod S3 ต้องการ
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
    # ปรับปรุง: ใช้ fps=2 และเพิ่มเฟรมเป็น 300 เพื่อให้ SFM มีข้อมูลพอ และคงคุณภาพไว้ที่ q:v 2
    print("🎞️ Extracting frames (Optimized)...", flush=True)
    success, err = run_command(f"ffmpeg -i {video_path} -q:v 2 -vf \"fps=2\" -frames:v 300 {images_dir}/frame_%04d.jpg")
    if not success:
        print(f"⚠️ FFMPEG Error: {err}. Retrying with fallback scale...", flush=True)
        run_command(f"ffmpeg -i {video_path} -q:v 4 -vf \"fps=2,scale=-1:720\" -frames:v 200 {images_dir}/frame_%04d.jpg")
    
    # 📝 Safety Check: ตรวจสอบจำนวนภาพที่ได้
    extracted_imgs = list(images_dir.glob("*.jpg"))
    print(f"📸 Extracted {len(extracted_imgs)} images.", flush=True)
    if len(extracted_imgs) < 10: raise Exception(f"Insufficient frames extracted ({len(extracted_imgs)}). SfM needs at least 20-30 frames.")

    # 3. SfM
    print("🎬 Running SfM Pipeline...", flush=True)
    # ใช้ scripts/run_glomap.py และตรวจสอบผลลัพธ์
    success, err = run_command(f"python3 scripts/run_glomap.py --images_dir {images_dir} --output_dir {output_dir}")
    if not success:
        # ลองใช้ตัวสำรองถ้า glomap/colmap มีปัญหา
        print(f"⚠️ SfM Pipeline failed: {err}. Retrying with basic COLMAP...", flush=True)
        success, err = run_command(f"python3 step2_colmap_sfm.py --images_dir {images_dir} --output_dir {output_dir}")
        if not success: raise Exception(f"SfM failed after retries: {err}")

    # 📝 Verify Output before packaging
    sparse_files = list(output_dir.rglob("*.bin")) + list(output_dir.rglob("*.txt"))
    if not sparse_files:
        raise Exception("SfM completed but produced no reconstruction files (*.bin or *.txt)!")

    # 4. Packaging
    print("📦 Packaging data...", flush=True)
    zip_path = work_dir / "processed.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # ใส่ไฟล์จาก output_dir (Sparse Model / Transforms)
        files_count = 0
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, output_dir)
                zipf.write(abs_path, rel_path)
                files_count += 1
        
        # ใส่รูปภาพ
        img_count = 0
        print(f"📁 Checking images in: {images_dir}", flush=True)
        for root, dirs, files in os.walk(images_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.join("images", os.path.relpath(abs_path, images_dir))
                    zipf.write(abs_path, rel_path)
                    img_count += 1
        
    print(f"✅ Packaged {files_count} data files and {img_count} images.", flush=True)
    
    # 📝 Final Integrity Check
    if img_count == 0:
        raise Exception("Packaging failed: No images were added to the zip file!")
    
    zip_size = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"📦 Final Zip Size: {zip_size:.2f} MB", flush=True)

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
    print(f"🧠 [TRAIN] Starting Job: {job_id}", flush=True)
    supabase = get_supabase_client()
    job_data = supabase.table("jobs").select("message").eq("id", job_id).single().execute()
    msg = job_data.data.get("message", "")
    
    print(f"📖 [TRAIN] DB Message: {msg}", flush=True)
    if "S3_PATH:" not in msg:
        raise Exception(f"S3 path missing in DB. Current msg: {msg}")
    
    remote_temp_path = msg.split("S3_PATH:")[1]
    print(f"🔗 [TRAIN] Remote Path: {remote_temp_path}", flush=True)
    
    update_status(job_id, "training", "Step 2: Training (Initializing...)")
    
    zip_path = work_dir / "processed.zip"
    raw_data_dir = work_dir / "raw_data"
    final_data_dir = work_dir / "data"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    final_data_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Download
    s3 = get_s3_client()
    if not s3: raise Exception("S3 client is None! Check credentials.")
    
    print(f"📥 [TRAIN] Downloading from S3: {remote_temp_path}...", flush=True)
    s3.download_file(S3_BUCKET, remote_temp_path, str(zip_path))
    print(f"✅ [TRAIN] Downloaded {os.path.getsize(zip_path)} bytes.", flush=True)
    
    # 2. Extract
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(raw_data_dir)
    print(f"📂 [TRAIN] Extracted files to {raw_data_dir}", flush=True)
    
    # 3. Restructure
    print("🛠️ [TRAIN] Restructuring data for Nerfstudio...", flush=True)
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

    print(f"🔍 [TRAIN] Data Check: Images={found_imgs}, COLMAP Bins={found_bins}", flush=True)
    if found_imgs == 0 or found_bins == 0:
        raise Exception(f"Missing critical data: Images={found_imgs}, Bins={found_bins}")

    # 4. Run Training
    print("🔥 [TRAIN] Starting ns-train (splatfacto)...", flush=True)
    # เพิ่ม --max-num-iterations เป็นค่าที่เห็นชัดเจน (เช่น 2000)
    success, err = run_command(f"ns-train splatfacto --data . --vis tensorboard --max-num-iterations 2000 colmap", cwd=str(final_data_dir))
    
    if not success:
        print(f"❌ [TRAIN] ns-train failed: {err}", flush=True)
        raise Exception(f"Training failed: {err}")

    # 5. Export
    print("📤 [TRAIN] Exporting model to PLY...", flush=True)
    update_status(job_id, "exporting", "Exporting model...")
    train_out = final_data_dir / "outputs"
    config_yml = list(train_out.rglob("config.yml"))
    if not config_yml:
        raise Exception("Training finished but config.yml not found!")
        
    config_file = config_yml[0]
    ply_path = work_dir / "result.ply"
    success, err = run_command(f"ns-export gaussian-splat --load-config {config_file} --output-path {ply_path}")
    if not success: raise Exception(f"Export failed: {err}")
    
    # 6. Upload Result
    print(f"☁️ [TRAIN] Uploading PLY to results/{job_id}/model.ply", flush=True)
    final_path = f"results/{job_id}/model.ply"
    s3.upload_file(str(ply_path), S3_BUCKET, final_path)
    res_url = f"{S3_ENDPOINT}/{S3_BUCKET}/{final_path}"
    
    update_status(job_id, "completed", "Job Finished!", result_url=res_url)
    print(f"🎉 [TRAIN] SUCCESS! Result: {res_url}", flush=True)
    return {"status": "success", "result_url": res_url}

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
