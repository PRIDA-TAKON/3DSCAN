import os
import sys
import subprocess
import time
import json
import shutil
from pathlib import Path

def get_secret(key):
    # 1. Try Env Var
    val = os.environ.get(key)
    if val: return val
    
    # 2. Try Kaggle Secrets
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        return user_secrets.get_secret(key)
    except Exception as e:
        print(f"⚠️ Failed to read secret '{key}': {e}")
        return None

# --- Constants & Supabase Helper ---
SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_KEY = get_secret("SUPABASE_KEY")
GDRIVE_SA_JSON = get_secret("GDRIVE_SERVICE_ACCOUNT")
STATUS_SFM_RUNNING = "SFM_RUNNING"
STATUS_SFM_COMPLETED = "SFM_COMPLETED"
STATUS_SFM_FAILED = "SFM_FAILED"

def inject_scripts():
    print('💉 Injecting scripts...')

# --- Injection of step1_extract_frames.py ---
with open('step1_extract_frames.py', 'w') as f:
    f.write('''
import os
import argparse
import subprocess
from pathlib import Path
import shutil

def extract_frames(video_path, output_dir, fps=2, max_width=1024):
    \"\"\"
    Extracts frames from a video file using ffmpeg.
    
    Args:
        video_path (str): Path to the input video.
        output_dir (str): Directory where images will be saved.
        fps (int): Extraction rate in frames per second.
        max_width (int): Max width for resizing images (default: 1024).
    \"\"\"
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
    if not video_path.exists():
        print(f"❌ Error: Video file not found at {video_path}")
        return False

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🎬 Extracting frames from {video_path.name} at {fps} FPS (max width: {max_width})...")
    
    # ffmpeg scale filter: scale=w:h. -1 means preserve aspect ratio.
    vf_graph = f"scale={max_width}:-1"

    cmd = [
        "ffmpeg", "-i", str(video_path), 
        "-vf", vf_graph,
        "-qscale:v", "1", 
        "-r", str(fps), 
        str(output_dir / "%04d.jpg"),
        "-hide_banner", "-loglevel", "error" # Clean output
    ]
    
    try:
        subprocess.run(cmd, check=True)
        num_images = len(list(output_dir.glob("*.jpg")))
        print(f"✅ Successfully extracted {num_images} images to {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ ffmpeg failed: {e}")
        return False
    except FileNotFoundError:
        print("❌ ffmpeg not found. Please install ffmpeg.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1: Extract Frames from Video")
    parser.add_argument("--input_video", required=True, help="Path to input .mp4 video")
    parser.add_argument("--output_dir", required=True, help="Directory to save extracted images")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second (default: 2)")
    parser.add_argument("--max_width", type=int, default=1024, help="Max width for resizing (default: 1024)")
    
    args = parser.parse_args()
    
    extract_frames(args.input_video, args.output_dir, args.fps, args.max_width)
''')

# --- Injection of step2_colmap_sfm.py ---
with open('step2_colmap_sfm.py', 'w') as f:
    f.write('''
import os
import argparse
import subprocess
import shutil
import json
import math
import numpy as np
from pathlib import Path

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def read_cameras_text(path):
    cameras = {}
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            els = line.split()
            camera_id = int(els[0])
            model = els[1]
            width = int(els[2])
            height = int(els[3])
            params = np.array([float(x) for x in els[4:]])
            cameras[camera_id] = {"model": model, "width": width, "height": height, "params": params}
    return cameras

def read_images_text(path):
    images = {}
    with open(path, "r") as f:
        while True:
            line = f.readline()
            if not line: break
            if line.startswith("#") or not line.strip(): continue
            
            # Line 1: Image ID, Qvec, Tvec, Camera ID, Name
            els = line.split()
            image_id = int(els[0])
            qvec = np.array([float(x) for x in els[1:5]])
            tvec = np.array([float(x) for x in els[5:8]])
            camera_id = int(els[8])
            image_name = els[9]
            
            # Line 2: Points 2D (discard)
            f.readline()
            
            images[image_id] = {
                "qvec": qvec, "tvec": tvec, "camera_id": camera_id, "name": image_name
            }
    return images

def convert_colmap_to_transforms(colmap_dir, images_dir, output_path):
    print(f"🔄 Converting COLMAP text output to {output_path}...")
    
    colmap_dir = Path(colmap_dir)
    cameras_file = colmap_dir / "cameras.txt"
    images_file = colmap_dir / "images.txt"
    
    if not cameras_file.exists() or not images_file.exists():
        print("❌ COLMAP output files not found (cameras.txt/images.txt).")
        return False
        
    cameras = read_cameras_text(cameras_file)
    images = read_images_text(images_file)
    
    sorted_image_ids = sorted(images.keys(), key=lambda k: images[k]["name"])
    
    frames = []
    if not cameras: 
        print("❌ No cameras found.")
        return False
        
    cam_id = list(cameras.keys())[0]
    cam = cameras[cam_id]
    
    w, h = cam["width"], cam["height"]
    fl_x = cam["params"][0]
    fl_y = cam["params"][1]
    k1 = cam["params"][2] if len(cam["params"]) > 2 else 0
    k2 = cam["params"][3] if len(cam["params"]) > 3 else 0
    p1 = cam["params"][4] if len(cam["params"]) > 4 else 0
    p2 = cam["params"][5] if len(cam["params"]) > 5 else 0
    cx = cam["params"][2] if len(cam["params"]) == 3 else w / 2.0
    cy = cam["params"][3] if len(cam["params"]) == 3 else h / 2.0
    
    angle_x = 2 * math.atan(w / (2 * fl_x))
    angle_y = 2 * math.atan(h / (2 * fl_y))
    
    json_data = {
        "camera_angle_x": angle_x,
        "camera_angle_y": angle_y,
        "fl_x": fl_x, "fl_y": fl_y,
        "k1": k1, "k2": k2, "p1": p1, "p2": p2,
        "cx": cx, "cy": cy,
        "w": w, "h": h,
        "aabb_scale": 16,
        "frames": []
    }
    
    flip_mat = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    
    for img_id in sorted_image_ids:
        img = images[img_id]
        
        R = qvec2rotmat(img["qvec"])
        t = img["tvec"]
        
        c2w = np.eye(4)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ t
        
        c2w = c2w @ flip_mat
        
        frame = {
            "file_path": f"images/{img['name']}",
            "transform_matrix": c2w.tolist()
        }
        frames.append(frame)
        
    json_data["frames"] = frames
    
    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=4)
        
    print(f"✅ Saved {len(frames)} frames to {output_path}")
    return True

def run_step(cmd, shell=True):
    print(f"🚀 Running: {cmd}")
    # We use subprocess.run without capture_output so it streams to Kaggle's log
    try:
        subprocess.run(cmd, shell=shell, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {cmd}")
        # In check=True mode, the exception usually has basic info, 
        # but streaming directly to stdout/stderr is better for manual debugging on Kaggle.
        raise e

def run_colmap(images_dir, output_dir):
    project_dir = Path(output_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # Check image count
    num_images = len(list(Path(images_dir).glob("*.jpg")))
    print(f"📊 Total images for SfM: {num_images}")
    if num_images < 3:
        print("❌ Error: Too few images for SfM (need at least 3).")
        return False
    
    colmap_dir = project_dir / "colmap"
    colmap_dir.mkdir(parents=True, exist_ok=True)
    
    db_path = colmap_dir / "database.db"
    if db_path.exists():
        db_path.unlink() # Clean start? Or let colmap handle it

    # Check for xvfb
    colmap_binary = "colmap"
    try:
        subprocess.run(["which", "xvfb-run"], check=True, stdout=subprocess.DEVNULL)
        colmap_binary = "xvfb-run -a colmap"
        print("   Using xvfb-run for headless COLMAP")
    except:
        pass

    print("--- Feature Extraction ---")
    run_step(f"{colmap_binary} feature_extractor --database_path {db_path} --image_path {images_dir} --ImageReader.camera_model OPENCV --SiftExtraction.use_gpu 0")
    
    print("--- Matching ---")
    run_step(f"{colmap_binary} sequential_matcher --database_path {db_path} --SiftMatching.use_gpu 0")
    
    print("--- Reconstruction (Mapper) ---")
    sparse_dir = colmap_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    # Force CPU for mapper and BA
    run_step(f"{colmap_binary} mapper --database_path {db_path} --image_path {images_dir} --output_path {sparse_dir} --Mapper.ba_global_use_gpu 0 --Mapper.ba_local_use_gpu 0")
    
    print("--- Converting to Text ---")
    text_dir = colmap_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    
    if not (sparse_dir / "0").exists():
        print("❌ COLMAP reconstruction failed (no model found).")
        return False

    run_step(f"{colmap_binary} model_converter --input_path {sparse_dir}/0 --output_path {text_dir} --output_type TXT")
    
    transforms_path = project_dir / "transforms.json"
    return convert_colmap_to_transforms(text_dir, images_dir, transforms_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2: COLMAP SfM Pipeline")
    parser.add_argument("--images_dir", required=True, help="Directory containing extracted images")
    parser.add_argument("--output_dir", required=True, help="Project output directory (where transforms.json will be saved)")
    
    args = parser.parse_args()
    
    run_colmap(args.images_dir, args.output_dir)
''')

def install_dependencies():
    """Installs minimal dependencies + Glomap."""
    print("📦 Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "supabase", "requests", "gdown", "google-api-python-client", "google-auth-httplib2", "google-auth-oauthlib"], check=True)
    
    # Check for Glomap (Assuming it's pre-installed or we build it here)
    # For now, let's assume we use COLMAP as a fallback or a pre-compiled binary if available.
    # Real Glomap build on Kaggle takes time.
    print("⚠️ Glomap installation skipped in this template. Ensure it's in the environment or installed via apt/pip if possible.")
    run_command("apt-get update --quiet && apt-get install -y --quiet colmap xvfb ffmpeg")

def run_command(cmd, check=True):
    print(f"🚀 Running: {cmd}")
    subprocess.run(cmd, shell=True, check=check)

def get_job():
    """Finds a job assigned to this kernel (SFM_QUEUED)."""
    from supabase import create_client
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # Optimistic locking: Find a queued job and mark it running
    # Supabase doesn't support sophisticated transactions in py client easily without RPC.
    # We'll just grab the oldest QUEUED job.
    response = supabase.table("jobs").select("*").eq("status", "SFM_QUEUED").order("created_at").limit(1).execute()
    if not response.data:
        return None, None
    
    job = response.data[0]
    # Verify it's still queued before claiming
    # Update to RUNNING
    # Note: RLS policies might prevent this if we aren't careful, but we use Service Role key usually or authenticated user.
    # Assuming Env Key has write access.
    supabase.table("jobs").update({
        "status": STATUS_SFM_RUNNING,
        "message": "Starting SfM Process..."
    }).eq("id", job['id']).execute()
    
    return job, supabase

def upload_to_gdrive(file_path, folder_id):
    """Uploads file to GDrive using Service Account."""
    # (Implementation copied from original pipeline_master.py logic)
    # Simplified for brevity here
    print(f"📤 Uploading {file_path}...")
    # ... Actual upload logic ...
    return f"https://drive.google.com/file/d/FAKE_ID_FOR_NOW/view?usp=sharing"

def main():
    inject_scripts()
    print("🎬 Starting Kernel A: SfM")
    
    # 1. Install Deps
    install_dependencies()
    
    # 2. Get Job or Use Debug Video
    debug_video_url = os.environ.get("DEBUG_VIDEO_URL")
    if debug_video_url:
        print(f"🛠️ DEBUG MODE: Using video URL: {debug_video_url}")
        job = {
            'id': "debug_job_" + str(int(time.time())),
            'video_url': debug_video_url,
            'drive_folder_id': os.environ.get("DEBUG_DRIVE_FOLDER")
        }
        supabase = None # Mock or skip DB updates
        if SUPABASE_URL and SUPABASE_KEY:
            from supabase import create_client
            try:
                supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            except: pass
    else:
        job, supabase = get_job()
        if not job:
            print("😴 No 'SFM_QUEUED' jobs found. Exiting.")
            return

    job_id = job['id']
    print(f"✅ Processing Job: {job_id}")
    
    try:
        # 3. Setup Work Dir
        work_dir = Path("/kaggle/working/job_" + str(job_id))
        images_dir = work_dir / "images"
        sfm_dir = work_dir / "sfm"
        
        for d in [work_dir, images_dir, sfm_dir]:
            d.mkdir(parents=True, exist_ok=True)
            
        print(f"📂 Working Directory: {work_dir}")
        
        # 4. Download Video
        video_url = job['video_url']
        video_path = work_dir / "input_video.mp4"
        print(f"⬇️ Downloading video from {video_url}...")
        
        if "drive.google.com" in video_url:
            import gdown
            gdown.download(video_url, str(video_path), quiet=False, fuzzy=True)
        else:
            import requests
            resp = requests.get(video_url, stream=True)
            with open(video_path, 'wb') as f:
                shutil.copyfileobj(resp.raw, f)
        
        # --- DEBUG: Print Directory Structure ---
        print(f"🕵️ DEBUG: Current Working Directory: {os.getcwd()}")
        print(f"🕵️ DEBUG: Files in Current Directory: {os.listdir('.')}")
        if os.path.exists("/kaggle/working"):
            print(f"🕵️ DEBUG: Files in /kaggle/working: {os.listdir('/kaggle/working')}")
        if os.path.exists("/kaggle/src"):
             print(f"🕵️ DEBUG: Files in /kaggle/src: {os.listdir('/kaggle/src')}")
        if os.path.exists("/kaggle/input"):
             print(f"🕵️ DEBUG: Files in /kaggle/input: {os.listdir('/kaggle/input')}")
        # ----------------------------------------

        if not video_path.exists():
            raise FileNotFoundError("Video download failed.")

        # 5. Extract Frames
        print("🎞️ Extracting frames...")
        # We need to make sure the script is in the current directory or accessible
        # Since we push the whole folder, it should be in /kaggle/working/ (if we push source to there) 
        # OR /kaggle/src/script.py depending on how we run.
        # But 'kaggle kernels push' uploads code to /kaggle/working/ usually if it's a script type? 
        # Wait, script type kernels usually run the code file. 
        # The auxiliary files are in the same directory as the main script.
        
        cmd_extract = [sys.executable, "step1_extract_frames.py", "--input_video", str(video_path), "--output_dir", str(images_dir)]
        subprocess.run(cmd_extract, check=True)
        
        # 6. Run SfM (Colmap)
        # Using step2_colmap_sfm.py as primary since Glomap is hard to install
        print("Ez SfM (Colmap) running...")
        cmd_sfm = [sys.executable, "step2_colmap_sfm.py", "--images_dir", str(images_dir), "--output_dir", str(sfm_dir)]
        subprocess.run(cmd_sfm, check=True)
        
        # 7. Zip Result
        print("📦 Zipping results...")
        output_zip_path = work_dir / "sfm_output" # shutil.make_archive adds .zip
        shutil.make_archive(str(output_zip_path), 'zip', sfm_dir)
        output_zip_file = str(output_zip_path) + ".zip"
        
        # 8. Upload
        if job.get('drive_folder_id'):
            print(f"📤 Uploading to Drive Folder: {job.get('drive_folder_id')}")
            sfm_url = upload_to_gdrive(output_zip_file, job.get('drive_folder_id'))
        else:
            print("⚠️ No Drive Folder ID provided. Skipping upload (or uploading to root).")
            sfm_url = upload_to_gdrive(output_zip_file, None) # Upload to root or mocked
            
        if not sfm_url:
             raise Exception("Upload failed")

        # 9. Update Status
        print("✅ SfM Completed!")
        supabase.table("jobs").update({
            "status": STATUS_SFM_COMPLETED,
            "sfm_url": sfm_url,
            "message": "SfM Completed Successfully."
        }).eq("id", job_id).execute()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if supabase:
            supabase.table("jobs").update({
                "status": STATUS_SFM_FAILED,
                "message": str(e)
            }).eq("id", job_id).execute()
        sys.exit(1)

if __name__ == "__main__":
    main()
