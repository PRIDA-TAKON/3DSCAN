import os
import subprocess
import sys
import shutil
import zipfile
import requests
from pathlib import Path

def download_file(url, dest_path):
    print(f"📥 Downloading: {url}")
    response = requests.get(url, stream=True, timeout=60)
    if response.status_code != 200:
        raise Exception(f"Download failed: {response.status_code}")
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192): f.write(chunk)

def main():
    s3_logic_url = os.environ.get("RUNPOD_S3_LOGIC_URL") # ตัวอย่าง: https://s3api-us-il-1.runpod.io/3d-scans/deploy/worker_logic.zip
    worker_script = os.environ.get("WORKER_SCRIPT", "takon_3d_worker.py")
    
    print("--- 🛠️ RunPod S3 Logic Loader ---")

    if s3_logic_url:
        try:
            tmp_zip = Path("/tmp/logic.zip")
            download_file(s3_logic_url, tmp_zip)
            print("📦 Extracting logic from S3 Store...")
            with zipfile.ZipFile(tmp_zip, 'r') as zip_ref:
                zip_ref.extractall(".")
            print("✅ Logic updated from S3 Store.")
        except Exception as e:
            print(f"⚠️ Failed to load logic from S3: {e}. Falling back to existing code.")
    else:
        print("ℹ️ No S3 Logic URL provided. Using local files.")

    # Execute the worker script
    if Path(worker_script).exists():
        print(f"🎬 Starting Worker: {worker_script}")
        # Flush outputs for RunPod logs
        sys.stdout.flush()
        sys.stderr.flush()
        os.execv(sys.executable, [sys.executable, worker_script])
    else:
        print(f"❌ ERROR: Worker script '{worker_script}' not found!")
        print("😴 Sleeping for 1 hour for debugging...")
        import time
        time.sleep(3600)

if __name__ == "__main__":
    main()
