import os
import subprocess
import sys
import shutil
import tarfile
import requests
import time
from pathlib import Path

def download_file(url, dest_path):
    print(f"📥 Downloading from Store: {url}")
    response = requests.get(url, stream=True, timeout=300)
    if response.status_code != 200:
        raise Exception(f"Download failed: {response.status_code}")
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024*1024): # 1MB chunks
            f.write(chunk)
    print(f"✅ Download complete.")

def main():
    s3_engine_url = os.environ.get("RUNPOD_S3_ENGINE_URL")
    repo_url = os.environ.get("GIT_REPO_URL")
    git_token = os.environ.get("GIT_TOKEN")
    branch = os.environ.get("GIT_BRANCH", "main")
    worker_script = os.environ.get("WORKER_SCRIPT", "takon_3d_worker.py")
    
    print("--- ⚡ Ultralight RunPod Loader (S3 + Git) ---")

    # 1. Load Heavy Engine from S3
    if s3_engine_url:
        engine_dir = Path("/app/engine")
        if not engine_dir.exists():
            try:
                tmp_tar = Path("/tmp/engine.tar.gz")
                download_file(s3_engine_url, tmp_tar)
                print("📦 Extracting Engine Environment...")
                with tarfile.open(tmp_tar, "r:gz") as tar:
                    tar.extractall(path="/")
                print("✅ Engine Ready.")
                os.environ["PATH"] = f"/app/engine/bin:{os.environ.get('PATH', '')}"
                os.environ["PYTHONPATH"] = f"/app/engine/lib/python3.10/site-packages:{os.environ.get('PYTHONPATH', '')}"
            except Exception as e:
                print(f"⚠️ S3 Engine Load failed: {e}")

    # 2. Sync Logic from Git
    if repo_url:
        auth_url = repo_url
        if git_token and "https://" in repo_url:
            auth_url = repo_url.replace("https://", f"https://{git_token}@")
        
        try:
            tmp_sync = Path("/tmp/git_sync")
            if tmp_sync.exists(): shutil.rmtree(tmp_sync)
            print(f"📥 Syncing Logic from Git: {repo_url}...")
            if subprocess.run(f"git clone --depth 1 -b {branch} {auth_url} {tmp_sync}", shell=True).returncode == 0:
                for item in tmp_sync.iterdir():
                    if item.name == ".git": continue
                    dest = Path(".") / item.name
                    if dest.exists():
                        if dest.is_dir(): shutil.rmtree(dest)
                        else: dest.unlink()
                    shutil.move(str(item), str(dest))
                print("✅ Logic Updated.")
        except Exception as e:
            print(f"⚠️ Git Sync failed: {e}")

    # 3. Start Worker
    if Path(worker_script).exists():
        print(f"🎬 Starting Worker: {worker_script}")
        sys.stdout.flush()
        os.execv(sys.executable, [sys.executable, worker_script])
    else:
        print(f"❌ ERROR: '{worker_script}' not found!")
        time.sleep(3600)

if __name__ == "__main__":
    main()
