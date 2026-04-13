import os
import subprocess
import sys
import shutil
import time
from pathlib import Path

def run_command(cmd, cwd=None):
    print(f"🚀 Running: {cmd}")
    try:
        # Use subprocess.run with capture_output to see what's happening
        result = subprocess.run(cmd, shell=True, text=True, cwd=cwd, capture_output=True)
        if result.returncode != 0:
            print(f"⚠️ Warning: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error executing command: {e}")
        return False

def main():
    repo_url = os.environ.get("GIT_REPO_URL")
    git_token = os.environ.get("GIT_TOKEN")
    branch = os.environ.get("GIT_BRANCH", "main")
    worker_script = os.environ.get("WORKER_SCRIPT", "takon_3d_worker.py")

    print("--- 🛠️ Git Code Loader (Fast Sync) ---")

    if not repo_url:
        print("⚠️ GIT_REPO_URL not set. Running with local files.")
    else:
        # Prepare Auth URL
        auth_url = repo_url
        if git_token and "https://" in repo_url:
            auth_url = repo_url.replace("https://", f"https://{git_token}@")

        try:
            # ใช้ /tmp/ เพื่อความสะอาดในการจัดการไฟล์
            tmp_sync = Path("/tmp/git_sync_dir")
            if tmp_sync.exists(): shutil.rmtree(tmp_sync)
            
            print(f"📥 Syncing from Git: {repo_url} [{branch}]...")
            if run_command(f"git clone --depth 1 -b {branch} {auth_url} {tmp_sync}"):
                print("🚚 Deploying new logic to workspace...")
                for item in tmp_sync.iterdir():
                    if item.name == ".git": continue # Don't move .git folder
                    
                    dest = Path(".") / item.name
                    if dest.exists():
                        if dest.is_dir(): shutil.rmtree(dest)
                        else: dest.unlink()
                    shutil.move(str(item), str(dest))
                print("✅ Logic sync successful.")
            else:
                print("❌ Git clone failed. Falling back to built-in code.")
        except Exception as e:
            print(f"❌ Critical Sync Error: {e}")

    # Execute the worker script
    if Path(worker_script).exists():
        print(f"🎬 Starting Worker: {worker_script}")
        sys.stdout.flush()
        sys.stderr.flush()
        os.execv(sys.executable, [sys.executable, worker_script])
    else:
        print(f"❌ ERROR: '{worker_script}' not found in workspace!")
        print("😴 Sleeping for 1 hour for debugging...")
        time.sleep(3600)

if __name__ == "__main__":
    main()
