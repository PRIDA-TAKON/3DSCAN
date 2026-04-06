import os
import subprocess
import sys
import shutil
from pathlib import Path

def run_command(cmd, cwd=None):
    print(f"🚀 Running: {cmd}")
    try:
        # Use subprocess.run with capture_output to see what's happening
        result = subprocess.run(cmd, shell=True, text=True, cwd=cwd)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Command failed: {e}")
        return False

def main():
    repo_url = os.environ.get("GIT_REPO_URL")
    git_token = os.environ.get("GIT_TOKEN")
    branch = os.environ.get("GIT_BRANCH", "main")
    worker_script = os.environ.get("WORKER_SCRIPT", "takon_3d_worker.py")

    print("--- 🛠️ Git Code Loader ---")

    if not repo_url:
        print("⚠️ GIT_REPO_URL not set. Skipping git sync.")
    else:
        # Inject token into URL for private repos
        auth_url = repo_url
        if git_token and "https://" in repo_url:
            auth_url = repo_url.replace("https://", f"https://{git_token}@")

        # Check if we are already in a git repo
        if Path(".git").exists():
            print("🔄 Updating existing repository...")
            run_command(f"git fetch origin {branch}")
            run_command(f"git reset --hard origin/{branch}")
        else:
            print(f"📥 Cloning repository: {repo_url} (branch: {branch})...")
            # Clone to a temporary folder then move files to avoid 'directory not empty' errors
            if run_command(f"git clone -b {branch} {auth_url} .tmp_repo"):
                print("🚚 Moving files to workspace...")
                tmp_path = Path(".tmp_repo")
                for item in tmp_path.iterdir():
                    dest = Path(".") / item.name
                    if dest.exists():
                        if dest.is_dir(): shutil.rmtree(dest)
                        else: dest.unlink()
                    shutil.move(str(item), str(dest))
                shutil.rmtree(".tmp_repo")
            else:
                print("❌ Failed to clone repository.")

    # Execute the worker script
    if Path(worker_script).exists():
        print(f"🎬 Starting Worker: {worker_script}")
        # Use execv to replace the current process with the worker
        os.execv(sys.executable, [sys.executable, worker_script])
    else:
        print(f"❌ ERROR: Worker script '{worker_script}' not found!")
        # Keep container alive for debugging if it fails
        print("😴 Sleeping for 1 hour to allow debugging...")
        import time
        time.sleep(3600)

if __name__ == "__main__":
    main()
