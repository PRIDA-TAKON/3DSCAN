import os
import json
import subprocess
from pathlib import Path

# Load env (simple loader since python-dotenv might not be there, or we assume .env is loaded in shell)
# But for safety, let's look for .env
def load_env():
    if Path(".env").exists():
        with open(".env") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    key, _, val = line.partition("=")
                    os.environ[key.strip()] = val.strip()

load_env()

def main():
    print("🚀 3DSCAN Kaggle Deployer")
    
    # 1. Get Kaggle Username
    username = os.environ.get("KAGGLE_USERNAME")
    if not username:
        print("⚠️ KAGGLE_USERNAME not found in .env.")
        username = input("👉 Enter your Kaggle Username: ").strip()
    
    if not username:
        print("❌ Username required. Exiting.")
        return

    kernels_dir = Path("kernels")
    if not kernels_dir.exists():
        print("❌ 'kernels' directory not found.")
        return

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    # 2. Iterate and Push
    for kernel_dir in kernels_dir.iterdir():
        if kernel_dir.is_dir():
            metadata_path = kernel_dir / "kernel-metadata.json"
            if metadata_path.exists():
                print(f"\n📦 Processing {kernel_dir.name}...")
                
                # Update Metadata
                with open(metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Check if ID needs user prefix
                current_id = data.get("id", "")
                slug = current_id.split("/")[-1] # Get the slug part
                new_id = f"{username}/{slug}"
                
                # Pre-emptively fix Title to match Slug to avoid Kaggle CLI warnings/crashes
                # This ensures clean URL resolution is identical
                new_title = slug.replace("-", " ").title()
                
                needs_save = False
                if current_id != new_id:
                    print(f"   ✏️ Updating ID: {current_id} -> {new_id}")
                    data["id"] = new_id
                    needs_save = True
                
                if data.get("title") != new_title:
                    print(f"   ✏️ Updating Title: {data.get('title')} -> {new_title}")
                    data["title"] = new_title
                    needs_save = True
                
                if needs_save:
                    with open(metadata_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                
                # Push
                try:
                    print(f"   ⬆️ Pushing to Kaggle...")
                    
                    # Use text=False to get bytes, then decode safely
                    # Pass clean env with PYTHONIOENCODING to prevent encoding issues in child process
                    result = subprocess.run(
                        ["kaggle", "kernels", "push", "-p", str(kernel_dir)], 
                        check=True,
                        capture_output=True,
                        text=False,
                        env=env
                    )

                    # Print success output if any (safely)
                    print(result.stdout.decode('utf-8', errors='replace'))
                    print(f"   ✅ Successfully pushed {new_id}")

                except subprocess.CalledProcessError as e:
                    print(f"   ❌ Push failed: {e}")
                    # Safely decode error output
                    err_msg = e.stderr.decode('utf-8', errors='replace') + e.stdout.decode('utf-8', errors='replace')
                    print(f"   Error: {err_msg}")
                    
                    if "does not resolve to the specified id" in err_msg:
                        print("   🔧 Title/ID mismatch detected. Updating Title to match Slug...")
                        # Auto-fix Title
                        valid_title = slug.replace("-", " ").title()
                        data["title"] = valid_title
                        with open(metadata_path, "w") as f:
                            json.dump(data, f, indent=2)
                        print(f"   🔄 Updated Title to: '{valid_title}'. Retrying push...")
                        
                        # Resize and Retry
                        try:
                            subprocess.run(["kaggle", "kernels", "push", "-p", str(kernel_dir)], check=True)
                            print(f"   ✅ Retry successful: {new_id}")
                        except subprocess.CalledProcessError as retry_e:
                             print(f"   ❌ Retry failed again.")

                    print("      Tip: Check if 'kaggle.json' is configured in your home directory.")

if __name__ == "__main__":
    main()
