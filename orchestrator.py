import os
import time
import json
import subprocess
from pathlib import Path
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME") # user/slug format for kernels
POLL_INTERVAL = 30  # seconds

# Ensure Kaggle Key is set (Backwards compat for KAGGLE_API_TOKEN)
if not os.environ.get("KAGGLE_KEY") and os.environ.get("KAGGLE_API_TOKEN"):
    print("🔑 Mapping KAGGLE_API_TOKEN to KAGGLE_KEY for CLI compatibility.")
    os.environ["KAGGLE_KEY"] = os.environ.get("KAGGLE_API_TOKEN")

# Kernel Slugs (Reflects the 3 stages)
KERNEL_A_SLUG = "takon-medical-3dgs-sfm"      # CPU/RAM
KERNEL_A_DIR = "kernel_a_sfm"
KERNEL_B_SLUG = "takon-medical-3dgs-train"    # GPU
KERNEL_B_DIR = "kernel_b_train"
KERNEL_C_SLUG = "takon-medical-3dgs-convert"  # GPU/CPU
KERNEL_C_DIR = "kernel_c_convert"

# Job Statuses
STATUS_PENDING = "PENDING"
STATUS_SFM_QUEUED = "SFM_QUEUED"
STATUS_SFM_RUNNING = "SFM_RUNNING"
STATUS_SFM_COMPLETED = "SFM_COMPLETED"
STATUS_TRAINING_QUEUED = "TRAINING_QUEUED"
STATUS_TRAINING_RUNNING = "TRAINING_RUNNING"
STATUS_TRAINING_COMPLETED = "TRAINING_COMPLETED"
STATUS_CONVERSION_QUEUED = "CONVERSION_QUEUED"
STATUS_CONVERSION_RUNNING = "CONVERSION_RUNNING"
STATUS_COMPLETED = "COMPLETED"
STATUS_FAILED = "FAILED"

def get_supabase_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("❌ Missing Supabase credentials in environment variables.")
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def run_kaggle_kernel(kernel_slug, local_dir):
    """Triggers a Kaggle kernel run using the API."""
    print(f"🚀 Triggering Kernel: {kernel_slug} (from {local_dir})...")
    try:
        kernel_path = Path(f"kernels/{local_dir}")
        if not kernel_path.exists():
             print(f"⚠️ Kernel path {kernel_path} does not exist. Skipping push.")
             return False
             
        # Push using the slug defined in metadata
        # We also need to pass the current environment to ensure KAGGLE_KEY is seen
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        
        result = subprocess.run(
            ["kaggle", "kernels", "push", "-p", str(kernel_path)], 
            check=True, 
            capture_output=True, 
            text=False, # Use bytes to avoid Windows encoding errors
            env=env
        )
        print(result.stdout.decode('utf-8', errors='replace'))
        print(f"✅ Kernel {kernel_slug} pushed/started successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to push kernel {kernel_slug}: {e}")
        err_msg = e.stderr.decode('utf-8', errors='replace') + e.stdout.decode('utf-8', errors='replace')
        print(f"   Error Output: {err_msg}")
        return False

def check_and_dispatch(supabase):
    """Main loop to check job statuses and dispatch kernels."""
    
    # 1. PENDING -> SFM
    # Find oldest PENDING job
    response = supabase.table("jobs").select("*").eq("status", STATUS_PENDING).order("created_at").limit(1).execute()
    if response.data:
        job = response.data[0]
        print(f"Found PENDING job: {job['id']}")
        
        # Dispatch Kernel A
        if run_kaggle_kernel(KERNEL_A_SLUG, KERNEL_A_DIR):
            supabase.table("jobs").update({
                "status": STATUS_SFM_QUEUED,
                "message": "Queued for SfM (Kernel A triggered)"
            }).eq("id", job['id']).execute()
            
    # 2. SFM_COMPLETED -> TRAINING
    response = supabase.table("jobs").select("*").eq("status", STATUS_SFM_COMPLETED).order("created_at").limit(1).execute()
    if response.data:
        job = response.data[0]
        print(f"Found SFM_COMPLETED job: {job['id']}")
        
        # Dispatch Kernel B
        if run_kaggle_kernel(KERNEL_B_SLUG, KERNEL_B_DIR):
            supabase.table("jobs").update({
                "status": STATUS_TRAINING_QUEUED,
                "message": "Queued for Training (Kernel B triggered)"
            }).eq("id", job['id']).execute()

    # 3. TRAINING_COMPLETED -> CONVERSION
    response = supabase.table("jobs").select("*").eq("status", STATUS_TRAINING_COMPLETED).order("created_at").limit(1).execute()
    if response.data:
        job = response.data[0]
        print(f"Found TRAINING_COMPLETED job: {job['id']}")
        
        # Dispatch Kernel C
        if run_kaggle_kernel(KERNEL_C_SLUG, KERNEL_C_DIR):
            supabase.table("jobs").update({
                "status": STATUS_CONVERSION_QUEUED,
                "message": "Queued for Conversion (Kernel C triggered)"
            }).eq("id", job['id']).execute()

def main():
    print("🤖 3DSCAN Orchestrator Started...", flush=True)
    supabase = get_supabase_client()
    
    print(f"👉 Monitoring jobs... (Poll Interval: {POLL_INTERVAL}s)", flush=True)

    while True:
        try:
            check_and_dispatch(supabase)
        except Exception as e:
            print(f"⚠️ Error in dispatch loop: {e}", flush=True)
        
        # Simple progress indicator
        print(".", end="", flush=True)
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
