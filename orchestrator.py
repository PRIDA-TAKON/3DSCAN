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

# Kernel Slugs (Reflects the 3 stages)
KERNEL_A_SLUG = "takon-medical-3dgs-sfm"      # CPU/RAM
KERNEL_B_SLUG = "takon-medical-3dgs-train"    # GPU
KERNEL_C_SLUG = "takon-medical-3dgs-convert"  # GPU/CPU

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

def run_kaggle_kernel(kernel_slug):
    """Triggers a Kaggle kernel run using the API."""
    print(f"🚀 Triggering Kernel: {kernel_slug}...")
    try:
        # Push the kernel (this assumes the local folder structure matches Kaggle's expectation)
        # But for 'triggering', we often just need to push the code. 
        # Ideally, we have local folders: kernels/A, kernels/B, kernels/C
        # And we run `kaggle kernels push -p kernels/A`
        
        kernel_path = Path(f"kernels/{kernel_slug}")
        if not kernel_path.exists():
             print(f"⚠️ Kernel path {kernel_path} does not exist. Skipping push.")
             return False
             
        subprocess.run(["kaggle", "kernels", "push", "-p", str(kernel_path)], check=True)
        print(f"✅ Kernel {kernel_slug} pushed/started successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to push kernel {kernel_slug}: {e}")
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
        if run_kaggle_kernel(KERNEL_A_SLUG):
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
        if run_kaggle_kernel(KERNEL_B_SLUG):
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
        if run_kaggle_kernel(KERNEL_C_SLUG):
            supabase.table("jobs").update({
                "status": STATUS_CONVERSION_QUEUED,
                "message": "Queued for Conversion (Kernel C triggered)"
            }).eq("id", job['id']).execute()

def main():
    print("🤖 3DSCAN Orchestrator Started...")
    supabase = get_supabase_client()
    
    while True:
        try:
            check_and_dispatch(supabase)
        except Exception as e:
            print(f"⚠️ Error in dispatch loop: {e}")
        
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
