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
KERNEL_A_SLUG = "takon-medical-3dgs-sfm"      # CPU/RAM (Colmap Standard)
KERNEL_A_DIR = "kernel_a_sfm"
KERNEL_A_GLOMAP_SLUG = "takon-medical-3dgs-glomap" # CPU/RAM (Glomap Optimized)
KERNEL_A_GLOMAP_DIR = "kernel_a_glomap"
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


def get_gdrive_service_account_json():
    """Reads the Service Account JSON content from local file."""
    sa_val = os.environ.get("GDRIVE_SERVICE_ACCOUNT")
    if not sa_val:
        return None
    
    # If it looks like JSON content (starts with {), return it
    if sa_val.strip().startswith("{"):
        return sa_val
        
    # If it looks like a filename or partial filename
    # The user has "mcp-gantt-127e2456eed8.json" and env has "127e2456eed8..."
    # Try to find a file matching the pattern
    for file in os.listdir("."):
        if file.endswith(".json") and sa_val in file:
             try:
                 with open(file, "r", encoding="utf-8") as f:
                     return f.read()
             except Exception as e:
                 print(f"⚠️ Failed to read Service Account file {file}: {e}")
                 return None
    return None

def inject_secrets(script_path, secrets):
    """Injects secrets into the python script as environment variables."""
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        injection_code = "import os\n"
        for key, value in secrets.items():
            if value:
                # Escape the value for safety, especially JSON
                # Use triple quotes for multiline values (like Private Key in JSON)
                safe_val = value.replace('\\', '\\\\').replace("'''", "\\'\\'\\'")
                injection_code += f"os.environ['{key}'] = '''{safe_val}'''\n"
        
        injection_code += "\n# --- End of Injected Secrets ---\n"
        
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(injection_code + content)
        return True
    except Exception as e:
        print(f"❌ Failed to inject secrets: {e}")
        return False

def inject_dependency_scripts(main_script_path, dependency_scripts):
    """
    Reads dependency scripts and injects code into main_script_path 
    to write them to disk at runtime.
    """
    try:
        if not dependency_scripts:
            return True

        print(f"💉 Injecting {len(dependency_scripts)} dependency scripts into main.py...")
        
        injection_code = "\n# --- Injected Dependency Scripts ---\n"
        injection_code += "import os\n"
        
        for script_path in dependency_scripts:
            script_path = Path(script_path)
            if not script_path.exists():
                print(f"⚠️ Warning: Dependency script {script_path} not found.")
                continue
                
            fname = script_path.name
            with open(script_path, "r", encoding="utf-8") as f:
                script_content = f.read()
            
            # use repr() to get a safe python string representation of the code
            safe_content = repr(script_content)
            
            injection_code += f"# Writing {fname}\n"
            injection_code += f"with open('{fname}', 'w', encoding='utf-8') as f:\n"
            injection_code += f"    f.write({safe_content})\n"
            injection_code += f"print(f'✅ Injected script {fname} written to disk.')\n"

        injection_code += "# -----------------------------------\n\n"

        with open(main_script_path, "r", encoding="utf-8") as f:
            original_content = f.read()
            
        with open(main_script_path, "w", encoding="utf-8") as f:
            f.write(injection_code + original_content)
            
        return True

    except Exception as e:
        print(f"❌ Failed to inject dependency scripts: {e}")
        return False

def run_kaggle_kernel(kernel_slug, local_dir):
    """Triggers a Kaggle kernel run using the API, with secret injection."""
    print(f"🚀 Triggering Kernel: {kernel_slug} (from {local_dir})...")
    
    # specific fix for "keys fly away" issue on Kaggle
    # We copy the kernel code to a temp dir, inject secrets, then push.
    import shutil
    temp_dir = Path(f"kernels/temp_{local_dir}")
    source_dir = Path(f"kernels/{local_dir}")
    
    if not source_dir.exists():
         print(f"⚠️ Kernel path {source_dir} does not exist. Skipping push.")
         return False
         
    try:
        # 1. Copy to temp
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        shutil.copytree(source_dir, temp_dir)
        
        # 1.1 Copy Dependency Scripts (from scripts/ folder)
        # We want to avoid committing duplicates in the repo, so we copy them at runtime.
        scripts_map = {
            KERNEL_A_DIR: ["step1_extract_frames.py", "step2_colmap_sfm.py"],
            KERNEL_A_GLOMAP_DIR: ["step1_extract_frames.py", "run_glomap.py"],
            KERNEL_B_DIR: ["step3_train_splatting.py"],
            KERNEL_C_DIR: ["step4_export.py"]
        }
        
        required_scripts = scripts_map.get(local_dir, [])
        dependency_script_paths = []

        for script_name in required_scripts:
            src_script = Path("scripts") / script_name
            dst_script = temp_dir / script_name
            
            if src_script.exists():
                shutil.copy(src_script, dst_script)
                print(f"   + Included script: {script_name}")
                dependency_script_paths.append(dst_script)
            else:
                print(f"⚠️ Warning: Required script {script_name} not found in scripts/ folder.")

        # 1.2 Inject Dependency Scripts into main.py
        main_script = temp_dir / "main.py"
        if main_script.exists():
            print(f"DEBUG: main_script found at {main_script.resolve()}")
            # inject dependency scripts BEFORE secrets, so they are at the top (or bottom)
            # Both prepend, so:
            # 1. inject_dependency_scripts prepends -> [Scripts] + [Original]
            # 2. inject_secrets prepends -> [Secrets] + [Scripts] + [Original]
            print(f"DEBUG: Calling inject_dependency_scripts with {len(dependency_script_paths)} scripts")
            if inject_dependency_scripts(main_script, dependency_script_paths):
                 print(f"✅ Injected {len(dependency_script_paths)} scripts into main.py")
            else:
                 print("⚠️ Script injection failed.")
        else:
            print(f"DEBUG: main_script NOT FOUND at {main_script.resolve()}")
        
        # 2. Prepare Secrets
        secrets = {
            "SUPABASE_URL": SUPABASE_URL,
            "SUPABASE_KEY": SUPABASE_KEY,
            "GDRIVE_SERVICE_ACCOUNT": get_gdrive_service_account_json()
        }
        
        # print(f"DEBUG: Injecting secrets...", flush=True) 

        # 3. Inject into main.py
        main_script = temp_dir / "main.py"
        if main_script.exists():
            if inject_secrets(main_script, secrets):
                 print(f"✅ Secrets injected into {kernel_slug}")
            else:
                print("⚠️ Secret injection failed, proceeding but kernel might fail.")
        else:
             print("⚠️ main.py not found in kernel dir, skipping secret injection.")

        # --- DEBUG: List files in temp dir before push ---
        print(f"📦 Files prepared for push in {temp_dir}:")
        for f in temp_dir.iterdir():
            print(f"   - {f.name}")
        # -------------------------------------------------

        # 4. Push using the slug defined in metadata
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        
        # We need to run push command on the TEMP dir
        # But `kaggle kernels push` command looks for metadata.json in the dir
        # It IS in the temp dir because we copied the whole tree.
        
        result = subprocess.run(
            ["kaggle", "kernels", "push", "-p", str(temp_dir)], 
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
    finally:
        # 5. Cleanup
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"⚠️ Failed to cleanup temp dir: {e}")
def check_and_dispatch(supabase):
    """Main loop to check job statuses and dispatch kernels."""
    
    # 1. PENDING -> SFM
    # Find oldest PENDING job
    response = supabase.table("jobs").select("*").eq("status", STATUS_PENDING).order("created_at").limit(1).execute()
    if response.data:
        job = response.data[0]
        print(f"Found PENDING job: {job['id']}")
        
        # Dispatch Kernel A (Standard COLMAP)
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
