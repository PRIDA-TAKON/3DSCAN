import os
import subprocess
from dotenv import load_dotenv

load_dotenv()

# CLI Compatibility
if not os.environ.get("KAGGLE_KEY") and os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_KEY"] = os.environ.get("KAGGLE_API_TOKEN")

username = os.environ.get("KAGGLE_USERNAME")
if not username:
    print("❌ KAGGLE_USERNAME not found in .env")
    exit(1)

slug = f"{username}/takon-medical-3dgs-sfm"

print(f"Fetching logs for: {slug}")

# Run kaggle output
env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"
env["PYTHONUTF8"] = "1"

try:
    # Download logs to current directory
    result = subprocess.run(
        ["kaggle", "kernels", "output", slug, "-p", "."],
        capture_output=True,
        text=False,
        env=env,
        check=True
    )
    print("✅ Logs downloaded successfully.")
    
    # Read and print the log file (usually named 'main.log' or similar, standard output is in the console output on web, 
    # but 'output' command downloads output files. We might need 'status' command's failure message or just check downloaded files.)
    # Actually, kaggle kernels output downloads the OUTPUT files produced by the kernel (like submission.csv or images), NOT the console logs.
    # To get console logs, we often have to rely on the web UI or 'kaggle kernels status' might give a failure reason string.
    # HOWEVER, let's see if there's a log file.
    
    files = os.listdir(".")
    print(f"Files in current dir: {files}")
    
    if "main.log" in files:
        with open("main.log", "r", encoding="utf-8") as f:
            print(f.read())
            
except subprocess.CalledProcessError as e:
    print(f"Error fetching logs for {slug}")
    print(f"Command output: {e.stdout.decode('utf-8', errors='replace')}")
    print(f"Error output: {e.stderr.decode('utf-8', errors='replace')}")
