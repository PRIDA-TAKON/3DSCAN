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

print(f"Checking status for: {slug}")

# Run kaggle status
env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"
env["PYTHONUTF8"] = "1"

try:
    result = subprocess.run(
        ["kaggle", "kernels", "status", slug],
        capture_output=True,
        text=False,
        env=env,
        check=True
    )
    print(result.stdout.decode('utf-8', errors='replace'))
except subprocess.CalledProcessError as e:
    print(f"Error checking status for {slug}")
    print(f"Command output: {e.stdout.decode('utf-8', errors='replace')}")
    print(f"Error output: {e.stderr.decode('utf-8', errors='replace')}")
