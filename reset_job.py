
import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
supabase = create_client(url, key)

job_id = "3e4e7e31-b42e-44da-9218-1904f0e2578b"
res = supabase.table("jobs").update({"status": "PENDING", "message": "Ready for headless re-run"}).eq("id", job_id).execute()
print(f"Updated job {job_id} to PENDING")
