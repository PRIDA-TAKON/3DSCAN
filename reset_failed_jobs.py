from dotenv import load_dotenv
import os
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Missing Supabase credentials in .env")
    exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Fetch stuck jobs (Queue or Running)
status_to_reset = ["SFM_QUEUED", "SFM_RUNNING", "TRAINING_QUEUED", "TRAINING_RUNNING", "CONVERSION_QUEUED", "CONVERSION_RUNNING"]

print("🔄 Checking for stuck jobs...")
try:
    response = supabase.table("jobs").select("id, status").in_("status", status_to_reset).execute()
    jobs = response.data

    if not jobs:
        print("✅ No stuck jobs found.")
    else:
        print(f"⚠️ Found {len(jobs)} jobs to reset: {[j['id'] for j in jobs]}")
        # Reset all to PENDING
        for job in jobs:
            try:
                supabase.table("jobs").update({"status": "PENDING"}).eq("id", job['id']).execute()
                print(f"✅ Reset job {job['id']} to PENDING.")
            except Exception as e:
                print(f"❌ Failed to reset job {job['id']}: {e}")
except Exception as e:
    print(f"❌ Error fetching jobs: {e}")
