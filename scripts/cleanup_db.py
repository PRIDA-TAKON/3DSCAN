import os
import requests
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL')
# Use Service Role Key for Delete permissions
SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_ROLL_KEY') 

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Missing Supabase credentials in .env")
    exit(1)

HEADERS = {
    'apikey': SUPABASE_KEY,
    'Authorization': f'Bearer {SUPABASE_KEY}',
    'Content-Type': 'application/json',
    'Prefer': 'return=representation'
}

def cleanup():
    # 1. Fetch all job IDs ordered by created_at desc
    url = f"{SUPABASE_URL}/rest/v1/jobs?select=id,created_at&order=created_at.desc"
    resp = requests.get(url, headers=HEADERS)
    
    if resp.status_code != 200:
        print(f"❌ Failed to fetch jobs: {resp.text}")
        return

    jobs = resp.json()
    print(f"📋 Total jobs found: {len(jobs)}")

    if len(jobs) <= 2:
        print("✅ 2 or fewer jobs exist. Nothing to clean.")
        return

    # 2. Identify jobs to delete (all except the first 2)
    jobs_to_delete = jobs[2:]
    ids_to_delete = [j['id'] for j in jobs_to_delete]
    
    print(f"🗑️ Deleting {len(ids_to_delete)} old jobs...")

    # 3. Perform Deletion
    # Note: We can delete by ID using 'in' operator in PostgREST
    delete_url = f"{SUPABASE_URL}/rest/v1/jobs?id=in.({','.join(ids_to_delete)})"
    del_resp = requests.delete(delete_url, headers=HEADERS)

    if del_resp.status_code in [200, 204]:
        print(f"✅ Successfully deleted {len(ids_to_delete)} jobs.")
    else:
        print(f"❌ Deletion failed: {del_resp.text}")

if __name__ == "__main__":
    cleanup()
