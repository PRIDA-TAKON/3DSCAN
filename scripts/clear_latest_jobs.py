import os
import requests
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_ROLL_KEY') or os.getenv('SUPABASE_SERVICE_ROLL_SECERT_KEY')

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Missing Supabase credentials")
    exit(1)

HEADERS = {
    'apikey': SUPABASE_KEY,
    'Authorization': f'Bearer {SUPABASE_KEY}',
    'Content-Type': 'application/json'
}

def clear_latest(limit=4):
    print(f"🚀 Fetching latest {limit} jobs...")
    # Fetch latest jobs ordered by created_at descending, limit to the number we want to delete
    url = f"{SUPABASE_URL}/rest/v1/jobs?select=id&order=created_at.desc&limit={limit}"
    resp = requests.get(url, headers=HEADERS)
    
    if resp.status_code != 200:
        print(f"❌ Failed to fetch jobs: {resp.text}")
        return

    jobs = resp.json()
    if not jobs:
        print("✅ No jobs found to delete.")
        return

    ids_to_delete = [j['id'] for j in jobs]
    print(f"🗑️ Deleting {len(ids_to_delete)} latest jobs: {ids_to_delete}")

    # Delete by ID in
    delete_url = f"{SUPABASE_URL}/rest/v1/jobs?id=in.({','.join(ids_to_delete)})"
    del_resp = requests.delete(delete_url, headers=HEADERS)

    if del_resp.status_code in [200, 204]:
        print(f"✅ Successfully cleared {len(ids_to_delete)} latest jobs.")
    else:
        print(f"❌ Deletion failed: {del_resp.text}")

if __name__ == "__main__":
    clear_latest(4)
