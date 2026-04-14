import os
import requests
import uuid
from dotenv import load_dotenv

load_dotenv(override=True)

# --- Configuration ---
RUNPOD_API_KEY = os.getenv('RUNPOD_API_KEY')
ENDPOINT_ID = os.getenv('RUNPOD_ENDPOINT_ID_PROCESSOR')
SUPABASE_URL = os.getenv('SUPABASE_URL')
# Try to find the correct key
SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_ROLE_KEY') or os.getenv('SUPABASE_KEY') or os.getenv('SUPABASE_ANON_PUBLIC_KEY')

VIDEO_URL = "https://nrkhqeavnypzklqqfatf.supabase.co/storage/v1/object/public/3d-scans/videos/video_car.mp4"

def trigger_new_job():
    job_id = str(uuid.uuid4())
    print(f"🎬 Initializing New Job: {job_id}")
    print(f"📡 Targeting Endpoint ID: {ENDPOINT_ID}")

    # 1. Create record in Supabase
    sb_url = f"{SUPABASE_URL}/rest/v1/jobs"
    sb_headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }
    sb_data = {
        "id": job_id,
        "video_url": VIDEO_URL,
        "status": "pending",
        "message": "Starting test with video_car.mp4 (Safe settings)"
    }
    
    try:
        resp = requests.post(sb_url, headers=sb_headers, json=sb_data)
        if resp.status_code not in [200, 201]:
            print(f"❌ Failed to create job record: {resp.status_code} - {resp.text}")
            return
        print("✅ Job record created in Supabase.")
    except Exception as e:
        print(f"❌ Supabase Error: {e}")
        return

    # 2. Trigger RunPod
    rp_url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"
    rp_headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }
    rp_data = {
        "input": {
            "id": job_id,
            "video_url": VIDEO_URL
        }
    }

    try:
        resp = requests.post(rp_url, headers=rp_headers, json=rp_data)
        if resp.status_code == 200:
            print(f"🚀 RunPod Triggered! RunPod Job ID: {resp.json().get('id')}")
            print(f"🔗 You can monitor progress in Supabase or RunPod logs.")
        else:
            print(f"❌ Failed to trigger RunPod: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"❌ RunPod Error: {e}")

if __name__ == "__main__":
    trigger_new_job()
