import os
import requests
import uuid
from dotenv import load_dotenv

# Load environment variables from root .env
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_PUBLIC_KEY")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")

def log(msg):
    print(f"[DEBUG] {msg}")

def test_step_1_storage():
    log("Step 1: Testing Supabase Storage...")
    if not SUPABASE_URL or not SUPABASE_KEY:
        log("❌ Missing Supabase credentials")
        return None
    
    bucket = "3d-scans"
    file_name = f"debug_{uuid.uuid4().hex[:8]}.txt"
    file_path = f"videos/{file_name}"
    url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{file_path}"
    
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "text/plain"
    }
    
    try:
        resp = requests.post(url, headers=headers, data="debug content")
        if resp.status_code == 200:
            log(f"✅ Storage Success: {file_path}")
            return f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{file_path}"
        else:
            log(f"❌ Storage Failed: {resp.status_code} - {resp.text}")
    except Exception as e:
        log(f"❌ Storage Exception: {e}")
    return None

def test_step_2_db(public_url):
    log("Step 2: Testing Supabase DB Insertion...")
    if not SUPABASE_URL or not SUPABASE_KEY:
        log("❌ Missing Supabase credentials")
        return None
    
    url = f"{SUPABASE_URL}/rest/v1/jobs"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }
    payload = {
        "video_url": public_url,
        "status": "PENDING",
        "message": "Debug test job"
    }
    
    try:
        resp = requests.post(url, headers=headers, json=payload)
        if resp.status_code in [200, 201]:
            data = resp.json()
            # Handle case where it might be a list
            job_id = data[0]['id'] if isinstance(data, list) else data.get('id')
            log(f"✅ DB Success: Job ID {job_id}")
            return job_id
        else:
            log(f"❌ DB Failed: {resp.status_code} - {resp.text}")
    except Exception as e:
        log(f"❌ DB Exception: {e}")
    return None

def test_step_3_runpod(job_id, public_url):
    log("Step 3: Testing RunPod API Trigger...")
    if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT_ID:
        log("❌ Missing RunPod credentials")
        return False
    
    url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/run"
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": {
            "id": job_id,
            "video_url": public_url,
            "mode": "PROCESS"
        }
    }
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            log(f"✅ RunPod Success: {resp.text}")
            return True
        else:
            log(f"❌ RunPod Failed: {resp.status_code} - {resp.text}")
    except Exception as e:
        log(f"❌ RunPod Exception: {e}")
    return False

if __name__ == "__main__":
    print("=== Starting End-to-End Debug Flow ===")
    public_url = test_step_1_storage()
    if public_url:
        job_id = test_step_2_db(public_url)
        if job_id:
            success = test_step_3_runpod(job_id, public_url)
            if success:
                print("\n🎉 RESULT: ALL SYSTEMS OPERATIONAL")
            else:
                print("\n🛑 RESULT: FAILED AT RUNPOD API")
        else:
            print("\n🛑 RESULT: FAILED AT SUPABASE DB")
    else:
        print("\n🛑 RESULT: FAILED AT SUPABASE STORAGE")
