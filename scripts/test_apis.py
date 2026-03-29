import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

# ข้อมูลจาก .env หรือใส่เองตรงนี้
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID")
VERCEL_URL = "https://3dscanfromvdo-pbrobolp3-prida-takons-projects.vercel.app" # เปลี่ยนเป็น URL จริงของคุณ

def test_runpod_direct():
    print("\n--- 1. Testing RunPod Direct API ---")
    if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT_ID:
        print("❌ Missing RUNPOD_API_KEY or RUNPOD_ENDPOINT_ID in .env")
        return

    url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/run"
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": {
            "id": "test_local_script",
            "video_url": "https://nrkhqeavnypzklqqfatf.supabase.co/storage/v1/object/public/3d-scans/videos/test-video.mp4"
        }
    }

    try:
        res = requests.post(url, json=payload, headers=headers)
        print(f"Status: {res.status_code}")
        print(f"Response: {res.text}")
        if res.status_code == 200:
            print("✅ RunPod API Key & ID are VALID")
            return res.json().get("id")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
    return None

def test_vercel_api(job_id="test_from_vercel"):
    print(f"\n--- 2. Testing Vercel API Route ({VERCEL_URL}) ---")
    url = f"{VERCEL_URL}/api/trigger-job"
    payload = {
        "jobId": job_id,
        "videoUrl": "https://nrkhqeavnypzklqqfatf.supabase.co/storage/v1/object/public/3d-scans/videos/test-video.mp4"
    }

    try:
        res = requests.post(url, json=payload)
        print(f"Status: {res.status_code}")
        print(f"Response: {res.text}")
        if res.status_code == 200:
            print("✅ Vercel FE successfully triggered RunPod")
        else:
            print("❌ Vercel API failed. Check Vercel Environment Variables.")
    except Exception as e:
        print(f"❌ Vercel connection failed: {e}")

if __name__ == "__main__":
    runpod_job_id = test_runpod_direct()
    test_vercel_api()
