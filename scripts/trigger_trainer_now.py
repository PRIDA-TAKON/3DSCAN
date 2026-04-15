import os
import requests
import json
import time
from dotenv import load_dotenv

load_dotenv(override=True)

# --- Config ---
JOB_ID = "c345c2a0-d781-4752-8250-60494b3d6c30" 
VIDEO_URL = "https://nrkhqeavnypzklqqfatf.supabase.co/storage/v1/object/public/3d-scans/videos/video_car.mp4"
ENDPOINT_ID = os.getenv('RUNPOD_ENDPOINT_ID_TRAINER')
if not ENDPOINT_ID:
    print("❌ ERROR: RUNPOD_ENDPOINT_ID_TRAINER not found in .env")
    exit(1)
API_KEY = os.getenv('RUNPOD_API_KEY')

def trigger_trainer():
    print(f"🧠 Triggering Trainer for Job: {JOB_ID}")
    url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "input": {
            "id": JOB_ID,
            "video_url": VIDEO_URL,
            "mode": "TRAIN"
        }
    }
    try:
        resp = requests.post(url, json=data, headers=headers)
        if resp.status_code == 200:
            rid = resp.json().get('id')
            print(f"🚀 Trainer Triggered! Job ID: {rid}")
            return rid
        else:
            print(f"❌ Failed: {resp.status_code} - {resp.text}")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def monitor_job(runpod_job_id):
    print(f"📺 Monitoring Job: {runpod_job_id}")
    url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{runpod_job_id}"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    last_status = ""
    while True:
        try:
            resp = requests.get(url, headers=headers)
            data = resp.json()
            status = data.get('status')
            
            if status != last_status:
                print(f"📊 Status changed: {status}")
                last_status = status
                
            if status == "COMPLETED":
                print("🎉 SUCCESS! Job finished.")
                if 'stdout' in data: print(data['stdout'])
                break
            elif status in ["FAILED", "CANCELLED"]:
                print(f"❌ JOB {status}!")
                print("🔻 Error Log:", data.get('error'))
                if 'stdout' in data:
                    print("\n--- 📝 Worker Stdout ---")
                    print(data['stdout'])
                break
            
            time.sleep(10)
        except Exception as e:
            print(f"⚠️ Monitor error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    job_id = trigger_trainer()
    if job_id:
        monitor_job(job_id)
