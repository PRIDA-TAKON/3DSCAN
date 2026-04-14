import os
import requests
import json
from dotenv import load_dotenv

load_dotenv(override=True)

# --- Config ---
JOB_ID = "f74a6064-8ebc-49a6-ba74-1785a4fe3fc0" # Latest Job ID
VIDEO_URL = "https://nrkhqeavnypzklqqfatf.supabase.co/storage/v1/object/public/3d-scans/videos/video_car.mp4"
ENDPOINT_ID = "grdg3rydqbsj9p" # New Trainer Endpoint
API_KEY = os.getenv('RUNPOD_API_KEY')

def trigger_trainer():
    print(f"🧠 Triggering Trainer for Job: {JOB_ID}")
    url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    # ส่ง ID และ Video URL ไปเพื่อให้ Worker รู้ว่าต้องไปดึง S3 Path ของงานไหนจาก Supabase
    data = {
        "input": {
            "id": JOB_ID,
            "video_url": VIDEO_URL
        }
    }

    try:
        resp = requests.post(url, headers=headers, json=data)
        if resp.status_code == 200:
            print(f"🚀 Trainer Triggered! Job ID: {resp.json().get('id')}")
        else:
            print(f"❌ Failed: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    trigger_trainer()
