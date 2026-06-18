import os
import requests
import json
from dotenv import load_dotenv

load_dotenv(override=True)

RUNPOD_API_KEY = os.getenv('RUNPOD_API_KEY')
ENDPOINT_ID = "grdg3rydqbsj9p" # Trainer Endpoint
# ใส่ RunPod Job ID ล่าสุดที่เพิ่งรันไป
JOB_ID = "47df52de-a8c6-4fd6-a2b7-822b6c02cd4a-e1" 

def fetch_runpod_output():
    print(f"📡 Fetching RunPod output for Job: {JOB_ID}")
    url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{JOB_ID}"
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        resp = requests.get(url, headers=headers)
        data = resp.json()
        print(f"📊 Status: {data.get('status')}")
        
        if 'error' in data:
            print(f"❌ Error Message: {data['error']}")
            
        if 'stdout' in data:
            print("\n--- 📝 Full Worker Stdout ---")
            print(data['stdout'])
        else:
            print("\n⚠️ No stdout found in response.")
            
        if 'output' in data:
            print("\n🔗 Output Data:", json.dumps(data['output'], indent=2))
            
    except Exception as e:
        print(f"❌ API Error: {e}")

if __name__ == "__main__":
    fetch_runpod_output()
