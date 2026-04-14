import os
import requests
import json
from dotenv import load_dotenv

load_dotenv(override=True)

JOB_ID = "622f11d2-a638-470a-ab11-7b5d933fbb5f-e2"
ENDPOINT_ID = "grdg3rydqbsj9p"
API_KEY = os.getenv('RUNPOD_API_KEY')

def fetch_fail_logs():
    print(f"📡 Fetching Fail Logs for RunPod Job: {JOB_ID}")
    url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{JOB_ID}"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        resp = requests.get(url, headers=headers)
        data = resp.json()
        print(f"📊 Status: {data.get('status')}")
        
        if 'error' in data and data['error']:
            print(f"❌ Error Message: {data['error']}")
        
        if 'output' in data and data['output']:
            print("📝 Output Log:")
            print(json.dumps(data['output'], indent=2))
        else:
            print("ℹ️ No detailed output log found via API.")
            
    except Exception as e:
        print(f"❌ Error fetching logs: {e}")

if __name__ == "__main__":
    fetch_fail_logs()
