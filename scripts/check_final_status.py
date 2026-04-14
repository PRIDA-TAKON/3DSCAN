import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

# ข้อมูลจากรอบที่แล้ว
JOB_ID = "86f7a715-e338-4d3f-ac26-f63164e53d85-e2"
ENDPOINT_ID = "grdg3rydqbsj9p"
API_KEY = os.getenv('RUNPOD_API_KEY')

print(f"📡 Checking Job: {JOB_ID}")
print(f"🔹 Endpoint ID in .env: {ENDPOINT_ID}")

url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{JOB_ID}"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

try:
    resp = requests.get(url, headers=headers)
    print(f"📊 Status Code: {resp.status_code}")
    print(f"📝 Response: {json.dumps(resp.json(), indent=2)}")
except Exception as e:
    print(f"❌ Error: {e}")
