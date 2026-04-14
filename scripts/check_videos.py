import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

URL = os.getenv('SUPABASE_URL')
KEY = os.getenv('SUPABASE_ANON_PUBLIC_KEY')

# List files in bucket 3d-scans inside folder videos/
endpoint = f"{URL}/storage/v1/object/list/3d-scans"
headers = {
    "apikey": KEY,
    "Authorization": f"Bearer {KEY}",
    "Content-Type": "application/json"
}
payload = {"prefix": "videos/", "limit": 10}

try:
    resp = requests.post(endpoint, headers=headers, json=payload)
    files = resp.json()
    print("--- 📹 Available Videos in Storage ---")
    for f in files:
        if f.get('name') != '.emptyFolderPlaceholder':
            print(f"- {f.get('name')}")
except Exception as e:
    print(f"Error: {e}")
