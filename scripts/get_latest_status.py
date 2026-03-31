import os
import requests
from dotenv import load_dotenv

load_dotenv()
url = f"{os.getenv('SUPABASE_URL')}/rest/v1/jobs?select=id,status,message,created_at&order=created_at.desc&limit=1"
key = os.getenv('SUPABASE_ANON_PUBLIC_KEY')
headers = {'apikey': key, 'Authorization': f'Bearer {key}'}

try:
    response = requests.get(url, headers=headers)
    data = response.json()
    if data and isinstance(data, list):
        job = data[0]
        print(f"🆔 Job ID: {job['id']}")
        print(f"📊 Status: {job['status']}")
        print(f"💬 Error Message: {job.get('message', 'No message')}")
        print(f"🕒 Created at: {job['created_at']}")
    else:
        print(f"❌ Error or No Data: {data}")
except Exception as e:
    print(f"⚠️ Failed to fetch status: {e}")
