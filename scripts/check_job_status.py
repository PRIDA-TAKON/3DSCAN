import os
import requests
from dotenv import load_dotenv

# โหลดค่าจาก .env
load_dotenv()

supabase_url = os.getenv('SUPABASE_URL')
# ลองหา key จากหลายๆ ชื่อที่อาจมีใน .env
supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY') or os.getenv('SUPABASE_KEY') or os.getenv('SUPABASE_SERVICE_ROLL_SECERT_KEY')

if not supabase_url or not supabase_key:
    print('❌ ไม่พบ SUPABASE_URL หรือ SUPABASE_KEY ใน .env')
    exit(1)

url = f'{supabase_url}/rest/v1/jobs?select=id,status,message,created_at&order=created_at.desc&limit=5'
headers = {
    'apikey': supabase_key,
    'Authorization': f'Bearer {supabase_key}'
}

try:
    print(f"🔍 Checking jobs at: {supabase_url}...")
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    jobs = response.json()
    
    if not jobs:
        print('❌ No jobs found in the system.')
    else:
        print(f"✅ Found {len(jobs)} recent jobs:\n")
        for job in jobs:
            print(f"🆔 ID: {job['id']}")
            print(f"🕒 Created: {job['created_at']}")
            print(f"📊 Status: {job['status']}")
            print(f"💬 Message: {job.get('message', '-')}")
            print("-" * 50)
            
except Exception as e:
    print(f'⚠️ Error: {e}')
    if hasattr(e, 'response') and e.response is not None:
        print(f"Details: {e.response.text}")
