import os
import requests
from dotenv import load_dotenv

load_dotenv()
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_ANON_PUBLIC_KEY')

def get_job_error(job_id):
    url = f"{supabase_url}/rest/v1/jobs?id=eq.{job_id}&select=*"
    headers = {
        'apikey': supabase_key,
        'Authorization': f'Bearer {supabase_key}'
    }
    try:
        res = requests.get(url, headers=headers)
        data = res.json()
        if data:
            print(f"🆔 Job: {job_id}")
            print(f"📊 Status: {data[0]['status']}")
            print(f"💬 Message: {data[0].get('message', 'No message')}")
        else:
            print("❌ Job not found")
    except Exception as e:
        print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    # เช็ค 10 งานล่าสุดเลย
    url = f"{supabase_url}/rest/v1/jobs?select=id,status,message&order=created_at.desc&limit=10"
    headers = {
        'apikey': supabase_key,
        'Authorization': f'Bearer {supabase_key}'
    }
    try:
        res_data = requests.get(url, headers=headers).json()
        if isinstance(res_data, list):
            for job in res_data:
                print(f"\n--- Job {job.get('id', 'N/A')} ---")
                print(f"Status: {job.get('status', 'N/A')}")
                print(f"Error: {job.get('message', 'N/A')}")
        else:
            print(f"⚠️ Response is not a list: {res_data}")
    except Exception as e:
        print(f"❌ Error fetching jobs: {e}")
