import os
import requests
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_PUBLIC_KEY")

def check_jobs():
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("❌ Missing credentials")
        return
    
    url = f"{SUPABASE_URL}/rest/v1/jobs?select=id,status,message,result_url&order=created_at.desc&limit=3"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }
    
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        jobs = resp.json()
        for i, job in enumerate(jobs, 1):
            print(f"Job {i} | Status: {job['status']} | Result: {job.get('result_url')}")
            print(f"Msg: {job.get('message')}")
            print("-" * 20)
    else:
        print(f"Error: {resp.status_code}")

if __name__ == "__main__":
    check_jobs()
