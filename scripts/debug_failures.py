import os
import requests
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_PUBLIC_KEY")

def check_specific_job(job_id):
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("❌ Missing credentials")
        return
    
    url = f"{SUPABASE_URL}/rest/v1/jobs?select=id,status,message,result_url&id=eq.{job_id}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }
    
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        jobs = resp.json()
        if jobs:
            job = jobs[0]
            print(f"Job ID: {job['id']}")
            print(f"Status: {job['status']}")
            print(f"Message: {job.get('message')}")
            print(f"Result: {job.get('result_url')}")
        else:
            print(f"Job {job_id} not found.")
    else:
        print(f"Error: {resp.status_code} - {resp.text}")

def check_all_failures():
    url = f"{SUPABASE_URL}/rest/v1/jobs?select=id,status,message&status=in.(.TRAINING_FAILED,.SFM_FAILED)&order=created_at.desc&limit=5"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        jobs = resp.json()
        print(f"Found {len(jobs)} recent failures:\n")
        for job in jobs:
            print(f"ID: {job['id']} | Status: {job['status']}")
            print(f"Msg: {job.get('message')}")
            print("-" * 20)
    else:
        print(f"Error: {resp.status_code}")

if __name__ == "__main__":
    # Check the one user mentioned
    print("--- Specific Job Check ---")
    check_specific_job("579c2e5c") 
    print("\n--- Recent Failures Check ---")
    check_all_failures()
