import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_ANON_PUBLIC_KEY')
HEADERS = {'apikey': SUPABASE_KEY, 'Authorization': f'Bearer {SUPABASE_KEY}'}

def get_latest_data():
    print("--- 📋 Latest Job Status ---")
    jobs_url = f"{SUPABASE_URL}/rest/v1/jobs?select=id,status,message,created_at&order=created_at.desc&limit=1"
    try:
        resp = requests.get(jobs_url, headers=HEADERS)
        jobs = resp.json()
        if jobs:
            print(json.dumps(jobs[0], indent=2))
        else:
            print("No jobs found.")
    except Exception as e:
        print(f"Error fetching jobs: {e}")

    print("\n--- 🛑 Latest Crash Logs (runpod_logs) ---")
    logs_url = f"{SUPABASE_URL}/rest/v1/runpod_logs?select=job_id,log_content,created_at&order=created_at.desc&limit=1"
    try:
        resp = requests.get(logs_url, headers=HEADERS)
        logs = resp.json()
        if logs:
            log = logs[0]
            print(f"Job ID: {log['job_id']}")
            print(f"Created At: {log['created_at']}")
            print(f"Content:\n{log['log_content'][:2000]}...") # Show first 2000 chars
        else:
            print("No crash logs found.")
    except Exception as e:
        print(f"Error fetching logs: {e}")

if __name__ == "__main__":
    get_latest_data()
