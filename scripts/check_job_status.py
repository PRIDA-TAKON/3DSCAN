import os
from supabase import create_client

def check_jobs():
    # Credentials derived from user's .env file
    url = "https://nrkhqeavnypzklqqfatf.supabase.co"
    key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5ya2hxZWF2bnlwemtscXFmYXRmIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2NzQ0NDk0OSwiZXhwIjoyMDgzMDIwOTQ5fQ.3WIY-fzFTI7vlZtQbTqNfgGbpp7Kf8qP_9Q2N-R0AtA"

    supabase = create_client(url, key)
    try:
        response = supabase.table("jobs").select("*").order("created_at", desc=True).limit(5).execute()
        jobs = response.data
        if not jobs:
            print("📭 No jobs found.")
            return

        print(f"{'ID':<15} | {'Status':<15} | {'Created At':<25} | {'Message'}")
        print("-" * 120)
        for job in jobs:
            id_short = str(job['id'])
            status = job.get('status', 'N/A')
            created_at = job.get('created_at', 'N/A')
            message = job.get('message', '')
            print(f"{id_short[:15]:<15} | {status:<15} | {created_at:<25} | {message}")
    except Exception as e:
        print(f"❌ Failed to query jobs: {e}")

if __name__ == "__main__":
    check_jobs()
