import os
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_ROLE_KEY') or os.getenv('SUPABASE_KEY') or os.getenv('SUPABASE_ANON_PUBLIC_KEY')

def fix_job_message():
    job_id = "c345c2a0-d781-4752-8250-60494b3d6c30"
    s3_path = f"temp/{job_id}/processed.zip"
    
    url = f"{SUPABASE_URL}/rest/v1/jobs?id=eq.{job_id}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    # ใส่เฉพาะ S3_PATH เพื่อให้ Worker ดึงไปใช้ง่ายๆ
    data = {
        "status": "ready_to_train",
        "message": f"S3_PATH:{s3_path}"
    }
    
    print(f"🛠️ Cleaning and fixing job {job_id} in Supabase...")
    try:
        resp = requests.patch(url, headers=headers, json=data)
        if resp.status_code in [200, 201, 204]:
            print("✅ Job message is now clean! S3_PATH set.")
        else:
            print(f"❌ Failed to fix job: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    fix_job_message()
