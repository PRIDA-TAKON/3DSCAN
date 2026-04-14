import os
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

URL = os.getenv('SUPABASE_URL')
KEY = os.getenv('SUPABASE_SERVICE_ROLL_KEY') or os.getenv('SUPABASE_KEY')
JOB_ID = "f74a6064-8ebc-49a6-ba74-1785a4fe3fc0"

def fix_job_data():
    print(f"🛠️ Fixing corrupted data for Job: {JOB_ID}")
    
    # กำหนดค่า S3_PATH ที่ถูกต้อง (ตามโครงสร้างที่เราออกแบบไว้)
    correct_message = f"S3_PATH:temp/{JOB_ID}/processed.zip"
    
    endpoint = f"{URL}/rest/v1/jobs?id=eq.{JOB_ID}"
    headers = {
        "apikey": KEY,
        "Authorization": f"Bearer {KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "status": "ready_to_train",
        "message": correct_message
    }

    try:
        resp = requests.patch(endpoint, headers=headers, json=data)
        if resp.status_code in [200, 204]:
            print("✅ Data restored successfully!")
        else:
            print(f"❌ Failed to fix data: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    fix_job_data()
