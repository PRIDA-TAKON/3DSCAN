import requests
import os
from dotenv import load_dotenv

# โหลดค่าจาก .env
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLL_SECERT_KEY") # ใช้ Service Key เพื่อข้าม RLS

def test_insert_job():
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        print("❌ ไม่พบ SUPABASE_URL หรือ SUPABASE_SERVICE_ROLL_SECERT_KEY ใน .env")
        return

    url = f"{SUPABASE_URL}/rest/v1/jobs"
    
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }

    # ข้อมูลจำลองสำหรับทดสอบ
    payload = {
        "video_url": "https://example.com/test_video.mp4",
        "status": "PENDING",
        "message": "Test webhook from script"
    }

    try:
        print(f"🚀 กำลังเพิ่มข้อมูลลงในตาราง jobs ที่ {url}...")
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        job_id = data[0]['id']
        print(f"✅ เพิ่มข้อมูลสำเร็จ! Job ID: {job_id}")
        print(f"💡 กรุณาตรวจสอบ logs ใน Supabase Edge Function (trigger-runpod)")
        print(f"💡 และตรวจสอบใน RunPod ว่ามีงานใหม่ถูกสร้างขึ้นหรือไม่")
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการเพิ่มข้อมูล: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"รายละเอียด: {e.response.text}")

if __name__ == "__main__":
    test_insert_job()
