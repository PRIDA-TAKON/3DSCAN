import os
import requests
from dotenv import load_dotenv

# โหลดค่าจาก .env
load_dotenv()

supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY') or os.getenv('SUPABASE_KEY') or os.getenv('SUPABASE_SERVICE_ROLL_SECERT_KEY')

if not supabase_url or not supabase_key:
    print('❌ ไม่พบ SUPABASE_URL หรือ SUPABASE_KEY ใน .env')
    exit(1)

headers = {
    'apikey': supabase_key,
    'Authorization': f'Bearer {supabase_key}',
    'Content-Type': 'application/json'
}

def cleanup_jobs():
    print("🧹 กำลังลบข้อมูลในตาราง jobs ทั้งหมด...")
    url = f"{supabase_url}/rest/v1/jobs?id=neq.00000000-0000-0000-0000-000000000000" # ลบทุก id
    try:
        res = requests.delete(url, headers=headers)
        res.raise_for_status()
        print("✅ ลบข้อมูลในตาราง jobs สำเร็จ!")
    except Exception as e:
        print(f"⚠️ ไม่สามารถลบ jobs ได้: {e}")

def cleanup_storage(bucket_name):
    print(f"🧹 กำลังล้างไฟล์ใน Bucket: {bucket_name}...")
    # 1. List ไฟล์ทั้งหมด (จำกัด 100 ไฟล์ต่อรอบ)
    list_url = f"{supabase_url}/storage/v1/object/list/{bucket_name}"
    try:
        # ลบไฟล์ในโฟลเดอร์หลัก
        res = requests.post(list_url, json={"prefix": ""}, headers=headers)
        files = res.json()
        
        file_names = [f['name'] for f in files]
        if file_names:
            del_url = f"{supabase_url}/storage/v1/object/{bucket_name}"
            requests.delete(del_url, json={"prefixes": file_names}, headers=headers)
            print(f"✅ ลบไฟล์ใน {bucket_name} (root) สำเร็จ: {len(file_names)} ไฟล์")

        # ลบไฟล์ในโฟลเดอร์ videos/
        res_vdo = requests.post(list_url, json={"prefix": "videos"}, headers=headers)
        vdo_files = res_vdo.json()
        vdo_names = [f"videos/{f['name']}" for f in vdo_files]
        if vdo_names:
            del_url = f"{supabase_url}/storage/v1/object/{bucket_name}"
            requests.delete(del_url, json={"prefixes": vdo_names}, headers=headers)
            print(f"✅ ลบไฟล์ใน {bucket_name}/videos สำเร็จ: {len(vdo_names)} ไฟล์")
            
        # ลบไฟล์ในโฟลเดอร์ results/
        res_res = requests.post(list_url, json={"prefix": "results"}, headers=headers)
        res_files = res_res.json()
        res_names = [f"results/{f['name']}" for f in res_files]
        if res_names:
            del_url = f"{supabase_url}/storage/v1/object/{bucket_name}"
            requests.delete(del_url, json={"prefixes": res_names}, headers=headers)
            print(f"✅ ลบไฟล์ใน {bucket_name}/results สำเร็จ: {len(res_names)} ไฟล์")

    except Exception as e:
        print(f"⚠️ ไม่สามารถล้าง storage {bucket_name} ได้: {e}")

if __name__ == "__main__":
    cleanup_jobs()
    cleanup_storage('3d-scans')
    cleanup_storage('scans')
    print("\n✨ ระบบสะอาดเอี่ยมเรียบร้อยครับ!")
