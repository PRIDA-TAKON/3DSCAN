import os
from supabase import create_client
from dotenv import load_dotenv

# โหลดค่าจาก .env
load_dotenv()

supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY') or os.getenv('SUPABASE_KEY') or os.getenv('SUPABASE_SERVICE_ROLL_SECERT_KEY')

if not supabase_url or not supabase_key:
    print('❌ ไม่พบ SUPABASE_URL หรือ SUPABASE_KEY ใน .env')
    exit(1)

supabase = create_client(supabase_url, supabase_key)

def list_files():
    try:
        bucket_name = '3d-scans'
        print(f"📁 Checking bucket: {bucket_name}...")
        
        # 1. ลอง List ไฟล์ใน Root
        files = supabase.storage.from_(bucket_name).list('')
        print(f"\n📄 Files in Root:")
        for f in files:
            print(f"  - {f['name']} (Type: {f.get('metadata', {}).get('mimetype', 'folder')})")
            
        # 2. ลอง List ไฟล์ใน videos/
        print(f"\n📂 Files in 'videos/' folder:")
        video_files = supabase.storage.from_(bucket_name).list('videos')
        if not video_files:
            print("  (Empty)")
        for f in video_files:
            print(f"  - {f['name']} (Size: {f.get('metadata', {}).get('size', '?')} bytes)")
            
    except Exception as e:
        print(f'⚠️ Error: {e}')

if __name__ == "__main__":
    list_files()
