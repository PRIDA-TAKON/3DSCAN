import os
import requests
import uuid
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_PUBLIC_KEY")
BUCKET_NAME = "3d-scans"

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Missing Supabase credentials in .env")
    exit(1)

def test_upload_to_supabase():
    print(f"🚀 Testing Upload to Supabase Storage (Bucket: {BUCKET_NAME})")
    
    # 1. Create a dummy video file content
    dummy_content = b"This is a dummy video file for testing upload."
    file_name = f"test_upload_{uuid.uuid4().hex[:8]}.mp4"
    file_path = f"videos/{file_name}"

    # 2. Upload to Supabase Storage via REST API
    # URL format: {SUPABASE_URL}/storage/v1/object/{bucket}/{path}
    upload_url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET_NAME}/{file_path}"
    
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "video/mp4"
    }

    print(f"📤 Uploading: {file_name}...")
    try:
        resp = requests.post(upload_url, headers=headers, data=dummy_content)
        
        if resp.status_code == 200:
            print(f"✅ Upload SUCCESS! File stored at: {file_path}")
            
            # 3. Verify Public URL
            public_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{file_path}"
            print(f"🔗 Public URL: {public_url}")
            
            # Check if file is accessible
            check_resp = requests.head(public_url)
            if check_resp.status_code == 200:
                print("✅ Public Access Verified.")
            else:
                print(f"⚠️ Public Access Check failed (Status: {check_resp.status_code}). Check bucket policy.")
            
            return True
        else:
            print(f"❌ Upload FAILED (Status: {resp.status_code})")
            print(f"📝 Response: {resp.text}")
            return False

    except Exception as e:
        print(f"❌ Error during upload: {e}")
        return False

def test_api_trigger():
    print("\n🚀 Testing API Worker Trigger (Localhost Check)")
    # This part tests if your local API route can trigger RunPod
    # Note: Requires the frontend to be running locally at port 3000
    local_api_url = "http://localhost:3000/api/run-worker"
    
    payload = {
        "jobId": str(uuid.uuid4()),
        "videoUrl": "https://example.com/test.mp4",
        "mode": "PROCESS"
    }
    
    try:
        resp = requests.post(local_api_url, json=payload, timeout=5)
        if resp.status_code == 200:
            print("✅ API Trigger SUCCESS (Local)")
        else:
            print(f"⚠️ API Trigger failed (Status: {resp.status_code}). Is the frontend running?")
    except:
        print("ℹ️ Local API not reachable (Frontend probably not running). This is normal if only testing storage.")

if __name__ == "__main__":
    success = test_upload_to_supabase()
    if success:
        test_api_trigger()
    else:
        print("\n❌ Storage test failed, skipping API test.")
