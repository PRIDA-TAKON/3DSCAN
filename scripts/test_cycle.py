import os
import time
import requests
import json
from dotenv import load_dotenv

# โหลด Environment Variables
load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_ROLE_KEY') or os.getenv('SUPABASE_ANON_PUBLIC_KEY')
RUNPOD_API_KEY = os.getenv('RUNPOD_API_KEY')
RUNPOD_ENDPOINT_ID = os.getenv('RUNPOD_ENDPOINT_ID')
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
REPO_NAME = "ramayana4/02_3DSCAN" # เปลี่ยนตามชื่อ Repo ของคุณ

headers_supabase = {
    'apikey': SUPABASE_KEY,
    'Authorization': f'Bearer {SUPABASE_KEY}',
    'Content-Type': 'application/json'
}

def get_latest_job():
    """ดึงข้อมูล Job ล่าสุดจาก Supabase"""
    print("🔍 Searching for the latest video job...")
    url = f"{SUPABASE_URL}/rest/v1/jobs?select=id,video_url&order=created_at.desc&limit=1"
    response = requests.get(url, headers=headers_supabase)
    data = response.json()
    if data:
        return data[0]
    return None

def wait_for_github_action():
    """รอจนกว่า GitHub Action จะบิลด์เสร็จ (ถ้ามี Token)"""
    if not GITHUB_TOKEN:
        print("⚠️ No GITHUB_TOKEN found, skipping build check (using current latest image)")
        return True
    
    print("⏳ Waiting for GitHub Action build to finish...")
    url = f"https://api.github.com/repos/{REPO_NAME}/actions/runs?per_page=1"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    
    while True:
        try:
            resp = requests.get(url, headers=headers)
            runs = resp.json().get('workflow_runs', [])
            if not runs:
                print("⚠️ No GitHub Action runs found yet. Skipping build check...")
                return True
                
            run = runs[0]
            status = run['status']
            conclusion = run['conclusion']
            
            if status == "completed":
                if conclusion == "success":
                    print("✅ Build SUCCESS! Proceeding to test...")
                    return True
                else:
                    print(f"❌ Build FAILED (conclusion: {conclusion})")
                    return False
            
            print(f"   - Current build status: {status}... (waiting 15s)")
            time.sleep(15)
        except Exception as e:
            print(f"⚠️ GitHub API Error: {e}")
            return True # ข้ามไปถ้ามีปัญหา

def trigger_runpod(job_id, video_url):
    """สั่ง RunPod ให้เริ่มทำงานด้วยวิดีโอเดิม"""
    print(f"🚀 Triggering RunPod for: {video_url} (Job ID: {job_id})")
    url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/run"
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": {
            "id": job_id,
            "video_url": video_url,
            "force_rebuild": True
        }
    }
    
    resp = requests.post(url, json=payload, headers=headers)
    if resp.status_code != 200:
        print(f"❌ RunPod API Error: {resp.status_code} - {resp.text}")
        return None
    return resp.json().get('id')

def monitor_job(runpod_job_id):
    """ติดตาม Log และสถานะจนกว่าจะจบ"""
    print(f"📺 Monitoring Job: {runpod_job_id}")
    url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status/{runpod_job_id}"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    
    last_status = ""
    while True:
        resp = requests.get(url, headers=headers)
        data = resp.json()
        status = data.get('status')
        
        if status != last_status:
            print(f"📊 Status changed: {status}")
            last_status = status
            
        if status == "COMPLETED":
            print("🎉 SUCCESS! Job finished perfectly.")
            print("🔗 Output:", json.dumps(data.get('output'), indent=2))
            break
        elif status == "FAILED":
            print("❌ JOB FAILED!")
            print("🔻 Error Log:", data.get('error'))
            break
            
        time.sleep(10)

if __name__ == "__main__":
    print("🚀 Starting Automated Test Cycle...")
    
    # 1. ดึงข้อมูล Job เดิม
    job_info = get_latest_job()
    if not job_info:
        print("❌ No previous video found in Supabase. Please upload at least one video via the web interface first.")
    else:
        job_id = job_info['id']
        video_url = job_info['video_url']
        print(f"✅ Found latest job: {job_id}")
        
        # 2. รอ Build จาก GitHub Actions
        if wait_for_github_action():
            print("⏳ Giving Docker Hub a few seconds to stabilize...")
            time.sleep(5)
            
            # 3. รันใหม่บน RunPod
            runpod_job_id = trigger_runpod(job_id, video_url)
            if runpod_job_id:
                print(f"🆔 RunPod Internal ID: {runpod_job_id}")
                # 4. ติดตามสถานะและ Log
                monitor_job(runpod_job_id)
            else:
                print("❌ Failed to trigger RunPod job.")
        else:
            print("⏭️ Skipping RunPod trigger due to build failure.")
