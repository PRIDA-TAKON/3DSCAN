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
REPO_NAME = "PRIDA-TAKON/3DSCAN" 
DOCKER_IMAGE_BASE = "ramayana4/worker-3d-scan"

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
    """รอจนกว่า GitHub Action จะบิลด์เสร็จ และคืนค่า SHA"""
    if not GITHUB_TOKEN:
        print("⚠️ GITHUB_TOKEN is empty. Using 'latest' tag.")
        return "latest"
    
    print(f"⏳ Waiting for GitHub Action build on {REPO_NAME}...")
    url = f"https://api.github.com/repos/{REPO_NAME}/actions/runs?per_page=5"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    while True:
        try:
            resp = requests.get(url, headers=headers)
            runs = resp.json().get('workflow_runs', [])
            if not runs:
                print("   - No GitHub Action runs found yet... (retrying in 10s)")
                time.sleep(10)
                continue
                
            run = runs[0]
            status = run['status']
            conclusion = run['conclusion']
            sha = run['head_sha']
            commit_msg = run.get('head_commit', {}).get('message', 'No msg')
            
            print(f"   - Found latest build: '{commit_msg}' | SHA: {sha[:7]} | Status: {status}")
            
            if status == "completed":
                if conclusion == "success":
                    print(f"✅ Build SUCCESS! SHA: {sha}")
                    return sha
                else:
                    print(f"❌ Build FAILED (conclusion: {conclusion})")
                    return None
            
            print(f"   - Still in progress... (waiting 30s)")
            time.sleep(30)
        except Exception as e:
            print(f"⚠️ GitHub API Exception: {e}")
            return "latest"

def update_runpod_endpoint_image(tag):
    """สั่ง RunPod ให้เปลี่ยน Image เป็น Tag ใหม่ (SHA)"""
    full_image_name = f"{DOCKER_IMAGE_BASE}:{tag}"
    print(f"🔄 Updating RunPod Endpoint Image to: {full_image_name}")
    
    url = "https://api.runpod.io/graphql"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }
    
    # GraphQL Mutation ที่ถูกต้องตามคำแนะนำของ RunPod
    mutation = """
    mutation SaveEndpoint($input: EndpointInput!) {
      saveEndpoint(input: $input) {
        id
      }
    }
    """
    
    variables = {
        "input": {
            "id": RUNPOD_ENDPOINT_ID,
            "name": "worker-3d-scan", # ชื่อ Endpoint
            "modelName": full_image_name, # Image URL
            "gpuIds": "3090,4090",
            "idleTimeout": 10,
            "locations": "CA-MTL-1,EU-RO-1,US-GA-1,US-TX-1"
        }
    }
    
    try:
        resp = requests.post(url, json={"query": mutation, "variables": variables}, headers=headers)
        result = resp.json()
        if "errors" in result:
            print(f"❌ GraphQL Errors: {json.dumps(result['errors'], indent=2)}")
            return False
            
        print(f"✅ RunPod Endpoint updated successfully! Tag: {tag[:7]}")
        # ให้เวลามัน Refresh สักพักเพื่อให้แน่ใจว่า Worker ใหม่พร้อมใช้
        time.sleep(20)
        return True
    except Exception as e:
        print(f"⚠️ RunPod Update Exception: {e}")
        return False

def trigger_runpod(job_id, video_url):
    """สั่ง RunPod ให้เริ่มทำงาน (v2 API)"""
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
            output = data.get('output', {})
            print("🔗 Output:", json.dumps(output, indent=2))
            if 'stdout' in data:
                print("📝 Worker Stdout:\n", data['stdout'])
            break
        elif status == "FAILED":
            print("❌ JOB FAILED!")
            print("🔻 Error Log:", data.get('error'))
            if 'stdout' in data:
                print("📝 Worker Stdout (before failure):\n", data['stdout'])
            break
            
        time.sleep(10)

if __name__ == "__main__":
    print("🚀 Starting Automated Test Cycle (SHA-Tagging Version)...")
    
    # 1. ดึงข้อมูล Job เดิม
    job_info = get_latest_job()
    if not job_info:
        print("❌ No previous video found in Supabase.")
    else:
        job_id = job_info['id']
        video_url = job_info['video_url']
        print(f"✅ Found latest job: {job_id}")
        
        # 2. รอ Build และดึงเลข SHA
        sha = wait_for_github_action()
        if sha:
            # 3. อัปเดต RunPod Endpoint ให้ใช้ภาพตามเลข SHA
            if update_runpod_endpoint_image(sha):
                # 4. รันใหม่บน RunPod
                runpod_job_id = trigger_runpod(job_id, video_url)
                if runpod_job_id:
                    print(f"🆔 RunPod Internal ID: {runpod_job_id}")
                    # 5. ติดตามสถานะและ Log
                    monitor_job(runpod_job_id)
                else:
                    print("❌ Failed to trigger RunPod job.")
        else:
            print("⏭️ Skipping test due to build failure.")
