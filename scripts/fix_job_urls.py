import os
import boto3
import requests
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLL_KEY")

S3_ACCESS_KEY = os.getenv("RUNPOD_S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("RUNPOD_S3_SECRET_KEY")
S3_ENDPOINT = "https://s3api-us-il-1.runpod.io"
S3_BUCKET = os.getenv("RUNPOD_BUCKET_NAME")

HEADERS = {
    'apikey': SUPABASE_KEY,
    'Authorization': f'Bearer {SUPABASE_KEY}',
    'Content-Type': 'application/json'
}

def get_s3_client():
    s3_config = Config(
        signature_version='s3v4',
        retries={'max_attempts': 3},
        s3={'addressing_style': 'path'}
    )
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        config=s3_config,
        region_name='us-il-1'
    )

def fix_urls():
    s3 = get_s3_client()

    # 1. Get COMPLETED jobs via REST API
    url = f"{SUPABASE_URL}/rest/v1/jobs?status=eq.COMPLETED&select=id,result_url"
    resp = requests.get(url, headers=HEADERS)
    
    if resp.status_code != 200:
        print(f"❌ Failed to fetch jobs: {resp.text}")
        return

    jobs = resp.json()
    if not jobs:
        print("No COMPLETED jobs found.")
        return

    for job in jobs:
        job_id = job['id']
        old_url = job.get('result_url', '')

        print(f"Checking Job: {job_id}")
        
        if old_url and "X-Amz-Signature" in old_url:
            print(f"  URL is already signed. Skipping.")
            continue

        # Generate new Presigned URL
        key = f"results/{job_id}/model.ply"
        
        try:
            new_url = s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': S3_BUCKET, 'Key': key},
                ExpiresIn=604800 # 7 days
            )
            
            # Update Supabase via PATCH
            update_url = f"{SUPABASE_URL}/rest/v1/jobs?id=eq.{job_id}"
            patch_resp = requests.patch(update_url, headers=HEADERS, json={"result_url": new_url})
            
            if patch_resp.status_code in [200, 204]:
                print(f"  ✅ Updated URL to Presigned version.")
            else:
                print(f"  ❌ Patch failed: {patch_resp.text}")
            
        except Exception as e:
            print(f"  ❌ Failed to generate URL for {job_id}: {e}")

if __name__ == "__main__":
    fix_urls()
