import os
import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

def check_s3_deployment():
    access_key = os.environ.get("RUNPOD_S3_ACCESS_KEY")
    secret_key = os.environ.get("RUNPOD_S3_SECRET_KEY")
    endpoint = "https://s3api-us-il-1.runpod.io"
    bucket = "3d-scans"

    print("📡 Connecting to RunPod S3 Store...")
    
    if not access_key or not secret_key:
        print("❌ Error: S3 Credentials missing in .env!")
        return

    s3 = boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version='s3v4')
    )

    try:
        target_key = "deploy/worker_logic.zip"
        response = s3.head_object(Bucket=bucket, Key=target_key)
        
        print(f"✅ Found deployed logic file: {target_key}")
        print(f"📦 Size:        {response['ContentLength'] / 1024:.2f} KB")
        print(f"🕒 Last Modified: {response['LastModified']}")
        print(f"🔗 Store URL:   {endpoint}/{bucket}/{target_key}")
    except Exception as e:
        print(f"⚠️ Logic file not found or error: {e}")

if __name__ == "__main__":
    check_s3_deployment()
