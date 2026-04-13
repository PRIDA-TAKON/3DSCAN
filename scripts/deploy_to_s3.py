import os
import boto3
import zipfile
from pathlib import Path
from botocore.config import Config

def zip_logic():
    print("📦 Packaging worker logic...")
    zip_path = "worker_logic.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # ใส่ไฟล์หลักที่รูท
        files_to_include = ["takon_3d_worker.py", "loader.py"]
        for f in files_to_include:
            if os.path.exists(f):
                zipf.write(f)
        
        # ใส่โฟลเดอร์ scripts
        for root, dirs, files in os.walk("scripts"):
            for file in files:
                if not file.endswith('.pyc'):
                    zipf.write(os.path.join(root, file))
    
    print(f"✅ Created {zip_path} ({os.path.getsize(zip_path) / 1024:.2f} KB)")
    return zip_path

def upload_to_s3(file_path):
    print("🚀 Uploading to RunPod S3 Store...")
    # ดึงค่าจาก Env (GitHub Secrets)
    access_key = os.environ.get("S3_ACCESS_KEY")
    secret_key = os.environ.get("S3_SECRET_KEY")
    endpoint = os.environ.get("S3_ENDPOINT", "https://s3api-us-il-1.runpod.io")
    bucket = os.environ.get("S3_BUCKET") or os.environ.get("RUNPOD_BUCKET_NAME")

    if not access_key or not secret_key:
        print("❌ Error: S3 Credentials missing in Environment!")
        return False

    s3 = boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name='us-il-1',
        config=Config(
            signature_version='s3v4',
            s3={'addressing_style': 'path'}
        )
    )

    remote_path = "deploy/worker_logic.zip"
    s3.upload_file(file_path, bucket, remote_path)
    print(f"🎉 Successfully deployed logic to: s3://{bucket}/{remote_path}")
    return True

if __name__ == "__main__":
    logic_zip = zip_logic()
    upload_to_s3(logic_zip)
