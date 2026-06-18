import os
import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv(override=True)

JOB_ID = "c345c2a0-d781-4752-8250-60494b3d6c30"
BUCKET = os.getenv('RUNPOD_BUCKET_NAME')
ACCESS_KEY = os.getenv('RUNPOD_S3_ACCESS_KEY') or os.getenv('S3_ACCESS_KEY')
SECRET_KEY = os.getenv('RUNPOD_S3_SECRET_KEY') or os.getenv('S3_SECRET_KEY')
ENDPOINT = os.getenv('RUNPOD_S3_ENDPOINT', 'https://s3api-us-il-1.runpod.io')

def download_final_result():
    s3_config = Config(
        signature_version='s3v4',
        s3={'addressing_style': 'path'}
    )
    s3 = boto3.client(
        's3',
        endpoint_url=ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        config=s3_config,
        region_name='us-il-1'
    )

    remote_path = f"results/{JOB_ID}/model.ply"
    local_path = f"result_{JOB_ID}.ply"

    print(f"📥 Downloading {remote_path} from S3...")
    try:
        s3.download_file(BUCKET, remote_path, local_path)
        size = os.path.getsize(local_path) / (1024 * 1024)
        print(f"✅ Success! Saved to: {local_path} ({size:.2f} MB)")
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    download_final_result()
