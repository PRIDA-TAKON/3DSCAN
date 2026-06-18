import os
import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv(override=True)

def test_s3_simple():
    access_key = os.environ.get("RUNPOD_S3_ACCESS_KEY")
    secret_key = os.environ.get("RUNPOD_S3_SECRET_KEY")
    endpoint = os.environ.get("RUNPOD_ENPOIN_URL", "https://s3api-us-il-1.runpod.io")
    
    print(f"📡 Testing S3 connection to: {endpoint}")
    print(f"🔑 Access Key prefix: {access_key[:5] if access_key else 'N/A'}...")

    s3 = boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version='s3v4')
    )

    try:
        response = s3.list_buckets()
        print("✅ Connection Successful!")
        print("📁 Buckets found:")
        for bucket in response['Buckets']:
            print(f"  - {bucket['Name']}")
    except Exception as e:
        print(f"❌ Connection Failed: {e}")

if __name__ == "__main__":
    test_s3_simple()
