import os
import boto3
import zipfile
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv(override=True)

def inspect_s3_zip():
    JOB_ID = "08732fe8-8c90-4107-918c-ea28cdf004d5"
    BUCKET = os.getenv('RUNPOD_BUCKET_NAME', '53kzs49kuf')
    KEY = f"temp/{JOB_ID}/processed.zip"
    LOCAL_ZIP = "inspect_processed.zip"
    
    print(f"📡 Downloading {KEY} from S3...")
    
    s3 = boto3.client(
        's3',
        endpoint_url="https://s3api-us-il-1.runpod.io",
        aws_access_key_id=os.getenv('RUNPOD_S3_ACCESS_KEY'),
        aws_secret_access_key=os.getenv('RUNPOD_S3_SECRET_KEY'),
        region_name='us-il-1',
        config=Config(signature_version='s3v4', s3={'addressing_style': 'path'})
    )

    try:
        s3.download_file(BUCKET, KEY, LOCAL_ZIP)
        print(f"✅ Downloaded to {LOCAL_ZIP}")
        
        print("\n--- 📦 Zip Content Structure ---")
        with zipfile.ZipFile(LOCAL_ZIP, 'r') as zip_ref:
            for file in zip_ref.namelist():
                print(f"  {file}")
                
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    inspect_s3_zip()
