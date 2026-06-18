import os
import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv(override=True)

def list_s3_contents():
    access_key = os.environ.get("RUNPOD_S3_ACCESS_KEY")
    secret_key = os.environ.get("RUNPOD_S3_SECRET_KEY")
    endpoint = "https://s3api-us-il-1.runpod.io"
    bucket_name = "53kzs49kuf"

    print(f"📡 Connecting to S3 Bucket: {bucket_name}")
    
    s3 = boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name='us-il-1',
        config=Config(signature_version='s3v4')
    )

    try:
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name)

        print("\n--- 📁 Files found in S3 ---")
        count = 0
        engines = []
        logic_backups = []
        processed_data = []
        
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    size_mb = obj['Size'] / (1024 * 1024)
                    last_mod = obj['LastModified']
                    
                    print(f"📄 {key} ({size_mb:.2f} MB) - {last_mod}")
                    
                    if "engine" in key.lower(): engines.append(key)
                    elif "logic" in key.lower(): logic_backups.append(key)
                    elif "processed.zip" in key.lower(): processed_data.append(key)
                    count += 1
        
        print(f"\n📊 Summary:")
        print(f"Total Files: {count}")
        print(f"Engines (.tar.gz): {len(engines)}")
        print(f"Logic Backups: {len(logic_backups)}")
        print(f"Processed Jobs: {len(processed_data)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    list_s3_contents()
