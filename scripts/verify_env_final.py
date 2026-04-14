import os
from dotenv import load_dotenv

load_dotenv()
print(f"🔹 RUNPOD_ENDPOINT_ID_PROCESSOR: {os.getenv('RUNPOD_ENDPOINT_ID_PROCESSOR')}")
print(f"🔹 RUNPOD_API_KEY:               {os.getenv('RUNPOD_API_KEY')[:5]}...") # แสดงแค่ 5 ตัวแรกเพื่อความปลอดภัย
