from dotenv import load_dotenv
import os
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Check for all jobs that are not completed/failed
response = supabase.table("jobs").select("*").neq("status", "COMPLETED").neq("status", "FAILED").execute()

print(f"Found {len(response.data)} active jobs:")
for job in response.data:
    print(f"ID: {job['id']}, Status: {job['status']}, Created: {job['created_at']}")
