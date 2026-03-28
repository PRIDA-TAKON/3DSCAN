import requests
import os

def trigger_runpod_job(job_id, video_url):
    """
    Trigger a RunPod Serverless job using their API.
    """
    api_key = os.environ.get("RUNPOD_API_KEY")
    endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID") # e.g., 'abc123xyz'
    
    if not api_key or not endpoint_id:
        print("❌ Missing RUNPOD_API_KEY or RUNPOD_ENDPOINT_ID")
        return None

    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": {
            "id": job_id,
            "video_url": video_url
        }
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        print(f"🚀 RunPod job triggered! ID: {data.get('id')}")
        return data.get('id')
    except Exception as e:
        print(f"❌ Failed to trigger RunPod: {e}")
        return None

if __name__ == "__main__":
    # Example usage (can be called from Supabase Edge Function or Backend)
    import sys
    if len(sys.argv) < 3:
        print("Usage: python trigger_runpod.py <job_id> <video_url>")
    else:
        trigger_runpod_job(sys.argv[1], sys.argv[2])
