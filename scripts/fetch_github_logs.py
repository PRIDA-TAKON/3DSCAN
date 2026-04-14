import requests
import os

repo = "PRIDA-TAKON/3DSCAN"
base_url = f"https://api.github.com/repos/{repo}/actions"
headers = {"Accept": "application/vnd.github.v3+json"}

def fetch_latest_logs():
    print(f"📡 Fetching latest workflow runs for {repo}...")
    runs_resp = requests.get(f"{base_url}/runs?per_page=1", headers=headers)
    runs = runs_resp.json().get('workflow_runs', [])
    
    if not runs:
        print("No runs found.")
        return

    latest_run = runs[0]
    run_id = latest_run['id']
    print(f"🆔 Run ID: {run_id} ({latest_run['status']})")
    print(f"🔗 URL: {latest_run['html_url']}\n")

    # Fetch jobs for this run
    jobs_resp = requests.get(f"{base_url}/runs/{run_id}/jobs", headers=headers)
    jobs = jobs_resp.json().get('jobs', [])

    for job in jobs:
        print(f"--- 🛠️ Job: {job['name']} ({job['status']} / {job['conclusion']}) ---")
        for step in job['steps']:
            status_icon = "✅" if step['conclusion'] == "success" else "❌" if step['conclusion'] == "failure" else "⏳"
            print(f"  {status_icon} {step['name']} ({step['status']})")
            
            if step['conclusion'] == "failure":
                print(f"    🚨 DETECTED FAILURE in step: {step['name']}")

if __name__ == "__main__":
    fetch_latest_logs()
