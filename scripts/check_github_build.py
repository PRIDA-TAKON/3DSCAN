import requests
import json

repo = "PRIDA-TAKON/3DSCAN"
url = f"https://api.github.com/repos/{repo}/actions/runs?per_page=1"
headers = {"Accept": "application/vnd.github.v3+json"}

try:
    resp = requests.get(url, headers=headers)
    data = resp.json()
    if 'workflow_runs' in data and len(data['workflow_runs']) > 0:
        run = data['workflow_runs'][0]
        print(f"--- 🛠️ GitHub Build Status ---")
        print(f"ID:         {run['id']}")
        print(f"Status:     {run['status']}")
        print(f"Conclusion: {run['conclusion']}")
        print(f"Created At: {run['created_at']}")
    else:
        print("No workflow runs found.")
except Exception as e:
    print(f"Error checking GitHub status: {e}")
