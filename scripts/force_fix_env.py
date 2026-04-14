import os

env_path = '.env'
new_processor_id = '72mhdo2ivp6bf7'

# อ่านไฟล์ .env เดิม
lines = []
if os.path.exists(env_path):
    with open(env_path, 'r') as f:
        lines = f.readlines()

new_lines = []
found_p = False
found_g = False

for line in lines:
    if line.startswith('RUNPOD_ENDPOINT_ID_PROCESSOR='):
        new_lines.append(f'RUNPOD_ENDPOINT_ID_PROCESSOR={new_processor_id}\n')
        found_p = True
    elif line.startswith('RUNPOD_ENDPOINT_ID='):
        new_lines.append(f'RUNPOD_ENDPOINT_ID={new_processor_id}\n')
        found_g = True
    else:
        new_lines.append(line)

if not found_p: new_lines.append(f'RUNPOD_ENDPOINT_ID_PROCESSOR={new_processor_id}\n')
if not found_g: new_lines.append(f'RUNPOD_ENDPOINT_ID={new_processor_id}\n')

with open(env_path, 'w') as f:
    f.writelines(new_lines)

print(f"✅ Successfully updated {env_path} to {new_processor_id}")
