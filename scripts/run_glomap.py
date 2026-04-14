
import os
import argparse
import subprocess
import shutil
import json
import numpy as np
import pandas as pd
from pathlib import Path
import collections
import struct

def run_command(cmd):
    """Run a shell command with updated PATH and headless Display."""
    print(f"🚀 Running: {cmd}")
    # Ensure standard Kaggle/Conda paths are included
    custom_env = os.environ.copy()
    paths = ["/opt/conda/bin", "/usr/local/bin", "/usr/bin", "/bin"]
    custom_env["PATH"] = ":".join(paths + [custom_env.get("PATH", "")])
    custom_env["QT_QPA_PLATFORM"] = "offscreen"
    
    try:
        subprocess.run(cmd, shell=True, check=True, env=custom_env)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {cmd}")
        return False

# --- COLMAP / GLOMAP Helpers (derived from prepare_colmap.py) ---

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def parse_parameters_dict(row):
    params = row['params']
    model = row['model']
    if model == 'SIMPLE_PINHOLE':
        return {'f': params[0], 'cx': params[1], 'cy': params[2]}
    elif model == 'PINHOLE':
        return {'fx': params[0], 'fy': params[1], 'cx': params[2], 'cy': params[3]}
    elif model == 'SIMPLE_RADIAL':
        return {'f': params[0], 'cx': params[1], 'cy': params[2], 'k1': params[3]}
    elif model == 'RADIAL':
        return {'f': params[0], 'cx': params[1], 'cy': params[2], 'k1': params[3], 'k2': params[4]}
    elif model == 'OPENCV':
        return {'fx': params[0], 'fy': params[1], 'cx': params[2], 'cy': params[3], 'k1': params[4], 'k2': params[5], 'p1': params[6], 'p2': params[7]}
    else:
        return {'params': params}

def get_intrinsic_matrix(params):
    if 'f' in params:
        f = params['f']
        cx = params['cx']
        cy = params['cy']
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    elif 'fx' in params:
        fx = params['fx']
        fy = params['fy']
        cx = params['cx']
        cy = params['cy']
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return None

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
}
CAMERA_MODEL_IDS = {m.model_id: m for m in CAMERA_MODELS}

def read_cameras_text(file):
    data = {}
    with open(file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith("#"): continue
        fields = line.split()
        if not fields: continue
        camera_id = int(fields[0])
        model = fields[1]
        width, height = int(fields[2]), int(fields[3])
        params = [float(x) for x in fields[4:]]
        data[camera_id] = {'model': model, 'width': width, 'height': height, 'params': params}
    df = pd.DataFrame.from_dict(data, orient='index')
    df['params_dict'] = df.apply(parse_parameters_dict, axis=1)
    df['K'] = df['params_dict'].apply(get_intrinsic_matrix)
    return df

def read_images_text(file):
    images = {}
    with open(file, 'r') as f:
        lines = f.readlines()
    for i in range(0, len(lines)):
        line = lines[i]
        if line.startswith("#"): continue
        fields = line.split()
        if not fields: continue
        if (i - 4) % 2 == 0: # Image info line
            image_id = int(fields[0])
            qvec = list(map(float, fields[1:5]))
            tvec = list(map(float, fields[5:8]))
            camera_id = int(fields[8])
            name = fields[9]
            images[name] = {'qvec': qvec, 'tvec': tvec, 'camera_id': camera_id}
    return images

def read_points3D_text(file):
    data = {}
    with open(file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith("#"): continue
        fields = line.split()
        if not fields: continue
        p_id = int(fields[0])
        x, y, z = map(float, fields[1:4])
        r, g, b = map(int, fields[4:7])
        data[p_id] = {'x': x, 'y': y, 'z': z, 'r': r, 'g': g, 'b': b}
    return pd.DataFrame.from_dict(data, orient='index')

def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])

# --- Main Glomap Logic ---

def run_glomap_sfm(images_dir, output_dir):
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    colmap_dir = output_dir / "colmap"
    colmap_dir.mkdir(parents=True, exist_ok=True)
    db_path = colmap_dir / "database.db"
    
    # 1. Feature Extraction
    # ปรับปรุง: ใช้ SIMPLE_RADIAL เพราะมือถือส่วนใหญ่มี Distortion และเพิ่มคุณภาพการสกัดจุด
    feat_cmd = f"colmap feature_extractor --database_path {db_path} --image_path {images_dir} --ImageReader.camera_model SIMPLE_RADIAL --SiftExtraction.max_num_features 8192"
    headless_feat_cmd = f"xvfb-run -a {feat_cmd}"
    
    print("🔥 Starting Feature Extraction (with Virtual Display)...")
    if not run_command(headless_feat_cmd):
        print("⚠️ Virtual Display failed. Retrying with CPU only...")
        if not run_command(f"{feat_cmd} --SiftExtraction.use_gpu 0"):
            return False
        
    # 2. Matching
    # ปรับปรุง: เพิ่มความพยายามในการ Match (overlap 20 ภาพย้อนหลัง)
    match_cmd = f"colmap sequential_matcher --database_path {db_path} --SequentialMatching.overlap 20"
    print("🔥 Starting Feature Matching...")
    if not run_command(f"xvfb-run -a {match_cmd}"):
        print("⚠️ Matching failed with GPU. Retrying with CPU...")
        if not run_command(f"{match_cmd} --SiftMatching.use_gpu 0"):
            return False
        
    # 3. Glomap Mapper
    sparse_dir = output_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    
    success = False
    # ลองใช้ Glomap ก่อน
    if shutil.which("glomap"):
        glomap_cmd = f"glomap mapper --database_path {db_path} --output_path {sparse_dir}"
        print("🔥 Starting Glomap Mapper...")
        success = run_command(glomap_cmd)

    if not success:
        print("⚠️ Glomap failed or not found. Falling back to COLMAP mapper...")
        # ปรับปรุง: เพิ่มความพยายามในการ Mapper (min_num_models 1)
        mapper_cmd = f"colmap mapper --database_path {db_path} --image_path {images_dir} --output_path {sparse_dir}"
        if not run_command(f"xvfb-run -a {mapper_cmd}"):
             print("⚠️ Mapper failed with GPU. Retrying with CPU only...")
             if not run_command(f"{mapper_cmd} --Mapper.ba_use_gpu 0"):
                 return False
            
    # ตรวจสอบหาโมเดลที่ได้ (ปกติ colmap จะสร้าง folder '0')
    model_dir = sparse_dir / "0"
    if not model_dir.exists():
        # ถ้าไม่มี '0' ให้ลองหา bin ใน root sparse_dir
        if (sparse_dir / "cameras.bin").exists():
            # ย้ายไฟล์เข้าโฟลเดอร์ '0' เพื่อให้ Nerfstudio หาเจอ
            model_dir.mkdir(parents=True, exist_ok=True)
            for f in ["cameras.bin", "images.bin", "points3D.bin", "project.ini"]:
                if (sparse_dir / f).exists():
                    shutil.move(str(sparse_dir / f), str(model_dir / f))
        else:
            print("❌ Reconstruction failed (no model output found).")
            return False

    # 4. Convert to Taichi/Transforms Format
    print("🔄 Generating transform metadata...")
    text_dir = output_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    run_command(f"colmap model_converter --input_path {model_dir} --output_path {text_dir} --output_type TXT")
    
    try:
        cameras = read_cameras_text(text_dir / "cameras.txt")
        images = read_images_text(text_dir / "images.txt")
        points = read_points3D_text(text_dir / "points3D.txt")
        
        data = []
        for name, image in images.items():
            cam = cameras.loc[int(image['camera_id'])]
            qvec = np.array(image['qvec'])
            tvec = np.array(image['tvec'])
            R = np.eye(4)
            R[:3, :3] = quaternion_to_rotation_matrix(qvec)
            R[:3, 3] = tvec
            T_pointcloud_camera = np.linalg.inv(R)
            
            data.append({
                'image_path': name, # relative to images dir
                'T_pointcloud_camera': T_pointcloud_camera.tolist(),
                'camera_intrinsics': cam['K'].tolist(),
                'camera_height': int(cam['height']),
                'camera_width': int(cam['width']),
                'camera_id': int(cam.name),
            })
            
        df = pd.DataFrame(data)
        df.to_json(output_dir / "train.json", orient="records")
        points.to_parquet(output_dir / "point_cloud.parquet")
        print(f"✅ Preprocessing complete. {len(df)} frames reconstructed.")
    except Exception as e:
        print(f"⚠️ Metadata generation skipped or failed: {e}")
        # แม้จะพลาดขั้นตอนนี้ แต่ถ้ามีไฟล์ .bin ใน sparse/0/ ก็ยังถือว่า SfM สำเร็จ (สำหรับ Nerfstudio)
    
    return True

if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser(description="Run Glomap SfM and Prepare for 3DGS")
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    
    success = run_glomap_sfm(args.images_dir, args.output_dir)
    if not success:
        sys.exit(1)
    sys.exit(0)
