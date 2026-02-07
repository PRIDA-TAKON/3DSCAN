
import os
import argparse
import subprocess
import shutil
import json
import math
import numpy as np
from pathlib import Path

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def read_cameras_text(path):
    cameras = {}
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            els = line.split()
            camera_id = int(els[0])
            model = els[1]
            width = int(els[2])
            height = int(els[3])
            params = np.array([float(x) for x in els[4:]])
            cameras[camera_id] = {"model": model, "width": width, "height": height, "params": params}
    return cameras

def read_images_text(path):
    images = {}
    with open(path, "r") as f:
        while True:
            line = f.readline()
            if not line: break
            if line.startswith("#") or not line.strip(): continue
            
            # Line 1: Image ID, Qvec, Tvec, Camera ID, Name
            els = line.split()
            image_id = int(els[0])
            qvec = np.array([float(x) for x in els[1:5]])
            tvec = np.array([float(x) for x in els[5:8]])
            camera_id = int(els[8])
            image_name = els[9]
            
            # Line 2: Points 2D (discard)
            f.readline()
            
            images[image_id] = {
                "qvec": qvec, "tvec": tvec, "camera_id": camera_id, "name": image_name
            }
    return images

def convert_colmap_to_transforms(colmap_dir, images_dir, output_path):
    print(f"🔄 Converting COLMAP text output to {output_path}...")
    
    colmap_dir = Path(colmap_dir)
    cameras_file = colmap_dir / "cameras.txt"
    images_file = colmap_dir / "images.txt"
    
    if not cameras_file.exists() or not images_file.exists():
        print("❌ COLMAP output files not found (cameras.txt/images.txt).")
        return False
        
    cameras = read_cameras_text(cameras_file)
    images = read_images_text(images_file)
    
    sorted_image_ids = sorted(images.keys(), key=lambda k: images[k]["name"])
    
    frames = []
    if not cameras: 
        print("❌ No cameras found.")
        return False
        
    cam_id = list(cameras.keys())[0]
    cam = cameras[cam_id]
    
    w, h = cam["width"], cam["height"]
    fl_x = cam["params"][0]
    fl_y = cam["params"][1]
    k1 = cam["params"][2] if len(cam["params"]) > 2 else 0
    k2 = cam["params"][3] if len(cam["params"]) > 3 else 0
    p1 = cam["params"][4] if len(cam["params"]) > 4 else 0
    p2 = cam["params"][5] if len(cam["params"]) > 5 else 0
    cx = cam["params"][2] if len(cam["params"]) == 3 else w / 2.0
    cy = cam["params"][3] if len(cam["params"]) == 3 else h / 2.0
    
    angle_x = 2 * math.atan(w / (2 * fl_x))
    angle_y = 2 * math.atan(h / (2 * fl_y))
    
    json_data = {
        "camera_angle_x": angle_x,
        "camera_angle_y": angle_y,
        "fl_x": fl_x, "fl_y": fl_y,
        "k1": k1, "k2": k2, "p1": p1, "p2": p2,
        "cx": cx, "cy": cy,
        "w": w, "h": h,
        "aabb_scale": 16,
        "frames": []
    }
    
    flip_mat = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    
    for img_id in sorted_image_ids:
        img = images[img_id]
        
        R = qvec2rotmat(img["qvec"])
        t = img["tvec"]
        
        c2w = np.eye(4)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ t
        
        c2w = c2w @ flip_mat
        
        frame = {
            "file_path": f"images/{img['name']}",
            "transform_matrix": c2w.tolist()
        }
        frames.append(frame)
        
    json_data["frames"] = frames
    
    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=4)
        
    print(f"✅ Saved {len(frames)} frames to {output_path}")
    return True

def run_step(cmd, shell=True):
    print(f"🚀 Running: {cmd}")
    try:
        subprocess.run(cmd, shell=shell, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {cmd}")
        raise e

def run_colmap(images_dir, output_dir):
    images_dir = Path(images_dir)
    project_dir = Path(output_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    
    colmap_dir = project_dir / "colmap"
    colmap_dir.mkdir(parents=True, exist_ok=True)
    
    db_path = colmap_dir / "database.db"
    if db_path.exists():
        db_path.unlink() # Clean start? Or let colmap handle it

    # Check for xvfb
    colmap_binary = "colmap"
    try:
        subprocess.run(["which", "xvfb-run"], check=True, stdout=subprocess.DEVNULL)
        colmap_binary = "xvfb-run -a colmap"
        print("   Using xvfb-run for headless COLMAP")
    except:
        pass

    print("--- Feature Extraction ---")
    run_step(f"{colmap_binary} feature_extractor --database_path {db_path} --image_path {images_dir} --ImageReader.camera_model OPENCV")
    
    print("--- Matching ---")
    run_step(f"{colmap_binary} sequential_matcher --database_path {db_path}")
    
    print("--- Reconstruction (Mapper) ---")
    sparse_dir = colmap_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    run_step(f"{colmap_binary} mapper --database_path {db_path} --image_path {images_dir} --output_path {sparse_dir}")
    
    print("--- Converting to Text ---")
    text_dir = colmap_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    
    if not (sparse_dir / "0").exists():
        print("❌ COLMAP reconstruction failed (no model found).")
        return False

    run_step(f"{colmap_binary} model_converter --input_path {sparse_dir}/0 --output_path {text_dir} --output_type TXT")
    
    transforms_path = project_dir / "transforms.json"
    return convert_colmap_to_transforms(text_dir, images_dir, transforms_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2: COLMAP SfM Pipeline")
    parser.add_argument("--images_dir", required=True, help="Directory containing extracted images")
    parser.add_argument("--output_dir", required=True, help="Project output directory (where transforms.json will be saved)")
    
    args = parser.parse_args()
    
    run_colmap(args.images_dir, args.output_dir)
