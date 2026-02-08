
import os
import pandas as pd
import json
import numpy as np
import argparse
import struct
import collections

# --- COLMAP Reading Utilities ---

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_images_txt(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    lines = lines[4:]
    images = {}
    for i in range(0, len(lines), 2):
        fields = lines[i].split()
        image_id = int(fields[0])
        qvec = list(map(float, fields[1:5]))
        tvec = list(map(float, fields[5:8]))
        camera_id = int(fields[8])
        name = " ".join(fields[9:])
        images[name] = {'qvec': qvec, 'tvec': tvec, 'camera_id': camera_id}
    return images

def read_images_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            read_next_bytes(fid, num_bytes=24*num_points2D, format_char_sequence="ddq"*num_points2D)
            images[image_name] = {'qvec': qvec, 'tvec': tvec, 'camera_id': camera_id}
    return images

def parse_parameters_dict(row):
    params = row['params']
    model = row['model']
    if model == 'SIMPLE_RADIAL' or model == 'SIMPLE_RADIAL_FISHEYE':
        return {'f': params[0], 'cx': params[1], 'cy': params[2], 'k1': params[3]}
    elif model == 'RADIAL' or model == 'RADIAL_FISHEYE':
        return {'f': params[0], 'cx': params[1], 'cy': params[2], 'k1': params[3], 'k2': params[4]}
    elif model == 'PINHOLE':
        return {'fx': params[0], 'fy': params[1], 'cx': params[2], 'cy': params[3]}
    elif model == 'SIMPLE_PINHOLE':
        return {'f': params[0], 'cx': params[1], 'cy': params[2]}
    elif model in ['OPENCV', 'OPENCV_FISHEYE', 'FULL_OPENCV']: 
        # fx, fy, cx, cy, k1, k2, p1, p2 (for OPENCV)
        # We only really need K for training often, distortion handled elsewhere or ignored here
        return {'fx': params[0], 'fy': params[1], 'cx': params[2], 'cy': params[3]}
    elif model == 'FOV':
        # fx, fy, cx, cy, omega
        return {'fx': params[0], 'fy': params[1], 'cx': params[2], 'cy': params[3]}
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

def read_cameras_txt(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    lines = lines[3:]
    data = {}
    for line in lines:
        if line.startswith("#"): continue
        fields = line.split()
        camera_id = int(fields[0])
        model = fields[1]
        width = int(fields[2])
        height = int(fields[3])
        params = [float(x) for x in fields[4:]]
        data[camera_id] = {'model': model, 'width': width, 'height': height, 'params': params}
    df = pd.DataFrame.from_dict(data, orient='index')
    df['params_dict'] = df.apply(parse_parameters_dict, axis=1)
    df['K'] = df['params_dict'].apply(get_intrinsic_matrix)
    return df

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS])

def read_cameras_binary(path_to_model_file):
    data = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            if model_id in CAMERA_MODEL_IDS:
                model_name = CAMERA_MODEL_IDS[model_id].model_name
                num_params = CAMERA_MODEL_IDS[model_id].num_params
            else:
                model_name = "UNKNOWN"
                # Fallback or error, but let's assume standard models for now or skip
                # Actually we need to read the params to advance the file pointer correctly
                # This is risky if we don't know num_params. 
                # Assuming OPENCV (8 params) or similar if unknown is risky.
                # Only supporting basic models for now as per original script logic
                 # Default to implicit assumption or error
                pass 

            # Re-implementing robustly based on original script which had more models
            # But for brevity, if it hits unknown, it might crash. 
            # Let's rely on the dictionary coverage.
            
            width = camera_properties[2]
            height = camera_properties[3]
            
            # Expanded support based on colmap definition if needed, but sticking to what was in the tool
            # The original tool had more models in CAMERA_MODELS set.
            # I will assume the provided set covers 3D scan use cases (SIMPLE_RADIAL is common).
            
            # Just in case, if model_id is not in our small subset, we might fail to read.
            # I'll expand the list to be safe.
            pass

            # ... To ensure I don't miss, I'll use the full list from previous view_file
    
    # Rereading the full list from my memory of the file content
    # ...
    return pd.DataFrame() # Placeholder, implementing full logic below

# Redefining with full list and logic
def read_cameras_binary_full(path_to_model_file):
    local_CAMERA_MODELS = {
        CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
        CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
        CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
        CameraModel(model_id=3, model_name="RADIAL", num_params=5),
        CameraModel(model_id=4, model_name="OPENCV", num_params=8),
        CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
        CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
        CameraModel(model_id=7, model_name="FOV", num_params=5),
        CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
        CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
        CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
    }
    local_CAMERA_MODEL_IDS = dict([(x.model_id, x) for x in local_CAMERA_MODELS])
    
    data = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            
            if model_id in local_CAMERA_MODEL_IDS:
                model_name = local_CAMERA_MODEL_IDS[model_id].model_name
                num_params = local_CAMERA_MODEL_IDS[model_id].num_params
            else:
                raise ValueError(f"Unknown camera model_id: {model_id}")

            params = read_next_bytes(fid, num_bytes=8*num_params, format_char_sequence="d"*num_params)
            data[camera_id] = {'model': model_name, 'width': width, 'height': height, 'params': params}
            
    df = pd.DataFrame.from_dict(data, orient='index')
    df['params_dict'] = df.apply(parse_parameters_dict, axis=1)
    df['K'] = df['params_dict'].apply(get_intrinsic_matrix)
    return df

def read_points3D_txt(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    lines = lines[3:]
    data = {}
    for line in lines:
        if line.startswith("#"): continue
        fields = line.split()
        point3d_id = int(fields[0])
        x, y, z = map(float, fields[1:4])
        r, g, b = map(int, fields[4:7])
        error = float(fields[7])
        # track = list(zip(map(int, fields[8::2]), map(int, fields[9::2]))) # Optimization: skip track loading
        data[point3d_id] = {'x': x, 'y': y, 'z': z, 'r': r, 'g': g, 'b': b, 'error': error}
    return pd.DataFrame.from_dict(data, orient='index')

def read_points3D_binary(path_to_model_file):
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        data = {}
        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = binary_point_line_properties[1:4]
            rgb = binary_point_line_properties[4:7]
            error = binary_point_line_properties[7]
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            read_next_bytes(fid, num_bytes=8*track_length, format_char_sequence="ii"*track_length) # Skip track
            data[p_id] = {'x': xyz[0], 'y': xyz[1], 'z': xyz[2], 'r': rgb[0], 'g': rgb[1], 'b': rgb[2], 'error': error}
    return pd.DataFrame.from_dict(data, orient='index')

def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])

# --- Main Conversion Function ---

def convert_colmap_to_taichi(base_path, image_path, output_dir, val_split=8):
    print(f"Converting COLMAP data from {base_path} to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    # Read Images
    if os.path.exists(os.path.join(base_path, 'images.bin')):
        images = read_images_binary(os.path.join(base_path, 'images.bin'))
    else:
        images = read_images_txt(os.path.join(base_path, 'images.txt'))
    
    # Read Cameras
    if os.path.exists(os.path.join(base_path, 'cameras.bin')):
        cameras = read_cameras_binary_full(os.path.join(base_path, 'cameras.bin'))
    else:
        cameras = read_cameras_txt(os.path.join(base_path, 'cameras.txt'))

    # Read Points
    if os.path.exists(os.path.join(base_path, 'points3D.bin')):
        points = read_points3D_binary(os.path.join(base_path, 'points3D.bin'))
    elif os.path.exists(os.path.join(base_path, 'points3d.bin')):
        points = read_points3D_binary(os.path.join(base_path, 'points3d.bin'))
    elif os.path.exists(os.path.join(base_path, 'points3D.txt')):
        points = read_points3D_txt(os.path.join(base_path, 'points3D.txt'))
    else:
        print("Warning: points3D.txt/.bin not found!")
        points = pd.DataFrame(columns=['x', 'y', 'z', 'r', 'g', 'b', 'error'])

    # Process Points
    if not points.empty:
        # Save points to parquet
        # Reorder to match expectation if needed, but the reader likely uses column names
        points.to_parquet(os.path.join(output_dir, "point_cloud.parquet"))
        print(f"Saved point_cloud.parquet with {len(points)} points.")
    else:
        print("Error: No points found to save.")

    # Process Images & Cameras
    data = []
    
    # Sort images by name to ensure consistent splitting
    sorted_image_names = sorted(images.keys())
    
    for name in sorted_image_names:
        image = images[name]
        camera = cameras.loc[int(image['camera_id'])]
        
        qvec = np.array(image['qvec'])
        tvec = np.array(image['tvec'])
        
        R = np.zeros((4, 4))
        R[:3, :3] = quaternion_to_rotation_matrix(qvec)
        R[:3, 3] = tvec
        R[3, 3] = 1.0
        
        T_pointcloud_camera = np.linalg.inv(R)
        
        # Check if intrinsics exist
        if 'K' not in camera or camera['K'] is None:
             print(f"Skipping image {name}: No intrinsic matrix K found.")
             continue

        image_full_path = os.path.join(image_path, name)
        # Normalize path separators
        image_full_path = str(image_full_path).replace("\\", "/")

        data.append({
            'image_path': image_full_path,
            'T_pointcloud_camera': T_pointcloud_camera.tolist(),
            'camera_intrinsics': camera['K'].tolist(),
            'camera_height': int(camera['height']),
            'camera_width': int(camera['width']),
            'camera_id': int(camera.name) if hasattr(camera, 'name') else int(image['camera_id']),
        })

    df = pd.DataFrame(data)
    
    # Split Train/Val
    # Simple split: every Nth image is val
    df["is_train"] = df.index % val_split != 0
    
    train_df = df[df["is_train"]].copy()
    val_df = df[~df["is_train"]].copy()
    
    train_df.drop(columns=["is_train"], inplace=True)
    val_df.drop(columns=["is_train"], inplace=True)
    
    train_json_path = os.path.join(output_dir, "train.json")
    val_json_path = os.path.join(output_dir, "val.json")
    
    train_df.to_json(train_json_path, orient="records")
    val_df.to_json(val_json_path, orient="records")
    
    print(f"Saved train.json ({len(train_df)} images) and val.json ({len(val_df)} images).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--colmap_path", required=True)
    parser.add_argument("--images_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    
    convert_colmap_to_taichi(args.colmap_path, args.images_path, args.output_dir)
