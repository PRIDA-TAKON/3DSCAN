import argparse
import json
import math
import struct
import os
import sys
from pathlib import Path
from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.optim as optim
import cv2
import taichi as ti

# Ensure taichi-splatting is in path if needed, or installed
import taichi_splatting
from taichi_splatting.data_types import Gaussians3D, RasterConfig
from taichi_splatting.renderer import render_gaussians
from taichi_splatting.perspective import CameraParams
from taichi_splatting.misc.encode_depth import encode_depth
from taichi_splatting.misc.radius import compute_radius
from taichi_splatting.misc.parameter_class import ParameterClass
from functools import partial

# Try importing plyfile, handle if missing
try:
    from plyfile import PlyData, PlyElement
except ImportError:
    PlyData, PlyElement = None, None

def read_points3d_binary(path_to_model_file):
    """
    Reads COLMAP points3D.bin file.
    Returns a dictionary of point3D_id -> (xyz, rgb).
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = struct.unpack("Q", fid.read(8))[0]
        for _ in range(num_points):
            binary_point_line_properties = fid.read(43)
            point3D_id = struct.unpack("Q", binary_point_line_properties[0:8])[0]
            xyz = struct.unpack("ddd", binary_point_line_properties[8:32])
            rgb = struct.unpack("BBB", binary_point_line_properties[32:35])
            error = struct.unpack("d", binary_point_line_properties[35:43])[0]
            track_length = struct.unpack("Q", fid.read(8))[0]
            track_elems = fid.read(track_length * 8) # Skip track info
            points3D[point3D_id] = (np.array(xyz, dtype=np.float32), np.array(rgb, dtype=np.uint8))
    return points3D

def read_points3d_txt(path_to_model_file):
    """
    Reads COLMAP points3D.txt file.
    """
    points3D = {}
    with open(path_to_model_file, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])), dtype=np.float32)
                rgb = np.array(tuple(map(int, elems[4:7])), dtype=np.uint8)
                points3D[point3D_id] = (xyz, rgb)
                # Skip error (elem 7) and track list (elems 8+)
    return points3D

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[1] * qvec[3] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[1] * qvec[3] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def inverse_sigmoid(x):
    return torch.log(x / (1 - x))

def get_projection_matrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

class SceneDataset:
    def __init__(self, project_dir, device='cuda'):
        self.project_dir = Path(project_dir)
        self.device = device
        self.frames = []
        self.cameras = []
        self.images = []
        self.points_xyz = []
        self.points_rgb = []

        self.load_transforms()
        self.load_points()

    def load_transforms(self):
        json_path = self.project_dir / "transforms.json"
        with open(json_path, 'r') as f:
            meta = json.load(f)

        w = int(meta.get('w', 1920))
        h = int(meta.get('h', 1080))
        fl_x = float(meta.get('fl_x', 1000))
        fl_y = float(meta.get('fl_y', 1000))
        cx = float(meta.get('cx', w/2))
        cy = float(meta.get('cy', h/2))

        # Build intrinsic matrix (K)
        # Note: taichi-splatting CameraParams uses T_image_camera which is K
        K = torch.tensor([
            [fl_x, 0, cx],
            [0, fl_y, cy],
            [0, 0, 1]
        ], dtype=torch.float32, device=self.device)

        for frame in meta['frames']:
            fname = frame['file_path']
            # Handle relative path
            img_path = self.project_dir / fname
            if not img_path.exists():
                # Try finding in images folder
                img_path = self.project_dir / "images" / Path(fname).name

            if not img_path.exists():
                print(f"Warning: Image {img_path} not found, skipping.")
                continue

            # Load Image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img).float() / 255.0
            img_tensor = img_tensor.to(self.device)
            self.images.append(img_tensor)

            # Load Pose
            # transforms.json is usually OpenGL convention: +Y up, -Z forward.
            # taichi-splatting/CameraParams usually expects World-to-Camera?
            # CameraParams doc: T_camera_world (4,4) camera view matrix (World -> Camera?)
            # Wait, doc says: "T_camera_world : torch.Tensor # (4, 4) camera view matrix"
            # But usually T_camera_world means T_{camera <- world} which is the View Matrix (inverse of Pose).
            # Let's check params.py:
            # camera_position property: T_world_camera = torch.inverse(self.T_camera_world); return T_world_camera[0:3, 3]
            # So T_camera_world is indeed the View Matrix (World to Camera).

            c2w = torch.tensor(frame['transform_matrix'], dtype=torch.float32, device=self.device)

            # Nerfstudio transforms are C2W (Camera to World).
            # So we need inverse.
            w2c = torch.inverse(c2w)

            # Create CameraParams
            # Check image size
            ih, iw = img.shape[:2]

            cam_param = CameraParams(
                T_image_camera=K,
                T_camera_world=w2c,
                near_plane=0.1,
                far_plane=100.0,
                image_size=(iw, ih)
            )
            self.cameras.append(cam_param)

        print(f"Loaded {len(self.images)} images.")

    def load_points(self):
        # Try finding points3D
        candidates = [
            self.project_dir / "sparse" / "0" / "points3D.bin",
            self.project_dir / "sparse" / "0" / "points3D.txt",
            self.project_dir / "sparse_pc.ply"
        ]

        points = {}
        for p in candidates:
            if p.exists():
                print(f"Loading points from {p}")
                if p.suffix == ".bin":
                    points = read_points3d_binary(p)
                elif p.suffix == ".txt":
                    points = read_points3d_txt(p)
                elif p.suffix == ".ply":
                    # TODO: Implement PLY loader if needed, or rely on plyfile
                    # Using simple heuristic for now or plyfile
                    if PlyData:
                        plydata = PlyData.read(str(p))
                        v = plydata['vertex']
                        xs = v['x']
                        ys = v['y']
                        zs = v['z']
                        # Assume colors exist
                        try:
                            rs = v['red']
                            gs = v['green']
                            bs = v['blue']
                        except:
                            rs = np.ones_like(xs) * 128
                            gs = np.ones_like(xs) * 128
                            bs = np.ones_like(xs) * 128

                        for i in range(len(xs)):
                            points[i] = (np.array([xs[i], ys[i], zs[i]]), np.array([rs[i], gs[i], bs[i]]))
                break

        if not points:
            print("Warning: No initial points found. Using random initialization.")
            # Random initialization
            for i in range(1000):
                points[i] = (np.random.rand(3)*10 - 5, np.random.randint(0, 255, 3))

        self.points_xyz = np.array([p[0] for p in points.values()])
        self.points_rgb = np.array([p[1] for p in points.values()]) / 255.0
        print(f"Loaded {len(self.points_xyz)} points.")

def export_ply(filepath, gaussians):
    """
    Export Gaussians to PLY file.
    """
    xyz = gaussians.position.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = gaussians.feature.detach().cpu().numpy() # Assume SH=0 (RGB) or SH. We only take DC.
    # If SH, feature is (N, 3, (D+1)**2) or (N, C)
    # If standard 3DGS, f_dc is (N, 3) (or colors)

    # Check shape
    if len(f_dc.shape) == 3:
        # (N, 3, K) -> take K=0
        f_dc = f_dc[:, :, 0]

    # f_dc is usually SH coefficients. 0-th band is color/0.28209
    # But here we might just store RGB directly if we trained that way.
    # Let's assume we trained RGB directly for simplicity, or convert.
    # Standard PLY for splat viewers expects f_dc_0, f_dc_1, f_dc_2

    opacities = gaussians.alpha.detach().cpu().numpy().flatten() # sigmoid(alpha_logit)
    scale = gaussians.scale.detach().cpu().numpy() # exp(log_scaling)
    rotation = gaussians.rotation.detach().cpu().numpy() # (w, x, y, z)

    # Create dtype
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
             ('opacity', 'f4'),
             ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
             ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')]

    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    elements['nx'] = normals[:, 0]
    elements['ny'] = normals[:, 1]
    elements['nz'] = normals[:, 2]
    elements['f_dc_0'] = f_dc[:, 0]
    elements['f_dc_1'] = f_dc[:, 1]
    elements['f_dc_2'] = f_dc[:, 2]
    elements['opacity'] = inverse_sigmoid(torch.tensor(opacities)).numpy() # Viewers often expect logit?
    # Wait, standard PLY:
    # opacity: logit or raw?
    # "opacity": usually passed through sigmoid in renderer. In PLY it is often stored as logit or raw value 0-1.
    # Standard 3DGS implementation stores 'opacity' as sigmoid(logit).
    # But some viewers might expect logit.
    # Original 3DGS stores opacity as passed through sigmoid?
    # No, original 3DGS source:
    # "opacity": sigmoid(logit)

    elements['opacity'] = opacities

    # Scale: log scale?
    # Original 3DGS stores log scale.
    elements['scale_0'] = np.log(scale[:, 0])
    elements['scale_1'] = np.log(scale[:, 1])
    elements['scale_2'] = np.log(scale[:, 2])

    elements['rot_0'] = rotation[:, 0] # w
    elements['rot_1'] = rotation[:, 1] # x
    elements['rot_2'] = rotation[:, 2] # y
    elements['rot_3'] = rotation[:, 3] # z

    if PlyData:
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(str(filepath))
        print(f"Exported to {filepath}")
    else:
        print("Error: plyfile not installed, cannot export.")

def train(args):
    device = torch.device('cuda')
    ti.init(arch=ti.cuda, device_memory_GB=4.0) # Optimized for Kaggle T4

    dataset = SceneDataset(args.project_path, device=device)

    # Initialize Gaussians
    xyz = torch.from_numpy(dataset.points_xyz).float().to(device)
    rgb = torch.from_numpy(dataset.points_rgb).float().to(device)

    # SH: We will stick to RGB (DC only) for simplicity and speed
    # Or initialize SH degree 0
    features = rgb # (N, 3)

    # Scale
    dist2 = torch.clamp_min(torch.ones_like(xyz[:, 0]) * 0.01, 0.0000001)
    scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)

    # Rotation (quaternions)
    rots = torch.zeros((xyz.shape[0], 4), device=device)
    rots[:, 0] = 1 # w=1, xyz=0 (identity)

    # Opacity
    opacities = inverse_sigmoid(0.1 * torch.ones((xyz.shape[0], 1), device=device))

    gaussians = Gaussians3D(
        position=xyz,
        log_scaling=scales,
        rotation=rots,
        alpha_logit=opacities,
        feature=features
    )

    # Learning rates setup
    learning_rates = dict(
        position=0.00016,
        feature=0.0025,
        log_scaling=0.005,
        rotation=0.001,
        alpha_logit=0.05
    )

    # Use ParameterClass for easier optimization and densification
    params = ParameterClass.create(
        gaussians.to_tensordict(),
        learning_rates,
        base_lr=1.0,
        optimizer=partial(optim.Adam, eps=1e-15)
    )

    config = RasterConfig(tile_size=16)

    print("Starting training...")

    grad_accumulator = torch.zeros((params.batch_size[0], 2), device=device)

    for iteration in range(1, args.iterations + 1):
        params.optimizer.zero_grad()

        # Pick random camera
        idx = np.random.randint(0, len(dataset.cameras))
        cam = dataset.cameras[idx]
        gt_image = dataset.images[idx]

        # Render
        # params acts as Gaussians3D
        rendering = render_gaussians(params, cam, config, compute_split_heuristics=True)

        image = rendering.image

        # Loss
        l1_loss = torch.abs(image - gt_image).mean()
        loss = l1_loss

        loss.backward()
        params.optimizer.step()

        # Accumulate gradients
        with torch.no_grad():
             if rendering.split_heuristics is not None:
                 # Resize accumulator if needed
                 if grad_accumulator.shape[0] != params.batch_size[0]:
                     grad_accumulator = torch.zeros((params.batch_size[0], 2), device=device)
                 grad_accumulator += rendering.split_heuristics

        if iteration % 100 == 0:
            print(f"Iter {iteration}: Loss {loss.item():.4f}, Points {params.batch_size[0]}")

        # Densification
        if iteration < args.densify_until_iter and iteration % args.densification_interval == 0:
            with torch.no_grad():
                stats = grad_accumulator / args.densification_interval
                visibility = stats[:, 0]
                grads = stats[:, 1]

                grad_accumulator.zero_() # Reset

                grad_thresh = 0.0002

                # Masks
                scales = params.log_scaling.exp().max(dim=1).values
                scene_extent = 5.0 # simplified

                split_mask = (grads > grad_thresh) & (scales > 0.01 * scene_extent)
                clone_mask = (grads > grad_thresh) & (scales <= 0.01 * scene_extent)

                # Split (sample new points)
                if split_mask.any():
                    splits = params[split_mask].clone()
                    splits.log_scaling -= np.log(1.6) # Scale down
                    # Perturb position
                    splits.position += torch.randn_like(splits.position) * splits.log_scaling.exp()

                    params = params.append_tensors(splits.to_tensordict())

                if clone_mask.any():
                    clones = params[clone_mask].clone()
                    params = params.append_tensors(clones.to_tensordict())

                print(f"Densification: {split_mask.sum()} splits, {clone_mask.sum()} clones")

    # Export
    out_dir = Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    export_ply(out_dir / "point_cloud.ply", params)
    export_ply(out_dir / "model.ply", params) # Backup name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_path", required=True, help="Path to project folder containing transforms.json")
    parser.add_argument("--output_path", default="outputs", help="Output directory")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--densify_until_iter", type=int, default=1500)
    parser.add_argument("--densification_interval", type=int, default=100)

    args = parser.parse_args()
    train(args)
