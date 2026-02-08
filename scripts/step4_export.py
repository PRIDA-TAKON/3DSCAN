
import os
import argparse
from pathlib import Path
import numpy as np
import sys
import subprocess
import glob

def run_command(cmd):
    print(f"🚀 Running: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {cmd}")
        return False

def export_nerfstudio_to_ply(config_path: Path, ply_output_path: Path):
    """
    Exports Nerfstudio model to PLY using ns-export.
    """
    print(f"⏳ Exporting Nerfstudio model to {ply_output_path}...")
    cmd = f"ns-export gaussian-splat --load-config {config_path} --output-dir {ply_output_path.parent}"
    # Note: ns-export gaussian-splat typically creates a folder or specific naming.
    # We might need to handle the output name.
    if run_command(cmd):
        # ns-export usually saves as splat.ply in the output-dir
        exported_ply = ply_output_path.parent / "splat.ply"
        if exported_ply.exists() and exported_ply != ply_output_path:
            shutil.move(str(exported_ply), str(ply_output_path))
        return True
    return False

import shutil

def convert_ply_to_splat(ply_file: Path, output_file: Path):
    """
    Converts a PLY file to a .splat file.
    """
    ply_file = Path(ply_file)
    output_file = Path(output_file)
    
    print(f"⏳ Converting {ply_file.name} to .splat format...")
    try:
        from plyfile import PlyData, PlyElement
    except ImportError:
        print("❌ Error: plyfile not found. Cannot convert.")
        return False

    if not ply_file.exists():
        print(f"❌ Error: Input PLY file {ply_file} not found.")
        return False

    try:
        plydata = PlyData.read(str(ply_file))
        vert = plydata["vertex"]

        # Sort by scale/opacity importance approximation
        # Nerfstudio PLY properties might differ slightly from Taichi
        # Common keys: x, y, z, opacity, scale_0, scale_1, scale_2, f_dc_0...
        
        scales_keys = [k for k in vert.data.dtype.names if k.startswith("scale_")]
        opacity_key = "opacity" if "opacity" in vert.data.dtype.names else None
        
        if not scales_keys or not opacity_key:
             print("⚠️ Warning: Could not find standard scale/opacity keys. Using default sorting.")
             sorted_indices = np.arange(len(vert["x"]))
        else:
             # Approximation of importance
             scale_sum = np.sum([vert[k] for k in scales_keys], axis=0)
             sorted_indices = np.argsort(
                -np.exp(scale_sum) / (1 / (1 + np.exp(-vert[opacity_key])))
             )

        n = len(sorted_indices)
        position = np.stack([vert["x"][sorted_indices], vert["y"][sorted_indices], vert["z"][sorted_indices]], axis=1).astype(np.float32)

        scales = np.stack([vert[k][sorted_indices] for k in scales_keys], axis=1).astype(np.float32)
        scales = np.exp(scales)

        rot_keys = ["rot_0", "rot_1", "rot_2", "rot_3"]
        if all(k in vert.data.dtype.names for k in rot_keys):
            rot = np.stack([vert[k][sorted_indices] for k in rot_keys], axis=1).astype(np.float32)
            length = np.sqrt(np.sum(rot ** 2, axis=1, keepdims=True))
            rot /= length
            rot_int = ((rot * 128 + 128).clip(0, 255)).astype(np.uint8)
        else:
            rot_int = np.zeros((n, 4), dtype=np.uint8)

        # Handle color/SH
        SH_C0 = 0.28209479177387814
        R = np.zeros(n, dtype=np.uint8)
        G = np.zeros(n, dtype=np.uint8)
        B = np.zeros(n, dtype=np.uint8)
        
        if "f_dc_0" in vert.data.dtype.names:
             R = (0.5 + SH_C0 * vert["f_dc_0"][sorted_indices]) * 255
             G = (0.5 + SH_C0 * vert["f_dc_1"][sorted_indices]) * 255
             B = (0.5 + SH_C0 * vert["f_dc_2"][sorted_indices]) * 255
        elif "red" in vert.data.dtype.names:
             R = vert["red"][sorted_indices]
             G = vert["green"][sorted_indices]
             B = vert["blue"][sorted_indices]

        R = np.clip(R, 0, 255).astype(np.uint8)
        G = np.clip(G, 0, 255).astype(np.uint8)
        B = np.clip(B, 0, 255).astype(np.uint8)
        A = np.full_like(R, 255, dtype=np.uint8)
        color = np.stack([R, G, B, A], axis=1)

        dtype_output = np.dtype([
            ('position', np.float32, 3),
            ('scale', np.float32, 3),
            ('color', np.uint8, 4),
            ('rot', np.uint8, 4)
        ])

        structured_data = np.empty(n, dtype=dtype_output)
        structured_data['position'] = position
        structured_data['scale'] = scales
        structured_data['color'] = color
        structured_data['rot'] = rot_int

        with open(output_file, "wb") as f:
            f.write(structured_data.tobytes())

        print(f"✅ Successfully converted to {output_file}")
        return True

    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Step 4: Export Nerfstudio Model to SPLAT")
    parser.add_argument("--input_config", help="Path to Nerfstudio config.yml")
    parser.add_argument("--input_ply", help="Path to input .ply file (if already exported)")
    parser.add_argument("--output_splat", required=True, help="Path to output .splat file")
    
    args = parser.parse_args()

    ply_path = args.input_ply
    
    if args.input_config:
        config_path = Path(args.input_config)
        if config_path.exists():
            # Create a temp PLY
            temp_ply = Path("temp_model.ply")
            if export_nerfstudio_to_ply(config_path, temp_ply):
                ply_path = str(temp_ply)
            else:
                print("❌ Failed to export Nerfstudio model to PLY.")
                return
        else:
             # Search for config.yml in the dir if path is a directory
             if config_path.is_dir():
                 configs = list(config_path.glob("**/config.yml"))
                 if configs:
                     # Use the latest one
                     latest_config = sorted(configs, key=lambda p: p.stat().st_mtime)[-1]
                     print(f"📂 Found config: {latest_config}")
                     temp_ply = Path("temp_model.ply")
                     if export_nerfstudio_to_ply(latest_config, temp_ply):
                         ply_path = str(temp_ply)
                     else: return
                 else:
                     print(f"❌ No config.yml found in {config_path}")
                     return
             else:
                 print(f"❌ Error: config.yml not found at {config_path}")
                 return

    if not ply_path:
        print("❌ Error: No input provided (--input_config or --input_ply)")
        return

    convert_ply_to_splat(Path(ply_path), Path(args.output_splat))
    
    # Cleanup temp_ply if created
    if args.input_config and Path("temp_model.ply").exists():
        os.remove("temp_model.ply")

if __name__ == "__main__":
    main()
