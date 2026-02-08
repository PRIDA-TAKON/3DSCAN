
import os
import argparse
from pathlib import Path
import numpy as np
import sys

# Ensure we can import taichi_3d_gaussian_splatting from temp_new_taichi OR taichi_3d_gaussian_splatting
sys.path.append(str(Path(__file__).parent.parent / "temp_new_taichi"))
sys.path.append(str(Path(__file__).parent.parent / "taichi_3d_gaussian_splatting"))

def convert_parquet_to_ply(parquet_path: Path, ply_path: Path):
    """
    Converts a Parquet file to a PLY file using taichi_3d_gaussian_splatting.
    """
    print(f"⏳ Converting {parquet_path.name} to {ply_path.name}...")
    try:
        from taichi_3d_gaussian_splatting.GaussianPointCloudScene import GaussianPointCloudScene
        # Load scene from parquet
        scene = GaussianPointCloudScene.from_parquet(
            str(parquet_path), 
            config=GaussianPointCloudScene.PointCloudSceneConfig(max_num_points_ratio=None)
        )
        # Save to PLY
        scene.to_ply(str(ply_path))
        print(f"✅ Successfully exported PLY to {ply_path}")
        return True
    except ImportError:
        print("❌ Error: taichi_3d_gaussian_splatting not installed. Cannot convert parquet.")
        return False
    except Exception as e:
        print(f"❌ Parquet conversion failed: {e}")
        return False

def convert_ply_to_splat(ply_file: Path, output_file: Path):
    """
    Converts a PLY file to a .splat file.
    """
    ply_file = Path(ply_file)
    output_file = Path(output_file)
    
    print(f"⏳ Converting {ply_file.name} to .splat format...")
    # Import plyfile locally to ensure it is available (installed in deps)
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
        sorted_indices = np.argsort(
            -np.exp(vert["scale_0"] + vert["scale_1"] + vert["scale_2"])
            / (1 / (1 + np.exp(-vert["opacity"])))
        )

        n = len(sorted_indices)
        x = vert["x"][sorted_indices]
        y = vert["y"][sorted_indices]
        z = vert["z"][sorted_indices]
        position = np.stack([x, y, z], axis=1).astype(np.float32)

        s0 = vert["scale_0"][sorted_indices]
        s1 = vert["scale_1"][sorted_indices]
        s2 = vert["scale_2"][sorted_indices]
        scales = np.stack([s0, s1, s2], axis=1).astype(np.float32)
        scales = np.exp(scales)

        r0 = vert["rot_0"][sorted_indices]
        r1 = vert["rot_1"][sorted_indices]
        r2 = vert["rot_2"][sorted_indices]
        r3 = vert["rot_3"][sorted_indices]
        rot = np.stack([r0, r1, r2, r3], axis=1).astype(np.float32)
        length = np.sqrt(np.sum(rot ** 2, axis=1, keepdims=True))
        rot /= length
        rot_int = ((rot * 128 + 128).clip(0, 255)).astype(np.uint8)

        # Handle color/SH
        # Implementation depends on what the PLY contains.
        # taichi_3d_gaussian_splatting PLY output might need checking.
        # Assuming it outputs f_dc_* or red/green/blue.
        
        SH_C0 = 0.28209479177387814
        R = np.zeros(n, dtype=np.uint8)
        G = np.zeros(n, dtype=np.uint8)
        B = np.zeros(n, dtype=np.uint8)
        
        if "f_dc_0" in vert:
             dc0 = vert["f_dc_0"][sorted_indices]
             dc1 = vert["f_dc_1"][sorted_indices]
             dc2 = vert["f_dc_2"][sorted_indices]
             R = (0.5 + SH_C0 * dc0) * 255
             G = (0.5 + SH_C0 * dc1) * 255
             B = (0.5 + SH_C0 * dc2) * 255
        elif "red" in vert:
             R = vert["red"][sorted_indices]
             G = vert["green"][sorted_indices]
             B = vert["blue"][sorted_indices]

        R = np.clip(R, 0, 255).astype(np.uint8)
        G = np.clip(G, 0, 255).astype(np.uint8)
        B = np.clip(B, 0, 255).astype(np.uint8)
        A = np.full_like(R, 255, dtype=np.uint8) # Full opacity for splat visualization
        color = np.stack([R, G, B, A], axis=1) # RGBA

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
        return False

def main():
    parser = argparse.ArgumentParser(description="Step 4: Export to SPLAT")
    parser.add_argument("--input_ply", help="Path to input .ply file")
    parser.add_argument("--input_parquet", help="Path to input .parquet file (from taichi training)")
    parser.add_argument("--output_splat", required=True, help="Path to output .splat file")
    
    args = parser.parse_args()

    ply_path = args.input_ply
    
    # Check for parquet input
    if args.input_parquet:
        parquet_path = Path(args.input_parquet)
        if parquet_path.exists():
            # Generate intermediate PLY path
            ply_path = parquet_path.with_suffix(".ply")
            if not convert_parquet_to_ply(parquet_path, ply_path):
                print("❌ Failed to convert Parquet to PLY.")
                return
        else:
             print(f"❌ Error: Input Parquet file {parquet_path} not found.")
             # Proceed to check ply_path if provided, else fail
             if not ply_path: return

    if not ply_path:
        print("❌ Error: No input file specified (--input_ply or --input_parquet)")
        return

    convert_ply_to_splat(ply_path, args.output_splat)

if __name__ == "__main__":
    main()
