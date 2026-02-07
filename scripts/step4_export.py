
import os
import argparse
from pathlib import Path
import numpy as np

def convert_ply_to_splat(ply_file: Path, output_file: Path):
    """
    Converts a PLY file to a .splat file.
    """
    ply_file = Path(ply_file)
    output_file = Path(output_file)
    
    print(f"⏳ Converting {ply_file.name} to .splat format...")
    # Import plyfile locally to ensure it is available (installed in deps)
    try:
        from plyfile import PlyData
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

        # Assuming DC0, DC1, DC2 are RGB
        SH_C0 = 0.28209479177387814
        if "f_dc_0" in vert:
             dc0 = vert["f_dc_0"][sorted_indices]
             dc1 = vert["f_dc_1"][sorted_indices]
             dc2 = vert["f_dc_2"][sorted_indices]
        elif "red" in vert: # Fallback to color ply
             R = vert["red"][sorted_indices]
             G = vert["green"][sorted_indices]
             B = vert["blue"][sorted_indices]
             # Already 0-255?
             pass

        # Handle simplified color export from train script
        # In our train script we save f_dc, so...
        R = (0.5 + SH_C0 * dc0) * 255
        G = (0.5 + SH_C0 * dc1) * 255
        B = (0.5 + SH_C0 * dc2) * 255
        
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
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4: Export PLY to SPLAT")
    parser.add_argument("--input_ply", required=True, help="Path to input .ply file")
    parser.add_argument("--output_splat", required=True, help="Path to output .splat file")
    
    args = parser.parse_args()
    
    convert_ply_to_splat(args.input_ply, args.output_splat)
