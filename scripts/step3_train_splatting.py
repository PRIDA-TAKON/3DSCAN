
import argparse
from pathlib import Path
import sys
import os
import subprocess
import shutil

def run_command(cmd):
    print(f"🚀 Running: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {cmd}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Train Nerfstudio Splatfacto")
    parser.add_argument("--project_path", type=str, required=True, help="Path to the project folder (containing transforms.json and images)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save outputs")
    parser.add_argument("--iterations", type=int, default=30000, help="Number of training iterations (Nerfstudio default is higher)")
    parser.add_argument("--machine", type=str, default="colmap", help="Nerfstudio data parser (colmap, nerfosm, etc.)")
    parser.add_argument("--vis", type=str, default="none", help="Visualizer to use (none, wandb, tensorboard, viewer)")
    args = parser.parse_args()

    project_path = Path(args.project_path).absolute()
    output_path = Path(args.output_path).absolute()
    output_path.mkdir(parents=True, exist_ok=True)

    # Nerfstudio needs to know where to find the data
    # We assume project_path contains images/ and transforms.json (from step 2)
    # If transforms.json exists, we use colmap parser.
    
    print(f"📂 Project Path: {project_path}")
    print(f"📂 Output Path: {output_path}")

    # 1. Run Nerfstudio Training
    # We use splatfacto for Gaussian Splatting
    # We set a deterministic base directory for outputs
    
    cmd = (
        f"ns-train splatfacto "
        f"--data {project_path} "
        f"--output-dir {output_path} "
        f"--max-num-iterations {args.iterations} "
        f"--vis {args.vis} "
        f"colmap " # Using colmap parser by default since step 2 provides compatible data
    )
    
    # Check if wandb is requested but not available
    if args.vis == "wandb":
        try:
            import wandb
        except ImportError:
            print("⚠️ wandb not installed, falling back to none.")
            cmd = cmd.replace("--vis wandb", "--vis none")

    print("🔥 Starting Nerfstudio Training...")
    if run_command(cmd):
        print("✅ Training sequence completed.")
        
        # Nerfstudio saves in output_path/nerfstudio_models/<config>/...
        # We want to find the latest config folder and provide it as a reference
        model_root = output_path
        print(f"ℹ️ Check {model_root} for training results.")
    else:
        print("❌ Training failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
