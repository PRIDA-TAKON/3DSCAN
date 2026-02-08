
import argparse
from pathlib import Path
import sys
import os

# Ensure we can import from scripts directory
sys.path.append(str(Path(__file__).parent))
# Ensure we can import taichi_3d_gaussian_splatting from temp_new_taichi OR taichi_3d_gaussian_splatting
sys.path.append(str(Path(__file__).parent.parent / "temp_new_taichi"))
sys.path.append(str(Path(__file__).parent.parent / "taichi_3d_gaussian_splatting"))
import colmap_to_taichi

try:
    from taichi_3d_gaussian_splatting.GaussianPointTrainer import GaussianPointCloudTrainer
except ImportError as e:
    import traceback
    traceback.print_exc()
    print(f"❌ Error: taichi_3d_gaussian_splatting could not be imported: {e}")
    print("Please install the library or check sys.path.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Train Taichi Gaussian Splatting (Migrated Version)")
    parser.add_argument("--project_path", type=str, required=True, help="Path to the project folder (containing sparse/0 and images)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save outputs")
    parser.add_argument("--iterations", type=int, default=3000, help="Number of training iterations")
    args = parser.parse_args()

    project_path = Path(args.project_path)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Paths setup
    colmap_path = project_path / "sparse" / "0"
    if not colmap_path.exists():
        # Fallback to sparse if 0 doesn't exist (some colmap versions)
        colmap_path = project_path / "sparse"
    
    images_path = project_path / "images"
    
    if not colmap_path.exists():
        print(f"❌ Error: COLMAP sparse reconstruction not found at {colmap_path}")
        return

    # 1. Convert Data to Taichi Format
    print("🚀 Converting COLMAP data for Taichi Splatting...")
    dataset_output_dir = output_path / "dataset"
    try:
        colmap_to_taichi.convert_colmap_to_taichi(
            base_path=str(colmap_path),
            image_path=str(images_path),
            output_dir=str(dataset_output_dir)
        )
    except Exception as e:
        print(f"❌ Data conversion failed: {e}")
        return

    # 2. Configure Training
    print("⚙️ Configuring Trainer...")
    config = GaussianPointCloudTrainer.TrainConfig()
    
    config.train_dataset_json_path = str(dataset_output_dir / "train.json")
    config.val_dataset_json_path = str(dataset_output_dir / "val.json")
    config.pointcloud_parquet_path = str(dataset_output_dir / "point_cloud.parquet")
    config.summary_writer_log_dir = str(output_path / "logs")
    config.output_model_dir = str(output_path / "models")
    config.num_iterations = args.iterations
    
    # Adjust for T4/Kaggle environment if needed
    config.val_interval = 500  # Frequent validation for monitoring
    config.print_metrics_to_console = True

    # 3. Run Training
    print("🔥 Starting Training...")
    trainer = GaussianPointCloudTrainer(config)
    trainer.train()
    
    print(f"✅ Training Complete. Models saved to {config.output_model_dir}")
    
    # Verify output
    best_scene_path = Path(config.output_model_dir) / "best_scene.parquet"
    if best_scene_path.exists():
         print(f"✅ Best scene found: {best_scene_path}")
         # Copy to model.parquet for step 4 convenience
         import shutil
         shutil.copy(best_scene_path, output_path / "model.parquet")
    else:
        print("⚠️ 'best_scene.parquet' not found. Checking for latest scene...")
        # Check for any scene_*.parquet
        scenes = list(Path(config.output_model_dir).glob("scene_*.parquet"))
        if scenes:
             latest_scene = sorted(scenes, key=lambda p: p.stat().st_mtime)[-1]
             print(f"✅ Using latest scene: {latest_scene}")
             shutil.copy(latest_scene, output_path / "model.parquet")
        else:
             print("❌ No model parquet files found!")

if __name__ == "__main__":
    main()
