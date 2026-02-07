
import os
import argparse
import subprocess
from pathlib import Path
import shutil

def extract_frames(video_path, output_dir, fps=2):
    """
    Extracts frames from a video file using ffmpeg.
    
    Args:
        video_path (str): Path to the input video.
        output_dir (str): Directory where images will be saved.
        fps (int): Extraction rate in frames per second.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
    if not video_path.exists():
        print(f"❌ Error: Video file not found at {video_path}")
        return False

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🎬 Extracting frames from {video_path.name} at {fps} FPS...")
    
    cmd = [
        "ffmpeg", "-i", str(video_path), 
        "-qscale:v", "1", 
        "-r", str(fps), 
        str(output_dir / "%04d.jpg"),
        "-hide_banner", "-loglevel", "error" # Clean output
    ]
    
    try:
        subprocess.run(cmd, check=True)
        num_images = len(list(output_dir.glob("*.jpg")))
        print(f"✅ Successfully extracted {num_images} images to {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ ffmpeg failed: {e}")
        return False
    except FileNotFoundError:
        print("❌ ffmpeg not found. Please install ffmpeg.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1: Extract Frames from Video")
    parser.add_argument("--input_video", required=True, help="Path to input .mp4 video")
    parser.add_argument("--output_dir", required=True, help="Directory to save extracted images")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second (default: 2)")
    
    args = parser.parse_args()
    
    extract_frames(args.input_video, args.output_dir, args.fps)
