
import os
import argparse
import subprocess
from pathlib import Path
import shutil

def extract_frames(video_path, output_dir, fps=2, max_width=1024):
    """
    Extracts frames from a video file using ffmpeg.
    
    Args:
        video_path (str): Path to the input video.
        output_dir (str): Directory where images will be saved.
        fps (int): Extraction rate in frames per second.
        max_width (int): Max width for resizing images (default: 1024).
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
    if not video_path.exists():
        print(f"❌ Error: Video file not found at {video_path}")
        return False

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🎬 Extracting frames from {video_path.name} at {fps} FPS (max width: {max_width})...")
    
    # ffmpeg scale filter: scale=w:h. -1 means preserve aspect ratio.
    vf_graph = f"scale={max_width}:-1"

    cmd = [
        "ffmpeg", "-i", str(video_path), 
        "-vf", vf_graph,
        "-qscale:v", "1", 
        "-r", str(fps), 
        str(output_dir / "%04d.jpg"),
        "-hide_banner", "-loglevel", "error" # Clean output
    ]
    
    try:
        print(f"Executing: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        # ตรวจสอบว่ามีไฟล์ภาพเกิดขึ้นจริงไหม
        images = list(output_dir.glob("*.jpg"))
        num_images = len(images)
        if num_images == 0:
            print("❌ No images were extracted. ffmpeg might have failed silently.")
            return False
            
        print(f"✅ Successfully extracted {num_images} images to {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ ffmpeg failed with return code {e.returncode}")
        return False
    except FileNotFoundError:
        print("❌ ffmpeg not found. Please install ffmpeg.")
        return False

if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser(description="Step 1: Extract Frames from Video")
    parser.add_argument("--input_video", required=True, help="Path to input .mp4 video")
    parser.add_argument("--output_dir", required=True, help="Directory to save extracted images")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second (default: 2)")
    parser.add_argument("--max_width", type=int, default=1024, help="Max width for resizing (default: 1024)")
    
    args = parser.parse_args()
    
    if not extract_frames(args.input_video, args.output_dir, args.fps, args.max_width):
        sys.exit(1) # คืนค่า error ให้ระบบภายนอกรู้
