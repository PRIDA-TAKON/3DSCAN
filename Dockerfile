# === Zone 1: Nerfstudio Base (Official & Optimized) ===
FROM nerfstudio/nerfstudio:latest

# We use root only for OS package installation
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    colmap xvfb ffmpeg libsm6 libxext6 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Switch back to the 'user' that comes with the base image
# This ensures all paths for nerfstudio are correct automatically
USER user
WORKDIR /app

# Install additional python dependencies into the user's environment
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir supabase runpod requests opencv-python-headless

# Copy our scripts
COPY step1_extract_frames.py .
COPY step2_colmap_sfm.py .
COPY runpod_worker.py .

# Use python3 to run the worker (already in path for user)
ENTRYPOINT ["python3", "runpod_worker.py"]
