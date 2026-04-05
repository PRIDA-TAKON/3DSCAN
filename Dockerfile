# === Zone 1: PyTorch Base (Already has CUDA 11.8 & PyTorch) ===
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
USER root

# === Zone 2: System Dependencies ===
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    xvfb \
    wget \
    colmap \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# === Zone 3: Nerfstudio & Splatting Stack ===
# Install dependencies for gsplat/nerfstudio build
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir gsplat==1.0.0 nerfstudio==1.1.1

# Install Worker core dependencies
RUN pip install --no-cache-dir supabase runpod requests opencv-python-headless

# === Zone 4: Application Logic ===
WORKDIR /app
COPY step1_extract_frames.py .
COPY step2_colmap_sfm.py .
COPY runpod_worker.py .

# Fix for nerfstudio cache
ENV NERFSTUDIO_CACHE=/tmp/nerfstudio_cache
RUN mkdir -p $NERFSTUDIO_CACHE && chmod 777 $NERFSTUDIO_CACHE

ENTRYPOINT ["python", "runpod_worker.py"]
