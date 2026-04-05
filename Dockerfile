# === Stage 1: Base Environment ===
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Avoid interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

USER root

# === Stage 2: System Dependencies ===
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    python3-dev \
    python3-pip \
    python3-setuptools \
    colmap \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    xvfb \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Fix python alias
RUN ln -s /usr/bin/python3 /usr/bin/python

# === Stage 3: Python Environment & PyTorch ===
RUN pip install --no-cache-dir --upgrade pip
# Install PyTorch 2.1.2 for CUDA 11.8
RUN pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# === Stage 4: Nerfstudio & Splatting Stack ===
# Install gsplat first (compatible with nerfstudio 1.1.x)
RUN pip install --no-cache-dir gsplat==1.0.0

# Install Nerfstudio
RUN pip install --no-cache-dir nerfstudio==1.1.1

# Install Worker dependencies
RUN pip install --no-cache-dir supabase runpod requests opencv-python-headless

# === Stage 5: Application Logic ===
WORKDIR /app
COPY step1_extract_frames.py .
COPY step2_colmap_sfm.py .
COPY runpod_worker.py .

# Pre-create some directories for volume/temp use
RUN mkdir -p /tmp/nerfstudio_cache && chmod 777 /tmp/nerfstudio_cache
ENV NERFSTUDIO_CACHE=/tmp/nerfstudio_cache

ENTRYPOINT ["python3", "runpod_worker.py"]
