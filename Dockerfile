# === Zone 1: Stable Base ===
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
USER root

# === Zone 2: OS Packages ===
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev python3-venv git colmap ffmpeg \
    libsm6 libxext6 libgl1-mesa-glx xvfb wget build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

# === Zone 3: Virtual Environment Setup ===
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# === Zone 4: Install Python Stack inside VENV ===
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Torch (CUDA 11.8 version)
RUN pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# Install gsplat & nerfstudio (Allowing dependencies to resolve naturally)
RUN pip install --no-cache-dir gsplat==1.0.0
RUN pip install --no-cache-dir nerfstudio==1.1.1

# *** Verification Step ***
RUN python -c "import nerfstudio; print('✅ SUCCESS: Nerfstudio version', nerfstudio.__version__)"

# === Zone 5: Worker Application ===
RUN pip install --no-cache-dir supabase runpod requests opencv-python-headless

WORKDIR /app
COPY step1_extract_frames.py .
COPY step2_colmap_sfm.py .
COPY runpod_worker.py .

# Setup Cache
ENV NERFSTUDIO_CACHE=/tmp/nerfstudio_cache
RUN mkdir -p $NERFSTUDIO_CACHE && chmod 777 $NERFSTUDIO_CACHE

# Use the VENV Python directly for entrypoint
ENTRYPOINT ["/opt/venv/bin/python", "runpod_worker.py"]
