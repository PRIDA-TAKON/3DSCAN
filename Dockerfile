# Use Official NVIDIA CUDA image with Ubuntu 22.04 base
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Avoid prompts from apt during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONIOENCODING=utf-8
ENV QT_QPA_PLATFORM=offscreen

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-venv python3.10-dev \
    git wget curl unzip build-essential \
    colmap xvfb \
    ffmpeg libsm6 libxext6 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Symlink python3.10 to python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Create working directory
WORKDIR /app

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install required heavy ML/CV python utilities 
# (dataclass-wizard and pytorch-msssim required by Taichi)
RUN pip install numpy pandas opencv-python plyfile pyyaml \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
    taichi dataclass-wizard pytorch-msssim \
    supabase requests runpod

# Copy source code (scripts, GLOMAP fallback)
COPY . /app

# Setup Taichi 3DGS local package if it exists
RUN if [ -d "taichi-splatting-kaggle" ]; then \
    if [ -f "taichi-splatting-kaggle/requirements.txt" ]; then \
        pip install -r taichi-splatting-kaggle/requirements.txt; \
    fi; \
    cd taichi-splatting-kaggle && pip install -e .; \
    fi

# Set the entrypoint to the RunPod worker script
ENTRYPOINT ["python", "runpod_worker.py"]
