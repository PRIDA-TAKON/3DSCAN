# Layer 1: OS & CUDA Base
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Avoid prompts from apt during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONIOENCODING=utf-8
ENV QT_QPA_PLATFORM=offscreen

# Layer 2: OS Libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-venv python3.10-dev \
    git wget curl unzip build-essential \
    colmap xvfb \
    ffmpeg libsm6 libxext6 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Layer 3: Python Base Tools
RUN python3 -m pip install --upgrade pip setuptools wheel

# Layer 4: PyTorch & Dependencies (Nerfstudio requires specific versions)
RUN pip3 install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Layer 5: Tiny-cuda-nn (Crucial for Nerfstudio)
# We use the pre-built wheels to save build time
RUN pip install ninja
RUN pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# Layer 6: Nerfstudio & gsplat
RUN pip install nerfstudio gsplat

# Layer 7: Project Dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Layer 8: Application Code
COPY . .

# Set entrypoint to our worker
ENTRYPOINT ["python3", "runpod_worker.py"]
