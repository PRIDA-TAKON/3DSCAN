# Layer 1: OS & CUDA Base
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Avoid prompts from apt during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONIOENCODING=utf-8
ENV QT_QPA_PLATFORM=offscreen

# Layer 2: OS Libraries (Rarely changes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-venv python3.10-dev \
    git wget curl unzip build-essential \
    colmap xvfb \
    ffmpeg libsm6 libxext6 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade pip

# Layer 3: Python Dependencies (Changes occasionally)
# Install heavy libraries first to cache them
RUN pip install numpy pandas opencv-python plyfile pyyaml \
    torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 \
    taichi dataclass-wizard pytorch-msssim \
    supabase requests runpod

# Layer 4: Specific Tooling Dependencies (e.g. for taichi-splatting)
# We copy only requirements first to keep this layer cached
WORKDIR /app
COPY taichi-splatting-kaggle/requirements.txt ./taichi-requirements.txt
RUN if [ -f "./taichi-requirements.txt" ]; then pip install -r ./taichi-requirements.txt; fi

# Layer 5: Application Code (Changes FREQUENTLY)
# Copy the rest of the app code
COPY . .

# Setup Taichi 3DGS local package (since it's in editable mode or needs setup)
RUN if [ -d "taichi-splatting-kaggle" ]; then \
    cd taichi-splatting-kaggle && pip install -e .; \
    fi

# Set the entrypoint to the RunPod worker script
ENTRYPOINT ["python3", "runpod_worker.py"]
