# === Zone 1: Stable Base ===
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
USER root

# === Zone 2: OS & Python ===
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev git colmap ffmpeg libsm6 libxext6 libgl1-mesa-glx xvfb wget build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# === Zone 3: Essential Python Stack ===
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# === Zone 4: Nerfstudio (The Critical Part) ===
# เราจะติดตั้ง dependencies ที่จำเป็นก่อน
RUN pip install --no-cache-dir gsplat==1.0.0
RUN pip install --no-cache-dir nerfstudio==1.1.1

# *** ตรวจสอบทันทีว่า import ได้ไหม ***
RUN python -c "import nerfstudio; print('✅ Nerfstudio found:', nerfstudio.__version__)" || exit 1

# === Zone 5: Worker App ===
RUN pip install --no-cache-dir supabase runpod requests opencv-python-headless

WORKDIR /app
COPY step1_extract_frames.py .
COPY step2_colmap_sfm.py .
COPY runpod_worker.py .

ENV NERFSTUDIO_CACHE=/tmp/nerfstudio_cache
RUN mkdir -p $NERFSTUDIO_CACHE && chmod 777 $NERFSTUDIO_CACHE

ENTRYPOINT ["python3", "runpod_worker.py"]
