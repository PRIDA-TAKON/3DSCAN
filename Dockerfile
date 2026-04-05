# === Zone 1: Nerfstudio Base ===
FROM nerfstudio/nerfstudio:latest

USER root

# === Zone 2: OS Packages ===
RUN apt-get update && apt-get install -y --no-install-recommends \
    colmap xvfb ffmpeg libsm6 libxext6 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# === Zone 3: Python Packages (Force Install to ensure Root access) ===
# เราจะติดตั้ง nerfstudio ทับอีกรอบเพื่อให้มั่นใจว่า Root เรียกใช้ได้ง่ายๆ
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir supabase runpod requests opencv-python-headless nerfstudio

# === Zone 4: Application Logic ===
WORKDIR /app
COPY step1_extract_frames.py .
COPY step2_colmap_sfm.py .
COPY runpod_worker.py .

# Ensure scripts are executable
RUN chmod +x *.py

ENTRYPOINT ["python3", "runpod_worker.py"]
