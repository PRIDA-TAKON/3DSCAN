# === Zone 1: Nerfstudio Base (Official & Optimized) ===
FROM nerfstudio/nerfstudio:latest

USER root

# Fix: Ensure nerfstudio is in the PATH for root
ENV PATH="/home/user/.local/bin:${PATH}"
ENV PYTHONPATH="/home/user/.local/lib/python3.10/site-packages:${PYTHONPATH}"

# === Zone 2: COLMAP & OS Binaries (Fixed Layer) ===
RUN apt-get update && apt-get install -y --no-install-recommends \
    colmap xvfb ffmpeg libsm6 libxext6 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# === Zone 3: Python Dependencies (Worker Core) ===
# ติดตั้ง Library ที่จำเป็นสำหรับการสื่อสารกับระบบภายนอก
# เพิ่ม opencv-python-headless เพื่อใช้จัดการวิดีโอ/เฟรม
RUN pip install --no-cache-dir supabase runpod requests opencv-python-headless

# === Zone 4: Your Application Logic (Fast Iteration Layer) ===
WORKDIR /app
COPY step1_extract_frames.py .
COPY step2_colmap_sfm.py .
COPY runpod_worker.py .

ENTRYPOINT ["python3", "runpod_worker.py"]
