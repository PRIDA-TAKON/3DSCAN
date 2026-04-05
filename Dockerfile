# === Zone 1: Nerfstudio Base (Official & Optimized) ===
FROM nerfstudio/nerfstudio:latest

USER root

# Fix: Use dynamic paths for python and binaries
ENV PATH="/home/user/.local/bin:${PATH}"
ENV PYTHONPATH="/home/user/.local/lib/python$(python3 --version | cut -d' ' -f2 | cut -d. -f1,2)/site-packages:${PYTHONPATH}"

# === Zone 2: COLMAP & OS Binaries (Fixed Layer) ===
RUN apt-get update && apt-get install -y --no-install-recommends \
    colmap xvfb ffmpeg libsm6 libxext6 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# === Zone 3: Python Dependencies (Worker Core) ===
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir supabase runpod requests opencv-python-headless

# === Zone 4: Your Application Logic (Fast Iteration Layer) ===
WORKDIR /app
COPY step1_extract_frames.py .
COPY step2_colmap_sfm.py .
COPY runpod_worker.py .

ENTRYPOINT ["python3", "runpod_worker.py"]
