# === Zone 1: Nerfstudio Base (Official & Optimized) ===
FROM nerfstudio/nerfstudio:latest

USER root

# Fix: Path for nerfstudio binaries and libraries (based on standard nerfstudio image)
ENV PATH="/home/user/.local/bin:${PATH}"
# Link the user's packages to root's site-packages to ensure accessibility
RUN ln -s /home/user/.local/lib/python3.10/site-packages/* /usr/local/lib/python3.10/dist-packages/ || true

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
