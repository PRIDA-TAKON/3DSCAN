# === Zone 1: Official Nerfstudio Base ===
FROM nerfstudio/nerfstudio:latest

# We need root only for system packages
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    colmap xvfb ffmpeg libsm6 libxext6 libgl1-mesa-glx \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# IMPORTANT: Switch back to 'user' to maintain the working nerfstudio environment
USER user
WORKDIR /app

# Fix: If nerfstudio is in a conda env, we need to find it and put it in the PATH
# Standard Nerfstudio Docker usually has it in /home/user/miniconda/bin/python or similar
ENV PATH="/home/user/.local/bin:/opt/conda/bin:/home/user/miniconda/bin:${PATH}"
ENV PYTHONPATH="/home/user/.local/lib/python3.10/site-packages:${PYTHONPATH}"

# Install our worker dependencies into the user space
# This avoids breaking the system python or nerfstudio installation
RUN pip install --no-cache-dir --user --upgrade pip && \
    pip install --no-cache-dir --user supabase runpod requests opencv-python-headless

# Copy ONLY the loader file
# All other worker files will be pulled from Git at runtime by loader.py
COPY --chown=user:user loader.py .

# Setup cache directory in a writable place for 'user'
ENV NERFSTUDIO_CACHE=/home/user/.cache/nerfstudio
RUN mkdir -p $NERFSTUDIO_CACHE

# Final verification before completion
RUN python3 -c "import nerfstudio; print('✅ SUCCESS: Nerfstudio is working!'); import runpod; print('✅ SUCCESS: RunPod is working!')"

ENTRYPOINT ["python3", "/app/loader.py"]
