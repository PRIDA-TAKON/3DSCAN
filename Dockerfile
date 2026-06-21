FROM ghcr.io/nerfstudio-project/nerfstudio:latest

USER root
# ติดตั้งเฉพาะเครื่องมือระบบ (System Tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    colmap xvfb ffmpeg libsm6 libxext6 libgl1-mesa-glx git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ใช้ Python ของอิมเมจโดยตรง ไม่ติดตั้งทับ 
# ติดตั้งแค่ตัวเชื่อมต่อเบาๆ (ถ้าจำเป็น)
RUN python3 -m pip install --no-cache-dir supabase runpod requests

# ส่งเฉพาะ loader.py เข้าไป
COPY loader.py .

ENTRYPOINT ["python3", "/app/loader.py"]
