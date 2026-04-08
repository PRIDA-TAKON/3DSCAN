FROM nerfstudio/nerfstudio:latest

USER root
# ติดตั้งเฉพาะเครื่องมือระบบ (System Tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    colmap xvfb ffmpeg libsm6 libxext6 libgl1-mesa-glx git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

USER user
WORKDIR /app

# ใช้ Python ของอิมเมจโดยตรง ไม่ติดตั้งทับ 
# ติดตั้งแค่ตัวเชื่อมต่อเบาๆ (ถ้าจำเป็น) ใน user space
RUN pip install --no-cache-dir --user supabase runpod requests

# ส่งเฉพาะ loader.py เข้าไป
COPY --chown=user:user loader.py .

# ตั้งค่า PATH ให้เรียกใช้เครื่องมือใน .local ได้
ENV PATH="/home/user/.local/bin:${PATH}"

ENTRYPOINT ["python3", "/app/loader.py"]
