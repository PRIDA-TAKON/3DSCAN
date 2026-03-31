# === Zone 1: Nerfstudio Base (Official & Optimized) ===
FROM nerfstudio/nerfstudio:latest

USER root

# === Zone 2: COLMAP & OS Binaries (Fixed Layer) ===
# เลเยอร์นี้จะถูก Cache ไว้ ไม่ต้องโหลดใหม่ถ้าไม่เพิ่มโปรแกรม OS
RUN apt-get update && apt-get install -y --no-install-recommends \
    colmap xvfb ffmpeg libsm6 libxext6 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# === Zone 3: Python Dependencies (Worker Core) ===
# ติดตั้ง Library ที่จำเป็นสำหรับการสื่อสารกับระบบภายนอก
# แยกออกมาเพื่อให้ไม่ต้องติดตั้งใหม่เวลาแก้โค้ดประมวลผล
RUN pip install --no-cache-dir supabase runpod requests

# === Zone 4: Your Application Logic (Fast Iteration Layer) ===
# โซนนี้คือส่วนที่คุณจะแก้ "หลายสิบรอบ" 
# เราจะ Copy เฉพาะไฟล์ที่จำเป็น เพื่อให้ Docker Cache ทำงานได้ดีที่สุด
WORKDIR /app

# หากคุณมีการใช้ไฟล์จากโฟลเดอร์ taichi ใน nerfstudio (เผื่อกรณีดึงโค้ดบางส่วนมาใช้)
# แต่ถ้าจะย้ายไป nerfstudio เต็มตัว เราจะเน้นที่สคริปต์หลักของเรา
COPY step1_extract_frames.py .
COPY step2_colmap_sfm.py .
COPY runpod_worker.py .
# COPY scripts/ ./scripts/ 

# ทุกครั้งที่คุณแก้สคริปต์ด้านบน Docker จะเริ่มรันใหม่ตั้งแต่บรรทัด COPY นี้ลงมาเท่านั้น
# ซึ่งใช้เวลาเพียง 1-2 วินาทีในการประกอบร่างครับ

ENTRYPOINT ["python3", "runpod_worker.py"]
