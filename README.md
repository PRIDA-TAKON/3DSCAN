# 📸 3D Scan & Gaussian Splatting (RunPod Serverless Edition)

โปรเจกต์สำหรับสร้างโมเดล 3D (.ply) จากไฟล์วิดีโอ โดยใช้เทคนิค **3D Gaussian Splatting (Nerfstudio)** ประมวลผลบน **RunPod GPU (Serverless)**

## 🚀 สถาปัตยกรรมปัจจุบัน: RunPod Serverless
เราใช้ระบบประมวลผลแบบ Serverless เพื่อประสิทธิภาพและความประหยัดสูงสุด:
- **Two-Step Pipeline:** แยกส่วนการคำนวณตำแหน่งกล้อง (SfM) และการเทรนโมเดล (Training)
- **Git-Synced Worker:** ใช้ `loader.py` ดึงโค้ดล่าสุดจาก GitHub ทุกครั้งที่เริ่มงาน (แก้ไขโค้ดได้ไม่ต้องรอ Build Image)
- **S3 Storage:** เก็บข้อมูลกลางทางและผลลัพธ์บน S3-compatible storage (RunPod S3)
- **Auto-Scale:** จ่ายเงินตามวินาทีที่ประมวลผลจริง และรองรับการทำงานพร้อมกันหลาย Job

---

## 📂 โครงสร้างโปรเจกต์ (Project Structure)

| ไฟล์/โฟลเดอร์ | หน้าที่ |
| :--- | :--- |
| **`frontend/`** | หน้าเว็บ Next.js 16/19 สำหรับอัปโหลดวีดีโอและติดตามสถานะงานแบบ Real-time |
| **`takon_3d_worker.py`** | **สคริปต์หลัก:** ควบคุม Logic ทั้งหมด (SfM -> Training -> Export) |
| **`loader.py`** | **Bootstrap:** อยู่ใน Docker Image ทำหน้าที่ Sync โค้ดล่าสุดและเตรียม Environment |
| **`Dockerfile`** | Base Image จาก `nerfstudio/nerfstudio` พร้อมติดตั้ง COLMAP/Glomap |
| **`scripts/`** | สคริปต์เสริมสำหรับการทดสอบและกระบวนการ SfM (`run_glomap.py`, `test_cycle.py`) |
| **`taichi-splatting-kaggle/`** | (Legacy/Option) ไลบรารี Taichi สำหรับการเทรนแบบทางเลือก |

---

## ⚙️ ขั้นตอนการทำงาน (Workflow)

ระบบทำงานอัตโนมัติ 2 ขั้นตอนหลัก:

1.  **Step 1: SFM (SfM Running)**
    - สกัดเฟรมจากวิดีโอด้วย `ffmpeg`
    - คำนวณตำแหน่งกล้องด้วย `Glomap` หรือ `COLMAP`
    - อัปโหลด `processed.zip` ขึ้น S3 เพื่อเตรียมเทรน
2.  **Step 2: Training (Training Running)**
    - ดึงข้อมูลจาก S3 มาเทรนด้วย `ns-train splatfacto` (Nerfstudio)
    - ส่งออกผลลัพธ์เป็นไฟล์ `.ply` ด้วย `ns-export`
    - อัปโหลดผลลัพธ์ขึ้น S3 และสร้าง Presigned URL (7 วัน) กลับไปที่ Supabase

---

## 🛠️ การติดตั้งและพัฒนา (Development Guide)

### 1. การตั้งค่า Environment (.env)
ต้องมีตัวแปรต่อไปนี้ใน Root และ RunPod Config:
- `SUPABASE_URL`, `SUPABASE_KEY`
- `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID`
- `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_BUCKET`
- `GIT_REPO_URL`, `GIT_TOKEN` (สำหรับ `loader.py`)

### 2. Autonomous Loop (ไม่ต้อง Build Image ใหม่)
หากแก้ไขเฉพาะไฟล์ `.py` สามารถทดสอบได้ทันที:
1. `git push` โค้ดขึ้น GitHub
2. รัน `python scripts/test_cycle.py` เพื่อสั่งรันงานและดู Log แบบ Real-time

---

## 📂 ลิงก์ที่เกี่ยวข้อง
* **Repository หลัก:** `https://github.com/PRIDA-TAKON/3DSCAN`
* **Viewer แนะนำ:** [Polycam Viewer](https://poly.cam/tools/viewer) หรือ [Splat Viewer](https://splat.antimatter.ai/)
