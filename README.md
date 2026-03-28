# 📸 3D Scan & Gaussian Splatting (Cloud Version)

โปรเจคสำหรับสร้างโมเดล 3D (.splat) จากไฟล์วิดีโอ โดยใช้เทคนิค **3D Gaussian Splatting** บน **Google Cloud Run (GPU L4)**

## 🚀 ระบบใหม่: Cloud Run Worker
เราได้ย้ายระบบจาก Kaggle มาเป็นการรันบน **Google Cloud Run Jobs** เพื่อความเสถียรและความเร็วที่เหนือกว่า:
- **One-pot Execution:** รัน 4 ขั้นตอน (Data Prep -> SfM -> Training -> Export) จบในคำสั่งเดียว
- **Auto-Scale:** รันงานได้พร้อมกันและจ่ายเงินตามการใช้งานจริง (Pay-per-use)
- **High Performance:** ใช้ GPU L4 สำหรับการเทรนที่รวดเร็ว

---

## 📂 โครงสร้างโปรเจค (Project Structure)

| ไฟล์/โฟลเดอร์ | หน้าที่ |
| :--- | :--- |
| **`frontend/`** | หน้าเว็บ Next.js สำหรับอัปโหลดวีดีโอและติดตามสถานะงาน |
| **`Dockerfile`** | สำหรับ Build เป็น Container เพื่อรันบน Google Cloud |
| **`cloud_run_worker.py`** | สคริปต์หลักที่รันบน Cloud ทำหน้าที่ Download -> SfM -> Train -> Upload |
| **`scripts/`** | โฟลเดอร์เก็บสคริปต์ประมวลผล (Extract frames, SfM, Export) |
| **`taichi-splatting-kaggle/`** | ไลบรารีหลักสำหรับการเทรน 3D Gaussian Splatting (Taichi) |

---

## 🛠️ วิธีการติดตั้งและใช้งาน

### 1. ฝั่งคลาวด์ (Cloud Setups)
1. **Google Cloud Project:** สร้างโปรเจคและเปิดใช้งาน Cloud Run API, Artifact Registry
2. **Build Image:**
   ```bash
   docker build -t gcr.io/[PROJECT_ID]/3d-scan-worker .
   docker push gcr.io/[PROJECT_ID]/3d-scan-worker
   ```
3. **Environment Variables:** ตั้งค่าตัวแปรดังนี้ใน Cloud Run Job:
   - `SUPABASE_URL`, `SUPABASE_KEY`
   - `GDRIVE_SERVICE_ACCOUNT` (JSON Content)
   - `GDRIVE_OUTPUT_FOLDER_ID` (ID โฟลเดอร์ปลายทาง)

### 2. ฝั่ง Frontend
1. เข้าไปที่โฟลเดอร์ `frontend/`
2. ตั้งค่า `.env.local` เพื่อเชื่อมต่อกับ Supabase
3. รันด้วย `npm run dev` หรือ Deploy ขึ้น Vercel

### 3. Workflow การใช้งาน
1. **Upload:** อัปโหลดวีดีโอ (.mp4) ขนาดไม่เกิน **50MB** ผ่านหน้าเว็บ
2. **Process:** ระบบจะสร้าง Job ใน Supabase
3. **Download:** เมื่อประมวลผลเสร็จ ลิงก์ดาวน์โหลดจาก **Google Drive** จะปรากฏขึ้นอัตโนมัติ

---

## 📂 ลิงก์ที่เกี่ยวข้อง
* **Repository หลัก:** `https://github.com/PRIDA-TAKON/3DSCAN`
* **Viewer แนะนำ:** [Polycam Viewer](https://poly.cam/tools/viewer) หรือ [Splat Viewer](https://splat.antimatter.ai/)
