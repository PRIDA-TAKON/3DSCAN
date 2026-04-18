# โครงสร้างโปรเจกต์ 3DSCAN (RunPod Edition)

โปรเจกต์นี้ถูกออกแบบมาเพื่อรันกระบวนการ **3D Gaussian Splatting** บน **RunPod Serverless** โดยใช้สถาปัตยกรรมแบบ 2-Step Pipeline ที่เน้นความยืดหยุ่นและการพัฒนาที่รวดเร็ว

## 📁 โครงสร้างไฟล์ (File Structure)

| ไฟล์/โฟลเดอร์ (File/Folder) | หน้าที่ (Description) |
| :--- | :--- |
| **`takon_3d_worker.py`** | **หัวใจหลัก:** สคริปต์ที่รันบน RunPod รับหน้าที่ดึงงานจาก Supabase, ดาวน์โหลดวิดีโอ, และรัน SfM หรือ Training ตาม Mode ที่ได้รับ |
| **`loader.py`** | **Git Sync:** ดึงโค้ดล่าสุดจาก GitHub เข้าสู่คอนเทนเนอร์ก่อนเริ่ม Worker เพื่อให้แก้ไขโค้ดได้ทันที |
| **`Dockerfile`** | **Container Config:** ใช้ฐานจาก `nerfstudio/nerfstudio` พร้อมติดตั้ง COLMAP และเครื่องมือที่จำเป็น |
| **`frontend/`** | **Dashboard:** หน้าเว็บ Next.js สำหรับให้ผู้ใช้ส่งงาน (Upload) และติดตามสถานะแบบ Real-time |
| **`scripts/`** | **Dev & Utils:** สคริปต์สำหรับทดสอบระบบ (`test_cycle.py`) และประมวลผล SfM (`run_glomap.py`) |
| **`taichi-splatting-kaggle/`** | **Legacy/Alternative:** โค้ดต้นฉบับสำหรับการเทรนด้วย Taichi |
| `supabase_schema.sql` | **Database:** โครงสร้างตาราง `jobs` และ `job_status` enum |

---

## ⚙️ ขั้นตอนการรันงาน (Workflow)

ระบบทำงานแบบ Serverless และแยกส่วนประมวลผล:

1. **User Action:** อัปโหลดวิดีโอผ่านหน้าเว็บ -> ข้อมูลบันทึกลง Supabase
2. **SFM Phase (Worker Mode: PROCESS):**
    - `Step 1:` แปลงวิดีโอเป็นเฟรมภาพ
    - `Step 2:` ทำ Sparse Reconstruction (SfM) ด้วย Glomap/COLMAP
    - `Step 3:` อัปโหลดไฟล์ประมวลผลเบื้องต้น (`processed.zip`) ไปยัง S3
3. **Training Phase (Worker Mode: TRAIN):**
    - `Step 1:` ดาวน์โหลดข้อมูลจาก S3
    - `Step 2:` เทรนโมเดลด้วย `ns-train splatfacto` (2,000 Iterations)
    - `Step 3:` ส่งออกไฟล์ `.ply` และอัปโหลดขึ้น S3
4. **Result Delivery:** สร้าง Presigned URL สำหรับดาวน์โหลดผลลัพธ์และอัปเดตสถานะเป็น `COMPLETED`

---

## 🚀 ข้อดีของสถาปัตยกรรมนี้
- **Fast Iteration:** แก้ไขโค้ด Python และ Push Git เพื่อทดสอบได้ทันที (ไม่ต้องรอ Build Docker)
- **Scalability:** RunPod Serverless รองรับการรันงานขนานกันได้ตามต้องการ
- **Separation of Concerns:** แยกขั้นตอน SfM ที่ใช้ CPU/GPU ปานกลาง ออกจาก Training ที่ใช้ GPU สูง
