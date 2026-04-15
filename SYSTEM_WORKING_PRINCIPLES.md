# 3DGS Worker & Frontend Integration Principles

เอกสารนี้รวบรวมหลักการทำงานของระบบประมวลผล 3D Gaussian Splatting (3DGS) ระหว่าง Worker และ Frontend เพื่อใช้ในการตรวจสอบและกู้คืนระบบ (Recovery).

## 1. ผังการทำงาน (Job Workflow)
ระบบทำงานแบบ 2 ขั้นตอน (Two-Step Pipeline) ผ่าน RunPod Serverless:

### Step 1: SFM Process (Image Extraction & Pose Estimation)
1.  **Input:** วิดีโอ (MP4/MOV) จาก Supabase Storage.
2.  **Logic:** `ffmpeg` สกัดเฟรมภาพ -> `Glomap`/`COLMAP` คำนวณตำแหน่งกล้อง (SfM).
3.  **Output:** ไฟล์ `processed.zip` (ประกอบด้วยภาพและข้อมูลตำแหน่งกล้อง) ถูกอัปโหลดไปที่ S3 (`temp/{job_id}/processed.zip`).
4.  **Status:** `SFM_RUNNING` -> `SFM_COMPLETED`.

### Step 2: Training (3DGS Model Generation)
1.  **Input:** ไฟล์ `processed.zip` จาก S3.
2.  **Logic:** `ns-train` (Nerfstudio) เทรนโมเดล -> `ns-export` ส่งออกไฟล์ `.ply`.
3.  **Output:** ไฟล์ `model.ply` ถูกอัปโหลดไปที่ S3 (`results/{job_id}/model.ply`).
4.  **Status:** `TRAINING_RUNNING` -> `COMPLETED`.

---

## 2. การควบคุมสถานะ (Supabase Enum Alignment)
เพื่อให้ Frontend ติดตามความคืบหน้าได้แม่นยำ Worker จะส่งสถานะตาม Enum ต่อไปนี้:

| สถานะใน DB | ความหมาย (Message) |
| :--- | :--- |
| `PENDING` | รอคิว Worker รับงาน |
| `SFM_RUNNING` | กำลังสกัดเฟรมภาพและคำนวณ SfM |
| `SFM_COMPLETED` | เตรียมข้อมูล SfM สำเร็จ (พร้อมเทรน) |
| `TRAINING_RUNNING` | กำลังเทรนโมเดล 3DGS (2000 Iterations) |
| `COMPLETED` | งานเสร็จสมบูรณ์ (พร้อมให้ดาวน์โหลด) |
| `FAILED` | เกิดข้อผิดพลาดในขั้นตอนใดขั้นตอนหนึ่ง |

---

## 3. ขั้นตอนการกู้คืนระบบ (Recovery Procedures)

### กรณีระบบพังหลังการแก้ไข (Reverting Code)
หากมีการแก้ไข Code และทำให้ระบบที่เคยรันได้พังลง ให้ใช้คำสั่งต่อไปนี้เพื่อกู้ไฟล์สำรอง (Backup):
```bash
# กู้ไฟล์ Worker
cp takon_3d_worker.py.bak takon_3d_worker.py

# กู้ไฟล์ Frontend Component
cp frontend/components/JobCard.tsx.bak frontend/components/JobCard.tsx
```

### กรณี Image บน RunPod มีปัญหา (Rollback Image)
1. ตรวจสอบเลข Version (Tag) ของ Docker Image ที่ทำงานได้ล่าสุดใน GitHub Action หรือ Artifact Registry.
2. อัปเดต `RUNPOD_ENDPOINT_ID` ให้ชี้ไปยัง Image Version เดิมที่เสถียร.

---

## 4. โครงสร้างไฟล์สำคัญ
- `takon_3d_worker.py`: สคริปต์หลักที่รันบน RunPod (Logic & S3 Integration).
- `frontend/app/api/run-worker/route.ts`: API สำหรับสั่งงาน Worker.
- `frontend/components/JobCard.tsx`: คอมโพเนนต์แสดงสถานะและปุ่มดาวน์โหลด.
- `scripts/run_glomap.py`: ส่วนขยายสำหรับการคำนวณ SfM แบบรวดเร็ว.
