# 3DGS Worker & Frontend Integration Principles

เอกสารนี้รวบรวมหลักการทำงานของระบบประมวลผล 3D Gaussian Splatting (3DGS) ระหว่าง Worker และ Frontend เพื่อใช้ในการตรวจสอบและกู้คืนระบบ (Recovery).

## 1. ผังการทำงาน (Job Workflow)
ระบบทำงานแบบขั้นตอนเดียวจบ (Single-step FULL Pipeline) ผ่าน RunPod Serverless เพื่อประสิทธิภาพและความประหยัด:

### FULL Mode Pipeline (End-to-End)
1.  **Input:** วิดีโอ (MP4/MOV) จาก Supabase Storage.
2.  **SfM Phase:** `ffmpeg` สกัดเฟรมภาพ -> `Glomap`/`COLMAP` คำนวณตำแหน่งกล้อง (SfM) เก็บไว้ในเครื่องชั่วคราว.
3.  **Training Phase:** `ns-train` (Nerfstudio) ทำการเทรนโมเดล (2000 Iterations) และแปลงผลลัพธ์ผ่าน `ns-export` ได้ไฟล์โมเดล `.ply`.
4.  **Output:** ไฟล์ผลลัพธ์ `model.ply` ถูกอัปโหลดขึ้นไปยัง **Supabase Storage** ใน Bucket `3d-scans` โฟลเดอร์ `results/{job_id}/model.ply`.
5.  **Status Transition:** `PENDING` -> `SFM_RUNNING` -> `TRAINING_RUNNING` -> `COMPLETED` (หรือ `FAILED`).

---

## 2. การควบคุมสถานะ (Supabase Enum Alignment)
เพื่อให้ Frontend ติดตามความคืบหน้าได้แม่นยำ Worker จะส่งสถานะตาม Enum ต่อไปนี้:

| สถานะใน DB | ความหมาย (Message) |
| :--- | :--- |
| `PENDING` | รอคิว Worker รับงาน |
| `SFM_RUNNING` | กำลังสกัดเฟรมภาพและคำนวณ SfM |
| `TRAINING_RUNNING` | กำลังเทรนโมเดล 3DGS (2000 Iterations) |
| `COMPLETED` | งานเสร็จสมบูรณ์ (พร้อมให้ดาวน์โหลดไฟล์ .ply จาก Supabase) |
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
