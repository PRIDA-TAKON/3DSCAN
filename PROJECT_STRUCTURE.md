# โครงสร้างโปรเจค 3DSCAN (Cloud Edition)

โปรเจคนี้ถูกออกแบบมาเพื่อรันกระบวนการ **3D Gaussian Splatting** บน **Google Cloud Run** โดยใช้สถาปัตยกรรมแบบ Worker-based ที่ทำงานครบวงจรในตัวเดียว

## 📁 โครงสร้างไฟล์ (File Structure)

| ไฟล์/โฟลเดอร์ (File/Folder) | หน้าที่ (Description) |
| :--- | :--- |
| **`cloud_run_worker.py`** | **หัวใจหลัก:** สคริปต์ที่รันบน Cloud รับหน้าที่ดึงงานจาก Supabase, ดาวน์โหลดวิดีโอ, และรัน 4 ขั้นตอนการประมวลผลจนจบ |
| **`Dockerfile`** | **Container Config:** สำหรับสร้างสถาพแวดล้อม Ubuntu + CUDA ที่ติดตั้ง COLMAP และ Taichi เรียบร้อยแล้ว |
| **`frontend/`** | **Dashboard:** หน้าเว็บ Next.js สำหรับให้ผู้ใช้ส่งงาน (Upload) และรอรับลิงก์ดาวน์โหลดผลลัพธ์ |
| **`scripts/`** | **Processing Scripts:** ไฟล์ Python ย่อยสำหรับงานเฉพาะทาง เช่น `step1_extract_frames.py` หรือ `run_glomap.py` |
| **`taichi-splatting-kaggle/`** | **Core Library:** ตัวเทรน 3D Gaussian Splatting ที่ใช้ภาษา Taichi เพื่อความรวดเร็ว |
| `supabase_schema.sql` | **Database:** โครงสร้างตาราง `jobs` สำหรับใช้ใน Supabase |

---

## ⚙️ ขั้นตอนการรันงาน (Workflow)

ระบบเปลี่ยนจากแบบ Manual บน Kaggle มาเป็นแบบอัตโนมัติ (Automated):

1. **User Action:** อัปโหลดวิดีโอผ่านหน้าเว็บ (จำกัด 50MB) -> ข้อมูลบันทึกลง Supabase
2. **Cloud Trigger:** เมื่อมีการสร้าง Job ใหม่ (สามารถใช้ Supabase Edge Function หรือ Trigger ภายนอก) สั่งรัน Cloud Run Job
3. **Worker Process:**
    - `Step 1:` แปลงวิดีโอเป็นเฟรมภาพ
    - `Step 2:` ทำ Sparse Reconstruction (SfM) เพื่อหาตำแหน่งกล้อง
    - `Step 3:` เทรนโมเดลด้วย Taichi 3DGS (30,000 Iterations)
    - `Step 4:` ส่งออกไฟล์ `.splat` และ Zip ผลลัพธ์
4. **Result Delivery:** อัปโหลดไฟล์ Zip ขึ้น **Google Drive** และส่งลิงก์กลับไปที่ Supabase
5. **Realtime Update:** หน้าเว็บของผู้ใช้เปลี่ยนสถานะเป็น `Ready` พร้อมปุ่มดาวน์โหลด

---

## 🚀 ข้อดีของการใช้ Cloud Run
- **No Idle Cost:** เสียเงินเฉพาะวินาทีที่รันงานประมวลผลจริง
- **GPU Power:** เข้าถึง GPU L4 ที่ประสิทธิภาพสูงกว่า Kaggle รุ่นฟรี
- **Clean Architecture:** แยกส่วนประมวลผล (Worker) ออกจากส่วนแสดงผล (Frontend) อย่างชัดเจน
