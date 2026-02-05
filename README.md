# 📸 3DSCAN - 3D Gaussian Splatting on Kaggle

โปรเจคสำหรับสร้างโมเดล 3D (.splat) จากไฟล์วิดีโอ โดยรันบน Kaggle ฟรี! รองรับทั้งการเริ่มทำใหม่และการทำต่อจากเดิม (Resume)

---

## 🚀 วิธีการใช้งาน (How to Use)

### 1. การเตรียมตัว (Setup)
1.  สร้าง **New Notebook** ใน Kaggle
2.  เลือก **File -> Import Notebook** แล้วเลือกไฟล์ `3d-scan-fixed.ipynb` (หลัก) หรือ `3d-scan-resume.ipynb` จากโปรเจคนี้
3.  เปิดการตั้งค่าด้านขวา (Settings):
    *   **Internet:** On (เปิดใช้งานอินเทอร์เน็ต)
    *   **Accelerator:** GPU P100 หรือ T4 x2

---

### 2. เลือกโหมดการทำงาน (Modes)

#### 🎬 2.1 โหมดเริ่มใหม่ (New Run)
*ใช้เมื่อ:* คุณมีไฟล์วิดีโอ (.mp4) และต้องการเริ่มกระบวนการตั้งแต่ต้น (แปลงไฟล์ -> สร้าง Point Cloud -> เทรนโมเดล)

1.  **Add Data:** อัปโหลดไฟล์วิดีโอของคุณไปที่ Kaggle Dataset
2.  **ตั้งค่า Path:** ไม่ต้องแก้ไข path วิดีโอ ระบบจะค้นหาไฟล์ .mp4 ใน `/kaggle/input` ให้อัตโนมัติ
3.  **ตั้งค่า Resume:** ปล่อยตัวแปร `RESUME_PATH` ให้ว่างไว้
    ```python
    RESUME_PATH = "" 
    ```
4.  กด **Run All**

#### 🔄 2.2 โหมดทำต่อ (Resume Mode)
*ใช้เมื่อ:* คุณเคยรัน Colmap (สร้าง Sparse Point Cloud) เสร็จแล้ว แต่เทรนไม่จบ หรือต้องการเทรนเพิ่มโดยไม่ต้องเสียเวลาทำ Colmap ใหม่

1.  **เตรียมข้อมูล:** ตรวจสอบว่า Dataset ของคุณมีไฟล์/โฟลเดอร์เหล่านี้ครบ:
    *   `sparse_pc.ply` (สำคัญมาก! Sparse Point Cloud)
    *   `transforms.json` (ข้อมูลตำแหน่งกล้อง)
    *   `images/` (โฟลเดอร์รูปภาพที่แปลงแล้ว)
    *   `sparse/` (โฟลเดอร์โมเดล Colmap)
    *   `database.db`
2.  **Add Data:** เพิ่ม Dataset งานเก่าของคุณเข้ามาใน Notebook
3.  **ตั้งค่า Path:** ในโค้ด `3d-scan-fixed.ipynb` (หรือ `3d-scan-resume.ipynb`) ให้ใส่ Path ของโฟลเดอร์นั้น
    ```python
    # ตัวอย่าง
    RESUME_PATH = "/kaggle/input/my-old-scan/car_scan"
    ```
4.  กด **Run All**
    *   *ระบบจะข้ามขั้นตอน Colmap ให้อัตโนมัติ และเริ่ม Training ทันที*

---

## 📂 ผลลัพธ์ (Outputs)
เมื่อทำงานเสร็จสิ้น ไฟล์โมเดลจะถูกบันทึกอยู่ที่:
`/kaggle/working/outputs/3d_scan/splatfacto/.../config.yml` และไฟล์ `.splat`

คุณสามารถดาวน์โหลดไฟล์ `.splat` ไปเปิดดูใน Viewer (เช่น Polycam หรือ Splat Viewer) ได้ทันที

---

## ❓ การแก้ปัญหา (Troubleshooting)
*   **Error: Read PLY failed / sparse_pc.ply not found:**
    *   ตรวจสอบว่าใน Dataset ที่นำมา Resume มีไฟล์ `sparse_pc.ply` อยู่จริง
    *   ถ้าไม่มี ให้ลองกลับไปรันโหมด New Run ใหม่ให้จบขั้นตอน Colmap
*   **Code ไม่อัปเดต (Old Code Detected):**
    *   โค้ดบน Kaggle อาจจะยังเป็นเวอร์ชั่นเก่า
    *   ให้ทำการ `git push` โค้ดล่าสุดจากเครื่องคอมของคุณขึ้น GitHub ก่อน แล้วค่อยกดรันใน Kaggle ใหม่

    ตัวอย่างผลลัพธ์ https://superspl.at/view?id=fa12962d
