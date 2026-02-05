# โครงสร้างโปรเจค 3DSCAN สำหรับ Kaggle

โปรเจคนี้ถูกออกแบบมาเพื่อรันกระบวนการ **3D Gaussian Splatting** บน **Kaggle** โดยอัตโนมัติ ตั้งแต่การเตรียมข้อมูลวิดีโอ ไปจนถึงการเทรนโมเดลและส่งออกไฟล์ `.splat` เพื่อนำไปใช้งานต่อ

## 📁 โครงสร้างไฟล์ (File Structure)

| ไฟล์ (File) | หน้าที่ (Description) |
| :--- | :--- |
| **`3d-scan-fixed.ipynb`** | **ตัวรันหลัก (Main Launcher):** Notebook สำหรับรันบน Kaggle ทำหน้าที่ Clone โค้ดจาก GitHub และสั่งรัน `3d-scan.py` |
| **`3d-scan-resume.ipynb`** | **ตัวรันโหมดทำต่อ (Resume Launcher):** Notebook สำหรับการรันต่อจากงานเดิม (Resume) โดยเฉพาะ |
| **`3d-scan.py`** | **โค้ดหลัก (Core Script):** สคริปต์ Python ที่ควบคุม Process การทำงานทั้งหมด ตั้งแต่จัดการ Environment, แปลงวิดีโอ, รัน COLMAP, เทรนโมเดล Splatfacto, และส่งออกไฟล์ .splat |
| `LICENSE` | ไลเซนส์ของโปรเจค (MIT License) |
| `*.txt` | ไฟล์ Log บันทึกผลการทำงาน (เช่น `3d-scan.log.txt`) |

---

## ⚙️ การทำงานบน Kaggle (Workflow)

การทำงานของโปรเจคนี้บน Kaggle ถูกแบ่งออกเป็น 2 ส่วนหลักที่ทำงานประสานกัน:

### 1. `3d-scan-fixed.ipynb` (Launcher)
ทำหน้าที่เป็นตัวจุดชนวนการทำงาน:
1.  **Clone Code:** ดึงโค้ดล่าสุดจาก GitHub `PRIDA-TAKON/3DSCAN` ลงมาใน Kaggle
2.  **Execute Script:** สั่งรันคำสั่ง `python 3d-scan.py` พร้อมบันทึก Log

### 2. `3d-scan.py` (Core Pipeline)
ไฟล์นี้เป็นหัวใจสำคัญ ทำหน้าที่จัดการทุกอย่างเมื่อถูกเรียกใช้:
1.  **Environment Management:**
    *   **Check GPU:** ตรวจสอบ GPU
    *   **Install Dependencies:** ติดตั้ง `colmap`, `ffmpeg`, `nerfstudio`, `plyfile` หากยังไม่มี
    *   **Apply Patches:**
        *   *Patch NumPy:* แก้ปัญหา `ImportError: cannot import name 'broadcast_to'`
        *   *Patch Nerfstudio:* แก้ปัญหาความเข้ากันได้กับ PyTorch 2.6+ (`weights_only=False`)
2.  **Process Data:**
    *   **Find Video:** ค้นหาวิดีโอ .mp4 ใน `/kaggle/input` อัตโนมัติ
    *   **Convert Video:** แปลงวิดีโอเป็นภาพนิ่ง (Extract Frames)
    *   **COLMAP:** รันกระบวนการ Photogrammetry (Structure for Motion)
    *   **Generate JSON:** สร้างไฟล์ `transforms.json` สำหรับ Nerfstudio
3.  **Train Model:** เทรนโมเดลด้วย `ns-train splatfacto`
4.  **Export:** ส่งออกผลลัพธ์เป็นไฟล์ `.splat` ไปยังโฟลเดอร์ Output ของ Kaggle

---

## 🚀 วิธีการใช้งานบน Kaggle

1.  **Create New Notebook:** สร้าง Notebook ใหม่บน Kaggle
2.  **Import Notebook:** อัปโหลดไฟล์ `3d-scan-fixed.ipynb` หรือ copy โค้ดลงไป
3.  **Add Data:**
    *   อัปโหลดวิดีโอที่ต้องการสแกนไปที่ Input Dataset (ระบบจะหาไฟล์อัตโนมัติ)
4.  **Settings:**
    *   เปิด **Internet: On**
    *   เลือก **Accelerator: GPU P100** หรือ **T4 x2**
5.  **Run All:** กด Run All เพื่อเริ่มกระบวนการทั้งหมด

ระบบจะทำงานอัตโนมัติและบันทึกไฟล์ `.splat` ไว้ในโฟลเดอร์ `/kaggle/working/outputs/3d_scan/splatfacto/` เมื่อเสร็จสิ้น
