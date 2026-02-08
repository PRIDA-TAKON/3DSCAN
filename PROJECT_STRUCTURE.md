# โครงสร้างโปรเจค 3DSCAN สำหรับ Kaggle

โปรเจคนี้ถูกออกแบบมาเพื่อ rันกระบวนการ **3D Gaussian Splatting** บน **Kaggle** โดยแบ่งการทำงานออกเป็น 2 ส่วน (Parts) เพื่อแก้ปัญหา Environment Conflict (NumPy version) และเพิ่มความเสถียร

## 📁 โครงสร้างไฟล์ (File Structure)

| ไฟล์/โฟลเดอร์ (File/Folder) | หน้าที่ (Description) |
| :--- | :--- |
| **`3d-scan-part1-data-prep.ipynb`** | **ส่วนที่ 1 (เตรียมข้อมูล):** รันบน Kaggle เพื่อแปลงวิดีโอเป็นภาพ และทำ Sparse Reconstruction (COLMAP) ส่งออกเป็นไฟล์ Zip |
| **`3d-scan-part2-training.ipynb`** | **ส่วนที่ 2 (เทรนโมเดล):** รันบน Kaggle โดยรับไฟล์ Zip จากส่วนที่ 1 มาเทรนโมเดลด้วย Taichi Splatting และส่งออกไฟล์ `.splat` |
| **`taichi-splatting-kaggle/`** | **ไลบรารีเสริม (Submodule):** Fork ของ `taichi_3d_gaussian_splatting` ที่ถูกแก้บั๊ก (Fixed) แล้ว สำหรับใช้ใน Part 2 |
| `scripts/` | **สคริปต์หลัก (Core Scripts):** โฟลเดอร์เก็บไฟล์ Python (`step1` ถึง `step4`) ที่ถูกเรียกใช้โดย Notebooks |
| `LICENSE` | ไลเซนส์ของโปรเจค (MIT License) |

---

## ⚙️ การทำงานบน Kaggle (Workflow)

การทำงานถูกแบ่งเป็น 2 ขั่นตอน เพื่อแยก Environment ออกจากกันอย่างชัดเจน:

### 1. `3d-scan-part1-data-prep.ipynb` (Data Preparation)

* **Environment:** Kaggle Default (รองรับ NumPy 2.x)
* **Input:** วิดีโอ (`.mp4`) หรือ ภาพถ่าย
* **Process:**
    1. Extract Frames (แปลงวิดีโอเป็นภาพ)
    2. COLMAP SfM (สร้าง Sparse Point Cloud)
* **Output:** ไฟล์ `3d_scan_data_part1.zip` (ต้องดาวน์โหลดเก็บไว้)

### 2. `3d-scan-part2-training.ipynb` (Training)

* **Environment:** Custom (Strict NumPy < 2.0, Taichi)
* **Input:** ไฟล์ `3d_scan_data_part1.zip` (จากขั้นตอนที่ 1)
* **Process:**
    1. Install Dependencies (จาก `taichi-splatting-kaggle` fork)
    2. Train Taichi Splatting (เทรนโมเดล)
    3. Export to .splat
* **Output:** ไฟล์ `3d_splat_model.zip` (โมเดล 3D พร้อมใช้งาน)

---

## 🚀 วิธีการใช้งาน (Step-by-Step)

1. **รัน Part 1:**
    * สร้าง Notebook ใหม่ -> Import `3d-scan-part1-data-prep.ipynb`
    * รันจนจบ -> ดาวน์โหลด `output/3d_scan_data_part1.zip`

2. **รัน Part 2:**
    * สร้าง Notebook ใหม่ -> Import `3d-scan-part2-training.ipynb`
    * สร้าง **New Dataset** ใน Kaggle โดยอัปโหลดไฟล์ `3d_scan_data_part1.zip` ที่ได้มา
    * เพิ่ม Dataset นี้เข้าใน Notebook
    * รันจนจบ -> ดาวน์โหลด `3d_splat_model.zip`
