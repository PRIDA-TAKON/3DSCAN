# 📸 3DSCAN - 3D Gaussian Splatting on Kaggle (Split Version)

โปรเจคสำหรับสร้างโมเดล 3D (.splat) จากไฟล์วิดีโอ โดยใช้เทคนิค **3D Gaussian Splatting** บน **Kaggle**
**เวอร์ชันใหม่:** แบ่งกระบวนการออกเป็น 2 ส่วน (Parts) เพื่อแก้ปัญหา NumPy Conflict และเพิ่มความเสถียรในการทำงาน

---

## 🚀 วิธีการใช้งาน (Step-by-Step Guide)

### ส่วนที่ 1: เตรียมข้อมูล (Part 1 - Data Prep)

*ไฟล์: `3d-scan-part1-data-prep.ipynb`*

1. **Create New Notebook** ใน Kaggle
2. **Import Notebook:** อัปโหลดหรือ Copy โค้ดจาก `3d-scan-part1-data-prep.ipynb`
3. **Add Data:** อัปโหลดวิดีโอ (`.mp4`) หรือโฟลเดอร์รูปภาพ
4. **Run All:** กดรันจนจบกระบวนการ
    * *สิ่งที่ทำ:* ระบบจะแยกเฟรมจากวิดีโอ และรัน COLMAP (Structure-from-Motion)
5. **Download Output:**
    * ให้ดาวน์โหลดไฟล์ `3d_scan_data_part1.zip` จากโฟลเดอร์ Output เก็บไว้ที่เครื่องคอมพิวเตอร์ของคุณ

---

### ส่วนที่ 2: เทรนโมเดล (Part 2 - Training)

*ไฟล์: `3d-scan-part2-training.ipynb`*

**สิ่งสำคัญ:** ส่วนนี้จะใช้ไลบรารีพิเศษที่ถูกแก้บั๊กแล้ว (`taichi-splatting-kaggle`) โดยอัตโนมัติ

1. **Create New Notebook** ใน Kaggle
2. **Import Notebook:** อัปโหลดหรือ Copy โค้ดจาก `3d-scan-part2-training.ipynb`
3. **Add Data (สำคัญ!):**
    * สร้าง **New Dataset** ใน Kaggle โดยอัปโหลดไฟล์ `3d_scan_data_part1.zip` ที่ได้จาก Part 1
    * กดปุ่ม **Add Data** ใน Notebook แล้วเลือก Dataset ที่เพิ่งสร้าง
4. **Settings:**
    * **Internet:** On (ต้องเปิดเน็ตเพื่อดาวน์โหลด Dependencies)
    * **Accelerator:** GPU T4 x2 (แนะนำ) หรือ P100
5. **Run All:** กดรันจนจบ
    * *สิ่งที่ทำ:* ระบบจะติดตั้ง Taichi Splatting (เวอร์ชัน Fixed), แตกไฟล์ Zip, และเริ่มเทรนโมเดล
6. **Download Model:**
    * เมื่อเสร็จสิ้น ให้ดาวน์โหลดไฟล์ `3d_splat_model.zip` (ข้างในมีไฟล์ `.splat`)
    * นำไฟล์ `.splat` ไปเปิดดูใน [Polycam Viewer](https://poly.cam/tools/viewer) หรือ [Splat Viewer](https://splat.antimatter.ai/) ได้เลย!

---

## ❓ คำถามที่พบบ่อย (FAQ)

* **ทำไมต้องแบ่งเป็น 2 ส่วน?**
  * เพื่อให้ **Environment ไม่ตีกัน** ครับ (Part 1 ใช้ NumPy รุ่นใหม่ได้ แต่ Part 2 ต้องการ NumPy รุ่นเก่า < 2.0 อย่างเคร่งครัด) การแยก Notebook ช่วยให้แต่ละส่วนทำงานได้อย่างมีประสิทธิภาพสูงสุด
* **Taichi Splatting คืออะไร?**
  * เป็นไลบรารีสำหรับสร้าง 3D Gaussian Splatting ที่เขียนด้วยภาษา Taichi ซึ่งทำงานได้เร็วกว่าและติดตั้งง่ายกว่าเวอร์ชันดั้งเดิม
* **มีบั๊ก "Non contiguous tensors" หรือไม่?**
  * **แก้แล้วครับ!** ใน Part 2 เราใช้ Repository พิเศษ (`taichi-splatting-kaggle`) ที่ถูกแก้บั๊กนี้เรียบร้อยแล้ว

---

## 📂 ลิงก์ที่เกี่ยวข้อง

* **Repository หลัก:** `https://github.com/PRIDA-TAKON/3DSCAN`
* **Library Fork:** `https://github.com/PRIDA-TAKON/taichi-splatting-kaggle` (สำหรับดู Code ที่แก้แล้ว)
