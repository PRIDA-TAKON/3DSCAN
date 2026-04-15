# 📚 เอกสารอ้างอิงการตั้งค่า Nerfstudio (Splatfacto)

รวบรวมข้อมูล Syntax และตัวเลือกที่ถูกต้องตามมาตรฐานเอกสารอย่างเป็นทางการของ **Nerfstudio** เพื่อใช้ในโปรเจกต์ 02_3DSCAN

---

## 🛠️ โครงสร้างคำสั่ง (Command Structure)

คำสั่ง `ns-train` มีลำดับการวาง Argument ที่เข้มงวด:

```bash
ns-train <method> [method_flags] <dataparser> [dataparser_flags]
```

### ตัวอย่างที่ถูกต้องสำหรับโปรเจกต์เรา:
```bash
ns-train splatfacto --max-num-iterations 2000 colmap --data . --downscale-factor 1
```

---

## ⚙️ ตัวเลือกที่สำคัญ (Key Options)

### 1. Method: `splatfacto` (Gaussian Splatting)
ตัวเลือกเหล่านี้ต้องวาง **"หน้า"** คำว่า `colmap`:
- `--max-num-iterations <int>`: จำนวนรอบการเทรน (แนะนำ 2000 สำหรับ Quick Test, 30000 สำหรับงานจริง)
- `--vis <string>`: ระบบแสดงผล (เช่น `tensorboard`, `wandb`, `viewer`)
- `--output-dir <path>`: โฟลเดอร์เก็บผลลัพธ์

### 2. Dataparser: `colmap`
ตัวเลือกเหล่านี้ต้องวาง **"หลัง"** คำว่า `colmap`:
- `--data <path>`: พาธที่เก็บข้อมูลรูปภาพและไฟล์ SfM (ค่าเริ่มต้นคือ `.`)
- `--colmap-path <path>`: พาธไปยังไฟล์ sparse ของ COLMAP (เช่น `colmap/sparse/0`)
- `--images-path <path>`: พาธไปยังโฟลเดอร์รูปภาพ (เช่น `images`)
- `--downscale-factor <int>`: **สำคัญ!** บังคับการลดขนาดภาพ หากระบุจะเป็นการปิดการถามตอบแบบอัตโนมัติ (ข้าม EOFError)
    - `1`: ขนาดเท่าเดิม (ไม่ย่อ)
    - `2`, `4`, `8`: ย่อตามสัดส่วน
- `--load-3D-points <True/False>`: ใช้ Point Cloud จาก COLMAP เป็นจุดเริ่มต้น (Gaussian Init)

---

## 🆘 วิธีแก้ปัญหาที่พบบ่อย (Troubleshooting)

### ปัญหา EOFError / ค้างที่คำถาม [y/n]
**สาเหตุ:** Nerfstudio ตรวจพบว่าภาพมีขนาดใหญ่ และต้องการย่อภาพโดยอัตโนมัติ จึงถามยืนยันกับผู้ใช้
**การแก้ไข:** ต้องระบุ `--downscale-factor` ในส่วนท้ายของคำสั่งเสมอเมื่อรันบน Serverless

### ปัญหา Unrecognized options
**สาเหตุ:** วางตำแหน่ง Argument ผิดระดับ (เช่น เอาตัวเลือกของ dataparser ไปวางหน้าชื่อ method)
**การแก้ไข:** ย้าย Flag ที่ฟ้อง Error ไปไว้หลังคำสั่ง `colmap`

---
*ข้อมูลอ้างอิงจาก: [Nerfstudio Documentation (docs.nerf.studio)](https://docs.nerf.studio)*
