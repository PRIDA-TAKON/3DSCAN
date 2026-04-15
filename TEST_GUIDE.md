# 🚀 คู่มือการรันคำสั่งทดสอบ (Test Guide)

รวบรวมคำสั่งที่จำเป็นสำหรับการรันระบบเทรนเนอร์และตรวจสอบสถานะงานในโปรเจกต์ **02_3DSCAN**

## 🛠️ 1. เตรียมสภาพแวดล้อม (Preparation)
ก่อนรันคำสั่งใดๆ ตรวจสอบให้แน่ใจว่าได้สร้าง Virtual Environment และติดตั้งไลบรารีที่จำเป็นแล้ว:

```powershell
# สร้าง venv (ถ้ายังไม่มี)
python -m venv venv

# ติดตั้งไลบรารีที่จำเป็น
.\venv\Scripts\pip.exe install requests python-dotenv
```

---

## 🧠 2. การสั่งรันเทรนเนอร์ (Trigger Trainer)
ใช้สำหรับสั่งให้ RunPod เริ่มกระบวนการเทรนทันที:

```powershell
.\venv\Scripts\python.exe scripts/trigger_trainer_now.py
```
*สคริปต์นี้จะส่ง Request พร้อมระบุ `mode: "TRAIN"` เพื่อบังคับให้ Worker ทำงานในโหมดเทรนทันที*

---

## ⚙️ 3. กลไกการเปลี่ยนโหมด (Worker Modes)
ระบบแบ่งการทำงานเป็น 2 โหมดหลัก โดยใช้โค้ดชุดเดียวกัน (`takon_3d_worker.py`):

### A. โหมด PROCESS (SfM / COLMAP)
- **หน้าที่:** ดาวน์โหลดวิดีโอ -> แตกเฟรม -> ทำ COLMAP -> รวมไฟล์เป็น `processed.zip` -> อัปโหลดขึ้น S3
- **การเปิดใช้งาน:** ตั้งค่า Env ใน RunPod `WORKER_MODE=PROCESS` (ค่าเริ่มต้น)

### B. โหมด TRAIN (Gaussian Splatting)
- **หน้าที่:** อ่าน Path จาก DB -> ดาวน์โหลด `processed.zip` จาก S3 -> รัน `ns-train` -> ส่งออกโมเดล `.ply`
- **การเปิดใช้งาน:** ตั้งค่า Env ใน RunPod `WORKER_MODE=TRAIN` หรือ **ส่งผ่าน Request**

### 🆘 วิธีแก้ปัญหาเมื่อโหมดไม่เปลี่ยน (Mode Override)
หากคุณแก้ไข Environment Variable ใน RunPod แล้วระบบยังทำงานผิดโหมด (เนื่องจากระบบ Cache หรือโดนเขียนทับ) ให้ใช้วิธี **Override ผ่าน Request** โดยการส่ง JSON Payload ดังนี้:

```json
{
  "input": {
    "id": "JOB_ID",
    "mode": "TRAIN" 
  }
}
```
*โค้ดใน `takon_3d_worker.py` จะให้ความสำคัญกับค่า `mode` ใน Request มากกว่าค่าใน Environment ของเครื่อง*

---

## 📊 4. ตรวจสอบสถานะงาน (Check Job Status)
ตรวจสอบสถานะงานที่รันอยู่ใน Supabase:

### ตรวจสอบ 5 งานล่าสุด:
```powershell
.\venv\Scripts\python.exe scripts/check_job_status.py
```

### ตรวจสอบงานล่าสุดเพียงงานเดียว:
```powershell
.\venv\Scripts\python.exe scripts/get_latest_status.py
```

---

## 📝 หมายเหตุ (Notes)
- **Version Tag:** โค้ดรุ่นปัจจุบันคือ `v1.0.2` (ตรวจสอบได้ใน Log ของ Worker)
- **Restart Pod:** ทุกครั้งที่แก้โค้ดหรือต้องการล้าง Cache ของโหมดเก่า แนะนำให้กด **Stop/Restart** Endpoint ใน RunPod เสมอ
- **PowerShell:** คำสั่งทั้งหมดออกแบบมาให้รันบน PowerShell ใน Windows
