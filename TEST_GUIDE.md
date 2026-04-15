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
ใช้สำหรับสั่งให้ RunPod เริ่มกระบวนการเทรนทันทีโดยใช้ `JOB_ID` และ `VIDEO_URL` ที่ระบุไว้ในสคริปต์:

```powershell
.\venv\Scripts\python.exe scripts/trigger_trainer_now.py
```
*สคริปต์นี้จะทำหน้าที่ส่ง Request ไปยัง RunPod API และ Monitor สถานะจนกว่าจะจบงาน*

---

## 📊 3. ตรวจสอบสถานะงานล่าสุด (Check Job Status)
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

## 🔄 4. รันการทดสอบครบวงจร (Full Test Cycle)
ใช้สำหรับรันสคริปต์ทดสอบภาพรวมของระบบ:

```powershell
.\venv\Scripts\python.exe scripts/test_cycle.py
```

---

## 📝 หมายเหตุ (Notes)
- **.env:** ตรวจสอบให้แน่ใจว่ามีไฟล์ `.env` ที่มีคีย์ `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID_TRAINER`, `SUPABASE_URL`, และ `SUPABASE_KEY` ครบถ้วน
- **PowerShell:** คำสั่งทั้งหมดออกแบบมาให้รันบน PowerShell ใน Windows
