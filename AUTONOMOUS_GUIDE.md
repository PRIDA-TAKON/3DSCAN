# 🤖 Autonomous Development & Deployment Guide (3DSCAN)

คู่มือนี้สรุปขั้นตอนการใช้งานระบบ **Autonomous Loop** สำหรับการพัฒนาและแก้ไขบั๊กบน Cloud GPU (RunPod) โดยอัตโนมัติ

---

## 🛠️ 1. โครงสร้างระบบ (System Architecture)

ระบบของเราถูกออกแบบมาเพื่อลดเวลาในการ Build Docker Image และแก้ไขบั๊กได้รวดเร็ว:

1.  **Code Loader (`loader.py`)**: อยู่ใน Docker Image ทำหน้าที่ดึงโค้ดล่าสุดจาก Git ทุกครั้งที่คอนเทนเนอร์เริ่มทำงาน ทำให้เราแก้ไขไฟล์ `.py` ได้โดยไม่ต้อง Build Image ใหม่
2.  **Test Cycle (`scripts/test_cycle.py`)**: สั่ง Trigger งานไปยัง RunPod และเฝ้าดู Log (stdout) จนกว่างานจะสำเร็จหรือล้มเหลว
3.  **Autonomous Orchestrator (`scripts/autonomous_orchestrator.py`)**: (Optional) บอทที่จะคอยเฝ้าดู Error Log ใน Supabase และเรียก AI มาแก้ไข `Dockerfile` อัตโนมัติ

---

## 🚀 2. ขั้นตอนการทำงาน (The Development Cycle)

เมื่อต้องการทดสอบหรือแก้ไขโค้ด ให้ทำตามลูปนี้:

### Step A: แก้ไขโค้ดและ Push ขึ้น Git
แก้ไขไฟล์ที่ต้องการ (เช่น `takon_3d_worker.py` หรือ `loader.py`) แล้วดันขึ้น GitHub:
```bash
git add .
git commit -m "update: fix logic in worker"
git push origin main
```

### Step B: สั่งเริ่มการทดสอบ (Trigger & Monitor)
รันสคริปต์เพื่อสั่ง RunPod ให้ทำงานและดึง Log มาแสดงผล:
```bash
# ตรวจสอบให้แน่ใจว่ามีไฟล์ .env พร้อมใช้งาน
python scripts/test_cycle.py
```
*สคริปต์นี้จะแสดงสถานะ (Status) และ Log (Stdout) แบบ Real-time*

### Step C: วิเคราะห์และแก้ไข (Analyze & Fix)
*   **หากสำเร็จ (COMPLETED)**: จบงาน! 🎉
*   **หากล้มเหลว (FAILED)**: 
    1. อ่าน Log ที่ปรากฏในหน้าจอ
    2. แก้ไขโค้ดในไฟล์ `.py` หรือ `Dockerfile` ตามสาเหตุที่พบ
    3. กลับไปที่ **Step A** เพื่อเริ่มรอบใหม่

---

## 📋 3. สิ่งที่จำเป็น (Prerequisites)

ต้องมีไฟล์ `.env` ใน Root Directory พร้อมค่าดังนี้:

```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_or_service_key
RUNPOD_API_KEY=your_runpod_api_key
RUNPOD_ENDPOINT_ID=your_endpoint_id
GITHUB_TOKEN=your_github_personal_access_token (สำหรับเช็ค Build Status)
```

---

## 💡 เคล็ดลับ (Pro-tips)

*   **ไม่ต้อง Build Image ใหม่**: หากคุณแก้ไขเฉพาะไฟล์ `.py` คุณไม่ต้องรอ GitHub Actions บิลด์เสร็จ (เพราะ `loader.py` จะดึงไฟล์ล่าสุดมาเอง) แค่สั่ง `test_cycle.py` ได้เลย
*   **Force Rebuild**: หากแก้ไข `Dockerfile` ต้องรอ GitHub Actions บิลด์เสร็จก่อน (ประมาณ 5-10 นาที) และควรสั่ง **Update Endpoint** ในหน้าเว็บ RunPod เพื่อล้าง Cache ของอิมเมจเก่า
*   **การดู Log ย้อนหลัง**: หากพลาด Log ในหน้าจอ สามารถเข้าไปดูได้ที่ตาราง `runpod_logs` ใน Supabase Dashboard

---
*เอกสารนี้สร้างขึ้นโดย Gemini CLI เพื่อเป็นคู่มือมาตรฐานสำหรับโครงการ 02_3DSCAN*
