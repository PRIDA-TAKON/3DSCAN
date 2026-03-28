# โปรเจกต์ 02_3DSCAN: รายงานสถานะและการดำเนินการต่อ

## 📅 ข้อมูลล่าสุด: 25 มีนาคม 2569

---

### ✅ สิ่งที่ทำไปแล้ว (Completed)

#### 1. สภาพแวดล้อมและโปรเจกต์ (Local Setup)
- [x] ติดตั้ง Node.js Dependencies (Root & Frontend)
- [x] สร้าง Python 3.11 Virtual Environment
- [x] ติดตั้ง Libraries สำคัญ (Taichi, Torch+CUDA, Supabase, RunPod, OpenCV)
- [x] ตั้งค่า `taichi-splatting-kaggle` เป็น editable mode

#### 2. ส่วนงานหลังบ้าน (Backend / Worker)
- [x] สร้าง `runpod_worker.py`: รองรับ RunPod Serverless + Supabase Storage
- [x] ปรับปรุง `Dockerfile`: เตรียมพร้อมสำหรับการ Deploy ขึ้น Cloud
- [x] สร้าง `scripts/trigger_runpod.py`: สคริปต์สำหรับสั่งงานผ่าน API

#### 3. ส่วนงานฐานข้อมูล (Database)
- [x] สร้างตาราง `jobs` ใน Supabase
- [x] ตั้งค่า Storage Bucket `scans` และสิทธิ์การเข้าถึง (Policies)

---

### 📋 สิ่งที่ผู้ใช้ต้องทำ (User Actions Needed)

#### 1. Docker (ที่เครื่องผู้ใช้)
- [ ] รันคำสั่ง `docker build` เพื่อสร้าง Image
- [ ] รันคำสั่ง `docker push` เพื่อส่ง Image ขึ้น Docker Hub

#### 2. RunPod Dashboard (ตั้งค่าผ่านเว็บ)
- [ ] สร้าง Serverless Endpoint ใหม่
- [ ] เลือก GPU (RTX 4090)
- [ ] ตั้งค่า Environment Variables:
    - `SUPABASE_URL`
    - `SUPABASE_KEY` (ใช้ Service Role Key)

#### 3. การตั้งค่าใน Env Local (เพื่อทดสอบ)
- [ ] ใส่ค่า `RUNPOD_API_KEY` และ `RUNPOD_ENDPOINT_ID` ในเครื่องเพื่อใช้ทดสอบการ Trigger

---

### 🛠️ ขั้นตอนถัดไป (Future Tasks)
1. แก้ไข Frontend ให้เรียกใช้ RunPod API อัตโนมัติ
2. สร้างหน้า Dashboard สำหรับดูสถานะงานและดาวน์โหลดผลลัพธ์
3. ตั้งค่าระบบลบไฟล์อัตโนมัติ (Cleanup Script)
