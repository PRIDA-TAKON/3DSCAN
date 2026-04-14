# 🌐 GCP Transition & Cloud Run GPU Plan

เอกสารฉบับนี้บันทึกสถานะการเตรียมความพร้อมในการย้ายระบบจาก RunPod มายัง Google Cloud Platform (GCP)

## 📍 สถานะปัจจุบัน (Current Status)
- [x] **Artifact Registry (Singapore):** สร้างแล้วที่ `asia-southeast1-docker.pkg.dev/mcp-gantt/worker-3d-scan-repo/worker-3d-scan`
- [x] **Artifact Registry (US-Central):** สร้างแล้วและทำการ Mirror Image มาไว้ที่ `us-central1-docker.pkg.dev/mcp-gantt/worker-3d-scan-us-repo/worker-3d-scan` เพื่อลดค่า Data Transfer
- [x] **Image Migration:** อิมเมจฐาน (10GB+) ถูกย้ายมาฝากไว้บน GCP เรียบร้อยแล้ว (ช่วยให้ Cold Start เร็วขึ้น)
- [ ] **GPU Quota (NVIDIA L4):** อยู่ระหว่างรอการอนุมัติจาก Google Cloud Support (ภูมิภาค `us-central1`)
- [ ] **Cloud Run GPU Service:** เตรียมไฟล์ `cloudrun_deploy.yaml` ไว้พร้อมสำหรับ Deploy ทันทีที่โควตาอนุมัติ

## 🧪 แผนการทดสอบปัจจุบัน (Current Testing)
- **Primary Environment:** ใช้ **RunPod Serverless** สำหรับการพัฒนาและทดสอบ (Development/Testing) เป็นหลักในตอนนี้
- **Workflow:** 
    1. แก้ไขโค้ดในเครื่อง/Git
    2. `git push` ขึ้น GitHub
    3. RunPod ดึงโค้ดผ่าน `loader.py` (Code Loader) อัตโนมัติ
    4. ตรวจสอบผลลัพธ์ผ่าน `scripts/test_cycle.py`

## 🚀 แผนการย้าย (Future Migration)
1. เมื่อได้รับอนุมัติโควตา GPU บน GCP ให้รันการ Deploy ผ่าน Cloud Build
2. เปลี่ยน Webhook หรือ Trigger จาก RunPod API มาเป็น Cloud Run URL
3. ใช้ระบบ **Code Loader** ต่อไปบน GCP เพื่อประหยัดค่า Build และเพิ่มความเร็วในการอัปเดต

---
*จัดทำโดย Gemini CLI - 7 เมษายน 2569*
