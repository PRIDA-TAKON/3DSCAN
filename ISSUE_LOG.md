# 📔 บันทึกการแก้ไขปัญหา (Issue & Fix Log) - 02_3DSCAN

เอกสารฉบับนี้ใช้สำหรับติดตามสถานะการพัฒนา ระบบจะอัปเดตเมื่อพบปัญหาใหม่หรือแก้ไขปัญหาเดิมสำเร็จ

---

## 🛰️ สถานะภาพรวม (Executive Summary)
- **ขั้นตอน PROCESS (SfM):** ✅ **สมบูรณ์ (Stable)** - ทำงานได้ 100% ไม่พบปัญหา
- **ขั้นตอน TRAIN (Splatting):** 🟡 **กำลังปรับปรุง (In Progress)** - ติดปัญหาเรื่อง Syntax คำสั่งและสภาพแวดล้อมบน Cloud

---

## 🛠️ รายรายการปัญหาและการแก้ไข (Issue Tracking)

### [ISSUE-001] Trainer ทำงานผิดโหมด (Reset to PROCESS)
- **อาการ:** สั่งเทรนแต่เครื่องกลับไปเริ่มแตกไฟล์วิดีโอและทำ COLMAP ใหม่
- **สาเหตุ:** Environment Variable `WORKER_MODE` ใน RunPod ถูก Cache หรือโดน CI/CD เขียนทับเป็นค่าเริ่มต้น
- **การแก้ไข:** เพิ่มระบบ **Request-Based Mode** ให้ Worker รับค่า `mode: "TRAIN"` จาก JSON Payload ได้โดยตรง (Override ค่าในเครื่อง)
- **สถานะ:** ✅ **แก้ไขแล้ว (v1.0.2)**

### [ISSUE-002] ดาวน์โหลดไฟล์จาก S3 ไม่สำเร็จ (SignatureDoesNotMatch)
- **อาการ:** Worker แจ้งว่ารหัส S3 ไม่ถูกต้องตอนพยายามโหลด `processed.zip`
- **สาเหตุ:** การก๊อบปี้ Access Key/Secret Key ลงใน Endpoint ใหม่มีความผิดพลาด
- **การแก้ไข:** ผู้ใช้ทำการตรวจสอบและอัปเดตค่าใน RunPod Dashboard ให้ถูกต้อง
- **สถานะ:** ✅ **แก้ไขแล้ว**

### [ISSUE-003] ข้อมูล S3_PATH ในฐานข้อมูลหายไป
- **อาการ:** เมื่อการเทรนพลาด ระบบเขียนทับช่อง `message` ด้วย Error ทำให้ลบ Path ของไฟล์ S3 ทิ้ง
- **สาเหตุ:** Logic การอัปเดตสถานะไม่ได้เก็บค่าเดิมไว้
- **การแก้ไข:** ปรับปรุงโค้ดให้ทำการแนบ `S3_PATH` กลับไปในทุกการอัปเดตสถานะ (Training, Failed, Exporting)
- **สถานะ:** ✅ **แก้ไขแล้ว (v1.0.4)**

### [ISSUE-004] ns-train ติดคำถามโต้ตอบ (EOFError)
- **อาการ:** โปรแกรมหยุดทำงานที่คำถาม `Would you like to downscale images? [y/n]`
- **สาเหตุ:** การรันบน Serverless ไม่มีหน้าจอให้โต้ตอบ
- **การแก้ไข:** เพิ่ม Flag บังคับไม่ให้ถาม (Non-interactive) และกำหนด Scale คงที่
- **สถานะ:** ✅ **แก้ไขแล้ว (v1.0.5)**

### [ISSUE-005] ns-train Syntax Error (Unrecognized options)
- **อาการ:** โปรแกรมฟ้องว่าไม่รู้จักคำสั่ง หรือวาง Argument ผิดตำแหน่ง
- **สาเหตุ:** Nerfstudio บังคับลำดับ Argument อย่างเคร่งครัด (Method Flags ต้องอยู่ก่อน Subcommand `colmap` และ Dataparser Flags ต้องอยู่หลังสุด)
- **การแก้ไข:** ปรับโครงสร้างคำสั่งใหม่ตามเอกสาร Nerfstudio Documentation (v1.0.7)
- **สถานะ:** ✅ **แก้ไขแล้ว (v1.0.7-doc-aligned)**

---

## 📈 ประวัติเวอร์ชัน (Version History)
- **v1.0.1:** เริ่มต้นระบบ Hybrid (Git + Docker)
- **v1.0.2:** เพิ่มระบบ Mode Override ผ่าน Request
- **v1.0.3:** พยายามแก้ปัญหา Non-interactive (ติดเรื่องลำดับ Syntax)
- **v1.0.4:** เพิ่ม Robust Logging และระบบรักษา S3_PATH ใน DB
- **v1.0.5:** แก้ไขลำดับ Syntax คำสั่ง `ns-train` ให้ถูกต้องตามมาตรฐาน Nerfstudio

---
*หมายเหตุ: หากพบปัญหาใหม่ ให้เพิ่มหัวข้อ [ISSUE-XXX] ต่อท้ายรายการนี้*
