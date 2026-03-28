# คู่มือการ Deploy ระบบ 02_3DSCAN

เอกสารนี้สรุปขั้นตอนการนำระบบ 02_3DSCAN ขึ้นใช้งานจริง (Production) ทั้งส่วนหน้าบ้าน (Frontend) และระบบประมวลผล (Backend Worker)

---

## 1. การเตรียมฐานข้อมูล (Supabase Setup)

ก่อนเริ่มการ Deploy ส่วนอื่น คุณต้องเตรียมฐานข้อมูลเพื่อเก็บสถานะของงาน (Jobs)

1.  สร้างโปรเจกต์ใหม่ใน [Supabase Dashboard](https://supabase.com/)
2.  ไปที่เมนู **SQL Editor**
3.  คัดลอกเนื้อหาจากไฟล์ `supabase_schema.sql` ในโปรเจกต์นี้ไปวางแล้วกด **Run**
4.  จดจำค่า **Project URL** และ **Anon Key** เพื่อใช้ในขั้นตอนถัดไป

---

## 2. การ Deploy Frontend (Next.js)

แนะนำให้ใช้ **Vercel** เนื่องจากรองรับ Next.js ได้สมบูรณ์ที่สุด

1.  Push โค้ดทั้งหมดขึ้น GitHub/GitLab
2.  Import โปรเจกต์เข้า Vercel โดยเลือก Root Directory เป็นโฟลเดอร์ `frontend`
3.  ตั้งค่า **Environment Variables** ดังนี้:
    *   `NEXT_PUBLIC_SUPABASE_URL`: (จาก Supabase Project)
    *   `NEXT_PUBLIC_SUPABASE_ANON_KEY`: (จาก Supabase Project)
4.  กด **Deploy**

---

## 3. การเตรียม Google Cloud สำหรับ Backend (Cloud Run Worker)

ระบบประมวลผล 3D ต้องใช้ GPU ดังนั้นต้องตั้งค่าใน GCP ดังนี้:

### ก. เปิดใช้งาน API ที่จำเป็น
ใช้คำสั่งผ่าน Google Cloud SDK (gcloud CLI):
```bash
gcloud services enable run.googleapis.com \
                       artifactregistry.googleapis.com \
                       cloudbuild.googleapis.com
```

### ข. สร้าง Artifact Registry
เพื่อเก็บ Docker Image ของ Worker:
```bash
gcloud artifacts repositories create 3d-scan-repo \
    --repository-format=docker \
    --location=asia-southeast1 \
    --description="Docker repository for 3D Scan Worker"
```

### ค. เตรียม Service Account สำหรับ Google Drive
หากระบบมีการอัปโหลดไฟล์ไปที่ Google Drive:
1.  สร้าง Service Account ใน GCP Console (IAM & Admin > Service Accounts)
2.  สร้าง JSON Key และดาวน์โหลดไว้
3.  นำเนื้อหาในไฟล์ JSON นั้นมาเตรียมไว้สำหรับใส่ใน Environment Variable `GDRIVE_SERVICE_ACCOUNT`

---

## 4. การ Deploy Backend Worker ไปยัง RunPod Serverless

การใช้ RunPod Serverless จะช่วยประหยัดค่าใช้จ่ายได้มาก เพราะจ่ายตามการใช้งานจริง (GPU Seconds)

### ก. การเตรียม Docker Image (Build on GCP):
เราจะใช้ **Google Cloud Build** ในการสร้างและเก็บ Image ไว้บน **Artifact Registry** ของคุณโดยตรง

1.  ตรวจสอบว่าคุณได้สร้าง Repository แล้ว (ตามขั้นตอนในข้อ 3 ข)
2.  รันคำสั่งเพื่อเริ่มการ Build บน Cloud:
    ```bash
    gcloud builds submit --config cloudbuild.yaml .
    ```
3.  เมื่อ Build เสร็จสิ้น Image ของคุณจะอยู่ที่:
    `asia-southeast1-docker.pkg.dev/[PROJECT_ID]/3d-scan-repo/worker:latest`

### ข. การตั้งค่าใน RunPod เพื่อดึง Image จาก GCP:
เนื่องจาก Artifact Registry เป็นแบบส่วนตัว คุณต้องอนุญาตให้ RunPod เข้าถึงได้:

1.  **สร้าง Service Account ใน GCP:**
    *   ไปที่ IAM & Admin > Service Accounts
    *   สร้างใหม่ชื่อ `runpod-puller`
    *   ให้สิทธิ์ (Role): `Artifact Registry Reader`
    *   ไปที่แท็บ **Keys** > **Add Key** > **Create new key** (เลือกประเภท JSON)
    *   ดาวน์โหลดไฟล์ JSON เก็บไว้

2.  **เพิ่ม Registry ใน RunPod Dashboard:**
    *   ไปที่ **User Settings** > **Container Registries** > **Add Registry**
    *   **Registry Domain Name:** `asia-southeast1-docker.pkg.dev`
    *   **Username:** `_json_key`
    *   **Password:** (คัดลอกเนื้อหาทั้งหมดในไฟล์ JSON ที่ดาวน์โหลดมาใส่ที่นี่)

3.  **สร้าง Endpoint ใน RunPod:**
    *   ไปที่ **Serverless** > **Endpoints** > **New Endpoint**
    *   ระบุ Image URL: `asia-southeast1-docker.pkg.dev/[PROJECT_ID]/3d-scan-repo/worker:latest`
    *   เลือก GPU ที่ต้องการ (เช่น **RTX 4090**)
    *   ตั้งค่า Environment Variables (`SUPABASE_URL`, `SUPABASE_KEY`, ฯลฯ)

### ค. การเรียกใช้งาน (Triggering):
เมื่อมีงานใหม่ใน Supabase คุณต้องส่ง POST Request ไปยัง RunPod Endpoint URL (สามารถใช้สคริปต์ `scripts/trigger_runpod.py` หรือตั้งค่าใน Supabase Edge Functions ได้)

---

## 5. สรุป Environment Variables ที่สำคัญ

| ส่วนงาน | ชื่อตัวแปร | รายละเอียด |
| :--- | :--- | :--- |
| **Frontend** | `NEXT_PUBLIC_SUPABASE_URL` | URL ของ Supabase Project |
| | `NEXT_PUBLIC_SUPABASE_ANON_KEY` | API Key (Anon) ของ Supabase |
| **Backend** | `SUPABASE_URL` | เหมือนด้านบน (ใช้เชื่อมต่อจาก Worker) |
| | `SUPABASE_KEY` | แนะนำให้ใช้ Service Role Key เพื่อสิทธิ์ในการเขียน Table |
| | `GDRIVE_SERVICE_ACCOUNT` | เนื้อหาในไฟล์ JSON Key ของ GCP Service Account |

---

## 6. การทดสอบหลัง Deploy
1.  เข้าหน้าเว็บ Frontend ที่ Deploy แล้ว
2.  ลองอัปโหลดวิดีโอทดสอบ
3.  ตรวจสอบใน Supabase Table `jobs` ว่ามีข้อมูลเข้าหรือไม่
4.  ตรวจสอบ Log ใน Cloud Run เพื่อดูการทำงานของ Worker
