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

## 3. การเตรียม Google Cloud สำหรับ Backend (Build Worker)

ระบบประมวลผล 3D ต้องใช้ GPU ดังนั้นเราจะใช้ GCP สำหรับการ Build เท่านั้น:

### ก. เปิดใช้งาน API ที่จำเป็น
ใช้คำสั่งผ่าน Google Cloud SDK (gcloud CLI):
```bash
gcloud services enable artifactregistry.googleapis.com \
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

---

## 4. การ Deploy Backend Worker ไปยัง RunPod Serverless

### ก. การเตรียม Docker Image (Build on GCP):
เรารันการ Build บน Cloud โดยใช้คำสั่งเดียว:
```bash
gcloud builds submit --config cloudbuild.yaml .
```

### ข. การตั้งค่าใน RunPod เพื่อดึง Image จาก GCP:
เพื่อให้ RunPod ดึง Image จาก Registry ส่วนตัวของคุณได้:

1.  **สร้าง Service Account ใน GCP:**
    *   ไปที่ IAM & Admin > Service Accounts > สร้างใหม่ชื่อ `runpod-puller`
    *   ให้สิทธิ์ (Role): `Artifact Registry Reader`
    *   สร้าง JSON Key และดาวน์โหลดไว้

2.  **เพิ่ม Registry ใน RunPod Dashboard:**
    *   ไปที่ **User Settings** > **Container Registries** > **Add Registry**
    *   **Registry Domain Name:** `asia-southeast1-docker.pkg.dev`
    *   **Username:** `_json_key`
    *   **Password:** (คัดลอกเนื้อหาทั้งหมดในไฟล์ JSON ที่ดาวน์โหลดมาใส่ที่นี่)

3.  **สร้าง Endpoint ใน RunPod:**
    *   ระบุ Image URL: `asia-southeast1-docker.pkg.dev/[PROJECT_ID]/3d-scan-repo/worker:latest`
    *   เลือก GPU ที่ต้องการ (เช่น **RTX 4090**)
    *   ตั้งค่า Environment Variables:
        *   `SUPABASE_URL`
        *   `SUPABASE_KEY` (แนะนำให้ใช้ Service Role Key)

---

## 5. การจัดการผลลัพธ์ (Retention Policy)

เพื่อให้เป็นไปตามนโยบายการเก็บข้อมูล 24 ชั่วโมง แนะนำให้ตั้งค่าใน **Supabase Storage**:

1.  ไปที่ **Storage** > **Buckets**
2.  เลือก Bucket `scans`
3.  ไปที่ **Policies** (หรือ Bucket Settings ขึ้นอยู่กับเวอร์ชัน Dashboard)
4.  ตั้งค่า **Auto-deletion** (ถ้ามี) หรือใช้ **Supabase Edge Functions** ที่ตั้งเวลา (Cron) ไว้ให้ลบไฟล์ที่มี `created_at` เกิน 24 ชั่วโมง

---

## 6. สรุป Environment Variables ที่สำคัญ

| ส่วนงาน | ชื่อตัวแปร | รายละเอียด |
| :--- | :--- | :--- |
| **Frontend** | `NEXT_PUBLIC_SUPABASE_URL` | URL ของ Supabase Project |
| | `NEXT_PUBLIC_SUPABASE_ANON_KEY` | API Key (Anon) ของ Supabase |
| **Backend** | `SUPABASE_URL` | เหมือนด้านบน |
| | `SUPABASE_KEY` | **Service Role Key** เพื่อสิทธิ์ในการเขียนไฟล์ลง Storage |

---

## 6. การทดสอบหลัง Deploy
1.  เข้าหน้าเว็บ Frontend ที่ Deploy แล้ว
2.  ลองอัปโหลดวิดีโอทดสอบ
3.  ตรวจสอบใน Supabase Table `jobs` ว่ามีข้อมูลเข้าหรือไม่
4.  ตรวจสอบ Log ใน Cloud Run เพื่อดูการทำงานของ Worker
