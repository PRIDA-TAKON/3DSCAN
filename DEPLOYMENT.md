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

## 3. การเตรียม GitHub สำหรับ Backend (Auto-Build)

เราจะใช้ **GitHub Actions** ในการบิ้ว Docker Image โดยอัตโนมัติทุกครั้งที่มีการ Push โค้ด:

1.  ไปที่ GitHub Repository ของคุณ > **Settings** > **Secrets and variables** > **Actions**
2.  เพิ่ม **Repository secrets** ใหม่ 2 ตัวดังนี้:
    *   `DOCKER_HUB_USERNAME`: (ชื่อผู้ใช้ Docker Hub ของคุณ เช่น `ramayana4`)
    *   `DOCKER_HUB_TOKEN`: (Access Token จาก Docker Hub > Account Settings > Security)
3.  ทุกครั้งที่คุณ `git push` ขึ้นกิ่ง `main` ระบบจะเริ่มบิ้ว Image และส่งขึ้น Docker Hub ให้ทันที

---

## 4. การ Deploy Backend Worker ไปยัง RunPod Serverless

### ก. การเตรียม Docker Image:
ระบบจะทำงานผ่าน GitHub Actions อัตโนมัติ โดยทำการบิวด์อิมเมจจาก `Dockerfile.colmap` ซึ่งเป็นอิมเมจตัวเดียวที่รวมทั้ง Nerfstudio และ Colmap เข้าไว้ด้วยกัน (ตรวจสอบสถานะการบิวด์ได้ที่เมนู **Actions** ใน GitHub)

### ข. การตั้งค่าใน RunPod เพื่อดึง Image จาก Docker Hub:
เนื่องจากเราเก็บ Image ไว้ที่ Docker Hub เราสามารถตั้งค่าใน RunPod ได้ดังนี้:

1.  **ถ้า Image เป็น Public:**
    *   ไม่ต้องตั้งค่า Container Registry ใน RunPod
2.  **ถ้า Image เป็น Private:**
    *   ไปที่ **User Settings** > **Container Registries** > **Add Registry**
    *   **Registry Domain Name:** `index.docker.io`
    *   **Username:** (ชื่อผู้ใช้ Docker Hub)
    *   **Password:** (Access Token หรือรหัสผ่าน)

3.  **สร้าง Endpoint ใน RunPod:**
    *   ระบุ Image URL: `ramayana4/worker-3d-scan:processor`
    *   เลือก GPU ที่ต้องการ (แนะนำ **RTX 3090** หรือ **4090** เพื่อให้ขั้นตอนการเทรนเร็วขึ้น)
    *   ตั้งค่า Environment Variables:
        *   `SUPABASE_URL`: (จาก Supabase Project)
        *   `SUPABASE_KEY`: (ต้องใช้ **Service Role Key** เพื่อสิทธิ์ในการเขียนไฟล์ลง Storage)
        *   `WORKER_MODE`: `FULL` (เพื่อรันกระบวนการตั้งแต่สกัดเฟรมภาพไปจนถึงเทรนและส่งออกโมเดลทีเดียวจบ)

*หมายเหตุ: ไม่จำเป็นต้องกำหนดค่าการเชื่อมต่อ S3 อีกต่อไป เนื่องจากตัวประมวลผลเวอร์ชันใหม่จะส่งไฟล์ผลลัพธ์ (.ply) ขึ้นไปเก็บบน Supabase Storage โดยตรง*

---

## 5. การจัดการผลลัพธ์ (Retention Policy)

เพื่อให้เป็นไปตามนโยบายการเก็บข้อมูล 24 ชั่วโมง แนะนำให้ตั้งค่าใน **Supabase Storage**:

1.  ไปที่ **Storage** > **Buckets**
2.  เลือก Bucket `3d-scans`
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
4.  ตรวจสอบ Log ใน RunPod เพื่อดูการทำงานของ Worker
