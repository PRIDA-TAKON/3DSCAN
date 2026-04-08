# สถาปัตยกรรมระบบ 02_3DSCAN (Modular & Lean)

## 🏗️ แนวคิดหลัก
เพื่อแก้ปัญหาการบิวด์อิมเมจล้มเหลวเนื่องจากซอฟต์แวร์ตีกัน (Dependency Conflicts) เราจึงใช้กลยุทธ์แยกส่วนการประมวลผลและการเก็บรักษาข้อมูล:

1.  **Isolation (แยกส่วนประมวลผล)**: ใช้ Docker Image ของ `nerfstudio` เป็นฐานหลักโดยไม่ติดตั้ง Library ที่ไม่จำเป็นทับ เพื่อให้เครื่องมือไม่ตีกันพัง
2.  **Data Orchestration (Supabase as Buffer)**: ใช้ Supabase Storage เป็นพื้นที่พักข้อมูลชั่วคราว (Temporary Buffer) ระหว่างขั้นตอนประมวลผล (Frames, COLMAP sparse data)
3.  **Small Results (ผลลัพธ์สุดท้ายขนาดเล็ก)**: เก็บเฉพาะไฟล์ Gaussian Splat (.ply, .splat) ที่ผ่านการประมวลผลจนจบลงในที่เก็บข้อมูลถาวร
4.  **Automatic Cleanup (ระบบล้างไฟล์ขยะ)**: ระบบจะสั่งลบไฟล์ชั่วคราวที่มีขนาดใหญ่ (วิดีโอต้นฉบับ, Frames ทั้งหมด) ใน Supabase ทันทีที่ได้ผลลัพธ์สุดท้าย

---

## 🔄 ลำดับขั้นตอน (Pipeline Flow)

1.  **Frontend**: อัปโหลดวิดีโอ -> Supabase
2.  **RunPod (Worker)**:
    *   ดึงวิดีโอ -> แปลงเป็นภาพ (Frames)
    *   รัน COLMAP (Reconstruction) -> บันทึก Sparse Data
    *   รัน Nerfstudio (Training) -> จนครบจำนวนรอบ
3.  **Export & Clean**:
    *   ส่งออกไฟล์โมเดลขนาดเล็ก (.ply) -> Supabase
    *   **คำสั่งลบโฟลเดอร์ชั่วคราว** ใน Supabase Storage ทิ้งทั้งหมด (เพื่อลดภาระค่าใช้จ่ายพื้นที่)

---

## 🎯 แผนการย้ายสู่ GCP (Future Roadmap)
*   เปลี่ยน **RunPod** เป็น **Google Cloud Run (GPU)** หรือ **Vertex AI**
*   เปลี่ยน **Supabase Storage** เป็น **Google Cloud Storage (GCS)** พร้อมตั้งค่า **Lifecycle Rules** (ลบไฟล์อัตโนมัติเมื่อครบกำหนด 1 วัน)
