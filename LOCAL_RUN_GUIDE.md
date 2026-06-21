# 💻 คู่มือการรันระบบแบบ Local (เครื่องตัวเอง)

คู่มือนี้สำหรับทดสอบการประมวลผลวิดีโอเป็นโมเดล 3DGS ด้วย Docker บนเครื่องคอมพิวเตอร์พกพาที่มีการ์ดจอ NVIDIA RTX 3050 (4GB) เป็นต้นไป

---

## 🛠️ 1. สิ่งที่ต้องเตรียม (Prerequisites)
1. **Docker Desktop:** ตรวจสอบว่า Docker Desktop บน Windows เปิดทำงานอยู่
2. **NVIDIA Container Toolkit:** ตรวจสอบว่าระบบรองรับ GPU Passthrough (สำหรับ Windows WSL2) โดยทดสอบด้วยคำสั่ง:
   ```powershell
   docker run --gpus all --rm nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```
   *หากแสดงข้อมูลการ์ดจอแปลว่าระบบพร้อมใช้งาน*

---

## 🚀 2. ขั้นตอนการรันประมวลผล (Running Pipeline)

การประมวลผลแบบโลคอลจะใช้โปรเซสแบบรันต่อกันทีเดียวจบ (End-to-End) ผ่านตู้คอนเทนเนอร์ตัวเดียว:

### ขั้นตอนที่ 1: เตรียมไฟล์วิดีโอ
คัดลอกไฟล์วิดีโอที่ต้องการทำ 3DGS ไปเก็บไว้ที่โฟลเดอร์ Workspace ของเครื่อง (เช่น `video_car.mp4`)

### ขั้นตอนที่ 2: รันคำสั่ง Docker เพื่อประมวลผล
เปิด PowerShell แล้วสั่งรันคำสั่งด้านล่างนี้ โดยเปลี่ยนพาร์ทไปยังพาร์ทโฟลเดอร์ Workspace จริงของคุณ:

```powershell
# ลบคอนเทนเนอร์ชื่อเก่าออก (ถ้ามีค้างอยู่)
docker rm -f nerfstudio

# รันดึงวิดีโอไปสกัดเฟรม และฝึกสอน (Train) ทันที
docker run --gpus all `
  -v "C:\Users\takon\OneDrive\Desktop\da\38_omniruntime\workspace:/workspace" `
  -p 7007:7007 `
  --shm-size=12gb `
  --name nerfstudio --rm `
  ghcr.io/nerfstudio-project/nerfstudio:latest `
  bash -c "ns-process-data video --data /workspace/video_car.mp4 --output-dir /workspace/nerf_data && ns-train splatfacto --data /workspace/nerf_data"
```

*คำอธิบายตัวแปร:*
* `-v "...:/workspace"`: เชื่อมต่อโฟลเดอร์เครื่องเราเข้าไปที่ระบบข้างในคอนเทนเนอร์
* `-p 7007:7007`: เปิดพอร์ต `7007` สำหรับการดูโมเดลสดบนเว็บเบราว์เซอร์
* `--shm-size=12gb`: จองพื้นที่ความจำเสมือนสำหรับตัวเทรนเนอร์ (สำคัญมากสำหรับ Nerfstudio)
* `ns-process-data video ...`: คำสั่งดึงเฟรมและรัน Colmap
* `ns-train splatfacto ...`: คำสั่งเทรนโมเดล Splatting

---

## 📺 3. การแสดงผล (Viewing Results)
เมื่อขึ้นคำแนะนำ `Use ctrl+c to quit` ในเทอร์มินัล แปลว่าการฝึกสอน (Training) 2000 Iterations เสร็จสิ้นแล้ว คุณสามารถเปิดหน้าเว็บเพื่อหมุนดูโมเดล 3D แบบเรียลไทม์ได้ที่:
👉 **[http://localhost:7007](http://localhost:7007)**

---

## 📤 4. การดึงไฟล์ผลลัพธ์ (.ply)
ให้เปิด PowerShell หน้าต่างใหม่ แล้วพิมพ์คำสั่งนี้เพื่อดึงผลงานออกมาบันทึกเป็นไฟล์ `.ply` (แทนที่จะดาวน์โหลดผ่านคลาวด์):

```powershell
docker exec -it nerfstudio ns-export gaussian-splat `
  --load-config /workspace/nerf_data/outputs/nerf_data/splatfacto/2026-06-21_092040/config.yml `
  --output-dir /workspace/export
```
*(หมายเหตุ: ให้เปลี่ยนพาร์ทโฟลเดอร์วันที่ `2026-06-21_092040` ใน `config.yml` ให้ตรงกับพาร์ทล่าสุดที่เกิดขึ้นจริงในระบบของคุณ)*

ชิ้นงานที่ได้จะไปบันทึกอยู่ใต้โฟลเดอร์เครื่องของคุณ:
📂 **`C:\Users\takon\OneDrive\Desktop\da\38_omniruntime\workspace\export`**
