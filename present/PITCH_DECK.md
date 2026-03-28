# Pitch Deck: 3DScan Cloud - The Future of 3D Reconstruction
**"High-Fidelity 3D Models from Any Video, Powered by Google Cloud & Taichi"**

---

## Slide 1: The Problem (ปัญหาที่เราเจอ)
*   **3D Content Gap:** ความต้องการคอนเทนต์ 3 มิติ (E-commerce, VR/AR, Games) สูงขึ้นมาก
*   **Traditional Methods are Hard:** การสแกน 3D แบบดั้งเดิมต้องใช้อุปกรณ์แพง หรือใช้เวลาประมวลผลนานหลายชั่วโมง (เช่น NeRF)
*   **Accessibility:** ผู้ใช้ทั่วไปไม่สามารถเข้าถึงการทำ 3D คุณภาพสูงได้ด้วยตัวเอง

## Slide 2: Our Solution (ทางออกของเรา)
*   **3DScan Cloud Platform:** แพลตฟอร์มที่เปลี่ยนวิดีโอจากสมาร์ทโฟนเพียง 1 นาที ให้กลายเป็นไฟล์โมเดล 3D คุณภาพระดับภาพถ่าย
*   **Ease of Use:** เพียงอัปโหลดวิดีโอผ่านหน้าเว็บ (Next.js Frontend)
*   **Cloud Processing:** ระบบจัดการหลังบ้านประมวลผลให้อัตโนมัติบน Google Cloud Run GPU L4

## Slide 3: Core Technology (เทคโนโลยีเบื้องหลัง)
*   **3D Gaussian Splatting (3DGS):** เทคโนโลยี Neural Rendering ล่าสุด (2023) ที่เรนเดอร์ได้เร็วกว่า NeRF ถึง 100 เท่า
*   **Taichi Implementation:** เราใช้ตัวเทรนที่ปรับปรุงประสิทธิภาพ (Optimization) ให้ใช้ทรัพยากรน้อยกว่าต้นฉบับถึง **75%** ในขณะที่ได้ภาพชัดเท่าเดิม
*   **Automated Pipeline:**
    1. Video Frame Extraction
    2. SfM (Camera Pose Estimation)
    3. Taichi 3DGS Training (30,000 iters)
    4. Auto-Export & Delivery

## Slide 4: Competitive Advantage (ทำไมต้องเป็นเรา?)
*   **Beyond LiDAR:** ในขณะที่ LiDAR (Laser Scan) ให้เพียงจุดสีที่ขาดความต่อเนื่อง แต่เทคโนโลยีของเราให้ภาพที่ **Photorealistic** และเก็บแสงสะท้อนได้สมจริงกว่า
*   **Zero Hardware Barrier:** ไม่ต้องใช้ iPhone Pro หรือเครื่องสแกนราคาหลักแสน ใช้เพียงวิดีโอจากมือถือทั่วไป
*   **Speed & Quality:** ใช้ GPU L4 ล่าสุดบน Cloud ประมวลผลเสร็จในเวลาไม่กี่นาที
*   **Optimized Output:** ไฟล์ขนาดเล็กกว่า (ใช้จุดน้อยลง 75%) แต่คุณภาพสูงกว่า เหมาะกับการใช้งานบน Mobile Web

## Slide 5: Business Use Cases (โอกาสทางธุรกิจ)
*   **E-Commerce:** สร้าง Viewable 3D สินค้าให้ลูกค้าหมุนดูได้รอบทิศทางสมจริงกว่ารูปถ่าย
*   **Real Estate / Tourism:** ทัวร์เสมือนจริง (Virtual Tour) จากวิดีโอเดินถ่ายห้อง
*   **Gaming & VFX:** เปลี่ยนวัตถุจริงให้เป็น Asset ในเกมได้อย่างรวดเร็ว
*   **Digital Twin:** เก็บข้อมูลสิ่งก่อสร้างหรือโบราณสถานในรูปแบบดิจิทัล

## Slide 6: Current Progress (ความคืบหน้า)
*   ✅ **Completed MVP:** ระบบหน้าเว็บเชื่อมต่อ Supabase และ Cloud Run ทำงานได้สมบูรณ์
*   ✅ **Optimized Training:** เทสผลลัพธ์ (PSNR) เทียบเท่าระดับโลก (SOTA)
*   ✅ **Pipeline Automation:** รัน 4 ขั้นตอนจบในปุ่มเดียว (One-pot Execution)

## Slide 7: Funding Request & Roadmap (เป้าหมายและการใช้ทุน)
*   **เป้าหมาย:** พัฒนาไปสู่การเป็น "SaaS 3D Solution" สำหรับองค์กร
*   **การใช้ทุน:**
    *   **Scale Infra:** รองรับการเทรนงานจำนวนมากพร้อมกัน
    *   **R&D:** พัฒนา Real-time Viewer และการปรับแต่งโมเดลหลังการเทรน (Post-editing)
    *   **Marketing:** เจาะกลุ่มตลาดนักออกแบบและเจ้าของธุรกิจ E-commerce

---
**"Empowering Everyone to Create the 3D World"**
*Contact: [Your Contact Info]*
