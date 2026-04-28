# VibeVoice ASR - Phiên bản cá nhân

<table>
  <tr>
    <td width="50%">
      <b>Video 1</b><br>
      <video src="https://github.com/user-attachments/assets/f3d77073-b91f-4c04-9920-2eaf65c58fdd" controls="controls" width="100%"></video>
    </td>
    <td width="50%">
      <b>Video 2</b><br>
      <video src="https://github.com/user-attachments/assets/f73333d2-e98d-4de2-b725-563df9c28730" controls="controls" width="100%"></video>
    </td>
  </tr>
</table>

 (VibeVoice Edit)

Dự án này là phiên bản tối ưu hóa và tinh chỉnh lại từ mã nguồn gốc **VibeVoice** của Microsoft. Mục tiêu chính là chuyển đổi một công cụ nghiên cứu kỹ thuật thành một sản phẩm (Product) hoàn thiện, thân thiện với người dùng cuối và hoạt động ổn định trên các tài nguyên miễn phí như Google Colab.

---

##  Các thay đổi và Nâng cấp chính

### 1. Giao diện Người dùng (UI/UX) Chuyên nghiệp
* **Tối giản hóa:** Loại bỏ toàn bộ các biểu tượng (emoji) và icon rườm rà.
* **Ngôn ngữ:** Việt hóa 100% các thông báo, nhãn (labels) và hướng dẫn sử dụng.
* **Bố cục:** Chuyển sang bố cục 2 cột (Input - Output) chuẩn doanh nghiệp.
* **Tiện ích:** Thêm nút "Copy to Clipboard" cho văn bản kết quả và tính năng nghe lại từng đoạn hội thoại đã phân rã.
* **Ẩn kỹ thuật:** Giấu các thông số phức tạp (Temperature, Top-p, Penalty...) vào phần ẩn để người dùng không bị rối mắt, nhưng vẫn giữ được độ chính xác của AI.

### 2. Mở rộng Khả năng Đa phương thức
* **Hỗ trợ Video trực tiếp:** Mở khóa khả năng nhận diện trực tiếp từ các định dạng video phổ biến như `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`.
* **Tích hợp Link Mạng xã hội:** Sử dụng thư viện `yt-dlp` để cho phép người dùng chỉ cần dán link YouTube, TikTok hoặc Facebook là hệ thống tự động tải, tách âm thanh và bóc băng.

### 3. Tối ưu hóa Hiệu suất & Phần cứng
* **Ép cân 4-bit:** Sử dụng kỹ thuật Quantization (`bitsandbytes`) để nén mô hình từ 18GB VRAM xuống còn khoảng 5GB, giúp chạy ổn định trên card đồ họa Tesla T4 (16GB) mà không bị lỗi "Out of Memory".
* **Liger Kernel:** Tích hợp nhân tối ưu hóa Liger để tăng tốc độ tính toán ma trận, giúp quá trình bóc băng diễn ra nhanh hơn và tiêu tốn ít tài nguyên hơn.
* **Quản lý Cache thông minh:** Cấu hình hệ thống tự động lưu trữ "não bộ" AI vào Google Drive, giúp bỏ qua bước tải 18GB dữ liệu ở những lần khởi động sau.

---

## 🛠 Các lỗi kỹ thuật đã được khắc phục (Bug Fixes)

Hệ thống đã được vá các lỗi quan trọng từ bản gốc và trong quá trình triển khai:
1.  **Lỗi Danh sách Định dạng:** Sửa lỗi `AttributeError: 'list' object has no attribute 'update'` bằng cách chuyển sang phương thức `.extend()` chuẩn cho danh sách đuôi file video.
2.  **Lỗi Giới hạn Slider:** Khắc phục lỗi `Value 32768 is greater than maximum value 100` bằng cách thiết lập lại phạm vi điều khiển của Gradio, cho phép xử lý các đoạn hội thoại dài với số lượng Token lớn.
3.  **Lỗi Lệch Tensor (Device Mapping):** Ép cứng quy trình nạp mô hình lên `cuda` thay vì để `auto`, giải quyết triệt để lỗi "Tensors are on different devices" khi bóc băng.
4.  **Lỗi Xung đột Tiến trình:** Cấu hình khởi chạy trên các cổng mạng (Port) linh hoạt để tránh bị treo khi khởi động lại ứng dụng nhiều lần trên cùng một phiên làm việc.

---

## 📂 Cấu trúc Tệp tin Quan trọng
* `app_pro.py`: File khởi chạy chính với giao diện chuyên nghiệp.
* `vibevoice/`: Thư mục chứa hạt nhân thuật toán (Core AI).
* `demo/vibevoice_asr_gradio_demo.py`: Mã nguồn giao diện đã được đại tu.

---
**Ghi chú:** Đây là dự án cá nhân được phát triển dựa trên nền tảng VibeVoice của Microsoft Research.
