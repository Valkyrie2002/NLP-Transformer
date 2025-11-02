# NLP-Transformer
Đồ án môn học: Xây dựng trợ lý phân loại cảm xúc Tiếng Việt sử dụng Transformer (PhoBERT/DistilBERT).

# [SGU] Xây dựng Trợ lý Phân loại Cảm xúc Tiếng Việt
_Đồ án môn học: Seminar chuyên đề (HK1 2025-2026)_ 

Dự án này xây dựng một ứng dụng web cho phép người dùng nhập vào một câu Tiếng Việt và phân loại cảm xúc của câu đó (ví dụ: POSITIVE, NEGATIVE, NEUTRAL,...)  sử dụng mô hình Transformer.

**Sinh viên thực hiện:**
* Nguyễn Hồ Khánh An - 3121410048

---

## 1. Tổng quan & Mục tiêu

* Mục tiêu 1: Ứng dụng phân loại cảm xúc đơn giản.
* Mục tiêu 2: Sử dụng Transformer (PhoBERT/DistilBERT).
* Mục tiêu 3: Huấn luyện với 6 nhãn cảm xúc.
* Mục tiêu 4: Đạt độ chính xác > 70%.

## 2. Giải pháp & Kiến trúc

*(Mô tả ngắn gọn về giải pháp của bạn)*
* **Mô hình:** Sử dụng PhoBERT (pre-trained) và fine-tune trên bộ dữ liệu [Tên-Dataset].
* **Tiền xử lý:** Xử lý teencode, viết tắt, và chuẩn hóa Tiếng Việt.
* **Ứng dụng:** Xây dựng bằng Streamlit/Flask.
* *(Vẽ sơ đồ kiến trúc nếu có thể)*

## 3. Hướng dẫn Cài đặt & Sử dụng

Đây là phần **quan trọng nhất** trong `README` (và cũng là mục 4 trong báo cáo).

### Yêu cầu
* Python 3.9+
* Git

### Cài đặt
1.  Clone repository này:
    ```bash
    git clone [https://github.com/ten-cua-ban/ten-repo.git](https://github.com/ten-cua-ban/ten-repo.git)
    cd ten-repo
    ```
2.  Tạo và kích hoạt môi trường ảo:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Trên Windows: venv\Scripts\activate
    ```
3.  Cài đặt các thư viện cần thiết:
    ```bash
    pip install -r requirements.txt
    ```
4.  Tải mô hình đã huấn luyện:
    * *(Nếu bạn không đẩy model lên Git)*
    * Tải file `model.joblib` từ [link Google Drive của bạn] và đặt vào thư mục `_models/`.

### Chạy ứng dụng
```bash
# Nếu dùng Streamlit
streamlit run app/main.py

# Nếu dùng Flask
python app/main.py
```

## 4. Kết quả & Đánh giá

* Mô hình đạt độ chính xác **[XX.X]%** trên tập kiểm thử.
* *(Hiển thị một vài hình ảnh chụp màn hình ứng dụng của bạn)*



---
*Ghi chú: Đừng quên tạo file `requirements.txt` bằng cách chạy `pip freeze > requirements.txt` sau khi cài đặt xong tất cả thư viện.*