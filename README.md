# Trợ lý Phân loại Cảm xúc Tiếng Việt

**Đồ án môn học: Seminar chuyên đề (HK1 2025-2026)**

Ứng dụng web phân loại cảm xúc văn bản Tiếng Việt sử dụng mô hình Transformer (PhoBERT), hỗ trợ 3 nhãn cảm xúc: **POSITIVE** (Tích cực), **NEUTRAL** (Trung lập), và **NEGATIVE** (Tiêu cực).

**Sinh viên thực hiện:**
* Nguyễn Hồ Khánh An - 3121410048

![Demo](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red)

---

## 📋 Mục lục

- [Tổng quan](#-tổng-quan--mục-tiêu)
- [Tính năng](#-tính-năng)
- [Kiến trúc](#-giải-pháp--kiến-trúc)
- [Cài đặt](#-hướng-dẫn-cài-đặt--sử-dụng)
- [Sử dụng](#-chạy-ứng-dụng)
- [Kết quả](#-kết-quả--đánh-giá)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)

---

## 🎯 Tổng quan & Mục tiêu

### Mục tiêu dự án

1. **Phân loại cảm xúc chính xác**: Xây dựng mô hình phân loại cảm xúc văn bản Tiếng Việt với độ chính xác cao
2. **Sử dụng Transformer**: Áp dụng PhoBERT (pre-trained Vietnamese BERT) cho bài toán NLP
3. **Ứng dụng thực tế**: Tạo giao diện web thân thiện, dễ sử dụng cho người dùng cuối
4. **Lưu trữ lịch sử**: Lưu lại kết quả phân loại để phân tích xu hướng

### Bài toán

Phân loại cảm xúc của văn bản Tiếng Việt thành 3 nhãn:
- 🟢 **POSITIVE**: Cảm xúc tích cực (vui vẻ, hài lòng, thích thú)
- ⚪ **NEUTRAL**: Trung lập (khách quan, không thiên vị)
- 🔴 **NEGATIVE**: Tiêu cực (thất vọng, tức giận, buồn bã)

---

## ✨ Tính năng

### Ứng dụng Web (Streamlit)
- ✅ Nhập văn bản và phân loại cảm xúc tức thì
- ✅ Hiển thị độ tin cậy của dự đoán (confidence score)
- ✅ Validation đầu vào (độ dài, ký tự hợp lệ)
- ✅ Toast notifications cho trải nghiệm người dùng tốt hơn

### Quản lý Lịch sử
- 📊 Lưu trữ tất cả phân loại vào SQLite database
- 📥 Xuất lịch sử ra file CSV
- 🗑️ Xóa lịch sử với xác nhận an toàn
- 🕐 Hiển thị 50 bản ghi gần nhất trong sidebar

### Thu thập Dữ liệu
- 🔄 Script tự động thu thập data từ Hugging Face
- 🔀 Merge và deduplicate datasets
- ✂️ Split data thành train/val/test (70%/10%/20%)

---

## 🏗️ Giải pháp & Kiến trúc

### Mô hình

**PhoBERT-base-v2** ([vinai/phobert-base-v2](https://huggingface.co/vinai/phobert-base-v2))
- Pre-trained trên 20GB dữ liệu tiếng Việt
- Architecture: RoBERTa-base (125M parameters)
- Fine-tuned cho 3-class sentiment classification

### Dataset

**UIT-VSMEC** (Vietnamese Social Media Emotion Corpus)
- Source: [SEACrowd/uit_vsmec](https://huggingface.co/datasets/SEACrowd/uit_vsmec)
- Converted từ 6 emotion labels → 3 sentiment labels
- Total: 4,500 samples (balanced distribution)
  - Train: 3,150 samples (70%)
  - Validation: 450 samples (10%)
  - Test: 900 samples (20%)

**Label Mapping:**
```
POSITIVE ← JOY, POSITIVE, SURPRISE
NEUTRAL  ← NEUTRAL, OTHER
NEGATIVE ← ANGER, FEAR, SADNESS, NEGATIVE
```

### Tiền xử lý

File: `src/preprocess.py`
- ✅ Chuẩn hóa Unicode (NFD → NFC)
- ✅ Xử lý teencode và viết tắt phổ biến
- ✅ Loại bỏ HTML tags, URLs, emails
- ✅ Chuẩn hóa dấu câu và khoảng trắng
- ✅ Lowercase và trim

### Tech Stack

```
Backend:     Python 3.9+, PyTorch, Transformers
Frontend:    Streamlit
Database:    SQLite
Model:       PhoBERT-base-v2
Training:    Hugging Face Trainer API
```

### Kiến trúc hệ thống

```
┌─────────────────┐
│   Streamlit     │ ← User Interface
│     (app.py)    │
└────────┬────────┘
         │
    ┌────▼────┐
    │ Models  │
    └────┬────┘
         │
    ┌────▼──────────┐
    │   Database    │ ← SQLite (history.db)
    │  (database.py)│
    └───────────────┘
```

---

## 🚀 Hướng dẫn Cài đặt & Sử dụng

### Yêu cầu hệ thống

- **Python**: 3.9 hoặc cao hơn
- **RAM**: Tối thiểu 4GB (khuyến nghị 8GB)
- **Disk**: ~2GB (bao gồm model và dependencies)
- **OS**: Windows, macOS, hoặc Linux

### Cài đặt

#### 1. Clone repository

```bash
git clone https://github.com/nguyenhokhanan/NLP-Transformer.git
cd NLP-Transformer
```

#### 2. Tạo môi trường ảo

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

#### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

> **Lưu ý**: Quá trình cài đặt có thể mất 5-10 phút do cần tải PyTorch và Transformers.

#### 4. Tải mô hình đã huấn luyện

Có 2 cách:

**Cách 1: Tải model sẵn có** (Khuyến nghị)

📥 **[Tải model từ Google Drive](https://drive.google.com/drive/folders/1LBy72fiDHi-u9GQqwO1ZwHStuTkjwj8W?usp=sharing)**

Sau khi tải về:
1. Giải nén file ZIP
2. Đặt thư mục vào `notebooks/_models/phobert-3-labels-final/`

**Cách 2: Huấn luyện từ đầu**
```bash
# Mở Jupyter notebook
jupyter notebook notebooks/02_model_training.ipynb

# Chạy tất cả cells để train model (~2-3 giờ trên CPU, ~30 phút trên GPU)
```

---

## 💻 Chạy ứng dụng

### Khởi động Streamlit App

```bash
streamlit run app.py
```

App sẽ mở tại: **http://localhost:8501**

### Sử dụng ứng dụng

1. **Nhập văn bản**: Gõ hoặc paste câu tiếng Việt vào ô text area
2. **Phân tích**: Nhấn nút "Phân tích cảm xúc 🚀"
3. **Xem kết quả**: 
   - Nhãn cảm xúc (POSITIVE/NEUTRAL/NEGATIVE)
   - Độ tin cậy (0-100%)
4. **Kiểm tra lịch sử**: Xem sidebar bên trái

### Ví dụ

**Input:**
```
Sản phẩm rất tốt, tôi rất hài lòng!
```

**Output:**
```
Cảm xúc: POSITIVE
Độ tin cậy: 98.5%
```

---

## 📊 Kết quả & Đánh giá

### Kết quả kiểm thử thực tế
Dự án tập trung đánh giá hiệu quả của mô hình pre-trained trên các dữ liệu thực tế (Test Cases) thay vì các chỉ số huấn luyện khô khan.

**Kết quả trên 10 câu mẫu:**
*   **Độ chính xác:** 9/10 (90%) - Đạt mục tiêu đề ra (≥ 65%).
*   **Phân tích:**
    *   ✅ Nhận diện tốt các từ khóa cảm xúc rõ ràng ("đẹp", "hỏng", "tuyệt vời").
    *   ✅ Hiểu được một số từ lóng và thành ngữ ("lồi lõm", "treo đầu dê bán thịt chó").
    *   ✅ Xử lý tốt các ký tự đặc biệt và icon cảm xúc (<3).
    *   ❌ Còn hạn chế với các câu có ranh giới mờ nhạt giữa Trung lập và Tích cực.

## 📁 Cấu trúc dự án

```
NLP_Transformer/
├── app.py                    # Streamlit application chính
├── requirements.txt          # Python dependencies
├── README.md                 # Documentation này
├── LICENSE                   # MIT License
│
├── data/                     # Dữ liệu
│   ├── history.db           # SQLite database (prediction history)
│   └── processed_3labels/   # Dataset đã xử lý
│       ├── train.csv        # 3,150 samples
│       ├── val.csv          # 450 samples
│       └── test.csv         # 900 samples
│
├── notebooks/               # Jupyter notebooks
│   ├── 01_data_exploration.ipynb      # Phân tích dữ liệu
│   ├── 02_model_training.ipynb        # Training script
│   └── _models/                       # Trained models
│       └── phobert-3-labels-final/
│           ├── config.json
│           ├── pytorch_model.bin
│           └── tokenizer files...
│
└── src/                     # Source code
    ├── __init__.py
    ├── collect_data.py      # Thu thập data từ Hugging Face
    ├── convert_to_3_labels.py  # Chuyển đổi 6→3 labels
    ├── database.py          # SQLite operations
    ├── predictor.py         # Sentiment prediction
    └── preprocess.py        # Text preprocessing
```

---

## 🔧 Development

### Chạy Data Exploration

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### Thu thập thêm dữ liệu

```bash
cd src
python collect_data.py
```

Script sẽ:
1. Tải datasets từ Hugging Face
2. Merge với dataset hiện tại
3. Deduplicate
4. Split thành train/val/test

### Chuyển đổi dataset 6→3 labels

```bash
cd src
python convert_to_3_labels.py
```

---

## 📝 Notes

### Limitations

- Model hoạt động tốt với văn bản ngắn (< 256 tokens)
- Cần GPU để inference nhanh với batch lớn
- Chưa hỗ trợ multilingual (chỉ Tiếng Việt)

### Future Improvements

- [ ] Thêm visualization cho confidence scores
- [ ] Export model sang ONNX để tăng tốc inference
- [ ] Hỗ trợ batch prediction (upload CSV)
- [ ] REST API endpoint cho integration
- [ ] Thêm các nhãn cảm xúc chi tiết hơn (6 labels)

---

## 📄 License

MIT License - Xem file [LICENSE](LICENSE) để biết thêm chi tiết.

---

## 🙏 Acknowledgments

- **Dataset**: [UIT-VSMEC](https://huggingface.co/datasets/SEACrowd/uit_vsmec)
- **Pre-trained Model**: [PhoBERT](https://github.com/VinAIResearch/PhoBERT) by VinAI Research
- **Framework**: Hugging Face [Transformers](https://huggingface.co/docs/transformers)

---

## 📞 Contact

**Nguyễn Hồ Khánh An**
- Email: khanhan280103@gmail.com
- GitHub: [@Valkyrie2002](https://github.com/Valkyrie2002)

---
