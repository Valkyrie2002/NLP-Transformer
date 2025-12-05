import streamlit as st
from src.predictor import SentimentPredictor
from src.database import HistoryDatabase
from datetime import datetime
import pandas as pd

# Cấu hình trang
st.set_page_config(
    page_title="Phân loại Cảm xúc Tiếng Việt (3 nhãn)", 
    page_icon="📊",
    layout="wide"
)

# Load model (Cache để không phải load lại mỗi lần bấm nút)
@st.cache_resource
def load_predictor():
    return SentimentPredictor()

# Khởi tạo database
@st.cache_resource
def load_database():
    return HistoryDatabase()

try:
    predictor = load_predictor()
    model_loaded = True
except Exception as e:
    st.error(f"Không tìm thấy model! Hãy chắc chắn bạn đã chạy notebook training. Lỗi: {e}")
    model_loaded = False

# Khởi tạo database
db = load_database()

# Sidebar - Lịch sử phân loại
with st.sidebar:
    st.header("Lịch sử Phân loại")
    
    # Thống kê
    total_count = db.get_count()
    st.metric("Tổng số phân tích", total_count)
    
    # Nút xuất CSV
    if total_count > 0:
        if st.button("Xuất CSV", use_container_width=True):
            try:
                csv_path = db.export_to_csv("history_export.csv")
                with open(csv_path, "rb") as f:
                    st.download_button(
                        label="Tải xuống CSV",
                        data=f,
                        file_name=f"sentiment_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                st.success("Đã tạo file CSV!")
            except Exception as e:
                st.error(f"Lỗi khi xuất CSV: {e}")
        
        # Nút xóa lịch sử
        if st.button("Xóa lịch sử", use_container_width=True):
            if 'confirm_clear' not in st.session_state:
                st.session_state.confirm_clear = True
        
        # Xác nhận xóa
        if st.session_state.get('confirm_clear', False):
            st.warning("Bạn có chắc chắn muốn xóa toàn bộ lịch sử?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Có", use_container_width=True):
                    db.clear_history()
                    st.session_state.confirm_clear = False
                    st.session_state.last_prediction = None
                    st.rerun()
            with col2:
                if st.button("Không", use_container_width=True):
                    st.session_state.confirm_clear = False
                    st.rerun()
        
        st.divider()
        
        # Hiển thị lịch sử gần đây (giới hạn 50 bản ghi)
        st.subheader("🕐 Lịch sử gần đây")
        history = db.get_history(limit=50)
        
        if history:
            for record in history:
                record_id, timestamp, text, label, score = record
                
                # Tô màu cho nhãn (3 labels)
                color_map = {
                    "POSITIVE": "🟢",
                    "NEUTRAL": "⚪",
                    "NEGATIVE": "🔴"
                }
                emoji = color_map.get(label, "🔵")
                
                with st.expander(f"{emoji} {label} - {timestamp}", expanded=False):
                    st.write(f"Văn bản: {text}")
                    st.progress(score, text=f"Độ tin cậy: {score*100:.2f}%")
        else:
            st.info("Chưa có lịch sử phân loại")
    else:
        st.info("Chưa có lịch sử phân loại")

# Main content
st.title("Trợ lý Phân loại Cảm xúc Tiếng Việt (3 nhãn)")
st.write("Nhập một câu tiếng Việt bên dưới để AI phân loại cảm xúc: POSITIVE  (Tích cực), NEUTRAL (Trung lập), NEGATIVE (Tiêu cực)")

# Giao diện nhập liệu
text_input = st.text_area("Nhập văn bản tại đây:", height=100, placeholder="VD: Hôm nay tôi rất vui...")

if st.button("Phân tích cảm xúc", type="primary"):
    # Kiểm tra input rỗng
    if not text_input.strip():
        st.toast("⚠️ Vui lòng nhập nội dung!", icon="⚠️")
        st.warning("Vui lòng nhập nội dung!")
    # Kiểm tra câu quá ngắn
    elif len(text_input.strip()) < 3:
        st.toast("⚠️ Câu quá ngắn! Vui lòng nhập ít nhất 3 ký tự.", icon="⚠️")
        st.warning("Câu quá ngắn! Vui lòng nhập ít nhất 3 ký tự.")
    # Kiểm tra câu quá dài
    elif len(text_input.strip()) > 500:
        st.toast("⚠️ Câu quá dài! Vui lòng nhập tối đa 500 ký tự.", icon="⚠️")
        st.warning("Câu quá dài! Vui lòng nhập tối đa 500 ký tự.")
    # Kiểm tra chỉ có số hoặc ký tự đặc biệt
    elif not any(c.isalpha() for c in text_input):
        st.toast("⚠️ Vui lòng nhập văn bản có chữ cái!", icon="⚠️")
        st.warning("Vui lòng nhập văn bản có chữ cái!")
    elif model_loaded:
        with st.spinner("Đang suy nghĩ..."):
            try:
                # Gọi hàm dự đoán
                label, score = predictor.predict(text_input)
                
                # Lưu vào database
                try:
                    db.save_prediction(text_input, label, score)
                    st.toast("✅ Đã lưu vào lịch sử!", icon="✅")
                except Exception as e:
                    st.error(f"Lỗi khi lưu vào database: {e}")
                    st.toast("❌ Lỗi khi lưu vào database!", icon="❌")
                
                # Lưu kết quả vào session state
                st.session_state.last_prediction = {
                    'text': text_input,
                    'label': label,
                    'score': score
                }
                
                # Reload để cập nhật sidebar lịch sử
                st.rerun()
            except Exception as e:
                st.toast(f"Lỗi khi phân tích: {str(e)}", icon="❌")
                st.error(f"Lỗi khi phân tích: {e}")

# Hiển thị kết quả (nếu có)
if 'last_prediction' in st.session_state and st.session_state.last_prediction is not None:
    result = st.session_state.last_prediction
    
    # Hiển thị kết quả
    st.success("Hoàn tất!")
    
    # Tô màu cho kết quả (3 labels)
    color_map = {
        "POSITIVE": "green",
        "NEUTRAL": "gray",
        "NEGATIVE": "red"
    }
    color = color_map.get(result['label'], "blue")
    
    st.markdown(f"### Cảm xúc: :{color}[{result['label']}]")
    st.progress(result['score'], text=f"Độ tin cậy: {result['score']*100:.2f}%")