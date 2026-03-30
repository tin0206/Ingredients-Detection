import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# --- Cấu hình Trang ---
st.set_page_config(page_title="Ingredients Detection Demo", layout="wide")
st.title("🍳 Hệ thống Nhận diện Nguyên liệu Thực phẩm")
st.write("Tải ảnh lên để trích xuất danh sách nguyên liệu (đã lọc trùng).")

# --- Load Model ---
@st.cache_resource # Cache để không phải load lại model mỗi khi bấm nút
def load_model():
    model_path = "runs/detect/train19/weights/best.pt"
    return YOLO(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Không tìm thấy model tại {model_path}. Vui lòng kiểm tra lại đường dẫn.")
    st.stop()

# --- Giao diện Sidebar ---
st.sidebar.header("Cấu hình Model")
conf_threshold = st.sidebar.slider("Độ tự tin (Confidence)", 0.01, 1.0, 0.01, step=0.01)
iou_threshold = st.sidebar.slider("Ngưỡng chồng lấp (IoU)", 0.1, 1.0, 0.45)

# --- Upload Ảnh ---
uploaded_file = st.file_uploader("Chọn một bức ảnh thực phẩm...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc ảnh
    image = Image.open(uploaded_file)
    
    # Tạo 2 cột để so sánh
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Ảnh gốc", use_container_width=True)
    
    # --- Dự đoán ---
    with st.spinner('Đang phân tích...'):
        # Chạy inference
        results = model.predict(
            source=image, 
            conf=conf_threshold, 
            iou=iou_threshold,
            augment=True # Bật augment để tăng độ chính xác khi demo
        )
        
        # Lấy ảnh đã vẽ bounding box
        res_plotted = results[0].plot()
        
        # Lấy danh sách các class name đã detect được
        detected_names = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            name = model.names[class_id]
            detected_names.append(name)
        
        # --- LOGIC LỌC TRÙNG ---
        # Sử dụng set() để loại bỏ phần tử trùng và sorted() để sắp xếp bảng chữ cái
        unique_ingredients = sorted(list(set(detected_names)))

    with col2:
        st.image(res_plotted, caption="Kết quả nhận diện", use_container_width=True)

    # --- Hiển thị Danh sách Nguyên liệu ---
    st.divider()
    st.subheader(f"📋 Danh sách nguyên liệu tìm thấy ({len(unique_ingredients)})")
    
    if unique_ingredients:
        # Hiển thị dạng tag/badge cho đẹp
        cols = st.columns(5)
        for idx, ingredient in enumerate(unique_ingredients):
            cols[idx % 5].info(f"**{ingredient}**")
    else:
        st.warning("Không tìm thấy nguyên liệu nào với mức Confidence hiện tại.")

# Chạy bằng lệnh: streamlit run demo.py