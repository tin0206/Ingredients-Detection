import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import torch # Dùng để xử lý mảng box nhanh hơn

# --- Cấu hình Trang ---
st.set_page_config(page_title="Ingredients Detection Demo", layout="wide")
st.title("🍳 Hệ thống Nhận diện Nguyên liệu (Tối ưu Label)")

@st.cache_resource 
def load_model():
    model_path = "runs/detect/train8/weights/best.pt"
    return YOLO(model_path)

model = load_model()

# --- Sidebar ---
st.sidebar.header("Cấu hình Model")
conf_threshold = st.sidebar.slider("Độ tự tin (Confidence)", 0.01, 1.0, 0.01, step=0.01)
# Ngưỡng IoU thấp (ví dụ 0.3) sẽ giúp loại bỏ các box chồng lấn mạnh mẽ hơn
iou_threshold = st.sidebar.slider("Ngưỡng chồng lấp (IoU NMS)", 0.1, 1.0, 0.30)

uploaded_file = st.file_uploader("Chọn ảnh...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Ảnh gốc", use_container_width=True)
    
    with st.spinner('Đang phân tích...'):
        # Chạy inference với NMS (Non-Maximum Suppression) chặt chẽ
        results = model.predict(
            source=image, 
            conf=conf_threshold, 
            iou=iou_threshold, # Giảm IoU ở đây sẽ lọc bớt box chồng nhau ngay từ đầu
            augment=True 
        )
        
        # --- LOGIC LỌC LẠI BOX VÀ DANH SÁCH ---
        best_ingredients = {}
        boxes = results[0].boxes
        
        # Nếu có nhiều box cho cùng 1 vị trí, YOLO đã lọc bằng NMS qua tham số 'iou' ở trên.
        # Bây giờ ta lọc danh sách hiển thị để mỗi loại nguyên liệu chỉ xuất hiện 1 lần với Conf cao nhất.
        for box in boxes:
            class_id = int(box.cls[0])
            name = model.names[class_id]
            conf = float(box.conf[0])
            
            if name not in best_ingredients or conf > best_ingredients[name]:
                best_ingredients[name] = conf

        # Tạo ảnh kết quả: Nếu bạn muốn ảnh TRÔNG SẠCH HƠN (ít box trùng), 
        # hãy điều chỉnh thanh trượt "Ngưỡng chồng lấp (IoU)" xuống thấp (về phía 0.1 - 0.3).
        res_plotted = results[0].plot()
        
    with col2:
        st.image(res_plotted, caption="Kết quả (Đã lọc NMS)", use_container_width=True)

    # --- HIỂN THỊ DANH SÁCH ĐÃ LỌC ---
    st.divider()
    unique_list = sorted(best_ingredients.items(), key=lambda x: x[1], reverse=True)
    st.subheader(f"📋 Danh sách nguyên liệu tìm thấy ({len(unique_list)})")
    
    if unique_list:
        # Hiển thị Top nguyên liệu có độ tự tin cao nhất trước
        cols = st.columns(5)
        for idx, (name, conf) in enumerate(unique_list):
            cols[idx % 5].success(f"**{name}**\n\nConf: {conf:.2f}")
    else:
        st.warning("Không tìm thấy nguyên liệu nào.")