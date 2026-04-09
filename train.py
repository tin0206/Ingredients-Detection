from ultralytics import YOLO

def train():
    # model = YOLO("yolo11s.pt")
    model = YOLO("yolo26s.pt")
    # model = YOLO("yolo26m.pt")
    # model = YOLO("runs/detect/ingredients_multi_v6_inc45/weights/best.pt")
    
    # current choice
    # model.train(
    #     data="data11.yaml",
    #     epochs=70,
    #     imgsz=640,
    #     batch=16,
    #     lr0=5e-4,
    #     mosaic=1.0,
    #     mixup=0.15,
    #     hsv_h=0.015,
    #     hsv_s=0.7,
    #     hsv_v=0.4,
    #     scale=0.5,
    #     degrees=10,
    #     close_mosaic=15,
    # )
    
    # current best
    # model.train(
    #     data="data12.yaml",
    #     epochs=70,              # Tăng epoch vì dataset lớn cần nhiều thời gian học hơn
    #     imgsz=640,               
    #     batch=20,
    #     workers=8,               # Tăng tốc load dữ liệu (thử 8, 12 hoặc 16 tùy CPU)
    #     device=0,                # Đảm bảo sử dụng GPU (0 là card đầu tiên)
        
    #     # --- Siêu tham số tối ưu hóa ---
    #     lr0=1e-3,                # Tăng nhẹ lr0 nếu dùng batch size lớn
    #     cos_lr=True,             # Giúp mAP ổn định ở cuối quá trình train
    #     label_smoothing=0.1,     # Cải thiện khả năng phân biệt class
    #     cls = 1.5,
        
    #     # --- Augmentation mạnh mẽ ---
    #     mosaic=1.0, 
    #     mixup=0.2,               # Tăng mixup để model học vật thể đè lên nhau tốt hơn
    #     scale=0.8,               # Cho phép zoom ảnh linh hoạt hơn
    #     flipud=0.5,              # Lật ảnh theo chiều dọc (nguyên liệu thực phẩm nhìn từ trên xuống)
    #     hsv_h=0.015, 
    #     hsv_s=0.4,            # Giảm bớt độ bão hòa màu để model nhìn rõ khối hơn
    #     hsv_v=0.4,
    #     degrees=15.0,
        
    #     # --- Kỹ thuật ---
    #     close_mosaic=20,         # Tắt mosaic sớm hơn để model tinh chỉnh vị trí box
    #     amp=True,                # Bật Mixed Precision
    # )
    
    # test yolo26s (train8)
    # model.train(
    #     data="data12.yaml",
    #     epochs=70,              # Tăng epoch vì dataset lớn cần nhiều thời gian học hơn
    #     imgsz=768,               
    #     batch=12,
    #     workers=8,               # Tăng tốc load dữ liệu (thử 8, 12 hoặc 16 tùy CPU)
    #     device=0,                # Đảm bảo sử dụng GPU (0 là card đầu tiên)
        
    #     # --- Siêu tham số tối ưu hóa ---
    #     lr0=1e-3,                # Tăng nhẹ lr0 nếu dùng batch size lớn
    #     cos_lr=True,             # Giúp mAP ổn định ở cuối quá trình train
    #     label_smoothing=0.1,     # Cải thiện khả năng phân biệt class
    #     cls = 1.5,
        
    #     # --- Augmentation mạnh mẽ ---
    #     mosaic=1.0, 
    #     mixup=0.2,               # Tăng mixup để model học vật thể đè lên nhau tốt hơn
    #     scale=0.8,               # Cho phép zoom ảnh linh hoạt hơn
    #     flipud=0.5,              # Lật ảnh theo chiều dọc (nguyên liệu thực phẩm nhìn từ trên xuống)
    #     hsv_h=0.015, 
    #     hsv_s=0.4,            # Giảm bớt độ bão hòa màu để model nhìn rõ khối hơn
    #     hsv_v=0.4,
    #     degrees=15.0,
        
    #     # --- Kỹ thuật ---
    #     close_mosaic=20,         # Tắt mosaic sớm hơn để model tinh chỉnh vị trí box
    #     amp=True,                # Bật Mixed Precision
    # )
    
    # test yolo26s
    model.train(
        data="data.yaml",
        project="D:/Code/Ingredients-Detection/output_yolo26s",
        name="YOLO26s_v2",
        
        epochs=70,              # Tăng epoch vì dataset lớn cần nhiều thời gian học hơn
        imgsz=768,               
        batch=12,
        workers=8,               # Tăng tốc load dữ liệu (thử 8, 12 hoặc 16 tùy CPU)
        device=0,                # Đảm bảo sử dụng GPU (0 là card đầu tiên)
        
        # --- Siêu tham số tối ưu hóa ---
        lr0=1e-3,                # Tăng nhẹ lr0 nếu dùng batch size lớn
        cos_lr=True,             # Giúp mAP ổn định ở cuối quá trình train
        label_smoothing=0.1,     # Cải thiện khả năng phân biệt class
        cls = 1.5,
        optimizer='auto',      # [MỚI] Để YOLO tự chọn AdamW hoặc SGD
        weight_decay=0.0005,   # [MỚI] Tránh overfitting cho các class hiếm
        
        # --- Augmentation mạnh mẽ ---
        mosaic=0.8,            # [GIẢM NHẸ] Vì dataset sinh ra đã có cảnh dense sẵn rồi
        mixup=0.1,             # [GIẢM NHẸ] 0.2 có thể làm mờ vật thể nhỏ quá mức
        scale=0.5,             # [TĂNG] Để model học vật thể ở nhiều khoảng cách hơn
        flipud=0.5, 
        fliplr=0.5,            # Đảm bảo lật cả trái phải
        hsv_h=0.015, 
        hsv_s=0.4, 
        hsv_v=0.4,
        degrees=15.0,
        
        # --- Kỹ thuật ---
        close_mosaic=10,
        patience=15,           # [MỚI] Tránh treo máy khi model đã đạt đỉnh
        amp=True,                # Bật Mixed Precision
    )

if __name__ == "__main__":
    train()