from ultralytics import YOLO, RTDETR

def train():
    # model = YOLO("yolo26s.pt")
    model = RTDETR("rtdetr-l.pt")

    # test yolo26s
    model.train(
        data="data.yaml",
        epochs=150,              # Tăng epoch vì dataset lớn cần nhiều thời gian học hơn
        imgsz=640,               
        batch=20,                # Tăng batch size nếu GPU đủ VRAM để cải thiện ổn định và tốc độ train
        workers=8,               # Tăng tốc load dữ liệu (thử 8, 12 hoặc 16 tùy CPU)
        device=0,                # Đảm bảo sử dụng GPU (0 là card đầu tiên)
        
        # --- Siêu tham số tối ưu hóa ---
        lr0=1e-3,                # Tăng nhẹ lr0 nếu dùng batch size lớn
        cos_lr=True,             # Giúp mAP ổn định ở cuối quá trình train
        label_smoothing=0.05,     # Cải thiện khả năng phân biệt class
        
        # --- Augmentation mạnh mẽ ---
        mosaic=1.0, 
        mixup=0.1,               # Tăng mixup để model học vật thể đè lên nhau tốt hơn
        scale=0.9,               # Cho phép zoom ảnh linh hoạt hơn
        flipud=0.5,              # Lật ảnh theo chiều dọc (nguyên liệu thực phẩm nhìn từ trên xuống)
        hsv_h=0.015, 
        hsv_s=0.4,            # Giảm bớt độ bão hòa màu để model nhìn rõ khối hơn
        hsv_v=0.4,
        degrees=15.0,
        
        # --- Kỹ thuật ---
        close_mosaic=30,         # Tắt mosaic sớm hơn để model tinh chỉnh vị trí box
        amp=True,                # Bật Mixed Precision
    )
    
    # test rtdetr-l
    # model.train(
    #     data="data.yaml",
    #     epochs=150,
    #     imgsz=640,               # Khuyến nghị 640 cho RT-DETR để tối ưu bộ nhớ VRAM
    #     batch=8,                 # RT-DETR tốn VRAM hơn, nếu 896 bị crash thì hạ xuống 640
    #     workers=8,
    #     device=0,
        
    #     # --- Siêu tham số cho Transformer ---
    #     lr0=1e-4,              # QUAN TRỌNG: Transformer cần LR thấp hơn YOLO (thường là 1e-4)
    #     lrf=1e-6,                # LR cuối cùng rất thấp để fine-tune tốt hơn
    #     cos_lr=True,
    #     warmup_epochs=5,
    #     label_smoothing=0.05,
        
    #     # --- Augmentation (Giữ nguyên để so sánh Data Engineering) ---
    #     mosaic=1.0, 
    #     mixup=0.1, 
    #     scale=0.9, 
    #     flipud=0.5, 
    #     hsv_h=0.015, 
    #     hsv_s=0.4, 
    #     hsv_v=0.4,
    #     degrees=15.0,
        
    #     # --- Kỹ thuật đặc thù ---
    #     close_mosaic=30,
    #     patience=15,         
    #     amp=True,         # Tiếp tục train từ checkpoint nếu có (rtdetr-l.pt hoặc last.pt
    # )

if __name__ == "__main__":
    train()