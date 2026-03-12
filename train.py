from ultralytics import YOLO

def train():
    model = YOLO("yolo11s.pt")
    # model = YOLO("runs/detect/ingredients_multi_v6_inc45/weights/best.pt")

    # model.train(
    #     data="data6.yaml",
        # epochs=80,
        # imgsz=512,
        # batch=16,
        # lr0=5e-4,
        # mosaic=1.0,
        # mixup=0.2,
        # hsv_h=0.02,
        # hsv_s=0.8,
        # hsv_v=0.5,
        # scale=0.7,
        # degrees=15,
        # close_mosaic=15,
    #     name="ingredients_multi_v6_inc4",
    # )
    
    # model.train(
    #     data="data11.yaml",
    #     epochs=70,
    #     imgsz=640,
    #     batch=16,
    #     lr0=5e-4,
    #     mosaic=1.0,
    #     mixup=0.2,
    #     name="finetune_dense_stage2_v2"
    # )
    
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
    
    # test
    model.train(
        data="data12.yaml",
        epochs=70,              # Tăng epoch vì dataset lớn cần nhiều thời gian học hơn
        imgsz=640,               
        batch=20,
        workers=8,               # Tăng tốc load dữ liệu (thử 8, 12 hoặc 16 tùy CPU)
        device=0,                # Đảm bảo sử dụng GPU (0 là card đầu tiên)
        
        # --- Siêu tham số tối ưu hóa ---
        lr0=1e-3,                # Tăng nhẹ lr0 nếu dùng batch size lớn
        cos_lr=True,             # Giúp mAP ổn định ở cuối quá trình train
        label_smoothing=0.1,     # Cải thiện khả năng phân biệt class
        cls = 1.5,
        
        # --- Augmentation mạnh mẽ ---
        mosaic=1.0, 
        mixup=0.2,               # Tăng mixup để model học vật thể đè lên nhau tốt hơn
        scale=0.8,               # Cho phép zoom ảnh linh hoạt hơn
        flipud=0.5,              # Lật ảnh theo chiều dọc (nguyên liệu thực phẩm nhìn từ trên xuống)
        hsv_h=0.015, 
        hsv_s=0.4,            # Giảm bớt độ bão hòa màu để model nhìn rõ khối hơn
        hsv_v=0.4,
        degrees=15.0,
        
        # --- Kỹ thuật ---
        close_mosaic=20,         # Tắt mosaic sớm hơn để model tinh chỉnh vị trí box
        amp=True,                # Bật Mixed Precision
    )

if __name__ == "__main__":
    train()