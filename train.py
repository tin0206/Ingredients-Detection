from ultralytics import YOLO

def train():
    # model = YOLO("yolo26s.pt")
    model = YOLO("runs/detect/test_v4/weights/best.pt")

    # test yolo26s
    # model.train(
    #     data="data_test_v4.yaml",

    #     epochs=50,
    #     imgsz=640,
    #     batch=20,
    #     workers=8,
    #     device=0,

    #     lr0=1e-3,
    #     lrf=0.01,
    #     cos_lr=True,

    #     mosaic=1.0,
    #     mixup=0.05,
    #     scale=0.5,
    #     degrees=10.0,
    #     translate=0.1,
    #     shear=2.0,

    #     hsv_h=0.015,
    #     hsv_s=0.25,
    #     hsv_v=0.25,

    #     fliplr=0.5,
    #     flipud=0.3,

    #     close_mosaic=30,
         
    #     amp=True,
    #     name="test_v4.1",
    #     exist_ok=True
    # )
    
    model.train(
        data="data_test_v4.2.yaml",
        
        epochs=50,
        imgsz=640,
        batch=16,
        workers=8,
        device=0,

        lr0=1e-4,        # giảm LR để không phá weight cũ
        lrf=0.01,
        cos_lr=True,

        mosaic=0.2,      # giảm mạnh
        mixup=0.0,
        scale=0.3,
        degrees=5.0,
        translate=0.05,
        shear=0.0,

        hsv_h=0.01,
        hsv_s=0.2,
        hsv_v=0.2,

        fliplr=0.5,
        flipud=0.0,      # ảnh thật món ăn không nên lật dọc nhiều

        close_mosaic=10,

        amp=True,
        name="test_v4.2",
        exist_ok=True
    )
    
if __name__ == "__main__":
    train()