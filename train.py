from ultralytics import YOLO

def train():
    model = YOLO("yolo11s.pt")
    # model = YOLO("runs/detect/ingredients_multi_v86/weights/best.pt")

    model.train(
        data="data6.yaml",
        epochs=80,
        imgsz=512,
        batch=16,
        lr0=5e-4,
        mosaic=1.0,
        mixup=0.2,
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.5,
        scale=0.7,
        degrees=15,
        close_mosaic=15,
        name="ingredients_multi_v6_inc4",
    )

if __name__ == "__main__":
    train()