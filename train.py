from ultralytics import YOLO

def train():
    # Load pretrained YOLOv8 model
    model = YOLO("yolov8s.pt")

    # Train
    model.train(
        data="dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=8,
        project="runs/detect",
        name="train_py",
    )

if __name__ == "__main__":
    train()
