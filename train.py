from ultralytics import YOLO

def train():
    model = YOLO("yolo11s.pt")

    model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=8,
        project="runs/detect",
        name="ingredients_multi"
    )

if __name__ == "__main__":
    train()