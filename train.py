from ultralytics import YOLO

def train():
    # model = YOLO("yolo11s.pt")
    model = YOLO("runs/detect/runs/detect/ingredients_multi/weights/best.pt")

    model.train(
        data="data3.yaml",
        epochs=100,
        freeze=10,
        lr0=1e-4,
        name="ingredients_multi_v2_inc"
    )

if __name__ == "__main__":
    train()