from ultralytics import YOLO

model = YOLO("runs/detect/runs/detect/ingredients_multi_v22/weights/best.pt")
print("NAMES:", model.names)
print("NC:", len(model.names))