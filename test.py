from ultralytics import YOLO

def test_image():
    # load model đã train
    
    #dataset v3
    # model = YOLO("runs/detect/ingredients_multi_v2_inc2/weights/best.pt")
    
    #dataset
    model = YOLO("runs/detect/runs/detect/ingredients_multi/weights/best.pt")

    # inference
    results = model(
        source="test.jpg",   # đường dẫn ảnh
        imgsz=640,
        conf=0.25,
        save=True            # lưu ảnh kết quả
    )

    print("Done inference!")

if __name__ == "__main__":
    test_image()
