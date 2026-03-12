from ultralytics import YOLO

def test_image():
    # load model đã train
    
    #dataset v3
    # model = YOLO("runs/detect/ingredients_multi_v2_inc2/weights/best.pt")
    
    #dataset
    # model = YOLO("runs/detect/ingredients_multi_v6_inc45/weights/best.pt")
    model = YOLO("runs/detect/train5/weights/best.pt")

    # inference
    results = model(
        source="test.jpg",   # đường dẫn ảnh
        imgsz=640,
        conf=0.01,
        save=True            # lưu ảnh kết quả
    )
    
    for r in results:
        print(r.boxes)

    print("Done inference!")

if __name__ == "__main__":
    test_image()
