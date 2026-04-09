from ultralytics import YOLO

def test_image():
    # load model đã train
    #current choice - yolo11s datasetv12
    # model = YOLO("runs/detect/train5/weights/best.pt")
    
    # model = YOLO("runs/detect/train8/weights/best.pt")
    model = YOLO("D:/Code/Ingredients-Detection/output_yolo26s/YOLO26s_v22/weights/best.pt")
    
    tests = ["test.jpg", "test2.jpg", "test3.jpg", "test4.jpg", "test5.jpg", "test6.jpg", "test7.jpg"]
    

    # inference
    results = model(
        source=tests,
        imgsz=640,
        conf=0.01,
        save=True            # lưu ảnh kết quả
    )
    
    for r in results:
        print(r.boxes)

    print("Done inference!")

if __name__ == "__main__":
    test_image()
