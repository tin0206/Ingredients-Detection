from ultralytics import YOLO

def test_image():
    # load model đã train
    #current choice - yolo11s datasetv12
    # model = YOLO("runs/detect/train5/weights/best.pt")
    
    # model = YOLO("runs/detect/train8/weights/best.pt")
    model = YOLO("runs/detect/train-4/weights/best.pt")
    
    # tests = ["test4.jpg"]
    tests = ["test.jpg", "test2.jpg", "test3.jpg", "test4.jpg", "test5.jpg", "test6.jpg", "test7.jpg", "test8.jpg", "test9.jpg", "test10.jpg", "test11.jpg", "test12.png", "test13.png"]

    # inference
    for img_path in tests:
        results = model(
            source=img_path,   # Truyền TỪNG ẢNH một thay vì truyền cả mảng
            imgsz=640,
            conf=0.1,         # Nên nâng conf lên một chút như đã bàn nhé
            save=True
        )
    
    for r in results:
        print(r.boxes)

    print("Done inference!")

if __name__ == "__main__":
    test_image()
