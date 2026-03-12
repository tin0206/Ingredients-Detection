import os
import json
import matplotlib.pyplot as plt
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.data import build_detection_test_loader
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

# 1. ĐĂNG KÝ DATASET (Trỏ đúng vào file JSON đã tạo ở Bước 1)
register_coco_instances("food_train", {}, "dataset_v11/train/train_annotations.json", "dataset_v11/train/images")
register_coco_instances("food_val", {}, "dataset_v11/val/val_annotations.json", "dataset_v11/val/images")

def setup_config():
    cfg = get_cfg()
    
    # 2. CHỌN MODEL (Ở đây chọn Faster R-CNN ResNet-50)
    # Nếu muốn dùng RetinaNet, thay bằng: "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

    # 3. CẤU HÌNH DỮ LIỆU 
    cfg.DATASETS.TRAIN = ("food_train",)
    cfg.DATASETS.TEST = ("food_val",)
    cfg.DATALOADER.NUM_WORKERS = 4

    # 4. THÔNG SỐ HUẤN LUYỆN
    cfg.SOLVER.IMS_PER_BATCH = 8      # Batch size (Giảm xuống 4 nếu thiếu VRAM)
    cfg.SOLVER.BASE_LR = 0.00025      # Learning rate thấp cho Faster R-CNN
    cfg.SOLVER.MAX_ITER = 20000       # Với 242 classes, nên train ít nhất 20k-50k iterations
    cfg.SOLVER.STEPS = (15000,)       # Điểm giảm Learning rate
    
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 242  # BẮT BUỘC KHỚP VỚI NC CỦA BẠN

    cfg.OUTPUT_DIR = "./output_faster_rcnn"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg

def setup_retinanet_config():
    cfg = get_cfg()
    
    # 1. CHỌN MODEL RETINANET
    # Chúng ta dùng ResNet-50 kết hợp FPN (Feature Pyramid Network)
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")

    # 2. DỮ LIỆU
    cfg.DATASETS.TRAIN = ("food_train",)
    cfg.DATASETS.TEST = ("food_val",)
    
    # 3. THÔNG SỐ HUẤN LUYỆN
    cfg.SOLVER.IMS_PER_BATCH = 4       # RetinaNet nhẹ hơn Faster R-CNN nhưng vẫn khá tốn VRAM
    cfg.SOLVER.BASE_LR = 0.0001        # RetinaNet thường cần LR nhỏ hơn và ổn định hơn
    cfg.SOLVER.MAX_ITER = 30000        # 242 classes cần nhiều thời gian để hội tụ
    
    # 4. CẤU HÌNH RIÊNG CHO RETINANET
    cfg.MODEL.RETINANET.NUM_CLASSES = 242  # KHỚP VỚI NC CỦA BẠN
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.4
    
    cfg.OUTPUT_DIR = "./output_retinanet"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg

def plot_metrics(output_dir):
    experiment_metrics = []
    with open(os.path.join(output_dir, "metrics.json")) as f:
        for line in f:
            experiment_metrics.append(json.loads(line))

    plt.figure(figsize=(12, 6))
    
    # Vẽ Total Loss
    plt.subplot(1, 2, 1)
    plt.plot(
        [x["iteration"] for x in experiment_metrics if "total_loss" in x], 
        [x["total_loss"] for x in experiment_metrics if "total_loss" in x]
    )
    plt.title("Total Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")

    # Vẽ Learning Rate (Để xem điểm giảm LR 15k)
    plt.subplot(1, 2, 2)
    plt.plot(
        [x["iteration"] for x in experiment_metrics if "lr" in x], 
        [x["lr"] for x in experiment_metrics if "lr" in x]
    )
    plt.title("Learning Rate")
    plt.xlabel("Iteration")
    plt.ylabel("LR")
    plt.tight_layout()
    plt.show()

# plot_metrics("./output_faster_rcnn")

if __name__ == "__main__":
    # 1. Khởi tạo cấu hình (Chọn RetinaNet như bạn đang làm)
    cfg = setup_retinanet_config() 
    
    # 2. Huấn luyện (Chạy xuyên suốt cho đến khi xong 30,000 iter)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    # 3. Sau khi train xong mới chạy Evaluation
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    
    # Check kiến trúc để set threshold đúng
    is_retinanet = "retinanet" in cfg.MODEL.META_ARCHITECTURE.lower()
    if is_retinanet:
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
        print("\n--- Đang đánh giá mô hình RetinaNet ---")
    else:
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        print("\n--- Đang đánh giá mô hình Faster R-CNN ---")
        
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("food_val", output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "food_val")
    
    results = inference_on_dataset(predictor.model, val_loader, evaluator)
    
    # 4. CUỐI CÙNG: In bảng điểm ra terminal và hiện biểu đồ
    print("\n--- KẾT QUẢ CUỐI CÙNG ---")
    print(results)
    
    print("\n--- ĐANG HIỂN THỊ BIỂU ĐỒ LOSS & LEARNING RATE ---")
    plot_metrics(cfg.OUTPUT_DIR)