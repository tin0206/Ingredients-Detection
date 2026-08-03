import os
from datetime import datetime

from ultralytics import YOLO


def log_metrics(f, title, metrics):
    f.write(f"\n{'=' * 60}\n")
    f.write(f"{title}\n")
    f.write(f"{'=' * 60}\n")
    f.write(f"Precision (B):   {metrics.box.mp:.4f}\n")
    f.write(f"Recall (B):      {metrics.box.mr:.4f}\n")
    f.write(f"mAP50 (B):       {metrics.box.map50:.4f}\n")
    f.write(f"mAP50-95 (B):    {metrics.box.map:.4f}\n")
    f.write(f"Fitness:         {metrics.fitness:.4f}\n")

    f.write("\nPer-class AP50:\n")
    names = metrics.names
    ap50_per_class = metrics.box.ap50
    ap_classes = metrics.box.ap_class_index
    for idx, cls_idx in enumerate(ap_classes):
        f.write(f"  {names[int(cls_idx)]:<20s} {ap50_per_class[idx]:.4f}\n")

    print(title)
    print(f"  P={metrics.box.mp:.4f}  R={metrics.box.mr:.4f}  "
          f"mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")


def test_real():
    model = YOLO("runs/detect/test_v4.2/weights/best.pt")

    output_dir = "output_v4"
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "results.log")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Model: runs/detect/test_v4.2/weights/best.pt\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")

        # Synthetic test set
        synthetic_metrics = model.val(
            data="data_test_v4.yaml",
            split="test",
            imgsz=640,
            batch=16,
            project=output_dir,
            name="synthetic_test",
            exist_ok=True,
        )
        log_metrics(f, "SYNTHETIC TEST SET (dataset_test_v4/test)", synthetic_metrics)

        # Real test set
        real_metrics = model.val(
            data="data_test_v4.2.yaml",
            split="test",
            imgsz=640,
            batch=16,
            project=output_dir,
            name="real_test",
            exist_ok=True,
        )
        log_metrics(f, "REAL TEST SET (real_dataset_1000_imgs/test)", real_metrics)

    print(f"\nDone! Results logged to {log_path}")


if __name__ == "__main__":
    test_real()
