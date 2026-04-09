import pandas as pd
from ultralytics import YOLO
import os

def run_evaluation():
    # 1. Define paths to your best weights
    model_configs = {
        "YOLO11s": {"path": "runs/detect/train5/weights/best.pt", "imgsz": 640},
        "YOLO26s": {"path": "runs/detect/train8/weights/best.pt", "imgsz": 640} 
    }

    data_yaml = "data12.yaml"
    results_list = []

    for name, config in model_configs.items():
        if not os.path.exists(config["path"]):
            print(f"Warning: {name} weights not found at {config['path']}")
            continue

        print(f"\n--- Evaluating {name} on Test Set ---")
        
        # Load the trained model
        model = YOLO(config["path"])
        
        # Run validation on the TEST split
        metrics = model.val(
            data=data_yaml,
            split='test',
            imgsz=config["imgsz"],
            batch=16,
            conf=0.001,
            iou=0.6
        )
        
        # Extract metrics
        results_list.append({
            "Model": name,
            "Precision": metrics.results_dict.get('metrics/precision(B)', 0),
            "Recall": metrics.results_dict.get('metrics/recall(B)', 0),
            "mAP50": metrics.results_dict.get('metrics/mAP50(B)', 0),
            "mAP50-95": metrics.results_dict.get('metrics/mAP50-95(B)', 0)
        })

    # 2. Display and Save Comparison Table
    if results_list:
        df = pd.DataFrame(results_list)
        print("\nFINAL TEST SET COMPARISON:")
        print(df.to_string(index=False))
        df.to_csv("test_set_comparison_results.csv", index=False)
        print("\nResults saved to 'test_set_comparison_results.csv'")

# This block is MANDATORY on Windows to avoid the RuntimeError
if __name__ == '__main__':
    run_evaluation()