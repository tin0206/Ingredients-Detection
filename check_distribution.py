import os
from collections import Counter
import yaml
from pathlib import Path

def check_distribution(label_dir, yaml_path):
    # Load tên class từ file yaml
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
        class_names = data['names']

    label_files = list(Path(label_dir).glob('*.txt'))
    stats = Counter()

    print(f"Scanning {len(label_files)} label files...")
    
    for lbl in label_files:
        with open(lbl, 'r') as f:
            for line in f:
                class_id = int(line.split()[0])
                stats[class_id] += 1

    # In kết quả theo thứ tự số lượng giảm dần
    print(f"{'ID':<5} | {'Class Name':<25} | {'Count':<10}")
    print("-" * 45)
    
    for cid, count in stats.most_common():
        name = class_names.get(cid, "Unknown")
        print(f"{cid:<5} | {name:<25} | {count:<10}")

    # Kiểm tra xem có class nào bị 0 instance không
    missing = set(class_names.keys()) - set(stats.keys())
    if missing:
        print(f"\n⚠ CẢNH BÁO: Có {len(missing)} class không có dữ liệu:")
        for m in missing:
            print(f"- {m}: {class_names[m]}")

# Chạy thử cho tập train
check_distribution("dataset_v12/train/labels", "data12.yaml")