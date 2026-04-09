import os
import yaml
from pathlib import Path

# --- Cấu hình đường dẫn ---
PROCESSED_DIR = Path("processed_ingredients")
DATA_YAML = "data.yaml"

def check_class_counts():
    # 1. Đọc tên class từ data.yaml
    with open(DATA_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        class_names = data.get("names", {})

    print(f"{'ID':<5} | {'Tên Class':<25} | {'Số lượng ảnh'}")
    print("-" * 50)

    total_images = 0
    # 2. Quét qua các thư mục ID trong processed_ingredients
    # Sắp xếp theo ID số để dễ nhìn
    subdirs = sorted([d for d in PROCESSED_DIR.iterdir() if d.is_dir()], 
                    key=lambda x: int(x.name) if x.name.isdigit() else 999)

    for subdir in subdirs:
        class_id = int(subdir.name) if subdir.name.isdigit() else subdir.name
        name = class_names.get(class_id, "Unknown")
        
        # Đếm các file ảnh (png, jpg, jpeg)
        image_count = len([f for f in subdir.glob("*") if f.suffix.lower() in [".png", ".jpg", ".jpeg"]])
        
        print(f"{str(class_id):<5} | {name:<25} | {image_count} ảnh")
        total_images += image_count

    print("-" * 50)
    print(f"Tổng cộng: {len(subdirs)} classes - {total_images} ảnh")

if __name__ == "__main__":
    if PROCESSED_DIR.exists():
        check_class_counts()
    else:
        print(f"Lỗi: Không tìm thấy thư mục {PROCESSED_DIR}")