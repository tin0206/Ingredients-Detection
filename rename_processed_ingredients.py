import os
from pathlib import Path
import uuid

def rename_ingredients_final(root_dir):
    root_path = Path(root_dir)
    
    if not root_path.exists():
        print(f"Lỗi: Không tìm thấy thư mục {root_dir}")
        return

    subfolders = sorted([f for f in root_path.iterdir() if f.is_dir()], 
                        key=lambda x: int(x.name) if x.name.isdigit() else x.name)

    for folder in subfolders:
        print(f"--- Đang xử lý folder: {folder.name} ---")
        
        # Lấy danh sách file hiện có
        files = list(folder.glob("*.*"))
        files.sort()

        if not files:
            continue

        # BƯỚC 1: Đổi tên sang UUID ngẫu nhiên để tránh xung đột tuyệt đối
        temp_files = []
        for file_path in files:
            temp_name = f"{uuid.uuid4()}.tmp"
            temp_path = file_path.with_name(temp_name)
            file_path.rename(temp_path)
            temp_files.append(temp_path)

        # BƯỚC 2: Bây giờ folder đã sạch, ta đổi tên sang số 0.png, 1.png...
        for i, temp_path in enumerate(temp_files):
            new_name = f"{i}.png"
            final_path = folder / new_name
            
            try:
                temp_path.rename(final_path)
            except Exception as e:
                print(f"  ⚠ Lỗi nghiêm trọng tại {folder.name} file {i}: {e}")

        print(f"  ✅ Thành công! Đã đánh số lại {len(temp_files)} file.")

if __name__ == "__main__":
    TARGET_DIR = "processed_ingredients_test_v2"
    rename_ingredients_final(TARGET_DIR)