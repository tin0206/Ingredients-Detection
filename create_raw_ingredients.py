import yaml
from pathlib import Path

def create_raw_ingredients_structure(yaml_path, output_root):
    # 1. Kiểm tra file YAML có tồn tại không
    yaml_file = Path(yaml_path)
    if not yaml_file.exists():
        print(f"Lỗi: Không tìm thấy file {yaml_path}")
        return

    # 2. Đọc nội dung file YAML
    try:
        with open(yaml_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        # Lấy dictionary 'names' (chứa index: tên_nguyên_liệu)
        class_names = data.get("names", {})
        
        if not class_names:
            print("Lỗi: Không tìm thấy mục 'names' trong file YAML.")
            return

        # 3. Tạo thư mục gốc raw_ingredients
        root_path = Path(output_root)
        root_path.mkdir(parents=True, exist_ok=True)
        print(f"Đã tạo/kiểm tra thư mục gốc: {root_path.absolute()}")

        # 4. Tạo các thư mục con theo Index
        count = 0
        for index in class_names.keys():
            # Tạo thư mục với tên là index (ép kiểu về string để làm tên folder)
            subfolder = root_path / str(index)
            subfolder.mkdir(exist_ok=True)
            count += 1
        
        print(f"Hoàn thành! Đã tạo {count} thư mục con tương ứng với các class trong {yaml_path}.")

    except Exception as e:
        print(f"Đã xảy ra lỗi khi xử lý: {e}")

if __name__ == "__main__":
    # Tên file cấu hình của bạn
    YAML_CONFIG = "data12.yaml"
    # Tên thư mục gốc muốn tạo
    RAW_DIR = "raw_ingredients"
    
    create_raw_ingredients_structure(YAML_CONFIG, RAW_DIR)