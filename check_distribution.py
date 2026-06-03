import os
import yaml
from pathlib import Path
from collections import defaultdict

# ================= CONFIG =================
DATA_YAML = "data.yaml"
DATASET_ROOT = Path("dataset")
SPLITS = ["train", "val", "test"]
OUTPUT_LOG = "dataset_distribution_report.txt"
# ==========================================

def log_distribution():
    # 1. Load tên class từ data.yaml
    if not Path(DATA_YAML).exists():
        print(f"❌ Không tìm thấy file {DATA_YAML}")
        return
        
    with open(DATA_YAML, "r", encoding="utf-8") as f:
        data_config = yaml.safe_load(f)
        id_to_name = {int(k): v for k, v in data_config["names"].items()}

    report_lines = []
    report_lines.append("==========================================")
    report_lines.append("   DATASET CLASS DISTRIBUTION REPORT")
    report_lines.append("==========================================\n")

    # 2. Duyệt qua từng split để đếm
    for split in SPLITS:
        label_dir = DATASET_ROOT / split / "labels"
        if not label_dir.exists():
            continue

        counts = defaultdict(int)
        label_files = list(label_dir.glob("*.txt"))
        
        for lbl in label_files:
            with open(lbl, "r") as f:
                for line in f:
                    parts = line.split()
                    if parts:
                        class_id = int(parts[0])
                        counts[class_id] += 1

        # 3. Ghi kết quả cho từng split vào danh sách dòng
        report_lines.append(f"--- SPLIT: {split.upper()} ---")
        report_lines.append(f"Tổng số ảnh: {len(label_files)}")
        report_lines.append(f"Tổng số vật thể (instances): {sum(counts.values())}")
        report_lines.append(f"{'ID':<5} | {'Class Name':<25} | {'Count':<8}")
        report_lines.append("-" * 45)

        # Sắp xếp theo ID hoặc theo số lượng (ở đây sắp theo ID)
        sorted_ids = sorted(counts.keys())
        for cid in sorted_ids:
            name = id_to_name.get(cid, "Unknown")
            count = counts[cid]
            report_lines.append(f"{cid:<5} | {name:<25} | {count:<8}")
        
        # Thống kê nhanh
        if counts:
            sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            report_lines.append(f"\nTop 3 nhiều nhất: {[(id_to_name.get(x[0]), x[1]) for x in sorted_counts[:3]]}")
            report_lines.append(f"Top 3 ít nhất: {[(id_to_name.get(x[0]), x[1]) for x in sorted_counts[-3:]]}")
        
        report_lines.append("\n" + "="*45 + "\n")

    # 4. Xuất ra file .txt
    with open(OUTPUT_LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print(f"✅ Đã lưu báo cáo phân phối tại: {OUTPUT_LOG}")

if __name__ == "__main__":
    log_distribution()