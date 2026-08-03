from pathlib import Path
from collections import Counter
import yaml

# ======================================================
# CONFIG
# ======================================================

DATASET_ROOT = Path("real_dataset")
YAML_FILE = "data_test_v4.1.yaml"

OUTPUT_FILE = "class_frequency.txt"

# ======================================================
# LOAD CLASSES
# ======================================================

with open(YAML_FILE, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

classes = data["names"]

counter = Counter()

total_boxes = 0
total_images = 0

# ======================================================
# COUNT LABELS
# ======================================================

for split in ["train", "val", "test"]:

    label_dir = DATASET_ROOT / split / "labels"

    if not label_dir.exists():
        continue

    txt_files = sorted(label_dir.glob("*.txt"))

    total_images += len(txt_files)

    for txt in txt_files:

        with open(txt, "r", encoding="utf-8") as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                try:
                    cls = int(parts[0])
                except:
                    continue

                counter[cls] += 1
                total_boxes += 1

# ======================================================
# WRITE RESULT
# ======================================================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    f.write("=" * 60 + "\n")
    f.write("REAL DATASET CLASS FREQUENCY\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Total images : {total_images}\n")
    f.write(f"Total boxes  : {total_boxes}\n")
    f.write(f"Total classes: {len(classes)}\n\n")

    f.write("-" * 60 + "\n")
    f.write(f"{'ID':<5}{'Class':<25}{'Frequency':>12}\n")
    f.write("-" * 60 + "\n")

    for cls_id in sorted(classes.keys()):

        freq = counter.get(cls_id, 0)

        f.write(
            f"{cls_id:<5}{classes[cls_id]:<25}{freq:>12}\n"
        )

    f.write("-" * 60 + "\n")

    missing = [
        classes[i]
        for i in classes
        if counter.get(i, 0) == 0
    ]

    f.write(f"\nMissing classes ({len(missing)}):\n")

    for m in missing:
        f.write(f"- {m}\n")

print(f"Saved to {OUTPUT_FILE}")