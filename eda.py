import os
import yaml
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

DATA_YAML = "data3.yaml"
DATASET_ROOT = "dataset_v3"
SPLITS = ["train", "valid", "test"]


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    data = load_yaml(DATA_YAML)
    names = data["names"]          # {id: class_name}
    id_to_name = {int(k): v for k, v in names.items()}

    class_count = defaultdict(int)

    # ---- count images containing each class
    for split in SPLITS:
        label_dir = os.path.join(DATASET_ROOT, split, "labels")
        if not os.path.exists(label_dir):
            continue

        for fname in os.listdir(label_dir):
            if not fname.endswith(".txt"):
                continue

            seen = set()   # reset for each image

            with open(os.path.join(label_dir, fname), "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue

                    cls_id = int(parts[0])

                    if cls_id not in seen:
                        class_count[cls_id] += 1
                        seen.add(cls_id)

    # ---- prepare data for plot
    class_ids = sorted(id_to_name.keys())
    class_names = [id_to_name[i] for i in class_ids]
    items = sorted(class_count.items(), key=lambda x: x[1], reverse=True)

    TOP_K = 30
    top_items = items[:TOP_K]

    labels = [id_to_name[k] for k, _ in top_items]
    values = [v for _, v in top_items]

    x = np.arange(len(labels))

    plt.figure(figsize=(14, 6))
    plt.bar(x, values, width=0.7)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.xlabel("Class")
    plt.ylabel("Number of images")
    plt.title(f"Top {TOP_K} Most Frequent Classes")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
