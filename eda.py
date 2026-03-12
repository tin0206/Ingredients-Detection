import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

# ==============================
# CONFIG
# ==============================
DATA_YAML = "data11.yaml"
DATASET_ROOT = "dataset_v11"
SPLITS = ["train", "val", "test"]
OUTPUT_DIR = "eda_results_v11"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================
# LOAD YAML
# ==============================
with open(DATA_YAML, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

names = data["names"]
id_to_name = {int(k): v for k, v in names.items()}
num_classes = len(id_to_name)

print("=" * 60)
print("DATASET BASIC INFO")
print("=" * 60)
print("Number of classes:", num_classes)


# ==============================
# INITIALIZE METRICS
# ==============================
instance_count = defaultdict(int)
image_count = defaultdict(int)
split_instance_count = {split: defaultdict(int) for split in SPLITS}

bbox_areas = []
bbox_widths = []
bbox_heights = []
objects_per_image = []

total_images = 0
total_instances = 0


# ==============================
# MAIN LOOP
# ==============================
for split in SPLITS:
    label_dir = os.path.join(DATASET_ROOT, split, "labels")

    if not os.path.exists(label_dir):
        continue

    for fname in os.listdir(label_dir):
        if not fname.endswith(".txt"):
            continue

        total_images += 1
        total_objects_in_image = 0
        seen_classes = set()

        with open(os.path.join(label_dir, fname), "r") as f:
            lines = f.readlines()
            total_instances += len(lines)

            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                cls_id = int(parts[0])
                w = float(parts[3])
                h = float(parts[4])

                instance_count[cls_id] += 1
                split_instance_count[split][cls_id] += 1
                total_objects_in_image += 1

                if cls_id not in seen_classes:
                    image_count[cls_id] += 1
                    seen_classes.add(cls_id)

                bbox_widths.append(w)
                bbox_heights.append(h)
                bbox_areas.append(w * h)

        objects_per_image.append(total_objects_in_image)


# ==============================
# OVERVIEW
# ==============================
print("\nTotal images:", total_images)
print("Total instances:", total_instances)
print("Average objects per image:", round(total_instances / total_images, 2))


# ==============================
# MISSING CLASS CHECK
# ==============================
missing_classes = set(id_to_name.keys()) - set(instance_count.keys())
print("\nMissing classes:", len(missing_classes))
for cls_id in sorted(missing_classes):
    print(" -", id_to_name[cls_id])


# ==============================
# CLASS DISTRIBUTION
# ==============================
counts = list(instance_count.values())

print("\nCLASS DISTRIBUTION")
print("Max instances:", max(counts))
print("Min instances:", min(counts))
print("Median:", int(np.median(counts)))
print("Imbalance ratio (max/min):", round(max(counts) / max(min(counts), 1), 2))

rare_threshold = np.median(counts) * 0.3
rare_classes = [id_to_name[c] for c, v in instance_count.items() if v < rare_threshold]

print("\nRare classes (<30% median):", len(rare_classes))


# ==============================
# SAVE CLASS DISTRIBUTION CSV
# ==============================
df_class = pd.DataFrame({
    "class_id": list(instance_count.keys()),
    "class_name": [id_to_name[i] for i in instance_count.keys()],
    "instances": list(instance_count.values()),
    "images_containing_class": [image_count[i] for i in instance_count.keys()]
})

df_class = df_class.sort_values("instances", ascending=False)
df_class.to_csv(os.path.join(OUTPUT_DIR, "class_distribution.csv"), index=False)


# ==============================
# HISTOGRAM: Instances per Class
# ==============================
plt.figure(figsize=(10, 6))
plt.hist(counts, bins=50)
plt.xlabel("Number of instances per class")
plt.ylabel("Number of classes")
plt.title("Instance Distribution (Linear)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "instance_distribution_linear.png"))
plt.close()

plt.figure(figsize=(10, 6))
plt.hist(counts, bins=50)
plt.yscale("log")
plt.xlabel("Number of instances per class")
plt.ylabel("Number of classes (log)")
plt.title("Instance Distribution (Log Scale)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "instance_distribution_log.png"))
plt.close()


# ==============================
# HISTOGRAM: Bounding Box Area
# ==============================
plt.figure(figsize=(10, 6))
plt.hist(bbox_areas, bins=50)
plt.xlabel("Bounding Box Area (normalized)")
plt.ylabel("Frequency")
plt.title("Bounding Box Area Distribution")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "bbox_area_distribution.png"))
plt.close()


# ==============================
# HISTOGRAM: Width / Height
# ==============================
plt.figure(figsize=(10, 6))
plt.hist(bbox_widths, bins=50)
plt.xlabel("Bounding Box Width")
plt.ylabel("Frequency")
plt.title("Bounding Box Width Distribution")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "bbox_width_distribution.png"))
plt.close()

plt.figure(figsize=(10, 6))
plt.hist(bbox_heights, bins=50)
plt.xlabel("Bounding Box Height")
plt.ylabel("Frequency")
plt.title("Bounding Box Height Distribution")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "bbox_height_distribution.png"))
plt.close()


# ==============================
# HISTOGRAM: Objects per Image
# ==============================
plt.figure(figsize=(10, 6))
plt.hist(objects_per_image, bins=30)
plt.xlabel("Objects per image")
plt.ylabel("Number of images")
plt.title("Object Density per Image")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "object_density.png"))
plt.close()


# ==============================
# PER-SPLIT REPORT
# ==============================
for split in SPLITS:
    split_counts = split_instance_count[split]
    if not split_counts:
        continue

    split_values = list(split_counts.values())

    print(f"\n--- {split.upper()} SPLIT ---")
    print("Images:", len(os.listdir(os.path.join(DATASET_ROOT, split, "labels"))))
    print("Instances:", sum(split_values))
    print("Max:", max(split_values))
    print("Min:", min(split_values))
    print("Median:", int(np.median(split_values)))


# ==============================
# SAVE SUMMARY REPORT
# ==============================
with open(os.path.join(OUTPUT_DIR, "summary_report.txt"), "w") as f:
    f.write("DATASET SUMMARY\n")
    f.write("="*40 + "\n")
    f.write(f"Total images: {total_images}\n")
    f.write(f"Total instances: {total_instances}\n")
    f.write(f"Average objects per image: {round(total_instances/total_images,2)}\n")
    f.write(f"Missing classes: {len(missing_classes)}\n")
    f.write(f"Rare classes (<30% median): {len(rare_classes)}\n")


print("\nEDA Completed.")
print("Results saved to:", OUTPUT_DIR)