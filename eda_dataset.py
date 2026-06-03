# =========================================================
# SIMPLE OBJECT DETECTION EDA
# 1. Objects per Image Distribution
# 2. Class Distribution over Dataset
# =========================================================

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import Counter

# =========================================================
# CONFIG
# =========================================================

DATASET_ROOT = "dataset"

SPLITS = ["train", "val", "test"]

OUTPUT_DIR = "eda_outputs_dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use("ggplot")

# =========================================================
# LOAD LABELS
# =========================================================

print("=" * 80)
print("LOADING LABEL FILES")
print("=" * 80)

objects_per_image = []

all_classes = []

for split in SPLITS:

    label_dir = os.path.join(
        DATASET_ROOT,
        split,
        "labels"
    )

    label_files = glob.glob(
        os.path.join(label_dir, "*.txt")
    )

    print(f"\n📂 {split}: {len(label_files)} label files")

    for label_path in tqdm(label_files):

        try:

            with open(label_path, "r") as f:
                lines = f.readlines()

            # -------------------------------------------------
            # OBJECTS PER IMAGE
            # -------------------------------------------------

            num_objects = len(lines)

            objects_per_image.append(num_objects)

            # -------------------------------------------------
            # CLASS DISTRIBUTION
            # -------------------------------------------------

            for line in lines:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                cls_id = int(parts[0])

                all_classes.append(cls_id)

        except:
            continue

print("\n✅ Dataset Loaded")

# =========================================================
# A. OBJECTS PER IMAGE DISTRIBUTION
# =========================================================

print("\n" + "=" * 80)
print("A. OBJECTS PER IMAGE DISTRIBUTION")
print("=" * 80)

plt.figure(figsize=(12, 6))

bins = range(
    1,
    max(objects_per_image) + 2
)

plt.hist(
    objects_per_image,
    bins=bins,
    edgecolor="black"
)

mean_objects = np.mean(objects_per_image)

plt.axvline(
    mean_objects,
    linestyle="--",
    linewidth=2,
    label=f"Mean = {mean_objects:.2f}"
)

plt.title(
    "Objects per Image Distribution",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel(
    "Number of Objects per Image",
    fontsize=13
)

plt.ylabel(
    "Number of Images",
    fontsize=13
)

plt.xticks(
    range(
        1,
        max(objects_per_image) + 1
    )
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "objects_per_image_distribution.png"
    ),
    dpi=300
)

plt.show()

# =========================================================
# B. CLASS DISTRIBUTION OVER DATASET
# =========================================================

print("\n" + "=" * 80)
print("B. CLASS DISTRIBUTION OVER DATASET")
print("=" * 80)

class_counter = Counter(all_classes)

sorted_classes = sorted(
    class_counter.items(),
    key=lambda x: x[1],
    reverse=True
)

class_ids = [x[0] for x in sorted_classes]
class_counts = [x[1] for x in sorted_classes]

fig_width = max(16, len(class_ids) * 0.25)

plt.figure(figsize=(fig_width, 8))

plt.bar(
    [str(c) for c in class_ids],
    class_counts
)

plt.title(
    "Class Distribution Across Entire Dataset",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel(
    "Class ID",
    fontsize=13
)

plt.ylabel(
    "Number of Object Instances",
    fontsize=13
)

plt.xticks(
    rotation=90,
    fontsize=8
)

plt.grid(axis="y")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "class_distribution_over_dataset.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =========================================================
# SUMMARY
# =========================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"Total Images : {len(objects_per_image)}")
print(f"Total Objects: {len(all_classes)}")
print(f"Average Objects/Image: {np.mean(objects_per_image):.2f}")

print(f"\n📁 Graphs saved to: {OUTPUT_DIR}")

print("\n✅ SIMPLE EDA COMPLETED")