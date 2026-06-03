# =========================================================
# ADVANCED EDA FOR INGREDIENT DATASETS
# Thesis-ready visualization version
# =========================================================

import os
import cv2
import imagehash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from io import BytesIO
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset, Image as HFImage
from collections import defaultdict
from difflib import get_close_matches

# =========================================================
# CONFIG
# =========================================================

HF_DATASETS = [
    "Scuccorese/food-ingredients-dataset"
]

EXTERNAL_DATASET_ROOT = "external_dataset"

OUTPUT_DIR = "eda_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_VISUAL_SAMPLES = 3000

plt.style.use("ggplot")

# =========================================================
# CLASS NORMALIZATION
# =========================================================

def map_class(name):

    name = name.lower().strip()
    name = name.replace("-", "_").replace(" ", "_")

    synonym_map = {
        "capsicum": "bell_pepper",
        "scallion": "green_onion",
        "spring_onion": "green_onion",
        "chilli": "chili",
    }

    return synonym_map.get(name, name)

# =========================================================
# LOAD HUGGINGFACE DATASET
# =========================================================

print("=" * 70)
print("LOADING HUGGINGFACE DATASETS")
print("=" * 70)

hf_records = []

for hf in HF_DATASETS:

    print(f"\n📥 Loading {hf}")

    ds = load_dataset(hf, split="train")

    if "image" in ds.features:
        ds = ds.cast_column("image", HFImage(decode=False))

    for s in tqdm(ds):

        # -------------------------------------------------
        # GET LABEL
        # -------------------------------------------------

        if "label" in s and "label" in ds.features:
            raw = ds.features["label"].names[s["label"]]

        elif "ingredient" in s:
            raw = s["ingredient"]

        else:
            continue

        if not raw:
            continue

        raw = raw.lower().strip()
        raw = raw.replace("-", "_").replace(" ", "_")

        norm = map_class(raw)

        # -------------------------------------------------
        # GET IMAGE
        # -------------------------------------------------

        img_info = s.get("image", None)

        if not img_info:
            continue

        img_bytes = img_info.get("bytes", None)

        if not img_bytes:
            continue

        hf_records.append({
            "dataset": hf,
            "ingredient": norm,
            "raw_label": raw,
            "image_bytes": img_bytes
        })

hf_df = pd.DataFrame(hf_records)

print(f"\n✅ HF Images: {len(hf_df)}")
print(f"✅ HF Classes: {hf_df['ingredient'].nunique()}")

# =========================================================
# LOAD EXTERNAL DATASET
# =========================================================

print("\n" + "=" * 70)
print("LOADING EXTERNAL DATASET")
print("=" * 70)

external_records = []

for class_dir in Path(EXTERNAL_DATASET_ROOT).iterdir():

    if not class_dir.is_dir():
        continue

    ingredient = map_class(class_dir.name)

    for img_path in class_dir.glob("*"):

        if img_path.suffix.lower() not in [
            ".jpg", ".jpeg", ".png", ".webp"
        ]:
            continue

        external_records.append({
            "ingredient": ingredient,
            "image_path": str(img_path)
        })

external_df = pd.DataFrame(external_records)

print(f"\n✅ External Images: {len(external_df)}")
print(f"✅ External Classes: {external_df['ingredient'].nunique()}")

# =========================================================
# 4.1 DATASET OVERVIEW
# =========================================================

print("\n" + "=" * 70)
print("4.1 DATASET OVERVIEW")
print("=" * 70)

overview_labels = [
    "HF Images",
    "HF Classes",
    "External Images",
    "External Classes"
]

overview_values = [
    len(hf_df),
    hf_df["ingredient"].nunique(),
    len(external_df),
    external_df["ingredient"].nunique()
]

plt.figure(figsize=(10, 6))

bars = plt.bar(
    overview_labels,
    overview_values
)

plt.title(
    "Dataset Overview Statistics",
    fontsize=16,
    fontweight="bold"
)

plt.ylabel("Count", fontsize=12)

for bar in bars:
    yval = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        yval + 10,
        int(yval),
        ha='center',
        fontsize=11
    )

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/dataset_overview.png",
    dpi=300
)

plt.show()

# =========================================================
# 4.2 INGREDIENT DISTRIBUTION ANALYSIS
# =========================================================

print("\n" + "=" * 70)
print("4.2 INGREDIENT DISTRIBUTION ANALYSIS")
print("=" * 70)

combined_labels = pd.concat([
    hf_df["ingredient"],
    external_df["ingredient"]
])

ingredient_counts = combined_labels.value_counts()

# ---------------------------------------------------------
# TOP 20 INGREDIENTS
# ---------------------------------------------------------

plt.figure(figsize=(16, 8))

top20 = ingredient_counts.head(20)

bars = plt.bar(
    top20.index,
    top20.values
)

plt.title(
    "Top 20 Most Frequent Ingredient Classes",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel("Ingredient Class", fontsize=13)
plt.ylabel("Number of Images", fontsize=13)

plt.xticks(rotation=45, ha="right")

for bar in bars:
    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 5,
        int(height),
        ha='center',
        fontsize=9
    )

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/top20_ingredients.png",
    dpi=300
)

plt.show()

# ---------------------------------------------------------
# LONG-TAIL DISTRIBUTION
# ---------------------------------------------------------

sorted_counts = ingredient_counts.sort_values(
    ascending=False
).values

plt.figure(figsize=(14, 6))

plt.plot(
    range(len(sorted_counts)),
    sorted_counts,
    linewidth=2
)

plt.title(
    "Long-tail Distribution of Ingredient Classes",
    fontsize=17,
    fontweight="bold"
)

plt.xlabel(
    "Ingredient Rank",
    fontsize=13
)

plt.ylabel(
    "Number of Images",
    fontsize=13
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/long_tail_distribution.png",
    dpi=300
)

plt.show()

# ---------------------------------------------------------
# CLASS FREQUENCY HISTOGRAM
# ---------------------------------------------------------

plt.figure(figsize=(12, 6))

plt.hist(
    sorted_counts,
    bins=50
)

plt.title(
    "Ingredient Class Frequency Histogram",
    fontsize=17,
    fontweight="bold"
)

plt.xlabel(
    "Images per Class",
    fontsize=13
)

plt.ylabel(
    "Number of Classes",
    fontsize=13
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/class_frequency_histogram.png",
    dpi=300
)

plt.show()

# =========================================================
# 4.3 SEMANTIC LABEL NORMALIZATION
# =========================================================

print("\n" + "=" * 70)
print("4.3 SEMANTIC LABEL NORMALIZATION")
print("=" * 70)

hf_labels = set(hf_df["raw_label"].unique())

ext_labels = set([
    x.lower().strip().replace("-", "_").replace(" ", "_")
    for x in external_df["ingredient"].unique()
])

duplicate_labels = hf_labels.intersection(ext_labels)

near_duplicates = []

for label in hf_labels:

    matches = get_close_matches(
        label,
        ext_labels,
        n=5,
        cutoff=0.8
    )

    for m in matches:

        if label != m:

            near_duplicates.append((label, m))

print(f"✅ Exact duplicate labels: {len(duplicate_labels)}")
print(f"✅ Near duplicate labels: {len(near_duplicates)}")

# ---------------------------------------------------------
# LABEL NORMALIZATION SUMMARY
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

categories = [
    "Exact Duplicates",
    "Near Duplicates"
]

values = [
    len(duplicate_labels),
    len(near_duplicates)
]

bars = plt.bar(categories, values)

plt.title(
    "Semantic Label Normalization Analysis",
    fontsize=16,
    fontweight="bold"
)

plt.ylabel("Number of Labels", fontsize=12)

for bar in bars:

    yval = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        yval + 1,
        int(yval),
        ha='center',
        fontsize=11
    )

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/semantic_normalization.png",
    dpi=300
)

plt.show()

# =========================================================
# 4.4 IMAGE QUALITY ANALYSIS
# =========================================================

print("\n" + "=" * 70)
print("4.4 IMAGE QUALITY ANALYSIS")
print("=" * 70)

widths = []
heights = []
aspect_ratios = []

corrupted_files = 0

# ---------------------------------------------------------
# HUGGINGFACE IMAGES
# ---------------------------------------------------------

for img_bytes in tqdm(hf_df["image_bytes"]):

    try:

        img = Image.open(BytesIO(img_bytes))

        w, h = img.size

        widths.append(w)
        heights.append(h)

        aspect_ratios.append(w / h)

    except:
        corrupted_files += 1

# ---------------------------------------------------------
# EXTERNAL IMAGES
# ---------------------------------------------------------

for path in tqdm(external_df["image_path"]):

    try:

        img = Image.open(path)

        w, h = img.size

        widths.append(w)
        heights.append(h)

        aspect_ratios.append(w / h)

    except:
        corrupted_files += 1

print(f"⚠️ Corrupted files: {corrupted_files}")

# ---------------------------------------------------------
# RESOLUTION DISTRIBUTION
# ---------------------------------------------------------

plt.figure(figsize=(10, 8))

plt.scatter(
    widths,
    heights,
    alpha=0.3
)

plt.title(
    "Image Resolution Distribution",
    fontsize=17,
    fontweight="bold"
)

plt.xlabel(
    "Image Width",
    fontsize=13
)

plt.ylabel(
    "Image Height",
    fontsize=13
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/resolution_distribution.png",
    dpi=300
)

plt.show()

# ---------------------------------------------------------
# ASPECT RATIO DISTRIBUTION
# ---------------------------------------------------------

plt.figure(figsize=(12, 6))

plt.hist(
    aspect_ratios,
    bins=50
)

plt.title(
    "Aspect Ratio Distribution",
    fontsize=17,
    fontweight="bold"
)

plt.xlabel(
    "Aspect Ratio (Width / Height)",
    fontsize=13
)

plt.ylabel(
    "Number of Images",
    fontsize=13
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/aspect_ratio_distribution.png",
    dpi=300
)

plt.show()

# ---------------------------------------------------------
# CORRUPTED FILES
# ---------------------------------------------------------

plt.figure(figsize=(6, 6))

labels = [
    "Valid Images",
    "Corrupted Images"
]

values = [
    len(widths),
    corrupted_files
]

plt.pie(
    values,
    labels=labels,
    autopct='%1.2f%%'
)

plt.title(
    "Corrupted File Analysis",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/corrupted_files_analysis.png",
    dpi=300
)

plt.show()

# =========================================================
# 4.5 VISUAL DIVERSITY ANALYSIS
# =========================================================

print("\n" + "=" * 70)
print("4.5 VISUAL DIVERSITY ANALYSIS")
print("=" * 70)

brightness_scores = []
edge_scores = []

hf_sample = hf_df.sample(
    min(MAX_VISUAL_SAMPLES, len(hf_df))
)

for img_bytes in tqdm(hf_sample["image_bytes"]):

    try:

        pil_img = Image.open(
            BytesIO(img_bytes)
        ).convert("RGB")

        img = np.array(pil_img)

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2GRAY
        )

        # -------------------------------------------------
        # LIGHTING ANALYSIS
        # -------------------------------------------------

        brightness = gray.mean()

        brightness_scores.append(brightness)

        # -------------------------------------------------
        # BACKGROUND COMPLEXITY
        # -------------------------------------------------

        edges = cv2.Canny(gray, 100, 200)

        edge_density = edges.mean()

        edge_scores.append(edge_density)

    except:
        continue

# ---------------------------------------------------------
# LIGHTING DISTRIBUTION
# ---------------------------------------------------------

plt.figure(figsize=(12, 6))

plt.hist(
    brightness_scores,
    bins=50
)

plt.title(
    "Lighting Condition Distribution",
    fontsize=17,
    fontweight="bold"
)

plt.xlabel(
    "Average Brightness",
    fontsize=13
)

plt.ylabel(
    "Number of Images",
    fontsize=13
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/lighting_distribution.png",
    dpi=300
)

plt.show()

# ---------------------------------------------------------
# BACKGROUND COMPLEXITY
# ---------------------------------------------------------

plt.figure(figsize=(12, 6))

plt.hist(
    edge_scores,
    bins=50
)

plt.title(
    "Background Complexity Distribution",
    fontsize=17,
    fontweight="bold"
)

plt.xlabel(
    "Edge Density",
    fontsize=13
)

plt.ylabel(
    "Number of Images",
    fontsize=13
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/background_complexity.png",
    dpi=300
)

plt.show()

# =========================================================
# DUPLICATE IMAGE DETECTION
# =========================================================

print("\n" + "=" * 70)
print("DUPLICATE IMAGE DETECTION")
print("=" * 70)

hashes = defaultdict(list)

sample_for_hash = hf_df.sample(
    min(2000, len(hf_df))
)

for idx, row in tqdm(sample_for_hash.iterrows()):

    try:

        img = Image.open(
            BytesIO(row["image_bytes"])
        )

        phash = str(
            imagehash.phash(img)
        )

        hashes[phash].append(idx)

    except:
        continue

duplicate_count = 0

for h, ids in hashes.items():

    if len(ids) > 1:
        duplicate_count += len(ids)

# ---------------------------------------------------------
# DUPLICATE IMAGE VISUALIZATION
# ---------------------------------------------------------

plt.figure(figsize=(6, 6))

labels = [
    "Unique Images",
    "Potential Duplicates"
]

values = [
    len(sample_for_hash) - duplicate_count,
    duplicate_count
]

plt.pie(
    values,
    labels=labels,
    autopct='%1.2f%%'
)

plt.title(
    "Duplicate Image Detection",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/duplicate_detection.png",
    dpi=300
)

plt.show()

# =========================================================
# 4.6 FINAL DATASET COMPOSITION
# =========================================================

print("\n" + "=" * 70)
print("4.6 FINAL DATASET COMPOSITION")
print("=" * 70)

merged_classes = set(
    hf_df["ingredient"]
).union(
    set(external_df["ingredient"])
)

final_labels = [
    "HF Images",
    "External Images",
    "Merged Images",
    "Final Classes"
]

final_values = [
    len(hf_df),
    len(external_df),
    len(hf_df) + len(external_df),
    len(merged_classes)
]

plt.figure(figsize=(12, 6))

bars = plt.bar(
    final_labels,
    final_values
)

plt.title(
    "Final Dataset Composition",
    fontsize=18,
    fontweight="bold"
)

plt.ylabel(
    "Count",
    fontsize=13
)

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 10,
        int(height),
        ha='center',
        fontsize=11
    )

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/final_dataset_composition.png",
    dpi=300
)

plt.show()

print("\n✅ EDA COMPLETED")
print(f"📁 Graphs saved to: {OUTPUT_DIR}")