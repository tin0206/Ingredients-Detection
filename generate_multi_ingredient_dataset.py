import random
import cv2
from pathlib import Path
import yaml
from datasets import load_dataset
import numpy as np

# ================= CONFIG =================
DATA_YAML = "data.yaml"
EXTERNAL_ROOT = Path("external_dataset")
OUT_ROOT = Path("dataset_v2")

HF_DATASETS = [
    "SunnyAgarwal4274/Food_and_Vegetables",
    "Scuccorese/food-ingredients-dataset"
]

IMG_SIZE = 640
SPLITS = {
    "train": 0.8,
    "valid": 0.1,
    "test": 0.1
}
# =========================================


def load_class_map():
    with open(DATA_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {name.lower(): i for i, name in enumerate(data["names"])}


def ensure_dirs():
    for split in SPLITS:
        (OUT_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)


def choose_split():
    r = random.random()
    acc = 0
    for split, p in SPLITS.items():
        acc += p
        if r <= acc:
            return split
    return "train"


def save_yolo(img, cls_id, img_id):
    split = choose_split()

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img_name = f"{img_id:06d}.jpg"
    label_name = f"{img_id:06d}.txt"

    cv2.imwrite(str(OUT_ROOT / split / "images" / img_name), img)

    with open(OUT_ROOT / split / "labels" / label_name, "w") as f:
        f.write(f"{cls_id} 0.5 0.5 1.0 1.0")


def main():
    class_map = load_class_map()
    ensure_dirs()

    img_id = 0

    # ================= 1️⃣ EXTERNAL DATASET =================
    for cls_dir in EXTERNAL_ROOT.iterdir():
        if not cls_dir.is_dir():
            continue

        cls_name = cls_dir.name.lower()
        if cls_name not in class_map:
            continue

        cls_id = class_map[cls_name]
        images = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))

        print(f"📦 External | {cls_name}: {len(images)} images")

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            save_yolo(img, cls_id, img_id)
            img_id += 1

    # ================= 2️⃣ + 3️⃣ HUGGINGFACE =================
    for hf_name in HF_DATASETS:
        print(f"\n📥 Loading HuggingFace: {hf_name}")
        ds = load_dataset(hf_name, split="train")

        label_names = ds.features["label"].names

        used = 0

        for sample in ds:
            cls_name = label_names[sample["label"]].lower()

            if cls_name not in class_map:
                continue

            cls_id = class_map[cls_name]

            img = np.array(sample["image"])
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            save_yolo(img, cls_id, img_id)
            img_id += 1
            used += 1

        print(f"✅ Used {used} images from {hf_name}")

    print(f"\n🎉 DONE! Total images: {img_id}")
    print(f"📁 Output: {OUT_ROOT.absolute()}")


if __name__ == "__main__":
    main()
