import random
import os
from pathlib import Path
import numpy as np
import cv2
from datasets import load_dataset
from rembg import remove
from PIL import Image

# ================= CONFIG =================
OUT_ROOT = Path("dataset")
BACKGROUNDS_DIR = Path("backgrounds")
IMG_SIZE = 640

NUM_IMAGES = {
    "train": 8000,
    "valid": 1000,
    "test": 1000
}

MIN_ING = 3
MAX_ING = 7
# =========================================


def ensure_dirs():
    # for split in ["train", "valid", "test"]:
    for split in ["test"]:
        (OUT_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)


def pil_to_cv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)


def normalize_bbox(x, y, w, h, W, H):
    return (
        (x + w / 2) / W,
        (y + h / 2) / H,
        w / W,
        h / H
    )
    
def get_start_index(split):
    img_dir = OUT_ROOT / split / "images"
    if not img_dir.exists():
        return 0

    existing = list(img_dir.glob("*.jpg"))
    if not existing:
        return 0

    ids = [int(p.stem) for p in existing if p.stem.isdigit()]
    return max(ids) + 1


def main():
    ensure_dirs()

    print("📥 Loading dataset...")
    ds = load_dataset("Scuccorese/food-ingredients-dataset", split="train")

    # build class map
    ingredients = sorted(set(ds["ingredient"]))
    class_map = {name: i for i, name in enumerate(ingredients)}

    with open("classes.txt", "w", encoding="utf-8") as f:
        for name in ingredients:
            f.write(name + "\n")

    print(f"✅ {len(class_map)} classes")

    bg_files = list(BACKGROUNDS_DIR.glob("*.jpg"))

    for split, total_imgs in NUM_IMAGES.items():
        print(f"\n🚀 Generating {split} set...")

        start_idx = get_start_index(split)
        for idx in range(start_idx, total_imgs):
            bg_path = random.choice(bg_files)
            bg = cv2.imread(str(bg_path))
            bg = cv2.resize(bg, (IMG_SIZE, IMG_SIZE))

            labels = []

            k = random.randint(MIN_ING, MAX_ING)
            samples = random.sample(range(len(ds)), k)

            for s in samples:
                sample = ds[s]
                cls_name = sample["ingredient"]
                cls_id = class_map[cls_name]

                # remove background
                fg = remove(sample["image"])
                fg = pil_to_cv(fg)

                h, w = fg.shape[:2]

                max_scale = min(
                    (IMG_SIZE * 0.6) / w,
                    (IMG_SIZE * 0.6) / h
                )

                scale = random.uniform(0.2, min(0.4, max_scale))

                nw, nh = int(w * scale), int(h * scale)

                if nw >= IMG_SIZE or nh >= IMG_SIZE:
                    continue

                fg = cv2.resize(fg, (nw, nh))
                
                if IMG_SIZE - nw <= 0 or IMG_SIZE - nh <= 0:
                    continue

                x = random.randint(0, IMG_SIZE - nw)
                y = random.randint(0, IMG_SIZE - nh)

                alpha = fg[:, :, 3] / 255.0
                for c in range(3):
                    bg[y:y+nh, x:x+nw, c] = (
                        alpha * fg[:, :, c] +
                        (1 - alpha) * bg[y:y+nh, x:x+nw, c]
                    )

                xc, yc, bw, bh = normalize_bbox(x, y, nw, nh, IMG_SIZE, IMG_SIZE)
                labels.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            img_name = f"{idx:06d}.jpg"
            label_name = f"{idx:06d}.txt"

            cv2.imwrite(str(OUT_ROOT / split / "images" / img_name), bg)

            with open(OUT_ROOT / split / "labels" / label_name, "w") as f:
                f.write("\n".join(labels))

            if idx % 100 == 0:
                print(f"  {idx}/{total_imgs}")

    print("\n🎉 DONE! Dataset generated successfully")


if __name__ == "__main__":
    main()
