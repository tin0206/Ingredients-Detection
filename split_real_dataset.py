from pathlib import Path
import random
import shutil

random.seed(42)

ROOT = Path("real_dataset_1000_imgs")

IMAGE_DIR = ROOT / "images"
LABEL_DIR = ROOT / "labels"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif"}

SPLITS = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1
}

# ==========================
# RESET / CREATE FOLDERS
# ==========================

for split in ["train", "val", "test"]:
    img_out = ROOT / split / "images"
    lbl_out = ROOT / split / "labels"

    if img_out.exists():
        shutil.rmtree(img_out)
    if lbl_out.exists():
        shutil.rmtree(lbl_out)

    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

# ==========================
# LOAD IMAGES
# ==========================

images = [
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in IMAGE_EXTENSIONS
]

images = sorted(images)

old_images = []
new_images = []

for img in images:
    try:
        img_id = int(img.stem)
    except ValueError:
        print(f"[SKIP] Invalid filename: {img.name}")
        continue

    if img_id <= 500:
        old_images.append(img)
    else:
        new_images.append(img)

random.shuffle(old_images)
random.shuffle(new_images)

# ==========================
# SPLIT FUNCTION
# ==========================

def split_group(group):
    n = len(group)

    n_train = int(n * SPLITS["train"])
    n_val = int(n * SPLITS["val"])

    train = group[:n_train]
    val = group[n_train:n_train + n_val]
    test = group[n_train + n_val:]

    return train, val, test


old_train, old_val, old_test = split_group(old_images)
new_train, new_val, new_test = split_group(new_images)

train_images = old_train + new_train
val_images = old_val + new_val
test_images = old_test + new_test

random.shuffle(train_images)
random.shuffle(val_images)
random.shuffle(test_images)

# ==========================
# COPY FUNCTION
# ==========================

def copy_split(image_list, split):
    copied_images = 0
    copied_labels = 0

    for img in image_list:
        label = LABEL_DIR / f"{img.stem}.txt"

        if not label.exists():
            print(f"[MISSING LABEL] {img.name}")
            continue

        shutil.copy2(img, ROOT / split / "images" / img.name)
        shutil.copy2(label, ROOT / split / "labels" / label.name)

        copied_images += 1
        copied_labels += 1

    return copied_images, copied_labels


train_i, train_l = copy_split(train_images, "train")
val_i, val_l = copy_split(val_images, "val")
test_i, test_l = copy_split(test_images, "test")

# ==========================
# LOG
# ==========================

print("=" * 50)
print(f"Old images: {len(old_images)}")
print(f"New images: {len(new_images)}")
print("-" * 50)
print(f"Train: {train_i} images, {train_l} labels")
print(f"Val  : {val_i} images, {val_l} labels")
print(f"Test : {test_i} images, {test_l} labels")
print("=" * 50)
print("Done!")