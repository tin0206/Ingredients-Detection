from pathlib import Path
import shutil

ROOT = Path("real_dataset")

OUT_IMAGES = ROOT / "images"
OUT_LABELS = ROOT / "labels"

OUT_IMAGES.mkdir(exist_ok=True)
OUT_LABELS.mkdir(exist_ok=True)

image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

image_count = 0
label_count = 0

for split in ["train", "val", "test"]:

    image_dir = ROOT / split / "images"
    label_dir = ROOT / split / "labels"

    # Copy images
    for img in image_dir.iterdir():

        if img.suffix.lower() not in image_exts:
            continue

        dst = OUT_IMAGES / img.name

        if dst.exists():
            print(f"[Duplicate Image] {img.name}")
            continue

        shutil.copy2(img, dst)
        image_count += 1

    # Copy labels
    for lbl in label_dir.glob("*.txt"):

        dst = OUT_LABELS / lbl.name

        if dst.exists():
            print(f"[Duplicate Label] {lbl.name}")
            continue

        shutil.copy2(lbl, dst)
        label_count += 1

print("=" * 50)
print(f"Images copied : {image_count}")
print(f"Labels copied : {label_count}")
print("Done!")