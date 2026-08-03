from pathlib import Path

ROOT = Path("real_dataset")

IMAGE_DIR = ROOT / "images"
LABEL_DIR = ROOT / "labels"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Lấy toàn bộ ảnh
images = sorted(
    [p for p in IMAGE_DIR.iterdir()
     if p.suffix.lower() in IMAGE_EXTS]
)

print(f"Found {len(images)} images")

# Rename tạm để tránh trùng tên
for i, img in enumerate(images):
    tmp = IMAGE_DIR / f"tmp_{i:06d}{img.suffix.lower()}"
    img.rename(tmp)

    label = LABEL_DIR / f"{img.stem}.txt"
    if label.exists():
        label.rename(LABEL_DIR / f"tmp_{i:06d}.txt")

# Rename chính thức
tmp_images = sorted(IMAGE_DIR.glob("tmp_*"))

for idx, img in enumerate(tmp_images, start=1):

    new_img = IMAGE_DIR / f"{idx:06d}{img.suffix.lower()}"
    img.rename(new_img)

    tmp_label = LABEL_DIR / f"{img.stem}.txt"

    if tmp_label.exists():
        tmp_label.rename(
            LABEL_DIR / f"{idx:06d}.txt"
        )

print("Done!")