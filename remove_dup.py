from pathlib import Path

# Thư mục chứa ảnh
folder = Path("real_dataset_1000_imgs/images/train/images")

# Các định dạng ảnh
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif"}

deleted = 0

for file in folder.iterdir():
    if (
        file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
        and "copy" in file.stem.lower()
    ):
        file.unlink()
        print(f"Deleted: {file.name}")
        deleted += 1

print(f"\nDeleted {deleted} images.")