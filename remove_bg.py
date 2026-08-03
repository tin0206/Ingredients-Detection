from pathlib import Path
from rembg import remove, new_session
from PIL import Image

INPUT_ROOT = Path("raw_ingredients_test_v2")
OUTPUT_ROOT = Path("processed_ingredients_test_v2")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}

# Load model một lần
session = new_session()

for class_dir in INPUT_ROOT.iterdir():
    if not class_dir.is_dir():
        continue

    print(f"\nProcessing class {class_dir.name}")

    out_dir = OUTPUT_ROOT / class_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [
        f for f in class_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    ]

    for i, img_path in enumerate(files, start=1):
        try:
            with open(img_path, "rb") as f:
                input_data = f.read()

            output_data = remove(
                input_data,
                session=session
            )

            # Luôn lưu PNG để giữ nền trong suốt
            out_file = out_dir / f"{img_path.stem}.png"

            with open(out_file, "wb") as f:
                f.write(output_data)

            # Kiểm tra ảnh có object hay không
            img = Image.open(out_file)

            # Nếu có alpha channel
            if "A" in img.getbands():
                alpha = img.getchannel("A")
                if alpha.getbbox() is None:
                    out_file.unlink()
                    print(f"✗ Empty mask: {img_path.name}")
                    continue

            print(
                f"✓ [{i}/{len(files)}] "
                f"{img_path.name} -> {out_file.name}"
            )

        except Exception as e:
            print(f"✗ Failed {img_path.name}: {e}")

print("\nDone!")