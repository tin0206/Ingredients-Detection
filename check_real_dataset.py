from pathlib import Path

ROOT = Path("temp")

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

count = 0

for ext in EXTS:
    count += len(list(ROOT.rglob(f"*{ext}")))

print(f"Total images: {count}")