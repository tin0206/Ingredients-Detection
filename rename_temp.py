from pathlib import Path

FOLDER = Path("temp")

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".avif",
}

START = 501

files = sorted(
    [f for f in FOLDER.iterdir() if f.suffix.lower() in IMAGE_EXTS]
)

for i, file in enumerate(files, start=START):
    new_name = f"{i:06d}{file.suffix.lower()}"
    file.rename(FOLDER / new_name)

print(f"Renamed {len(files)} files.")