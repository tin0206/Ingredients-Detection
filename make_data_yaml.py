from pathlib import Path

classes = Path("classes.txt").read_text(encoding="utf-8").splitlines()

with open("data.yaml", "w", encoding="utf-8") as f:
    f.write("path: dataset\n")
    f.write("train: train/images\n")
    f.write("val: valid/images\n")
    f.write("test: test/images\n\n")
    f.write(f"nc: {len(classes)}\n\n")
    f.write("names:\n")
    for c in classes:
        f.write(f"  - {c}\n")

print("✅ data.yaml created")
