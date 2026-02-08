import os
from collections import Counter, defaultdict

LABEL_DIR = "train/labels"   # đổi path nếu cần
IMAGE_DIR = "train/images"

class_count = Counter()
image_with_label = set()
bbox_per_image = defaultdict(int)

for label_file in os.listdir(LABEL_DIR):
    if not label_file.endswith(".txt"):
        continue

    image_name = label_file.replace(".txt", "")
    label_path = os.path.join(LABEL_DIR, label_file)

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            print(f"⚠️ Label lỗi format: {label_file}")
            continue

        cls = int(parts[0])
        class_count[cls] += 1
        bbox_per_image[image_name] += 1
        image_with_label.add(image_name)

print("📊 SỐ LƯỢNG BBOX THEO CLASS")
for cls, cnt in class_count.items():
    print(f"Class {cls}: {cnt} bbox")

print("\n📸 SỐ BBOX TRUNG BÌNH / ẢNH")
print(sum(bbox_per_image.values()) / len(bbox_per_image))

images = {os.path.splitext(f)[0] for f in os.listdir(IMAGE_DIR)}
labels = {os.path.splitext(f)[0] for f in os.listdir(LABEL_DIR)}

no_label_images = images - labels

print(f"\n❌ Ảnh không có label: {len(no_label_images)}")
for img in list(no_label_images)[:10]:
    print("  ", img)

no_image_labels = labels - images

print(f"\n❌ Label không có ảnh: {len(no_image_labels)}")
for lbl in list(no_image_labels)[:10]:
    print("  ", lbl)
