import os
from collections import Counter
from pathlib import Path

# chỉnh path nếu cần
dataset_path = Path("dataset_v5")

splits = ["train", "val"]  # có thể thêm "test"
class_counter = Counter()

for split in splits:
    labels_dir = dataset_path / split / "labels"
    
    for label_file in labels_dir.glob("*.txt"):
        with open(label_file, "r") as f:
            for line in f:
                if line.strip():
                    class_id = int(line.split()[0])
                    class_counter[class_id] += 1

# In kết quả
print("Total instances:", sum(class_counter.values()))
print("Number of classes found:", len(class_counter))

total_classes = 317
missing = [i for i in range(total_classes) if i not in class_counter]

print("Missing classes:", missing)
print("Number missing:", len(missing))

print("\nTop 20 most common classes:")
for cls, count in class_counter.most_common(20):
    print(f"Class {cls}: {count}")

print("\nLeast common classes:")
for cls, count in sorted(class_counter.items(), key=lambda x: x[1])[:20]:
    print(f"Class {cls}: {count}")
