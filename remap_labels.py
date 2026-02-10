import yaml
from pathlib import Path

DATA_YAML = "data.yaml"
ALIAS_YAML = "class_alias.yaml"
DATASET_ROOT = Path("dataset")  # train/valid/test

# load canonical classes
with open(DATA_YAML, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

canonical_names = data["names"]
canonical_id = {name: i for i, name in enumerate(canonical_names)}

# load alias map
with open(ALIAS_YAML, "r", encoding="utf-8") as f:
    alias_map = yaml.safe_load(f)

# build reverse map: alias -> canonical
alias_to_canonical = {}
for canon, aliases in alias_map.items():
    for a in aliases:
        alias_to_canonical[a] = canon

def remap_label_file(label_path):
    new_lines = []

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            old_id = int(parts[0])
            bbox = parts[1:]

            old_name = old_names[old_id]

            # map alias → canonical
            if old_name in alias_to_canonical:
                new_name = alias_to_canonical[old_name]
            else:
                new_name = old_name

            if new_name not in canonical_id:
                continue  # skip unknown

            new_id = canonical_id[new_name]
            new_lines.append(" ".join([str(new_id)] + bbox))

    with open(label_path, "w") as f:
        f.write("\n".join(new_lines))


# load old names BEFORE cleanup
with open("data_old.yaml", "r", encoding="utf-8") as f:
    old_names = yaml.safe_load(f)["names"]

for split in ["train", "valid", "test"]:
    label_dir = DATASET_ROOT / split / "labels"
    for label_file in label_dir.glob("*.txt"):
        remap_label_file(label_file)

print("✅ Labels remapped successfully")
