import yaml
from pathlib import Path
from datasets import load_dataset

DATA_YAML = "data.yaml"
EXTERNAL_ROOT = "external_dataset"
HF_DATASET = "SunnyAgarwal4274/Food_and_Vegetables"


def load_existing_classes():
    with open(DATA_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = [n.lower() for n in data["names"]]
    return data, set(names)


def parse_hf_label(label):
    return label.split(" ", 1)[1].strip().lower()


def get_external_classes(existing):
    new_classes = []
    for p in Path(EXTERNAL_ROOT).iterdir():
        if p.is_dir():
            name = p.name.lower()
            if name not in existing:
                new_classes.append(name)
    return new_classes


def get_hf_classes(existing):
    print("📥 Loading HuggingFace dataset...")
    ds = load_dataset(HF_DATASET, split="train")

    label_names = ds.features["label"].names

    return sorted(
        name.lower()
        for name in label_names
        if name.lower() not in existing
    )


def main():
    data, existing = load_existing_classes()

    added = []

    # 1️⃣ external_dataset
    ext_classes = get_external_classes(existing)
    added.extend(ext_classes)
    existing.update(ext_classes)

    # 2️⃣ HuggingFace
    hf_classes = get_hf_classes(existing)
    added.extend(hf_classes)
    existing.update(hf_classes)

    if not added:
        print("✅ No new classes found")
        return

    # append to yaml
    data["names"].extend(added)
    data["nc"] = len(data["names"])

    with open(DATA_YAML, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True)

    print(f"✅ Added {len(added)} new classes")
    print("➕", added)


if __name__ == "__main__":
    main()
