import yaml
from datasets import load_dataset

# =========================
# CONFIG
# =========================
HF_DATASETS = [
    "SunnyAgarwal4274/Food_and_Vegetables",
    "Scuccorese/food-ingredients-dataset"
]

OUTPUT_YAML = "data5.yaml"
# =========================


def normalize_name(name):
    name = name.lower().strip()

    # chuẩn hóa đặc biệt
    special_map = {
        "iodized salt": "salt",
        "sea salt": "salt",
        "raw sugar": "sugar",
        "canned olives": "olives",
        "jarred olives": "olives",
        "french lentils": "lentils",
    }

    if name in special_map:
        return special_map[name]

    # bỏ prefix phổ biến
    remove_words = ["canned", "jarred", "raw", "iodized"]
    for w in remove_words:
        if name.startswith(w + " "):
            name = name[len(w) + 1:]

    name = name.strip()
    name = name.replace(" ", "_")

    return name


def collect_classes():
    class_set = set()

    for hf in HF_DATASETS:
        print(f"Scanning {hf}")
        ds = load_dataset(hf, split="train", streaming=True)

        for sample in ds:
            if "label" in sample:
                raw = ds.features["label"].names[sample["label"]]
            elif "ingredient" in sample:
                raw = sample["ingredient"]
            else:
                continue

            norm = normalize_name(raw)
            if norm:
                class_set.add(norm)

    return sorted(class_set)


def build_yaml(classes):
    yaml_dict = {
        "path": "dataset_v5",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(classes)}
    }

    with open(OUTPUT_YAML, "w", encoding="utf-8") as f:
        yaml.dump(yaml_dict, f, allow_unicode=True)

    print(f"\n✅ Built {OUTPUT_YAML}")
    print(f"Total classes: {len(classes)}")


if __name__ == "__main__":
    classes = collect_classes()
    build_yaml(classes)
