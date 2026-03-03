import yaml
from datasets import load_dataset
from pathlib import Path

# ===== CONFIG =====
DATA5_YAML = "data5.yaml"
ALIAS_YAML = "class_alias.yaml"
GROUP_YAML = "class_groups.yaml"

HF_DATASETS = [
    "SunnyAgarwal4274/Food_and_Vegetables",
    "Scuccorese/food-ingredients-dataset"
]
# ===================


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_resolvers():
    data5 = load_yaml(DATA5_YAML)
    alias_yaml = load_yaml(ALIAS_YAML)
    group_yaml = load_yaml(GROUP_YAML)

    class_map = {v.lower(): int(k) for k, v in data5["names"].items()}

    alias_map = {}
    for canonical, aliases in alias_yaml.items():
        canonical = canonical.lower()
        alias_map[canonical] = canonical
        for a in aliases:
            alias_map[a.lower()] = canonical

    member_to_group = {}
    for group, members in group_yaml.items():
        group = group.lower()
        for m in members:
            member_to_group[m.lower()] = group

    return class_map, alias_map, member_to_group


def resolve(raw, class_map, alias_map, member_to_group):
    name = raw.lower().strip()
    name = alias_map.get(name, name)
    name = member_to_group.get(name, name)
    return class_map.get(name)


def main():
    class_map, alias_map, member_to_group = build_resolvers()

    yaml_classes = set(class_map.keys())
    hf_raw_names = set()
    mapped_classes = set()
    unmapped_raw = set()

    # ===== Scan HF =====
    for hf in HF_DATASETS:
        print(f"Scanning {hf}")
        ds = load_dataset(hf, split="train", streaming=True)

        for s in ds:
            if "label" in s:
                raw = ds.features["label"].names[s["label"]]
            elif "ingredient" in s:
                raw = s["ingredient"]
            else:
                continue

            raw = raw.lower().strip()
            hf_raw_names.add(raw)

            cid = resolve(raw, class_map, alias_map, member_to_group)
            if cid is None:
                unmapped_raw.add(raw)
            else:
                mapped_classes.add(raw)

    print("\n========== RESULTS ==========")
    print("Total YAML classes:", len(yaml_classes))
    print("Unique raw names in HF:", len(hf_raw_names))
    print("Mapped raw names:", len(mapped_classes))
    print("Unmapped raw names:", len(unmapped_raw))

    # Classes in YAML but never appear in HF
    hf_after_mapping = set(
        alias_map.get(r, r) for r in mapped_classes
    )
    hf_after_mapping = set(
        member_to_group.get(r, r) for r in hf_after_mapping
    )

    missing_from_hf = yaml_classes - hf_after_mapping

    print("YAML classes not found in HF:", len(missing_from_hf))

    print("\nSample unmapped raw names:")
    print(list(unmapped_raw)[:20])

    print("\nSample YAML classes missing from HF:")
    print(list(missing_from_hf)[:20])


if __name__ == "__main__":
    main()
