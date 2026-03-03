import yaml
from pathlib import Path

DATA2_YAML = "data2.yaml"
GROUPS_YAML = "class_groups.yaml"
OUT_YAML = "data3.yaml"


def normalize(name: str) -> str:
    return name.strip().lower()


def main():
    # ---------- load data2.yaml ----------
    with open(DATA2_YAML, "r", encoding="utf-8") as f:
        data2 = yaml.safe_load(f)

    old_names = data2["names"]  # dict {id: name}
    old_names_norm = {i: normalize(n) for i, n in old_names.items()}

    # ---------- load class_groups.yaml ----------
    with open(GROUPS_YAML, "r", encoding="utf-8") as f:
        groups = yaml.safe_load(f)

    # normalize groups
    group_map = {}
    for group_name, members in groups.items():
        group_map[group_name] = set(normalize(m) for m in members)

    # ---------- build mapping old_class -> new_class ----------
    old_to_new = {}
    new_names = []

    # 1️⃣ add group classes first (stable order)
    for group_name in group_map:
        new_names.append(group_name)

    # 2️⃣ add remaining standalone classes
    for _, cls_name in old_names.items():
        n = normalize(cls_name)
        if any(n in members for members in group_map.values()):
            continue
        new_names.append(cls_name)

    # ---------- assign new ids ----------
    new_name_to_id = {name: i for i, name in enumerate(new_names)}

    for old_id, old_name in old_names.items():
        n = normalize(old_name)

        assigned = False
        for group_name, members in group_map.items():
            if n in members:
                old_to_new[old_id] = new_name_to_id[group_name]
                assigned = True
                break

        if not assigned:
            old_to_new[old_id] = new_name_to_id[old_name]

    # ---------- build data3.yaml ----------
    data3 = {
        "path": data2.get("path", ""),
        "train": data2.get("train", ""),
        "val": data2.get("val", ""),
        "test": data2.get("test", ""),
        "nc": len(new_names),
        "names": {i: name for i, name in enumerate(new_names)},
    }

    with open(OUT_YAML, "w", encoding="utf-8") as f:
        yaml.dump(data3, f, sort_keys=False, allow_unicode=True)

    # ---------- summary ----------
    print("✅ data3.yaml created")
    print(f"Old classes: {len(old_names)}")
    print(f"New classes: {len(new_names)}")
    print("\nGroup mapping:")
    for g in group_map:
        print(f"  {g} -> {new_name_to_id[g]}")


if __name__ == "__main__":
    main()
