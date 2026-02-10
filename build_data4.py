import yaml

DATA_YAML = "data.yaml"
ALIAS_YAML = "class_alias.yaml"
GROUP_YAML = "class_groups.yaml"


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(obj, f, sort_keys=False, allow_unicode=True)


def main():
    data = load_yaml(DATA_YAML)
    alias_yaml = load_yaml(ALIAS_YAML)
    group_yaml = load_yaml(GROUP_YAML)

    # ---------- alias lookup ----------
    alias_map = {}
    for canonical, aliases in alias_yaml.items():
        alias_map[canonical] = canonical
        for a in aliases:
            alias_map[a] = canonical

    # ---------- group lookup ----------
    member_to_group = {}
    for group, members in group_yaml.items():
        for m in members:
            member_to_group[m] = group

    # ---------- normalize classes ----------
    new_names = []
    for name in data["names"]:
        # step 1: alias
        name = alias_map.get(name, name)

        # step 2: group
        name = member_to_group.get(name, name)

        if name not in new_names:
            new_names.append(name)

    data4 = {
        "path": "dataset_v4",
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(new_names),
        "names": {i: n for i, n in enumerate(new_names)},
    }

    save_yaml(data4, "data4.yaml")

    print(f"✅ data4.yaml created")
    print(f"Old classes: {len(data['names'])}")
    print(f"New classes: {len(new_names)}")


if __name__ == "__main__":
    main()
