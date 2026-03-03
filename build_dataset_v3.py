import os
import shutil
import yaml

DATA2_YAML = "data2.yaml"
CLASS_GROUPS_YAML = "class_groups.yaml"

SRC_DATASET = "dataset_v2"
DST_DATASET = "dataset_v3"

SPLITS = ["train", "valid", "test"]


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(obj, f, sort_keys=False, allow_unicode=True)


def main():
    data2 = load_yaml(DATA2_YAML)
    class_groups = load_yaml(CLASS_GROUPS_YAML)

    old_names = data2["names"]  # {id: name}
    old_id_to_name = {int(k): v for k, v in old_names.items()}
    name_to_old_id = {v: int(k) for k, v in old_names.items()}

    # ---- build group lookup: member -> group_name
    member_to_group = {}
    for group, members in class_groups.items():
        for m in members:
            member_to_group[m] = group

    # ---- build new class list (order stable)
    new_names = []
    old_to_new = {}

    for old_id, name in old_id_to_name.items():
        if name in member_to_group:
            new_name = member_to_group[name]
        else:
            new_name = name  # keep original class

        if new_name not in new_names:
            new_names.append(new_name)

        old_to_new[old_id] = new_names.index(new_name)

    print(f"Old classes: {len(old_names)}")
    print(f"New classes: {len(new_names)}")

    # ---- create folders
    for split in SPLITS:
        os.makedirs(f"{DST_DATASET}/{split}/images", exist_ok=True)
        os.makedirs(f"{DST_DATASET}/{split}/labels", exist_ok=True)

        img_src = f"{SRC_DATASET}/{split}/images"
        lbl_src = f"{SRC_DATASET}/{split}/labels"

        # copy images
        for fname in os.listdir(img_src):
            shutil.copy(
                os.path.join(img_src, fname),
                os.path.join(f"{DST_DATASET}/{split}/images", fname)
            )

        # remap labels
        for lbl in os.listdir(lbl_src):
            src_lbl = os.path.join(lbl_src, lbl)
            dst_lbl = os.path.join(f"{DST_DATASET}/{split}/labels", lbl)

            new_lines = []

            with open(src_lbl, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue

                    old_id = int(parts[0])
                    new_id = old_to_new[old_id]

                    new_lines.append(
                        " ".join([str(new_id)] + parts[1:])
                    )

            with open(dst_lbl, "w") as f:
                f.write("\n".join(new_lines))

    # ---- write data3.yaml
    data3 = {
        "path": DST_DATASET,
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(new_names),
        "names": {i: name for i, name in enumerate(new_names)},
    }

    save_yaml(data3, "data3.yaml")

    print("✅ dataset_v3 & data3.yaml created correctly")


if __name__ == "__main__":
    main()
