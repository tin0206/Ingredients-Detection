import cv2
import yaml
import numpy as np
import gc
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset
from rembg import remove, new_session
from PIL import Image

# ================= CONFIG =================
DATA5_YAML = "data5.yaml"
ALIAS_YAML = "class_alias.yaml"
GROUP_YAML = "class_groups.yaml"

HF_DATASETS = [
    "SunnyAgarwal4274/Food_and_Vegetables",
    "Scuccorese/food-ingredients-dataset"
]

MAX_PER_CLASS = 40
OUT_DIR = Path("preprocessed_objects")
OUT_DIR.mkdir(exist_ok=True)

REM_BG_SESSION = new_session("u2netp")
# =========================================


# ---------- YAML ----------
def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------- CLASS RESOLVER ----------
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


def resolve_class_id(raw_name, class_map, alias_map, member_to_group):
    name = raw_name.lower().strip()
    name = alias_map.get(name, name)
    name = member_to_group.get(name, name)
    return class_map.get(name)


# ---------- REMOVE BACKGROUND (giữ nguyên style v4) ----------
def remove_bg(img_bgr):
    h0, w0 = img_bgr.shape[:2]
    max_dim = 384
    scale = min(max_dim / max(h0, w0), 1.0)

    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale)

    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    try:
        fg = remove(pil, session=REM_BG_SESSION)
    except:
        return None

    fg = cv2.cvtColor(np.array(fg), cv2.COLOR_RGBA2BGRA)

    # Kiểm tra alpha tránh lưu ảnh chết
    if fg.shape[2] != 4:
        return None

    alpha = fg[:, :, 3]
    if alpha.sum() < 800:  # threshold cao hơn để tránh object mờ
        return None

    return fg


# ---------- MAIN ----------
def main():
    class_map, alias_map, member_to_group = build_resolvers()
    counts = defaultdict(int)

    for hf in HF_DATASETS:
        print(f"📥 Streaming {hf}")
        ds = load_dataset(hf, split="train", streaming=True)

        for s in ds:

            # Resolve class
            if "label" in s:
                raw = ds.features["label"].names[s["label"]]
            elif "ingredient" in s:
                raw = s["ingredient"]
            else:
                continue

            cid = resolve_class_id(raw, class_map, alias_map, member_to_group)
            if cid is None:
                continue

            if counts[cid] >= MAX_PER_CLASS:
                continue

            # Load image
            try:
                img = cv2.cvtColor(np.array(s["image"]), cv2.COLOR_RGB2BGR)
            except:
                continue

            if img is None:
                continue

            # Remove background
            fg = remove_bg(img)
            if fg is None:
                continue

            # Save
            class_dir = OUT_DIR / str(cid)
            class_dir.mkdir(exist_ok=True)

            cv2.imwrite(
                str(class_dir / f"{counts[cid]:03d}.png"),
                fg
            )

            counts[cid] += 1

            del img, fg
            gc.collect()

        print(f"Done {hf}")

    print("✅ Preprocess complete")
    print("Classes saved:", len(counts))


if __name__ == "__main__":
    main()
