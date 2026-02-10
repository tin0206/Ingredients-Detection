import random
import cv2
import yaml
import numpy as np
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset
from rembg import remove
from PIL import Image

# ================= CONFIG =================
DATA4_YAML = "data4.yaml"
ALIAS_YAML = "class_alias.yaml"
GROUP_YAML = "class_groups.yaml"

EXTERNAL_ROOT = Path("external_dataset")
OUT_ROOT = Path("dataset_v4")

HF_DATASETS = [
    "SunnyAgarwal4274/Food_and_Vegetables",
    "Scuccorese/food-ingredients-dataset"
]

BACKGROUND_DIR = Path("backgrounds")

CANVAS_MIN = 640
CANVAS_MAX = 1024

INGREDIENT_MIN = 3
INGREDIENT_MAX = 7

MIN_PER_CLASS = 200

SPLITS = {
    "train": 35700,
    "valid": 10000,
    "test": 10000
}
# =========================================


# ---------- YAML ----------
def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------- CLASS RESOLVER ----------
def build_resolvers():
    data4 = load_yaml(DATA4_YAML)
    alias_yaml = load_yaml(ALIAS_YAML)
    group_yaml = load_yaml(GROUP_YAML)

    class_map = {v: int(k) for k, v in data4["names"].items()}

    alias_map = {}
    for canonical, aliases in alias_yaml.items():
        alias_map[canonical] = canonical
        for a in aliases:
            alias_map[a] = canonical

    member_to_group = {}
    for group, members in group_yaml.items():
        for m in members:
            member_to_group[m] = group

    return class_map, alias_map, member_to_group


def resolve_class_id(raw_name, class_map, alias_map, member_to_group):
    name = alias_map.get(raw_name, raw_name)
    name = member_to_group.get(name, name)
    return class_map.get(name)


# ---------- IMAGE UTILS ----------
def remove_bg(img_bgr):
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    fg = remove(pil)
    return cv2.cvtColor(np.array(fg), cv2.COLOR_RGBA2BGRA)


def load_random_background(size):
    bgs = list(BACKGROUND_DIR.glob("*.*"))
    if not bgs:
        return np.ones((size, size, 3), dtype=np.uint8) * 255

    bg = cv2.imread(str(random.choice(bgs)))
    if bg is None:
        return np.ones((size, size, 3), dtype=np.uint8) * 255

    h, w = bg.shape[:2]
    scale = max(size / w, size / h)
    bg = cv2.resize(bg, None, fx=scale, fy=scale)

    y0 = random.randint(0, bg.shape[0] - size)
    x0 = random.randint(0, bg.shape[1] - size)

    return bg[y0:y0 + size, x0:x0 + size]


def place_object(canvas, fg, boxes):
    h, w = fg.shape[:2]
    H, W = canvas.shape[:2]

    if h >= H or w >= W:
        return None

    for _ in range(50):
        x = random.randint(0, W - w)
        y = random.randint(0, H - h)
        rect = (x, y, x + w, y + h)

        if all(
            rect[2] < bx[0] or rect[0] > bx[2] or
            rect[3] < bx[1] or rect[1] > bx[3]
            for bx in boxes
        ):
            alpha = fg[:, :, 3] / 255.0
            for c in range(3):
                canvas[y:y+h, x:x+w, c] = (
                    alpha * fg[:, :, c] +
                    (1 - alpha) * canvas[y:y+h, x:x+w, c]
                )
            boxes.append(rect)
            return rect

    return None

def resume_state(img_dir, lbl_dir, num_classes):
    class_count = defaultdict(int)
    indices = []

    if not lbl_dir.exists():
        return class_count, 0

    for p in lbl_dir.glob("*.txt"):
        indices.append(int(p.stem))
        with open(p) as f:
            for line in f:
                cid = int(line.split()[0])
                class_count[cid] += 1

    next_idx = max(indices) + 1 if indices else 0
    return class_count, next_idx


# ---------- LOAD INGREDIENT POOL ----------
def load_all_ingredients(class_map, alias_map, member_to_group):
    pool = defaultdict(list)

    # External dataset
    if EXTERNAL_ROOT.exists():
        for cls_dir in EXTERNAL_ROOT.iterdir():
            if not cls_dir.is_dir():
                continue

            cid = resolve_class_id(
                cls_dir.name, class_map, alias_map, member_to_group
            )
            if cid is None:
                continue

            for p in cls_dir.glob("*.*"):
                img = cv2.imread(str(p))
                if img is not None:
                    pool[cid].append(img)

    # HuggingFace datasets
    for hf in HF_DATASETS:
        print(f"📥 Loading {hf}")
        ds = load_dataset(hf, split="train")
        feats = ds.features.keys()

        if "label" in feats:
            names = ds.features["label"].names
            for s in ds:
                raw = names[s["label"]]
                cid = resolve_class_id(raw, class_map, alias_map, member_to_group)
                if cid is None:
                    continue
                img = cv2.cvtColor(np.array(s["image"]), cv2.COLOR_RGB2BGR)
                pool[cid].append(img)

        elif "ingredient" in feats:
            for s in ds:
                raw = s["ingredient"]
                cid = resolve_class_id(raw, class_map, alias_map, member_to_group)
                if cid is None:
                    continue
                img = cv2.cvtColor(np.array(s["image"]), cv2.COLOR_RGB2BGR)
                pool[cid].append(img)

    return pool


# ---------- MAIN GENERATOR ----------
def main():
    class_map, alias_map, member_to_group = build_resolvers()
    pool = load_all_ingredients(class_map, alias_map, member_to_group)

    num_classes = len(class_map)
    print(f"✅ Classes: {num_classes}")

    for split, target in SPLITS.items():
        print(f"\n🚀 Generating {split}")
        img_dir = OUT_ROOT / split / "images"
        lbl_dir = OUT_ROOT / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        class_count, img_idx = resume_state(img_dir, lbl_dir, num_classes)
        
        while (
            min(class_count.values(), default=0) < MIN_PER_CLASS
            or img_idx < target
        ):
            for cid in range(num_classes):
                if img_idx >= target:
                    break

                size = random.randint(CANVAS_MIN, CANVAS_MAX)
                canvas = load_random_background(size)

                boxes, labels = [], []

                # mandatory class
                imgs = pool.get(cid)
                if not imgs:
                    continue

                selected = [cid]
                others = list(set(pool.keys()) - {cid})
                random.shuffle(others)
                selected += others[:random.randint(2, INGREDIENT_MAX - 1)]

                for scid in selected:
                    img = random.choice(pool[scid])
                    fg = remove_bg(img)

                    h0, w0 = img.shape[:2]
                    scale = random.uniform(
                        0.1,
                        min((size * 0.4) / w0, (size * 0.4) / h0)
                    )
                    fg = cv2.resize(fg, None, fx=scale, fy=scale)

                    box = place_object(canvas, fg, boxes)
                    if box is None:
                        continue

                    x1, y1, x2, y2 = box
                    cx = (x1 + x2) / 2 / size
                    cy = (y1 + y2) / 2 / size
                    w = (x2 - x1) / size
                    h = (y2 - y1) / size

                    labels.append(
                        f"{scid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                    )

                name = f"{img_idx:06d}"
                cv2.imwrite(str(img_dir / f"{name}.jpg"), canvas)
                with open(lbl_dir / f"{name}.txt", "w") as f:
                    f.write("\n".join(labels))

                class_count[cid] += 1
                img_idx += 1

        print(f"✅ Done {split} ({img_idx} images)")


if __name__ == "__main__":
    main()
