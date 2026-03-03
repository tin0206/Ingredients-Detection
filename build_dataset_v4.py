import random
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
DATA4_YAML = "data4.yaml"
ALIAS_YAML = "class_alias.yaml"
GROUP_YAML = "class_groups.yaml"

EXTERNAL_ROOT = Path("external_dataset")
OUT_ROOT = Path("dataset_v4")

HF_DATASETS = [
    "SunnyAgarwal4274/Food_and_Vegetables",
    "Scuccorese/food-ingredients-dataset"
]

MAX_PER_CLASS = 40

BACKGROUND_DIR = Path("backgrounds")
REM_BG_SESSION = new_session("u2netp")

CANVAS_MIN = 640
CANVAS_MAX = 800

INGREDIENT_MIN = 3
INGREDIENT_MAX = 7

SPLITS = {
    "train": 11000,
    "valid": 5000,
    "test": 5000
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
    h0, w0 = img_bgr.shape[:2]
    max_dim = 384
    scale = min(max_dim / max(h0, w0), 1.0)

    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale)

    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    fg = remove(pil, session=REM_BG_SESSION)

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


# ---------- LOAD INGREDIENT REFERENCES ----------
def load_all_ingredients(class_map, alias_map, member_to_group):
    pool = defaultdict(list)

    # External → store PATH only
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
                pool[cid].append(("external", p))

    # HF datasets → store reference only
    for hf in HF_DATASETS:
        print(f"📥 Streaming {hf}")
        ds = load_dataset(hf, split="train", streaming=True)

        for s in ds:
            if "label" in s:
                raw = ds.features["label"].names[s["label"]]
            elif "ingredient" in s:
                raw = s["ingredient"]
            else:
                continue

            cid = resolve_class_id(raw, class_map, alias_map, member_to_group)
            if cid is None:
                continue

            if len(pool[cid]) >= MAX_PER_CLASS:
                continue

            pool[cid].append(("hf", s))

        print(f"Done {hf}")

    return pool


# ---------- MAIN ----------
def main():
    class_map, alias_map, member_to_group = build_resolvers()
    pool = load_all_ingredients(class_map, alias_map, member_to_group)

    print(f"✅ Loaded classes: {len(pool)}")

    for split, target in SPLITS.items():

        img_dir = OUT_ROOT / split / "images"
        lbl_dir = OUT_ROOT / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        img_idx = 0

        while img_idx < target:

            size = random.randint(CANVAS_MIN, CANVAS_MAX)
            canvas = load_random_background(size)

            boxes, labels = [], []

            selected_classes = random.sample(
                list(pool.keys()),
                min(random.randint(INGREDIENT_MIN, INGREDIENT_MAX), len(pool))
            )

            for cid in selected_classes:

                source_type, data = random.choice(pool[cid])

                if source_type == "external":
                    img = cv2.imread(str(data))
                else:
                    img = cv2.cvtColor(np.array(data["image"]), cv2.COLOR_RGB2BGR)

                if img is None:
                    continue

                fg = remove_bg(img)

                h0, w0 = fg.shape[:2]
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
                    f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                )

                del img, fg
                gc.collect()

            if not labels:
                continue

            name = f"{img_idx:06d}"
            cv2.imwrite(str(img_dir / f"{name}.jpg"), canvas)

            with open(lbl_dir / f"{name}.txt", "w") as f:
                f.write("\n".join(labels))

            img_idx += 1

            del canvas
            gc.collect()

        print(f"✅ Done {split} ({img_idx} images)")


if __name__ == "__main__":
    main()
