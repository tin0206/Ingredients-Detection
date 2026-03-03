import random
import cv2
import yaml
import numpy as np
import gc
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset, Image as HFImage
from rembg import remove, new_session
from PIL import Image

# ================= CONFIG =================
DATA_YAML = "data8.yaml"
CLASS_MAPPING_YAML = "class_mapping.yaml"

OUT_ROOT = Path("dataset_v8")
BACKGROUND_DIR = Path("backgrounds")

HF_DATASETS = [
    "Scuccorese/food-ingredients-dataset"
]

MAX_PER_CLASS = 60

CANVAS_MIN = 640
CANVAS_MAX = 800

INGREDIENT_MIN = 3
INGREDIENT_MAX = 7

SPLITS = {
    "train": 25000,
    "val": 5000,
    "test": 5000
}

REM_BG_SESSION = new_session("u2netp")
# ==========================================


# ---------- LOAD CLASS MAP ----------
def load_class_map():
    with open(DATA_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return {v: int(k) for k, v in data["names"].items()}


# ---------- LOAD SPECIAL MAPPING ----------
def load_special_mapping():
    if not Path(CLASS_MAPPING_YAML).exists():
        return {}
    with open(CLASS_MAPPING_YAML, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


SPECIAL_MAP = load_special_mapping()


# ---------- NORMALIZE ----------
def normalize_name(name):
    if not name:
        return None

    name = name.lower().strip()
    name = name.replace("-", "_")
    name = name.replace(" ", "_")

    # remove canned_ / jarred_
    for prefix in ["canned_", "jarred_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]

    # mapping file override
    if name in SPECIAL_MAP:
        return SPECIAL_MAP[name]

    # safe singular
    if name.endswith("s") and not name.endswith("ss"):
        if name not in ["peas", "lentils", "olives"]:
            name = name[:-1]

    return name


# ---------- REMOVE BG ----------
def remove_bg(img_bgr):
    h0, w0 = img_bgr.shape[:2]
    max_dim = 384
    scale = min(max_dim / max(h0, w0), 1.0)

    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale)

    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    fg = remove(pil, session=REM_BG_SESSION)

    return cv2.cvtColor(np.array(fg), cv2.COLOR_RGBA2BGRA)


# ---------- LOAD RANDOM BACKGROUND ----------
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


# ---------- PLACE OBJECT ----------
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


# ---------- LOAD HF POOL ----------
def load_hf_pool(class_map):
    pool = defaultdict(list)

    for hf in HF_DATASETS:
        print(f"📥 Loading {hf}")
        ds = load_dataset(hf, split="train")

        if "image" in ds.features:
            ds = ds.cast_column("image", HFImage(decode=False))

        for s in ds:

            if "label" in s:
                raw = ds.features["label"].names[s["label"]]
            elif "ingredient" in s:
                raw = s["ingredient"]
            else:
                continue

            norm = normalize_name(raw)

            if norm not in class_map:
                continue

            cid = class_map[norm]

            if len(pool[cid]) >= MAX_PER_CLASS:
                continue

            img_info = s.get("image", None)
            if not img_info:
                continue

            img_bytes = img_info.get("bytes", None)
            if not img_bytes:
                continue

            pool[cid].append(img_bytes)

        print(f"✅ Done {hf}")

    return pool


# ---------- MAIN ----------
def main():
    class_map = load_class_map()

    pool = load_hf_pool(class_map)

    # remove empty classes
    pool = {k: v for k, v in pool.items() if len(v) > 0}

    print(f"✅ Usable classes: {len(pool)}")

    for split, target in SPLITS.items():

        img_dir = OUT_ROOT / split / "images"
        lbl_dir = OUT_ROOT / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        existing = list(img_dir.glob("*.jpg"))
        img_idx = max([int(p.stem) for p in existing], default=-1) + 1

        while img_idx < target:

            size = random.randint(CANVAS_MIN, CANVAS_MAX)
            canvas = load_random_background(size)

            boxes, labels = [], []

            available_classes = list(pool.keys())
            if not available_classes:
                print("❌ No usable classes!")
                return

            selected_classes = random.sample(
                available_classes,
                min(random.randint(INGREDIENT_MIN, INGREDIENT_MAX), len(available_classes))
            )

            for cid in selected_classes:

                s = random.choice(pool[cid])
                img_array = np.frombuffer(s, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if img is None:
                    continue

                fg = remove_bg(img)

                h0, w0 = fg.shape[:2]
                scale = random.uniform(
                    0.15,
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

                labels.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

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