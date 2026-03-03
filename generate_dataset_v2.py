import random
import cv2
import yaml
import numpy as np
from pathlib import Path
from datasets import load_dataset
from rembg import remove
from PIL import Image

# ================= CONFIG =================
DATA_YAML = "data.yaml"
ALIAS_YAML = "class_alias.yaml"

EXTERNAL_ROOT = Path("external_dataset")
OUT_ROOT = Path("dataset_v2")

HF_DATASETS = [
    "SunnyAgarwal4274/Food_and_Vegetables",
    "Scuccorese/food-ingredients-dataset"
]

BACKGROUND_DIR = Path("backgrounds")

CANVAS_MIN = 640
CANVAS_MAX = 1024
INGREDIENT_MIN = 3
INGREDIENT_MAX = 7

SPLITS = {
    "train": 8000,
    "valid": 1000,
    "test": 1000
}
TOTAL_IMAGES = sum(SPLITS.values())
# =========================================


# ---------- LOAD YAML ----------
def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------- CLASS MAPS ----------
def load_class_maps():
    data = load_yaml(DATA_YAML)
    alias_yaml = load_yaml(ALIAS_YAML)

    class_map = {name: i for i, name in enumerate(data["names"])}

    alias_map = {}
    for canonical, aliases in alias_yaml.items():
        alias_map[canonical.lower()] = canonical
        for a in aliases:
            alias_map[a.lower()] = canonical

    return class_map, alias_map


def resolve_class(name, alias_map):
    return alias_map.get(name.lower(), name)


# ---------- IMAGE UTILS ----------
def remove_bg(img_bgr):
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    fg = remove(pil)
    return cv2.cvtColor(np.array(fg), cv2.COLOR_RGBA2BGRA)

def load_random_background(size):
    bg_files = list(BACKGROUND_DIR.glob("*.*"))
    if not bg_files:
        # fallback trắng
        return np.ones((size, size, 3), dtype=np.uint8) * 255

    bg_path = random.choice(bg_files)
    bg = cv2.imread(str(bg_path))

    if bg is None:
        return np.ones((size, size, 3), dtype=np.uint8) * 255

    h, w = bg.shape[:2]

    # scale để phủ đủ canvas
    scale = max(size / w, size / h)
    bg = cv2.resize(bg, None, fx=scale, fy=scale)

    # random crop
    y0 = random.randint(0, bg.shape[0] - size)
    x0 = random.randint(0, bg.shape[1] - size)

    return bg[y0:y0+size, x0:x0+size]


def load_all_ingredients(class_map, alias_map):
    pool = []

    # External dataset
    if EXTERNAL_ROOT.exists():
        for cls_dir in EXTERNAL_ROOT.iterdir():
            if not cls_dir.is_dir():
                continue

            canonical = resolve_class(cls_dir.name, alias_map)
            if canonical not in class_map:
                continue

            for img_path in cls_dir.glob("*.*"):
                img = cv2.imread(str(img_path))
                if img is not None:
                    pool.append((img, canonical))

    # HuggingFace datasets
    for hf in HF_DATASETS:
        print(f"📥 Loading {hf}")
        ds = load_dataset(hf, split="train")
        features = ds.features.keys()

        # ===== SunnyAgarwal: image + label =====
        if "label" in features:
            label_names = ds.features["label"].names

            for s in ds:
                raw = label_names[s["label"]]
                canonical = resolve_class(raw, alias_map)

                if canonical not in class_map:
                    continue

                img = np.array(s["image"])
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                pool.append((img, canonical))

        # ===== Scuccorese: image + ingredients =====
        elif "ingredient" in features:
            for s in ds:
                raw = s["ingredient"]
                canonical = resolve_class(raw, alias_map)

                if canonical not in class_map:
                    continue

                img = np.array(s["image"])
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                pool.append((img, canonical))

        else:
            print(f"⚠️ Unsupported schema in {hf}, skipped")

    return pool


def place_object(canvas, fg, boxes):
    h, w = fg.shape[:2]
    H, W = canvas.shape[:2]

    # ❌ object quá to → skip
    if w >= W or h >= H:
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

def get_start_index():
    max_idx = -1

    for split in SPLITS:
        img_dir = OUT_ROOT / split / "images"
        if not img_dir.exists():
            continue

        for p in img_dir.glob("*.jpg"):
            try:
                idx = int(p.stem)
                max_idx = max(max_idx, idx)
            except ValueError:
                pass

    return max_idx + 1

def count_existing(split):
    img_dir = OUT_ROOT / split / "images"
    if not img_dir.exists():
        return 0
    return len(list(img_dir.glob("*.jpg")))



# ---------- MAIN ----------
def main():
    class_map, alias_map = load_class_maps()

    for split in SPLITS:
        (OUT_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)

    pool = load_all_ingredients(class_map, alias_map)
    print(f"✅ Ingredient pool size: {len(pool)}")

    idx = get_start_index()
    for split, target in SPLITS.items():
        existing = count_existing(split)

        if existing >= target:
            print(f"⏭️ Skip {split} (already {existing}/{target})")
            continue

        need = target - existing
        print(f"\n🚀 Generating {split}: +{need} images")

        for _ in range(need):
            size = random.randint(CANVAS_MIN, CANVAS_MAX)
            canvas = load_random_background(size)

            n = random.randint(INGREDIENT_MIN, INGREDIENT_MAX)
            selected = random.sample(pool, n)

            boxes = []
            labels = []

            for img, cls in selected:
                fg = remove_bg(img)
                h0, w0 = img.shape[:2]
                max_scale = min(
                    (size * 0.5) / w0,
                    (size * 0.5) / h0,
                    0.4
                )
                scale = random.uniform(0.1, max_scale)
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
                    f"{class_map[cls]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                )

            name = f"{idx:06d}"
            cv2.imwrite(str(OUT_ROOT / split / "images" / f"{name}.jpg"), canvas)
            with open(OUT_ROOT / split / "labels" / f"{name}.txt", "w") as f:
                f.write("\n".join(labels))

            idx += 1

        print(f"✅ Done {split}")

    print("\n🎉 DATASET SYNTHESIS COMPLETED")


if __name__ == "__main__":
    main()
