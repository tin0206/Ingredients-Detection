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
DATA_YAML = "data11.yaml"
CLASS_MAPPING_YAML = "class_mapping.yaml"

OUT_ROOT = Path("dataset_v11")
BACKGROUND_DIR = Path("backgrounds")
EXTERNAL_PATH = Path("external_dataset")

HF_DATASETS = [
    "Scuccorese/food-ingredients-dataset"
]

MAX_PER_CLASS = 90

CANVAS_MIN = 640
CANVAS_MAX = 800

# 🔥 Dense objects
SCENE_TYPES = {
    "single": (1, 1),
    "medium": (3, 7),
    "dense": (12, 18)
}

SCENE_PROBS = {
    "single": 0.25,
    "medium": 0.35,
    "dense": 0.40
}

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

    for prefix in ["canned_", "jarred_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]

    if name in SPECIAL_MAP:
        return SPECIAL_MAP[name]

    if name.endswith("s") and not name.endswith("ss"):
        if name not in ["peas", "lentils", "olives"]:
            name = name[:-1]

    return name

def sample_scene_type():
    r = random.random()
    cum = 0
    for k, p in SCENE_PROBS.items():
        cum += p
        if r <= cum:
            return k
    return "dense"


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

# ---------- IOU ----------
def iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0

# ---------- PLACE OBJECT (Clustered + Allow Overlap) ----------
def place_object(canvas, fg, boxes, cluster_center):
    h, w = fg.shape[:2]
    H, W = canvas.shape[:2]

    if h >= H or w >= W:
        return None

    cx_cluster, cy_cluster = cluster_center

    for _ in range(50):

        # Gaussian around cluster center
        x = int(np.random.normal(cx_cluster, W * 0.08))
        y = int(np.random.normal(cy_cluster, H * 0.08))

        x = max(0, min(W - w, x))
        y = max(0, min(H - h, y))

        rect = (x, y, x + w, y + h)

        # allow slight overlap (IoU < 0.2)
        if all(iou(rect, bx) < 0.2 for bx in boxes):

            alpha = fg[:, :, 3] / 255.0
            for c in range(3):
                canvas[y:y+h, x:x+w, c] = (
                    alpha * fg[:, :, c] +
                    (1 - alpha) * canvas[y:y+h, x:x+w, c]
                )

            boxes.append(rect)
            return rect

    return None


# ---------- LOAD HF ----------
def load_hf_pool(class_map):
    pool = defaultdict(list)

    for hf in HF_DATASETS:
        print(f"Loading {hf}")
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

    return pool

# ---------- LOAD EXTERNAL DATASET ----------
def load_external_dataset(pool, class_map):
    if not EXTERNAL_PATH.exists():
        print("⚠ No external_dataset found.")
        return pool

    print("📥 Loading external_dataset")

    for folder in EXTERNAL_PATH.iterdir():
        if not folder.is_dir():
            continue

        raw_name = folder.name
        norm = normalize_name(raw_name)

        if norm not in class_map:
            print(f"⛔ Skip {raw_name} (not in data.yaml)")
            continue

        cid = class_map[norm]

        images = list(folder.glob("*.*"))
        if not images:
            continue

        for img_path in images:
            try:
                with open(img_path, "rb") as f:
                    img_bytes = f.read()

                pool[cid].append(img_bytes)

            except Exception as e:
                print(f"Error reading {img_path}: {e}")
                continue

    print("✅ Done external_dataset")
    return pool


# ---------- MAIN ----------
def main():
    class_map = load_class_map()
    pool = load_hf_pool(class_map)
    pool = load_external_dataset(pool, class_map)

    pool = {k: v for k, v in pool.items() if len(v) > 0}
    print(f"Usable classes: {len(pool)}")

    for split, target in SPLITS.items():

        img_dir = OUT_ROOT / split / "images"
        lbl_dir = OUT_ROOT / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        existing = len(list(img_dir.glob("*.jpg")))
        img_idx = existing

        while img_idx < target:

            size = random.randint(CANVAS_MIN, CANVAS_MAX)
            if random.random() < 0.2:
                canvas = np.ones((size, size, 3), dtype=np.uint8) * 255
            else:
                canvas = load_random_background(size)

            boxes, labels = [], []

            scene = sample_scene_type()
            
            if scene == "single":
                cluster_center = (size // 2, size // 2)
            else:
                cluster_center = (
                    random.randint(int(size * 0.3), int(size * 0.7)),
                    random.randint(int(size * 0.3), int(size * 0.7))
                )

            min_i, max_i = SCENE_TYPES[scene]

            num_ing = min(random.randint(min_i, max_i), len(pool))

            selected_classes = random.sample(list(pool.keys()), num_ing)

            for cid in selected_classes:

                s = random.choice(pool[cid])
                img_array = np.frombuffer(s, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is None:
                    continue

                fg = remove_bg(img)

                h0, w0 = fg.shape[:2]
                if scene == "single":
                    scale = random.uniform(
                        0.35,
                        min((size * 0.9) / w0, (size * 0.9) / h0)
                    )
                elif scene == "medium":
                    scale = random.uniform(
                        0.18,
                        min((size * 0.5) / w0, (size * 0.5) / h0)
                    )
                else:
                    scale = random.uniform(
                        0.08,
                        min((size * 0.3) / w0, (size * 0.3) / h0)
                    )

                fg = cv2.resize(fg, None, fx=scale, fy=scale)

                box = place_object(canvas, fg, boxes, cluster_center)
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

        print(f"Done {split} ({img_idx} images)")


if __name__ == "__main__":
    main()