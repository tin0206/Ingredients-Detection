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
import io

# ================= CONFIG =================
DATA_YAML = "data.yaml"
CLASS_MAPPING_YAML = "class_mapping.yaml"

OUT_ROOT = Path("dataset")

BACKGROUND_DIR = Path("backgrounds")

EXTERNAL_PATH = Path("external_dataset")

ROBOFLOW_PATH = [
    Path("roboflow_dataset1/Roboflow_Generated_Dataset"),
    Path("roboflow_dataset2/Roboflow_Generated_Dataset"),
]

HF_DATASETS = ["Scuccorese/food-ingredients-dataset"]

MAX_PER_CLASS = 50

CANVAS_MIN = 640

CANVAS_MAX = 800

SCENE_TYPES = {"single": (1, 1), "medium": (3, 7), "dense": (12, 18)}

SPLITS = {"train": 25000, "val": 5000, "test": 5000}


# Initialize rembg session
REM_BG_SESSION = new_session("u2netp")
# ==========================================

# ---------- UTILS & MAPPING ----------
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

def normalize_name(name, class_map):
    if not name: return None
    if "flour" in name: return "flour"
    if "salt" in name: return "salt"
    if "sugar" in name: return "sugar"
    if "lentil" in name: return "lentils"
    if "tomato" in name:
        if "sun" in name: return "sun_dried_tomato"
        return "tomato"

    if "anchov" in name: return "anchovy"
    if "octopus" in name: return "octopus"

    if "asparagus" in name: return "asparagus"

    name = name.lower().strip().replace("-", "_").replace(" ", "_")

    for char in ["(", ")", "[", "]", "{", "}"]:
        name = name.replace(char, "")

    name = name.strip("_")

    for prefix in ["canned_", "jarred_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]

    if name in SPECIAL_MAP:
        name = SPECIAL_MAP[name]

    if name.endswith("s") and not name.endswith("ss"):
        if name not in ["peas", "lentils", "olives", "couscous"]:
            name = name[:-1]

    return name if name in class_map else None

# ---------- QUEUE SYSTEM (Ensures 100% Coverage) ----------
def create_shuffled_queues(pool):
    queues = {}
    for cid, img_paths in pool.items():
        paths = list(img_paths)
        random.shuffle(paths)
        queues[cid] = {"paths": paths, "index": 0}

    return queues

def get_next_image(cid, queues):
    q = queues[cid]
    # If we reached the end of the list, reshuffle and start over
    if q["index"] >= len(q["paths"]):
        random.shuffle(q["paths"])
        q["index"] = 0

    path = q["paths"][q["index"]]
    q["index"] += 1

    return path

# ---------- IMAGE PROCESSING ----------
def remove_bg(img_bgr):
    h0, w0 = img_bgr.shape[:2]
    scale = min(384 / max(h0, w0), 1.0)
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale)
        
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    fg = remove(pil, session=REM_BG_SESSION)

    return cv2.cvtColor(np.array(fg), cv2.COLOR_RGBA2BGRA)

def preprocess_pool(pool):
    processed_path = Path("processed_ingredients")
    processed_path.mkdir(exist_ok=True)
    new_pool = defaultdict(list)
    print("✂ Preprocessing ingredients: Removing backgrounds...")
    for cid, img_list in pool.items():
        class_dir = processed_path / str(cid)
        class_dir.mkdir(exist_ok=True)
        
        for i, img_data in enumerate(img_list):
            out_file = class_dir / f"{i}.png"
            if not out_file.exists():
                img = cv2.imdecode(
                    np.frombuffer(
                        img_data,
                        np.uint8),
                    cv2.IMREAD_COLOR)

                if img is None:
                    continue

                fg = remove_bg(img)
                cv2.imwrite(str(out_file), fg)

            new_pool[cid].append(str(out_file))

    return new_pool

# ---------- AUGMENT & PLACEMENT ----------
def augment_fg(fg):
    # Random Rotation
    angle = random.randint(0, 360)
    h, w = fg.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    nw, nh = int((h * sin) + (w * cos)), int((h * cos) + (w * sin))
    M[0, 2] += (nw / 2) - (w / 2)
    M[1, 2] += (nh / 2) - (h / 2)

    fg = cv2.warpAffine(
        fg, M, (nw, nh), borderMode=cv2.BORDER_CONSTANT, borderValue=(
            0, 0, 0, 0))

    if random.random() > 0.5:
        fg = cv2.flip(fg, 1)

    # Brightness/Contrast adjustment
    alpha = random.uniform(0.8, 1.2)
    beta = random.randint(-15, 15)
    fg[:, :, :3] = cv2.convertScaleAbs(fg[:, :, :3], alpha=alpha, beta=beta)
    
    return fg

def iou(b1, b2):

    xA, yA, xB, yB = (
        max(b1[0], b2[0]),
        max(b1[1], b2[1]),
        min(b1[2], b2[2]),
        min(b1[3], b2[3]),
    )

    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (area1 + area2 - inter + 1e-6)


def place_object(canvas, fg, boxes, cluster_center):
    h, w = fg.shape[:2]
    H, W = canvas.shape[:2]

    if h >= H or w >= W:
        return None

    cx, cy = cluster_center

    for _ in range(50):
        # Gaussian distribution around cluster center
        x = max(0, min(W - w, int(np.random.normal(cx, W * 0.08))))
        y = max(0, min(H - h, int(np.random.normal(cy, H * 0.08))))
        rect = (x, y, x + w, y + h)
        
        if all(iou(rect, bx) < 0.2 for bx in boxes):
            alpha = fg[:, :, 3] / 255.0
            for c in range(3):
                canvas[y: y + h,
                       x: x + w,
                       c] = (alpha * fg[:,
                                        :,
                                        c] + (1 - alpha) * canvas[y: y + h,
                                                                  x: x + w,
                                                                  c])

            boxes.append(rect)
            return rect

    return None

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
    y0, x0 = (bg.shape[0] - size) // 2, (bg.shape[1] - size) // 2
    return bg[y0: y0 + size, x0: x0 + size]

# ---------- SOURCE LOADING ----------
def load_all_sources(class_map):
    pool = defaultdict(list)
    
    # --- 1. HUGGINGFACE SOURCE ---
    print("🌐 Loading from HuggingFace...")
    for hf in HF_DATASETS:
        try:
            ds = load_dataset(hf, split="train")
            ds_items = list(ds)
            random.shuffle(ds_items)
            for s in ds_items:
                norm = normalize_name(
                    s.get("label_name") or s.get("ingredient") or "", class_map
                )

                if norm:
                    cid = class_map[norm]
                    img_data = s["image"]
                    if isinstance(img_data, Image.Image):
                        img_byte_arr = io.BytesIO()
                        img_data.save(img_byte_arr, format="PNG")
                        pool[cid].append(img_byte_arr.getvalue())
                        
                    elif isinstance(img_data, dict) and img_data.get("bytes"):
                        pool[cid].append(img_data["bytes"])

        except Exception as e:

            print(f"⚠️ Error loading HF dataset {hf}: {e}")

    # --- 2. EXTERNAL DIRECTORY ---
    if EXTERNAL_PATH.exists():
        print(f"📂 Loading from External Path: {EXTERNAL_PATH}")
        for f in EXTERNAL_PATH.iterdir():
            if f.is_dir():
                norm = normalize_name(f.name, class_map)
                if norm:
                    cid = class_map[norm]
                    img_paths = list(f.glob("*.*"))
                    random.shuffle(img_paths)
                    for img_p in img_paths:
                        try:
                            pool[cid].append(open(img_p, "rb").read())

                        except BaseException:
                            continue

    # --- 3. ROBOFLOW DATASETS ---
    for rb_root in ROBOFLOW_PATH:
        if not rb_root.exists():
            continue

        print(f"🤖 Loading from Roboflow Path: {rb_root}")

        is_strict = "roboflow_dataset2" in str(rb_root)

        for f in rb_root.iterdir():
            if not f.is_dir():
                continue

            if is_strict:
                simple = f.name.lower().strip().replace(" ", "_").replace("-", "_")
                norm = simple if simple in class_map else None

            else:
                norm = normalize_name(f.name, class_map)

            if norm:
                cid = class_map[norm]
                img_paths = list(f.glob("**/*.*"))
                random.shuffle(img_paths)
                for img_p in img_paths:
                    try:
                        pool[cid].append(open(img_p, "rb").read())

                    except BaseException:
                        continue

    # --- FINAL RANDOM SELECTION (CRITICAL) ---
    print(
        f"🎲 Finalizing pool with Random Selection (Max: {MAX_PER_CLASS} per class)..."
    )

    final_pool = defaultdict(list)
    for cid, all_samples in pool.items():
        if len(all_samples) > MAX_PER_CLASS:
            final_pool[cid] = random.sample(all_samples, MAX_PER_CLASS)

        else:
            final_pool[cid] = all_samples

    return final_pool

# ---------- MAIN LOOP ----------
def main():
    class_map = load_class_map()
    raw_pool = load_all_sources(class_map)
    processed_pool = preprocess_pool(raw_pool)
    pool_counts = {cid: len(img_list)
                   for cid, img_list in processed_pool.items()}

    # Initialize Shuffled Queues for cross-split coverage

    queues = create_shuffled_queues(processed_pool)

    usable_cids = list(processed_pool.keys())

    print(f"🚀 Classes loaded: {len(usable_cids)}")

    for split, target in SPLITS.items():
        print(f"--- Generating Split: {split} ({target} images) ---")
        
        img_dir = OUT_ROOT / split / "images"
        lbl_dir = OUT_ROOT / split / "labels"

        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        split_counts = defaultdict(int)

        img_idx = 0

        while img_idx < target:
            size = random.randint(CANVAS_MIN, CANVAS_MAX)

            canvas = (
                load_random_background(size)
                if random.random() > 0.15
                else np.ones((size, size, 3), dtype=np.uint8) * 255
            )

            boxes, labels = [], []

            r = random.random()

            scene = "single" if r < 0.25 else (
                "medium" if r < 0.6 else "dense")

            cluster_center = (
                (size // 2, size // 2)
                if scene == "single"
                else (
                    random.randint(int(size * 0.3), int(size * 0.7)),
                    random.randint(int(size * 0.3), int(size * 0.7)),
                )
            )

            min_i, max_i = SCENE_TYPES[scene]

            num_to_add = random.randint(min_i, max_i)

            # 1. Calculate base rarity weight (To prioritize classes with fewer
            # total images in the pool)

            base_rarity = []

            for cid in usable_cids:
                count = pool_counts[cid]

                if count < 10:
                    # Special boost: Inverse square and multiply by factor 10
                    weight = (1.0 / (count**2)) * 10

                else:
                    weight = 1.0 / count

                base_rarity.append(weight)

            base_rarity = np.array(base_rarity)

            # 2. Calculate current balance weight (To balance the output based
            # on already generated images)
            current_balance = [1.0 / (split_counts[cid] + 1)
                               for cid in usable_cids]

            # 3. Combine both (Multiply to prioritize rare classes)
            combined_weights = np.array(
                base_rarity) * np.array(current_balance)

            # Normalize to ensure the sum of weights = 1
            p_normalized = combined_weights / combined_weights.sum()

            sel_cids = np.random.choice(
                usable_cids,
                size=min(num_to_add, len(usable_cids)),
                p=p_normalized,
                replace=False,
            )

            for cid in sel_cids:
                # GET THE NEXT IMAGE FROM THE QUEUE
                fg_path = get_next_image(cid, queues)
                fg = cv2.imread(fg_path, cv2.IMREAD_UNCHANGED)

                if fg is None:
                    continue

                if pool_counts[cid] < 10:
                    # Boost color distortion (light)
                    # Convert to HSV to adjust brightness/saturation without
                    # affecting alpha channel
                    fg_hsv = cv2.cvtColor(
                        fg[:, :, :3], cv2.COLOR_BGR2HSV).astype(np.float32)

                    fg_hsv[:, :, 1] *= random.uniform(0.7, 1.3)  # Saturation

                    # Value (Brightness)
                    fg_hsv[:, :, 2] *= random.uniform(0.7, 1.3)

                    fg_hsv = np.clip(fg_hsv, 0, 255).astype(np.uint8)

                    fg[:, :, :3] = cv2.cvtColor(fg_hsv, cv2.COLOR_HSV2BGR)

                    # Add some blur if needed
                    if random.random() > 0.5:

                        fg = cv2.GaussianBlur(fg, (3, 3), 0)

                fg = augment_fg(fg)

                s_min, s_max = {
                    "single": (0.4, 0.7),
                    "medium": (0.2, 0.4),
                    "dense": (0.1, 0.25),
                }[scene]

                scale = random.uniform(s_min, s_max) * \
                    (size / max(fg.shape[:2]))

                fg = cv2.resize(fg, None, fx=scale, fy=scale)

                box = place_object(canvas, fg, boxes, cluster_center)

                if box:
                    split_counts[cid] += 1
                    x1, y1, x2, y2 = box
                    labels.append(
                        f"{cid} {(x1+x2)/2/size:.6f} {(y1+y2)/2/size:.6f} {(x2-x1)/size:.6f} {(y2-y1)/size:.6f}"
                    )

            if labels:
                name = f"{img_idx:06d}"
                cv2.imwrite(str(img_dir / f"{name}.jpg"), canvas)
                with open(lbl_dir / f"{name}.txt", "w") as f:
                    f.write("\n".join(labels))
                img_idx += 1

            if img_idx % 500 == 0:
                print(f"Progress: {img_idx}/{target}")
            gc.collect()


if __name__ == "__main__":
    main()
