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
DATA_YAML = "data12.yaml"
CLASS_MAPPING_YAML = "class_mapping.yaml"

OUT_ROOT = Path("dataset_v12")
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
    
    if "flour" in name:
        return "flour"

    # SALT
    if "salt" in name:
        return "salt"

    # SUGAR
    if "sugar" in name:
        return "sugar"

    # LENTILS
    if "lentil" in name:
        return "lentils"

    # TOMATO
    if "tomato" in name:
        if "sun" in name:
            return "sun_dried_tomato"
        return "tomato"

    # ANCHOVY
    if "anchov" in name:
        return "anchovy"

    # OCTOPUS
    if "octopus" in name:
        return "octopus"

    # ASPARAGUS
    if "asparagus" in name:
        return "asparagus"

    # COUSCOUS
    if "couscous" in name:
        return "couscous"

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

def augment_fg(fg):
    # 1. Xoay ngẫu nhiên 0-360 độ
    angle = random.randint(0, 360)
    h, w = fg.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Tính toán kích thước mới sau khi xoay để không mất góc
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    fg = cv2.warpAffine(fg, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

    # 2. Lật ngẫu nhiên
    if random.random() > 0.5:
        fg = cv2.flip(fg, 1) # Lật ngang

    # 3. Thay đổi độ sáng/độ tương phản nhẹ
    alpha = random.uniform(0.8, 1.2) # Contrast
    beta = random.randint(-20, 20)   # Brightness
    # Chỉ áp dụng lên kênh RGB (3 kênh đầu), giữ nguyên kênh Alpha (kênh 4)
    fg[:, :, :3] = cv2.convertScaleAbs(fg[:, :, :3], alpha=alpha, beta=beta)

    return fg

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

def preprocess_pool(pool):
    processed_path = Path("processed_ingredients")
    processed_path.mkdir(exist_ok=True)
    
    new_pool = defaultdict(list)
    
    for cid, img_list in pool.items():
        class_dir = processed_path / str(cid)
        class_dir.mkdir(exist_ok=True)
        
        for i, img_bytes in enumerate(img_list):
            out_file = class_dir / f"{i}.png"
            if out_file.exists():
                new_pool[cid].append(str(out_file))
                continue
                
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None: continue
            
            fg = remove_bg(img) # Gọi hàm remove_bg cũ của bạn
            cv2.imwrite(str(out_file), fg)
            new_pool[cid].append(str(out_file))
            
    return new_pool

# ---------- MAIN ----------
def main():
    class_map = load_class_map()
    pool = load_hf_pool(class_map)
    pool = load_external_dataset(pool, class_map)
    pool = {k: v for k, v in pool.items() if len(v) > 0}
    
    print(f"Usable classes: {len(pool)}")

    for split, target in SPLITS.items():
        print(f"--- Starting Split: {split} ---")
        
        # Quan trọng: Reset bộ đếm cho mỗi split
        split_counts = {
            "single": defaultdict(int),
            "medium": defaultdict(int),
            "dense": defaultdict(int)
        }

        img_dir = OUT_ROOT / split / "images"
        lbl_dir = OUT_ROOT / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        existing_files = list(img_dir.glob("*.jpg"))
        img_idx = len(existing_files)
        
        print(f"Restoring counts from {img_idx} labels...")
        for lbl_file in lbl_dir.glob("*.txt"):
            with open(lbl_file, "r") as f:
                for line in f:
                    cid = int(line.split()[0])
                    # Vì không biết scene cũ, ta cộng đều vào các loại để giữ trọng số
                    for s_type in SCENE_TYPES:
                        split_counts[s_type][cid] += 1

        while img_idx < target:
            size = random.randint(CANVAS_MIN, CANVAS_MAX)
            canvas = load_random_background(size) if random.random() > 0.2 else np.ones((size, size, 3), dtype=np.uint8) * 255

            boxes, labels = [], []
            scene = sample_scene_type()
            
            # Chọn tâm cụm
            cluster_center = (size // 2, size // 2) if scene == "single" else \
                             (random.randint(int(size*0.3), int(size*0.7)), random.randint(int(size*0.3), int(size*0.7)))

            min_i, max_i = SCENE_TYPES[scene]
            num_ing = min(random.randint(min_i, max_i), len(pool))

            # --- Weighted Sampling dựa trên split_counts ---
            all_cids = list(pool.keys())
            weights = [1.0 / (split_counts[scene][cid] + 1) for cid in all_cids]
            
            prob = np.array(weights) / sum(weights)
            selected_classes = np.random.choice(all_cids, size=num_ing, replace=False, p=prob)
            # (Thêm logic set() để tránh trùng nếu bạn muốn 1 ảnh không có 2 quả cà chua giống hệt nhau)

            for cid in selected_classes:
                s = random.choice(pool[cid])
                img = cv2.imdecode(np.frombuffer(s, np.uint8), cv2.IMREAD_COLOR)
                if img is None: continue

                fg = remove_bg(img)
                
                # 🔥 BƯỚC MỚI: Augmentation vật thể
                fg = augment_fg(fg)

                # Tính toán scale tùy theo scene
                h0, w0 = fg.shape[:2]
                scale_map = {"single": 0.35, "medium": 0.18, "dense": 0.08}
                max_scale_map = {"single": 0.9, "medium": 0.5, "dense": 0.3}
                
                scale = random.uniform(scale_map[scene], min((size * max_scale_map[scene]) / w0, (size * max_scale_map[scene]) / h0))
                fg = cv2.resize(fg, None, fx=scale, fy=scale)

                box = place_object(canvas, fg, boxes, cluster_center)
                if box:
                    split_counts[scene][cid] += 1
                    x1, y1, x2, y2 = box
                    labels.append(f"{cid} {(x1+x2)/2/size:.6f} {(y1+y2)/2/size:.6f} {(x2-x1)/size:.6f} {(y2-y1)/size:.6f}")

                del img, fg
                gc.collect()

            if labels:
                name = f"{img_idx:06d}"
                cv2.imwrite(str(img_dir / f"{name}.jpg"), canvas)
                with open(lbl_dir / f"{name}.txt", "w") as f:
                    f.write("\n".join(labels))
                img_idx += 1
            
            del canvas
            gc.collect()

        # In báo cáo độ phủ sau mỗi split
        avg_per_class = sum(split_counts["dense"].values()) / len(pool)
        print(f"Finish {split}. Avg instances per class in 'dense': {avg_per_class:.2f}")


if __name__ == "__main__":
    main()