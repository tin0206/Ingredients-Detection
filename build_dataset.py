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
from tqdm import tqdm

# ================= CONFIG =================
DATA_YAML = "data.yaml"
CLASS_MAPPING_YAML = "class_mapping.yaml"

OUT_ROOT = Path("dataset")
BACKGROUND_DIR = Path("backgrounds")
EXTERNAL_PATH = Path("external_dataset")
PROCESSED_PATH = Path("processed_ingredients") # Thư mục cache

HF_DATASETS = ["Scuccorese/food-ingredients-dataset"]

MAX_PER_CLASS = 90
CANVAS_MIN = 640
CANVAS_MAX = 800

# Ngưỡng pixel tối thiểu để nguyên liệu không bị mờ (pixelated)
MIN_OBJ_SIZE = 45 

SCENE_TYPES = {
    "single": (1, 1),
    "medium": (3, 7),
    "dense": (12, 18)
}

SCENE_PROBS = {
    "single": 0.20,
    "medium": 0.35,
    "dense": 0.45 # Tăng tỷ lệ dense để model học bối cảnh phức tạp
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
    if not Path(CLASS_MAPPING_YAML).exists(): return {}
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

# ---------- REMOVE BG & CACHE ----------
def remove_bg(img_bgr):
    h0, w0 = img_bgr.shape[:2]
    max_dim = 512 # Tăng lên một chút để giữ detail cho cache
    scale = min(max_dim / max(h0, w0), 1.0)
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    fg = remove(pil, session=REM_BG_SESSION)
    return cv2.cvtColor(np.array(fg), cv2.COLOR_RGBA2BGRA)

def preprocess_pool(pool):
    """Tách nền 1 lần và lưu vào disk để dùng lại cực nhanh"""
    PROCESSED_PATH.mkdir(exist_ok=True)
    cached_pool = defaultdict(list)
    
    print("🚀 Pre-processing & Caching Background Removal...")
    for cid, img_bytes_list in tqdm(pool.items()):
        class_dir = PROCESSED_PATH / str(cid)
        class_dir.mkdir(exist_ok=True)
        
        for i, img_bytes in enumerate(img_bytes_list):
            out_file = class_dir / f"{i}.png"
            if not out_file.exists():
                img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                if img is None: continue
                fg = remove_bg(img)
                cv2.imwrite(str(out_file), fg)
            cached_pool[cid].append(str(out_file))
    return cached_pool

# ---------- AUGMENTATION & PLACEMENT ----------
def augment_fg(fg):
    # 1. Rotate 0-360
    angle = random.randint(0, 360)
    h, w = fg.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    new_w, new_h = int((h * sin) + (w * cos)), int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - (w // 2)
    M[1, 2] += (new_h / 2) - (h // 2)
    
    fg = cv2.warpAffine(fg, M, (new_w, new_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

    # 2. Random Flip
    if random.random() > 0.5: fg = cv2.flip(fg, 1)

    # 3. Lightness/Contrast (Chỉ áp dụng kênh RGB)
    alpha = random.uniform(0.8, 1.2)
    beta = random.randint(-15, 15)
    fg[:, :, :3] = cv2.convertScaleAbs(fg[:, :, :3], alpha=alpha, beta=beta)
    return fg

def smart_resize(fg, canvas_size, scene):
    h0, w0 = fg.shape[:2]
    scale_map = {"single": 0.35, "medium": 0.18, "dense": 0.10}
    max_scale_map = {"single": 0.85, "medium": 0.45, "dense": 0.25}
    
    scale = random.uniform(scale_map[scene], max_scale_map[scene])
    
    # Kiểm tra ngưỡng sắc nét
    if min(h0, w0) * scale < MIN_OBJ_SIZE:
        scale = MIN_OBJ_SIZE / min(h0, w0)
        
    # Đảm bảo không to hơn canvas
    scale = min(scale, (canvas_size * 0.9) / max(h0, w0))
    
    return cv2.resize(fg, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

def iou(box1, box2):
    xA, yA = max(box1[0], box2[0]), max(box1[1], box2[1])
    xB, yB = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    return inter / (area1 + area2 - inter + 1e-6)

def place_object(canvas, fg, boxes, cluster_center):
    h, w = fg.shape[:2]
    H, W = canvas.shape[:2]
    cx_c, cy_c = cluster_center

    for _ in range(40):
        # Cluster-based placement
        x = int(np.random.normal(cx_c, W * 0.12))
        y = int(np.random.normal(cy_c, H * 0.12))
        x, y = max(0, min(W-w, x)), max(0, min(H-h, y))

        rect = (x, y, x + w, y + h)
        if all(iou(rect, bx) < 0.25 for bx in boxes):
            alpha = fg[:, :, 3] / 255.0
            for c in range(3):
                canvas[y:y+h, x:x+w, c] = (alpha * fg[:, :, c] + (1 - alpha) * canvas[y:y+h, x:x+w, c])
            boxes.append(rect)
            return rect
    return None

# ---------- MAIN WORKFLOW ----------
def main():
    class_map = load_class_map()
    
    # 1. Thu thập dữ liệu thô
    pool = defaultdict(list)
    # Load HF
    for hf in HF_DATASETS:
        ds = load_dataset(hf, split="train")
        if "image" in ds.features: ds = ds.cast_column("image", HFImage(decode=False))
        for s in ds:
            raw = s.get("label") or s.get("ingredient")
            if isinstance(raw, int): raw = ds.features["label"].names[raw]
            norm = normalize_name(raw)
            if norm in class_map:
                img_bytes = s["image"].get("bytes")
                if img_bytes: pool[class_map[norm]].append(img_bytes)

    # Load External
    if EXTERNAL_PATH.exists():
        for folder in EXTERNAL_PATH.iterdir():
            if not folder.is_dir(): continue
            norm = normalize_name(folder.name)
            if norm in class_map:
                for img_p in folder.glob("*.*"):
                    with open(img_p, "rb") as f: pool[class_map[norm]].append(f.read())

    # 2. Tiền xử lý Cache (Chỉ chạy 1 lần)
    pool = {k: v for k, v in pool.items() if len(v) > 0}
    cached_pool = preprocess_pool(pool)
    
    # 3. Tạo Dataset
    for split, target in SPLITS.items():
        print(f"--- Generating {split} ({target} images) ---")
        img_dir, lbl_dir = OUT_ROOT/split/"images", OUT_ROOT/split/"labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        # Counter để cân bằng rare class
        global_counts = defaultdict(int)
        
        for img_idx in tqdm(range(target)):
            size = random.randint(CANVAS_MIN, CANVAS_MAX)
            # Load background
            bgs = list(BACKGROUND_DIR.glob("*.*"))
            if bgs and random.random() > 0.15:
                bg = cv2.imread(str(random.choice(bgs)))
                bg = cv2.resize(bg, (size, size))
            else:
                bg = np.ones((size, size, 3), dtype=np.uint8) * random.randint(200, 255)

            canvas, boxes, labels = bg.copy(), [], []
            scene = random.choices(list(SCENE_PROBS.keys()), weights=list(SCENE_PROBS.values()))[0]
            cluster_center = (random.randint(int(size*0.2), int(size*0.8)), random.randint(int(size*0.2), int(size*0.8)))

            num_ing = random.randint(*SCENE_TYPES[scene])
            
            # 🔥 Rare Class Priority Sampling
            all_cids = list(cached_pool.keys())
            # Trọng số nghịch đảo: class xuất hiện càng ít, xác suất được chọn càng cao
            weights = [1.0 / (np.sqrt(global_counts[cid]) + 1.0) for cid in all_cids]
            probs = np.array(weights) / sum(weights)
            
            selected_cids = np.random.choice(all_cids, size=min(num_ing, len(all_cids)), replace=False, p=probs)

            for cid in selected_cids:
                img_path = random.choice(cached_pool[cid])
                fg = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if fg is None: continue
                
                fg = augment_fg(fg)
                fg = smart_resize(fg, size, scene)
                
                box = place_object(canvas, fg, boxes, cluster_center)
                if box:
                    global_counts[cid] += 1
                    x1, y1, x2, y2 = box
                    labels.append(f"{cid} {(x1+x2)/2/size:.6f} {(y1+y2)/2/size:.6f} {(x2-x1)/size:.6f} {(y2-y1)/size:.6f}")

            if labels:
                name = f"{split}_{img_idx:06d}"
                cv2.imwrite(str(img_dir / f"{name}.jpg"), canvas)
                with open(lbl_dir / f"{name}.txt", "w") as f:
                    f.write("\n".join(labels))
            
            if img_idx % 1000 == 0: gc.collect()

if __name__ == "__main__":
    main()