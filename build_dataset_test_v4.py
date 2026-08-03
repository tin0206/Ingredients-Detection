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
DATA_YAML = "data_test_v4.yaml"
CLASS_MAPPING_YAML = "class_mapping_test_v4.yaml"

OUT_ROOT = Path("dataset_test_v4")
BACKGROUND_DIR = Path("backgrounds")
EXTERNAL_PATH = Path("external_dataset")
PROCESSED_DIR = Path("processed_ingredients_test_v2")

HF_DATASETS = [
    "Scuccorese/food-ingredients-dataset"
]
LAYOUT_PROBS = {
    "cluster": 0.50,
    "spread": 0.25,
    "edge": 0.10,
    "mixed": 0.15
}

MAX_PER_CLASS = 150

CANVAS_MIN = 640
CANVAS_MAX = 800

# 🔥 Dense objects
SCENE_TYPES = {
    "single": (1, 1),
    "medium": (3, 5),
    "dense": (6, 12)
}

SCENE_PROBS = {
    "single": 0.35,
    "medium": 0.45,
    "dense": 0.20
}

SPLITS = {
    "train": 30000,
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
# ===================== MERGE GROUP =====================
MERGE_GROUPS = {
    # ===== OIL =====
    "oil": [
        "olive_oil", "canola_oil", "grapeseed_oil", "peanut_oil",
        "sesame_oil", "sunflower_oil", "vegetable_oil",
        "avocado_oil", "flaxseed_oil", "coconut_oil"
    ],

    # ===== PASTA =====
    "pasta": [
        "spaghetti", "penne", "fusilli", "rigatoni",
        "linguine", "fettuccine", "macaroni", "rotini", "farfalle"
    ],

    # ===== GRAIN =====
    "grain": [
        "barley", "oat", "millet", "sorghum", "teff",
        "spelt", "farro", "emmer", "einkorn", "corn_grit", 
        "cracked_wheat", "freekeh", "polenta", "wheat_bran", 
        "barley", "oat", "millet", "sorghum", "teff", "spelt", "quinoa", "einkorn", "emmer", "farro", "kamut"
    
    ],

    # ===== BEAN =====
    "bean": [
        "black_bean", "kidney_bean", "navy_bean", "pinto_bean",
        "mung_bean", "adzuki_bean", "lima_bean",
        "fava_bean", "cannellini_bean", "refried_bean"
    ],
    
    # ===== CHICKEN =====
    "chicken": [
        "chicken", "chicken_breast", "chicken_thigh"
    ],

    # ===== GARLIC =====
    "garlic": [
        "garlic", "garlic_bulb"
    ],

    # ===== BROCCOLI =====
    "broccoli": [
        "broccoli", "broccoli_stem"
    ],
    
    "cherry": [
        "black_cherry", "sour_cherry"
    ],
    
    "berry": [
        "blackberry", "blueberry", "cranberry", "raspberry", "elderberry", "huckleberry", "mulberry", "boysenberry", "goji_berry"
    ],
}

MERGE_LOOKUP = {}
for target, sources in MERGE_GROUPS.items():
    for s in sources:
        MERGE_LOOKUP[s] = target
        
REMOVE_CLASSES = {
    "artichoke_heart", "black_sapote", "bison", "buffalo", "bulgur", "buckwheat", 
    "caribou", "chard_stalk", "cornmeal", "elk", "deer", 
    "grouse", "guinea_fowl", "pawpaw",
    "partridge", "pheasant", "quail", "salsa", "squab",
    "squirrel", "semolina", "wild_boar", "ostrich", "venison"
}

# =========================
def normalize_name(name):
    if not name:
        return None

    name = name.lower().strip()
    name = name.replace("-", "_")
    name = name.replace(" ", "_")

    # remove canned / jarred
    for prefix in ["canned_", "jarred_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]

    # ===================== SPECIAL MAP =====================
    special_map = {

        # ===================== SINGULAR FIX =====================
        "apples": "apple",
        "apricots": "apricot",
        "beets": "beet",
        "carrots": "carrot",
        "cherries": "cherry",
        "mushrooms": "mushroom",
        "peaches": "peach",
        "pineapples": "pineapple",
        "pears": "pear",
        "tomatoes": "tomato",
        "mandarin_oranges": "mandarin",

        # ================= SALT =================
        "sea_salt": "salt",
        "kosher_salt": "salt",
        "black_salt": "salt",
        "pink_salt": "salt",
        "table_salt": "salt",
        "smoked_salt": "salt",
        "iodized_salt": "salt",
        "celtic_salt": "salt",
        "pickling_salt": "salt",

        # ================= SUGAR =================
        "brown_sugar": "sugar",
        "white_sugar": "sugar",
        "powdered_sugar": "sugar",
        "cane_sugar": "sugar",
        "coconut_sugar": "sugar",
        "raw_sugar": "sugar",
        "demerara_sugar": "sugar",
        "muscovado_sugar": "sugar",
        "turbinado_sugar": "sugar",
        "date_sugar": "sugar",

        # ================= LENTILS =================
        "beluga_lentils": "lentils",
        "black_lentils": "lentils",
        "brown_lentils": "lentils",
        "french_lentils": "lentils",
        "golden_lentils": "lentils",
        "green_lentils": "lentils",
        "orange_lentils": "lentils",
        "red_lentils": "lentils",
        "spanish_pardina_lentils": "lentils",
        "yellow_lentils": "lentils",
        "sprouted_lentils": "lentils",

        # ================= PEAS =================
        "green_peas": "peas",
        "field_peas": "peas",
        "pigeon_peas": "peas",
        "snap_peas": "peas",
        "snow_peas": "peas",
        "split_peas": "peas",
        "white_peas": "peas",
        "yellow_peas": "peas",
        "black_eyed_peas": "peas",
        "sprouted_green_peas": "peas",

        # ================= OLIVES =================
        "castelvetrano_olives": "olives",
        "cerignola_olives": "olives",
        "gaeta_olives": "olives",
        "kalamata_olives": "olives",
        "ligurian_olives": "olives",
        "manzanilla_olives": "olives",
        "nicoise_olives": "olives",
        "picholine_olives": "olives",
        "black_olives": "olives",
        "green_olives": "olives",

        # ================= FLOUR =================
        "all_purpose_flour": "flour",
        "bread_flour": "flour",
        "cake_flour": "flour",
        "oat_flour": "flour",
        "rye_flour": "flour",
        "gluten_free_flour": "flour",
        "self_rising_flour": "flour",
        "white_flour": "flour",
        "whole_wheat_flour": "flour",
        "almond_flour": "flour",
        "coconut_flour": "flour",

        # ================= ONION =================
        "spring_onion": "green_onion",
        "scallion": "green_onion",
        "pearl_onion": "onion",

        # ================= GARLIC =================
        "elephant_garlic": "garlic",

        # ================= GINGER =================
        "ginger_root": "ginger",

        # ================= SPROUTED BEANS =================
        "sprouted_adzuki_beans": "adzuki_bean",
        "sprouted_black_beans": "black_bean",
        "sprouted_chickpeas": "chickpea",
        "sprouted_kidney_beans": "kidney_bean",
        "sprouted_mung_beans": "mung_bean",
        "sprouted_navy_beans": "navy_bean",
        "sprouted_pinto_beans": "pinto_bean",
        "sprouted_soybeans": "soybean",

        # ================= BEAN UNIFY =================
        "adzuki_beans": "adzuki_bean",
        "black_beans": "black_bean",
        "kidney_beans": "kidney_bean",
        "mung_beans": "mung_bean",
        "navy_beans": "navy_bean",
        "pinto_beans": "pinto_bean",
        "soybeans": "soybean",
        "chickpeas": "chickpea",
        
        "white_rice": "rice",
        
        "glass_noodles": "glass_noodle",
    }

    if name in special_map:
        name = special_map[name]

    # ================= SAFE SINGULAR RULE =================
    # chỉ áp dụng cho fruit/veg phổ biến
    if name.endswith("s") and not name.endswith("ss"):
        if name not in ["peas", "lentils", "olives"]:
            name = name[:-1]
            
    # ================= SPELLING FIX =================
    spelling_fix = {
        "anchovie": "anchovy",
        "octopu": "octopus",
        "couscou": "couscous",
        "asparagu": "asparagus",
        "sun_dried_tomatoe": "sun_dried_tomato",
    }

    if name in spelling_fix:
        name = spelling_fix[name]
        
    if name == "couscouscucumber":
        return None
    
    if name in REMOVE_CLASSES:
        return None
    
    if name in MERGE_LOOKUP:
        name = MERGE_LOOKUP[name]
        
    return name

def map_class(raw):
    if not raw:
        return None

    raw = raw.lower().strip()
    raw = raw.replace("-", "_").replace(" ", "_")

    # 🔥 PRIORITY 1: mapping yaml
    if raw in SPECIAL_MAP:
        return SPECIAL_MAP[raw]

    # 🔥 PRIORITY 2: fallback nhẹ (optional)
    return normalize_name(raw)

def sample_scene_type():
    r = random.random()
    cum = 0
    for k, p in SCENE_PROBS.items():
        cum += p
        if r <= cum:
            return k
    return "dense"

def get_all_backgrounds():
    exts = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
    files = []

    for p in BACKGROUND_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)

    random.shuffle(files)
    return files

BG_FILES = get_all_backgrounds()
BG_INDEX = 0


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
    global BG_INDEX

    if not BG_FILES:
        return np.ones((size, size, 3), dtype=np.uint8) * 255

    bg_path = BG_FILES[BG_INDEX % len(BG_FILES)]
    BG_INDEX += 1

    bg = cv2.imread(str(bg_path))

    if bg is None:
        return np.ones((size, size, 3), dtype=np.uint8) * 255

    h, w = bg.shape[:2]
    scale = max(size / w, size / h)
    bg = cv2.resize(bg, None, fx=scale, fy=scale)

    y0 = random.randint(0, bg.shape[0] - size)
    x0 = random.randint(0, bg.shape[1] - size)
    bg = bg[y0:y0 + size, x0:x0 + size]

    if random.random() < 0.5:
        k = random.choice([3, 5])
        bg = cv2.GaussianBlur(bg, (k, k), 0)

    if random.random() < 0.7:
        alpha = random.uniform(0.75, 1.25)
        beta = random.randint(-25, 25)
        bg = cv2.convertScaleAbs(bg, alpha=alpha, beta=beta)

    if random.random() < 0.25:
        noise = np.random.normal(0, 8, bg.shape).astype(np.int16)
        bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return bg

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

def sample_layout_type():
    r = random.random()
    cum = 0
    for k, p in LAYOUT_PROBS.items():
        cum += p
        if r <= cum:
            return k
    return "cluster"

# ---------- PLACE OBJECT (Clustered + Allow Overlap) ----------
def place_object(canvas, fg, boxes, cluster_center, layout_type="cluster"):
    h, w = fg.shape[:2]
    H, W = canvas.shape[:2]

    if h >= H or w >= W:
        return None

    cx_cluster, cy_cluster = cluster_center

    for _ in range(60):

        if layout_type == "cluster":
            x = int(np.random.normal(cx_cluster, W * 0.14))
            y = int(np.random.normal(cy_cluster, H * 0.14))

        elif layout_type == "spread":
            x = random.randint(0, W - w)
            y = random.randint(0, H - h)

        elif layout_type == "edge":
            side = random.choice(["top", "bottom", "left", "right"])

            if side == "top":
                x = random.randint(0, W - w)
                y = random.randint(0, int(H * 0.15))
            elif side == "bottom":
                x = random.randint(0, W - w)
                y = random.randint(max(0, int(H * 0.85) - h), H - h)
            elif side == "left":
                x = random.randint(0, int(W * 0.15))
                y = random.randint(0, H - h)
            else:
                x = random.randint(max(0, int(W * 0.85) - w), W - w)
                y = random.randint(0, H - h)

        else:  # mixed
            if random.random() < 0.6:
                x = int(np.random.normal(cx_cluster, W * 0.18))
                y = int(np.random.normal(cy_cluster, H * 0.18))
            else:
                x = random.randint(0, W - w)
                y = random.randint(0, H - h)

        x = max(0, min(W - w, x))
        y = max(0, min(H - h, y))

        rect = (x, y, x + w, y + h)

        # Cho overlap nhẹ để giống cảnh bếp thật hơn
        if all(iou(rect, bx) < 0.35 for bx in boxes):

            if fg.shape[2] != 4:
                continue

            alpha = fg[:, :, 3] / 255.0

            for c in range(3):
                canvas[y:y+h, x:x+w, c] = (
                    alpha * fg[:, :, c] +
                    (1 - alpha) * canvas[y:y+h, x:x+w, c]
                )

            alpha = fg[:, :, 3] / 255.0

            for c in range(3):
                canvas[y:y+h, x:x+w, c] = (
                    alpha * fg[:, :, c] +
                    (1 - alpha) * canvas[y:y+h, x:x+w, c]
                )

            alpha_2d = fg[:, :, 3]
            ys, xs = np.where(alpha_2d > 10)

            if len(xs) == 0 or len(ys) == 0:
                return None

            tight_rect = (
                x + int(xs.min()),
                y + int(ys.min()),
                x + int(xs.max()),
                y + int(ys.max())
            )

            boxes.append(tight_rect)
            return tight_rect

    return None


# ---------- LOAD HF ----------
def load_hf_pool(class_map):
    pool = defaultdict(list)

    for hf in HF_DATASETS:
        print(f"📥 Loading {hf}")
        ds = load_dataset(hf, split="train")
        skip_count = 0

        # Tối ưu: không decode ảnh ngay
        if "image" in ds.features:
            ds = ds.cast_column("image", HFImage(decode=False))

        for s in ds:
            # ===== LẤY RAW LABEL =====
            if "label" in s and "label" in ds.features:
                raw = ds.features["label"].names[s["label"]]
            elif "ingredient" in s:
                raw = s["ingredient"]
            else:
                continue

            if not raw:
                continue

            raw = raw.lower().strip()
            raw = raw.replace("-", "_").replace(" ", "_")

            norm = map_class(raw)

            if not norm:
                continue

            # ===== CHECK TRONG data.yaml =====
            if norm not in class_map:
                print(f"⛔ Skip {raw} → {norm} (not in data.yaml)")
                skip_count += 1
                continue

            cid = class_map[norm]

            # ===== LIMIT PER CLASS =====
            if len(pool[cid]) >= MAX_PER_CLASS:
                continue

            # ===== LẤY IMAGE =====
            img_info = s.get("image", None)
            if not img_info:
                continue

            img_bytes = img_info.get("bytes", None)
            if not img_bytes:
                continue

            pool[cid].append(img_bytes)

    print("✅ Done HF loading")
    print(f"⛔ Total skipped (HF): {skip_count}")

    # Debug thống kê
    print("\n📊 HF Pool stats:")
    for cid, imgs in pool.items():
        print(f"Class {cid}: {len(imgs)} images")

    return pool

# ---------- LOAD EXTERNAL DATASET ----------
def load_external_dataset(pool, class_map):
    if not EXTERNAL_PATH.exists():
        print("⚠ No external_dataset found.")
        return pool
    
    skip_count = 0

    print("📥 Loading external_dataset")

    for folder in EXTERNAL_PATH.iterdir():
        if not folder.is_dir():
            continue

        raw_name = folder.name.lower().strip()

        # 🔥 dùng mapping yaml
        norm = map_class(raw_name)

        if norm not in class_map:
            print(f"⛔ Skip {raw_name} → mapped to {norm} (not in data.yaml)")
            skip_count += 1
            continue

        cid = class_map[norm]

        for img_path in folder.glob("*.*"):
            try:
                with open(img_path, "rb") as f:
                    pool[cid].append(f.read())
            except:
                continue

    print("✅ Done external_dataset")
    print(f"⛔ Total skipped (external): {skip_count}")
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

def get_processed_pool():
    """
    Quét trực tiếp thư mục processed_ingredients để lấy danh sách file đã có.
    """
    pool = defaultdict(list)
    if not PROCESSED_DIR.exists():
        return pool
    
    print(f"🔍 Đang quét thư mục: {PROCESSED_DIR}")
    for class_folder in PROCESSED_DIR.iterdir():
        if class_folder.is_dir():
            cid = int(class_folder.name)
            images = list(class_folder.glob("*.png"))
            pool[cid].extend([str(img) for img in images])
            
    return pool

# ---------- MAIN ----------
# ---------- MAIN ----------
def main():
    class_map = load_class_map()
    
    # 1. Lấy danh sách ảnh đã tách nền sẵn từ folder của bạn
    if PROCESSED_DIR.exists() and any(PROCESSED_DIR.iterdir()):
        pool = get_processed_pool()
        valid_ids = set(class_map.values())
        pool = {k: v for k, v in pool.items() if k in valid_ids}
    else:
        pool = load_hf_pool(class_map)
        pool = load_external_dataset(pool, class_map)
        pool = preprocess_pool(pool)
        valid_ids = set(class_map.values())
        pool = {k: v for k, v in pool.items() if k in valid_ids and len(v) > 0}

    for cid in pool:
        if len(pool[cid]) > MAX_PER_CLASS:
            pool[cid] = random.sample(pool[cid], MAX_PER_CLASS)
    
    if not pool:
        print("⚠ Thư mục processed_ingredients trống hoặc không tồn tại!")
        return
    
    print("\n📊 Final usable classes:")
    for cid, imgs in pool.items():
        print(f"Class {cid}: {len(imgs)} images")

    print(f"Usable classes: {len(pool)}")

    for split, target in SPLITS.items():
        print(f"--- Starting Split: {split} ---")
        
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
        STEP = 500 
        
        class_indices = defaultdict(int)

        for cid in pool:
            random.shuffle(pool[cid])

        gc_counter = 0
        while img_idx < target:
            if img_idx > 0 and img_idx % STEP == 0:
                print(f"   🚀 Tiến độ: [{img_idx}/{target}] - Split: {split}")
                
            size = random.randint(CANVAS_MIN, CANVAS_MAX)
            canvas = load_random_background(size) if random.random() > 0.2 else np.ones((size, size, 3), dtype=np.uint8) * 255

            boxes, labels = [], []

            scene = sample_scene_type()
            layout_type = sample_layout_type()

            cluster_center = (
                random.randint(int(size * 0.3), int(size * 0.7)),
                random.randint(int(size * 0.3), int(size * 0.7))
            )
            
            min_i, max_i = SCENE_TYPES[scene]
            valid_cids = [cid for cid in pool if len(pool[cid]) > 0]
            num_ing = min(random.randint(min_i, max_i), len(valid_cids))

            all_cids = list(pool.keys())
            weights = [1.0 / (split_counts[scene][cid] + 1) for cid in all_cids]
            prob = np.array(weights) / sum(weights)
            selected_classes = np.random.choice(all_cids, size=num_ing, replace=False, p=prob)

            for cid in selected_classes:
                # Lấy đường dẫn file từ pool
                idx = class_indices[cid] % len(pool[cid])
                img_path = pool[cid][idx]
                class_indices[cid] += 1

                fg = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

                # Kiểm tra ảnh hợp lệ
                if fg is None or fg.ndim != 3 or fg.shape[2] != 4:
                    print(f"⚠ Invalid RGBA image: {img_path}")
                    continue
            
                
                if fg is None: 
                    print(f"⚠️ Failed to read: {img_path}")
                    continue

                # Augmentation vật thể (xoay, lật, v.v.)
                fg = augment_fg(fg)

                # Tính toán scale
                h0, w0 = fg.shape[:2]
                scale_map = {"single": 0.35, "medium": 0.20, "dense": 0.12}
                max_scale_map = {"single": 0.9, "medium": 0.5, "dense": 0.3}
                
                max_scale = min(
                    (size * max_scale_map[scene]) / w0,
                    (size * max_scale_map[scene]) / h0
                )
                
                min_scale = scale_map[scene]
                if max_scale <= min_scale:
                    continue

                scale = random.uniform(min_scale, max_scale)
                
                fg = cv2.resize(fg, None, fx=scale, fy=scale)

                box = place_object(canvas, fg, boxes, cluster_center, layout_type)
                if box:
                    split_counts[scene][cid] += 1
                    x1, y1, x2, y2 = box
                    labels.append(f"{cid} {(x1+x2)/2/size:.6f} {(y1+y2)/2/size:.6f} {(x2-x1)/size:.6f} {(y2-y1)/size:.6f}")

            if labels:
                name = f"{img_idx:06d}_{random.randint(0, 9999)}"
                cv2.imwrite(str(img_dir / f"{name}.jpg"), canvas)
                with open(lbl_dir / f"{name}.txt", "w") as f:
                    f.write("\n".join(labels))
                img_idx += 1
                
            gc_counter += 1
            if gc_counter % 100 == 0:
                gc.collect()

        print(f"✅ Finish {split}.")


if __name__ == "__main__":
    main()