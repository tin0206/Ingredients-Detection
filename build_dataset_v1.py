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
DATA_YAML = "data.yaml"
CLASS_MAPPING_YAML = "class_mapping.yaml"

OUT_ROOT = Path("dataset")
BACKGROUND_DIR = Path("backgrounds")
EXTERNAL_PATH = Path("external_dataset")
PROCESSED_DIR = Path("processed_ingredients")

HF_DATASETS = [
    "Scuccorese/food-ingredients-dataset"
]

MAX_PER_CLASS = 150

CANVAS_MIN = 640
CANVAS_MAX = 800

# 🔥 Tối ưu hóa số lượng và kịch bản sinh ảnh
SCENE_TYPES = {
    "single": (1, 1),
    "medium": (3, 6),
    "dense": (7, 12)  # Giảm bớt số lượng tối đa để tránh quá tải không gian nhưng tăng chồng lấn
}

SCENE_PROBS = {
    "single": 0.35,
    "medium": 0.45,
    "dense": 0.20
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


# ===================== MERGE GROUP =====================
MERGE_GROUPS = {
    "oil": ["olive_oil", "canola_oil", "grapeseed_oil", "peanut_oil", "sesame_oil", "sunflower_oil", "vegetable_oil", "avocado_oil", "flaxseed_oil", "coconut_oil"],
    "pasta": ["spaghetti", "penne", "fusilli", "rigatoni", "linguine", "fettuccine", "macaroni", "rotini", "farfalle"],
    "grain": ["barley", "oat", "millet", "sorghum", "teff", "spelt", "farro", "emmer", "einkorn", "corn_grit", "cracked_wheat", "freekeh", "polenta", "wheat_bran", "quinoa", "kamut"],
    "bean": ["black_bean", "kidney_bean", "navy_bean", "pinto_bean", "mung_bean", "adzuki_bean", "lima_bean", "fava_bean", "cannellini_bean", "refried_bean"],
    "chicken": ["chicken", "chicken_breast", "chicken_thigh"],
    "garlic": ["garlic", "garlic_bulb"],
    "broccoli": ["broccoli", "broccoli_stem"],
    "cherry": ["black_cherry", "sour_cherry"],
    "berry": ["blackberry", "blueberry", "cranberry", "raspberry", "elderberry", "huckleberry", "mulberry", "boysenberry", "goji_berry"],
}

MERGE_LOOKUP = {}
for target, sources in MERGE_GROUPS.items():
    for s in sources:
        MERGE_LOOKUP[s] = target
        
REMOVE_CLASSES = {
    "artichoke_heart", "black_sapote", "bison", "buffalo", "bulgur", "buckwheat", 
    "caribou", "chard_stalk", "cornmeal", "elk", "deer", 
    "grouse", "guinea_fowl", "pawpaw", "partridge", "pheasant", 
    "quail", "salsa", "squab", "squirrel", "semolina", "wild_boar", "ostrich", "venison"
}


def normalize_name(name):
    if not name:
        return None

    name = name.lower().strip()
    name = name.replace("-", "_").replace(" ", "_")

    for prefix in ["canned_", "jarred_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]

    special_map = {
        "apples": "apple", "apricots": "apricot", "beets": "beet", "carrots": "carrot",
        "cherries": "cherry", "mushrooms": "mushroom", "peaches": "peach", "pineapples": "pineapple",
        "pears": "pear", "tomatoes": "tomato", "mandarin_oranges": "mandarin",
        "sea_salt": "salt", "kosher_salt": "salt", "black_salt": "salt", "pink_salt": "salt",
        "table_salt": "salt", "smoked_salt": "salt", "iodized_salt": "salt", "celtic_salt": "salt", "pickling_salt": "salt",
        "brown_sugar": "sugar", "white_sugar": "sugar", "powdered_sugar": "sugar", "cane_sugar": "sugar",
        "coconut_sugar": "sugar", "raw_sugar": "sugar", "demerara_sugar": "sugar", "muscovado_sugar": "sugar",
        "turbinado_sugar": "sugar", "date_sugar": "sugar",
        "beluga_lentils": "lentils", "black_lentils": "lentils", "brown_lentils": "lentils", "french_lentils": "lentils",
        "golden_lentils": "lentils", "green_lentils": "lentils", "orange_lentils": "lentils", "red_lentils": "lentils",
        "spacing_pardina_lentils": "lentils", "yellow_lentils": "lentils", "sprouted_lentils": "lentils",
        "green_peas": "peas", "field_peas": "peas", "pigeon_peas": "peas", "snap_peas": "peas",
        "snow_peas": "peas", "split_peas": "peas", "white_peas": "peas", "yellow_peas": "peas",
        "black_eyed_peas": "peas", "sprouted_green_peas": "peas",
        "castelvetrano_olives": "olives", "cerignola_olives": "olives", "gaeta_olives": "olives",
        "kalamata_olives": "olives", "ligurian_olives": "olives", "manzanilla_olives": "olives",
        "nicoise_olives": "olives", "picholine_olives": "olives", "black_olives": "olives", "green_olives": "olives",
        "all_purpose_flour": "flour", "bread_flour": "flour", "cake_flour": "flour", "oat_flour": "flour",
        "rye_flour": "flour", "gluten_free_flour": "flour", "self_rising_flour": "flour", "white_flour": "flour",
        "whole_wheat_flour": "flour", "almond_flour": "flour", "coconut_flour": "flour",
        "spring_onion": "green_onion", "scallion": "green_onion", "pearl_onion": "onion",
        "elephant_garlic": "garlic", "ginger_root": "ginger",
        "sprouted_adzuki_beans": "adzuki_bean", "sprouted_black_beans": "black_bean", "sprouted_chickpeas": "chickpea",
        "sprouted_kidney_beans": "kidney_bean", "sprouted_mung_beans": "mung_bean", "sprouted_navy_beans": "navy_bean",
        "sprouted_pinto_beans": "pinto_bean", "sprouted_soybeans": "soybean",
        "adzuki_beans": "adzuki_bean", "black_beans": "black_bean", "kidney_beans": "kidney_bean",
        "mung_beans": "mung_bean", "navy_beans": "navy_bean", "pinto_beans": "pinto_bean",
        "soybeans": "soybean", "chickpeas": "chickpea", "white_rice": "rice", "glass_noodles": "glass_noodle"
    }

    if name in special_map:
        name = special_map[name]

    if name.endswith("s") and not name.endswith("ss"):
        if name not in ["peas", "lentils", "olives"]:
            name = name[:-1]
            
    spelling_fix = {
        "anchovie": "anchovy", "octopu": "octopus", "couscou": "couscous",
        "asparagu": "asparagus", "sun_dried_tomatoe": "sun_dried_tomato"
    }

    if name in spelling_fix:
        name = spelling_fix[name]
        
    if name == "couscouscucumber" or name in REMOVE_CLASSES:
        return None
    
    if name in MERGE_LOOKUP:
        name = MERGE_LOOKUP[name]
        
    return name


def map_class(raw):
    if not raw:
        return None
    raw = raw.lower().strip().replace("-", "_").replace(" ", "_")
    if raw in SPECIAL_MAP:
        return SPECIAL_MAP[raw]
    return normalize_name(raw)


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
    # 1. Xoay ngẫu nhiên từ 0 - 360 độ đầy đủ
    angle = random.randint(0, 360)
    h, w = fg.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    fg = cv2.warpAffine(fg, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

    # 2. Lật phối cảnh trái/phải ngẫu nhiên
    if random.random() > 0.5:
        fg = cv2.flip(fg, 1)

    # 3. Tinh chỉnh nhẹ Brightness / Contrast
    alpha = random.uniform(0.85, 1.15)
    beta = random.randint(-15, 15)
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
    bg = bg[y0:y0 + size, x0:x0 + size]

    # Augment Nền
    if random.random() < 0.5:
        k = random.choice([3, 5])
        bg = cv2.GaussianBlur(bg, (k, k), 0)

    if random.random() < 0.6:
        alpha = random.uniform(0.8, 1.2)
        beta = random.randint(-20, 20)
        bg = cv2.convertScaleAbs(bg, alpha=alpha, beta=beta)

    return bg


# ---------- IOU KINH ĐIỂN ----------
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


# ---------- PLACE OBJECT (Sửa đổi: Cho phép đè chồng thực tế + Mờ ranh giới) ----------
def place_object(canvas, fg, boxes, cluster_center, scene):
    h, w = fg.shape[:2]
    H, W = canvas.shape[:2]

    if h >= H or w >= W:
        return None

    cx_cluster, cy_cluster = cluster_center

    # 🔥 SỬA ĐỔI 1: Tăng ngưỡng IoU cho phép tùy theo độ dày đặc của scene nhằm kích hoạt hiện tượng occlusion (che khuất)
    # Gói kịch bản 'dense' tăng hẳn lên 0.45 để các vật thể chấp nhận đè chồng lên nhau tự nhiên
    max_iou_allowed = 0.45 if scene == "dense" else (0.25 if scene == "medium" else 0.05)

    for _ in range(100): # Tăng số lần thử tìm vị trí phù hợp quanh cụm
        x = int(np.random.normal(cx_cluster, W * 0.12))
        y = int(np.random.normal(cy_cluster, H * 0.12))

        x = max(0, min(W - w, x))
        y = max(0, min(H - h, y))

        rect = (x, y, x + w, y + h)

        if all(iou(rect, bx) < max_iou_allowed for bx in boxes):
            if fg.shape[2] != 4:
                continue
                
            alpha_mask = fg[:, :, 3] / 255.0
            
            # Khởi tạo vùng canvas chuẩn bị hòa trộn
            roi = canvas[y:y+h, x:x+w]
            
            # Trộn kênh Alpha phối hợp vật thể vào nền
            for c in range(3):
                roi[:, :, c] = (alpha_mask * fg[:, :, c] + (1.0 - alpha_mask) * roi[:, :, c])
            
            # 🔥 SỬA ĐỔI 3: Kỹ thuật xử lý Domain Gap - Làm mịn viền cục bộ (Edge Blending)
            # Quét mặt nạ nhị phân quanh vùng ranh giới biên để làm mờ nhẹ, tránh răng cưa cắt dán thô
            blur_mask = cv2.GaussianBlur((alpha_mask * 255).astype(np.uint8), (3, 3), 0)
            refined_roi = cv2.GaussianBlur(roi, (3, 3), 0)
            
            # Chỉ áp dụng mờ ở viền vật thể (nơi mask chuyển sắc giữa 0 và 255)
            edge_indices = (blur_mask > 10) & (blur_mask < 245)
            roi[edge_indices] = refined_roi[edge_indices]
            
            canvas[y:y+h, x:x+w] = roi
            boxes.append(rect)
            return rect

    return None


# ---------- LOAD POOL DATASET ----------
def load_hf_pool(class_map):
    pool = defaultdict(list)
    for hf in HF_DATASETS:
        print(f"📥 Loading {hf}")
        ds = load_dataset(hf, split="train")
        skip_count = 0

        if "image" in ds.features:
            ds = ds.cast_column("image", HFImage(decode=False))

        for s in ds:
            if "label" in s and "label" in ds.features:
                raw = ds.features["label"].names[s["label"]]
            elif "ingredient" in s:
                raw = s["ingredient"]
            else:
                continue

            if not raw: continue
            norm = map_class(raw)

            if not norm or norm not in class_map:
                skip_count += 1
                continue

            cid = class_map[norm]
            if len(pool[cid]) >= MAX_PER_CLASS:
                continue

            img_info = s.get("image", None)
            if not img_info: continue
            img_bytes = img_info.get("bytes", None)
            if not img_bytes: continue

            pool[cid].append(img_bytes)

    print(f"✅ Done HF loading. Skipped: {skip_count}")
    return pool


def load_external_dataset(pool, class_map):
    if not EXTERNAL_PATH.exists():
        return pool
    
    print("📥 Loading external_dataset")
    for folder in EXTERNAL_PATH.iterdir():
        if not folder.is_dir(): continue
        norm = map_class(folder.name)

        if norm not in class_map:
            continue

        cid = class_map[norm]
        for img_path in folder.glob("*.*"):
            try:
                with open(img_path, "rb") as f:
                    pool[cid].append(f.read())
            except:
                continue
    return pool


def preprocess_pool(pool):
    PROCESSED_DIR.mkdir(exist_ok=True)
    new_pool = defaultdict(list)
    
    for cid, img_list in pool.items():
        class_dir = PROCESSED_DIR / str(cid)
        class_dir.mkdir(exist_ok=True)
        
        for i, img_bytes in enumerate(img_list):
            out_file = class_dir / f"{i}.png"
            if out_file.exists():
                new_pool[cid].append(str(out_file))
                continue
                
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None: continue
            
            fg = remove_bg(img)
            cv2.imwrite(str(out_file), fg)
            new_pool[cid].append(str(out_file))
            
    return new_pool


def get_processed_pool():
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


# ---------- PROCESS MAIN PIPELINE ----------
def main():
    class_map = load_class_map()
    
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
        print("⚠ Thư mục dữ liệu trống!")
        return

    for split, target in SPLITS.items():
        print(f"--- Starting Split: {split} ---")
        
        split_counts = {"single": defaultdict(int), "medium": defaultdict(int), "dense": defaultdict(int)}

        img_dir = OUT_ROOT / split / "images"
        lbl_dir = OUT_ROOT / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        img_idx = len(list(img_dir.glob("*.jpg")))
        STEP = 500 
        gc_counter = 0

        while img_idx < target:
            if img_idx > 0 and img_idx % STEP == 0:
                print(f"   🚀 Tiến độ: [{img_idx}/{target}] - Split: {split}")
                
            size = random.randint(CANVAS_MIN, CANVAS_MAX)
            canvas = load_random_background(size) if random.random() > 0.15 else np.ones((size, size, 3), dtype=np.uint8) * 255

            boxes, labels = [], []
            scene = sample_scene_type()
            
            # Điểm neo trung tâm cho cụm nguyên liệu (mô phỏng lòng đĩa thức ăn)
            cluster_center = (
                random.randint(int(size * 0.35), int(size * 0.65)),
                random.randint(int(size * 0.35), int(size * 0.65))
            )
            
            min_i, max_i = SCENE_TYPES[scene]
            valid_cids = [cid for cid in pool if len(pool[cid]) > 0]
            num_ing = min(random.randint(min_i, max_i), len(valid_cids))

            all_cids = list(pool.keys())
            weights = [1.0 / (split_counts[scene][cid] + 1) for cid in all_cids]
            prob = np.array(weights) / sum(weights)
            selected_classes = np.random.choice(all_cids, size=num_ing, replace=True if scene == "dense" else False, p=prob)

            # 🔥 SỬA ĐỔI 4: Duyệt dán nhãn theo thứ tự diện tích giảm dần (Lớn trước -> Nhỏ sau)
            # Giúp cho tỏi hay củ cải nhỏ nằm đè lên trên thịt sườn, tránh hiện tượng vật thể nhỏ bị nuốt chửng hoàn toàn dưới góc nhìn 2D
            objects_to_place = []
            for cid in selected_classes:
                img_path = random.choice(pool[cid])
                fg = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if fg is None: continue
                
                fg = augment_fg(fg)
                h0, w0 = fg.shape[:2]
                
                # 🔥 SỬA ĐỔI 2: Đồng bộ kích thước thực tế (Scale Inconsistency Fix)
                # Thay vì bóp nhỏ vật thể lại khi ảnh đông đúc, ta giữ dải scale đồng đều, cho phép chúng tự tranh chấp không gian
                min_scale = 0.16
                max_scale = min((size * 0.38) / w0, (size * 0.38) / h0)
                
                if max_scale <= min_scale: 
                    scale = min_scale
                else:
                    scale = random.uniform(min_scale, max_scale)
                
                fg = cv2.resize(fg, None, fx=scale, fy=scale)
                area = fg.shape[0] * fg.shape[1]
                objects_to_place.append((area, fg, cid))
            
            # Sắp xếp giảm dần theo diện tích bề mặt pixel
            objects_to_place.sort(key=lambda item: item[0], reverse=True)

            # Tiến hành thả vật thể đã xếp thứ tự vào canvas đĩa nền
            for _, fg, cid in objects_to_place:
                box = place_object(canvas, fg, boxes, cluster_center, scene)
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
            if gc_counter % 120 == 0:
                gc.collect()

        print(f"✅ Finish Split: {split}.")


if __name__ == "__main__":
    main()