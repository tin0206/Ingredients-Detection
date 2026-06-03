import yaml
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset, Image as HFImage
from rembg import remove, new_session
from PIL import Image

# ================= CONFIG =================
DATA_YAML = "data.yaml"
CLASS_MAPPING_YAML = "class_mapping.yaml"

EXTERNAL_PATH = Path("external_dataset")
PROCESSED_DIR = Path("processed_ingredients_v2")
# Thư mục mới lưu ảnh gốc trước khi remove background
RAW_INGREDIENTS_DIR = Path("raw_ingredients_v2") 

HF_DATASETS = ["Scuccorese/food-ingredients-dataset"]
MAX_PER_CLASS = 150
REM_BG_SESSION = new_session("u2netp")
# ==========================================

def load_class_map():
    with open(DATA_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {v: int(k) for k, v in data["names"].items()}

def load_special_mapping():
    if not Path(CLASS_MAPPING_YAML).exists(): return {}
    with open(CLASS_MAPPING_YAML, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

SPECIAL_MAP = load_special_mapping()

# [Giữ nguyên hàm normalize_name và MERGE_GROUPS từ code cũ của bạn để map tên]
def normalize_name(name):
    # ... (Copy nguyên hàm normalize_name cũ của bạn vào đây)
    return name

def map_class(raw):
    if not raw: return None
    raw = raw.lower().strip().replace("-", "_").replace(" ", "_")
    if raw in SPECIAL_MAP: return SPECIAL_MAP[raw]
    return normalize_name(raw)

def remove_bg(img_bgr):
    h0, w0 = img_bgr.shape[:2]
    max_dim = 384
    scale = min(max_dim / max(h0, w0), 1.0)
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale)
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    fg = remove(pil, session=REM_BG_SESSION)
    return cv2.cvtColor(np.array(fg), cv2.COLOR_RGBA2BGRA)

def load_hf_pool(class_map):
    pool = defaultdict(list)
    for hf in HF_DATASETS:
        print(f"📥 Loading {hf}")
        ds = load_dataset(hf, split="train")
        if "image" in ds.features:
            ds = ds.cast_column("image", HFImage(decode=False))

        for s in ds:
            if "label" in s and "label" in ds.features:
                raw = ds.features["label"].names[s["label"]]
            elif "ingredient" in s:
                raw = s["ingredient"]
            else: continue

            norm = map_class(raw)
            if not norm or norm not in class_map: continue
            cid = class_map[norm]

            if len(pool[cid]) >= MAX_PER_CLASS: continue
            img_info = s.get("image", None)
            if not img_info: continue
            img_bytes = img_info.get("bytes", None)
            if not img_bytes: continue

            pool[cid].append(img_bytes)
    return pool

def load_external_dataset(pool, class_map):
    if not EXTERNAL_PATH.exists(): return pool
    print("📥 Loading external_dataset")
    for folder in EXTERNAL_PATH.iterdir():
        if not folder.is_dir(): continue
        norm = map_class(folder.name)
        if norm not in class_map: continue
        cid = class_map[norm]

        for img_path in folder.glob("*.*"):
            try:
                with open(img_path, "rb") as f:
                    pool[cid].append(f.read())
            except: continue
    return pool

def process_and_save_pool(pool):
    PROCESSED_DIR.mkdir(exist_ok=True)
    RAW_INGREDIENTS_DIR.mkdir(exist_ok=True)
    
    for cid, img_list in pool.items():
        # Tạo folder theo Class ID cho cả 2 loại
        class_processed_dir = PROCESSED_DIR / str(cid)
        class_raw_dir = RAW_INGREDIENTS_DIR / str(cid)
        
        class_processed_dir.mkdir(exist_ok=True)
        class_raw_dir.mkdir(exist_ok=True)
        
        print(f"⏳ Processing Class {cid} ({len(img_list)} images)...")
        for i, img_bytes in enumerate(img_list):
            out_processed = class_processed_dir / f"{i}.png"
            out_raw = class_raw_dir / f"{i}.jpg"
            
            # Nếu đã tồn tại cả 2 file thì bỏ qua
            if out_processed.exists() and out_raw.exists():
                continue
                
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None: continue
            
            # 1. Lưu ảnh gốc trước khi xóa nền (.jpg)
            cv2.imwrite(str(out_raw), img)
            
            # 2. Xóa nền và lưu ảnh đã xử lý (.png)
            fg = remove_bg(img)
            cv2.imwrite(str(out_processed), fg)

def main():
    class_map = load_class_map()
    pool = load_hf_pool(class_map)
    pool = load_external_dataset(pool, class_map)
    
    print("\n📦 Starting preprocessing and background removal...")
    process_and_save_pool(pool)
    print("✅ Preprocessing completed successfully!")

if __name__ == "__main__":
    main()