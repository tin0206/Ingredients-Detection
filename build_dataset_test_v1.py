import random
import cv2
import yaml
import numpy as np
import gc
from pathlib import Path
from collections import defaultdict

# ================= CONFIG =================
DATA_YAML = "data_test_v1.yaml"
PROCESSED_DIR = Path("processed_ingredients")
BACKGROUND_DIR = Path("backgrounds")
OUT_ROOT = Path("dataset_test_v1")

IMG_SIZE = 640

SCENE_TYPES = {
    "single": (1, 1),
    "medium": (3, 7),
    "dense": (6, 10)
}

SCENE_PROBS = {
    "single": 0.35,
    "medium": 0.45,
    "dense": 0.20
}

SPLITS = {
    "train": 20000,
    "val": 4000,
    "test": 4000
}

SIZE_GROUP = {
    "small": ["salt","sugar","pepper","cumin","turmeric","nutmeg","clove","cardamom"],
    "medium": ["tomato","onion","carrot","apple","lemon","garlic","mushroom","chili_pepper"],
    "large": ["chicken","beef","fish","salmon","pasta","bread","duck","turkey","lamb"]
}

CONFUSION_MAP = {
    "tomato": ["apple","cherry"],
    "lemon": ["lime","orange"],
    "garlic": ["onion","shallot"],
    "chicken": ["turkey","duck"]
}

# ================= CLASS MAP =================
def load_class_map():
    with open(DATA_YAML, "r") as f:
        data = yaml.safe_load(f)
    return {v: int(k) for k, v in data["names"].items()}

# ================= POOL =================
def load_processed_pool(class_map):
    pool = defaultdict(list)

    for class_dir in PROCESSED_DIR.iterdir():
        if not class_dir.is_dir():
            continue

        name = class_dir.name.lower().strip()

        if name.isdigit():
            cid = int(name)
            cname = None
        else:
            if name not in class_map:
                continue
            cid = class_map[name]
            cname = name

        imgs = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
        pool[cid].extend([(str(p), cname) for p in imgs])

    return pool

# ================= WEIGHTS (FIXED) =================
def build_class_weights(pool):
    keys = list(pool.keys())

    counts = np.array([len(pool[c]) for c in keys], dtype=np.float64)

    weights = 1.0 / (np.log1p(counts) + 1e-6)

    # FIX: clean numerical stability
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights = weights + 1e-12
    weights = weights / np.sum(weights)

    return keys, weights

# ================= SIM2REAL =================
def sim2real(img):
    h, w = img.shape[:2]

    if img.shape[2] == 4:
        rgb = img[:, :, :3]
        alpha = img[:, :, 3]
    else:
        rgb = img
        alpha = np.ones((h, w), dtype=np.uint8) * 255

    rgb = cv2.convertScaleAbs(
        rgb,
        alpha=random.uniform(0.75, 1.25),   # FIX: reduce extreme
        beta=random.randint(-20, 20)        # FIX: reduce shift
    )

    gamma = random.uniform(0.7, 1.5)       # FIX: narrower range
    inv = 1.0 / gamma
    lut = np.array([((i/255.0)**inv)*255 for i in range(256)]).astype(np.uint8)
    rgb = cv2.LUT(rgb, lut)

    if random.random() < 0.5:
        rgb = rgb.astype(np.int16)
        shift = random.randint(-15, 15)   # FIX
        rgb[:, :, 0] = np.clip(rgb[:, :, 0] + shift, 0, 255)
        rgb[:, :, 2] = np.clip(rgb[:, :, 2] - shift, 0, 255)
        rgb = rgb.astype(np.uint8)

    if random.random() < 0.25:            # FIX: lower noise
        noise = np.random.normal(0, 5, rgb.shape).astype(np.int16)
        rgb = np.clip(rgb + noise, 0, 255).astype(np.uint8)

    # 🔥 IMPORTANT: reduce blur probability
    if random.random() < 0.12:
        k = random.choice([3])            # FIX: remove 5
        rgb = cv2.GaussianBlur(rgb, (k, k), 0)

    if random.random() < 0.25:
        q = random.randint(60, 95)        # FIX: less compression
        _, enc = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        rgb = cv2.imdecode(enc, 1)

    return np.dstack((rgb, alpha))

# ================= ROTATE =================
def random_rotate(img):
    angle = random.gauss(0, 25)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

# ================= AUGMENT =================
def global_light(img):
    return np.clip(img.astype(np.float32) * random.uniform(0.7,1.3), 0, 255).astype(np.uint8)

def apply_perspective(img):
    h,w = img.shape[:2]
    dx,dy = int(0.08*w), int(0.08*h)

    src = np.float32([[0,0],[w,0],[w,h],[0,h]])
    dst = np.float32([
        [random.randint(0,dx), random.randint(0,dy)],
        [w-random.randint(0,dx), random.randint(0,dy)],
        [w-random.randint(0,dx), h-random.randint(0,dy)],
        [random.randint(0,dx), h-random.randint(0,dy)]
    ])

    M = cv2.getPerspectiveTransform(src,dst)
    return cv2.warpPerspective(img,M,(w,h),borderMode=cv2.BORDER_REFLECT)

def motion_blur(img):
    if random.random() < 0.18:
        k = random.randint(3, 7)
        kernel = np.zeros((k, k))
        kernel[k//2, :] = 1
        kernel /= k
        img = cv2.filter2D(img, -1, kernel)
    return img

# ================= PIPELINE =================
def production_pipeline(img):
    img = random_rotate(img)
    img = apply_perspective(img)

    img = global_light(img)

    img = sim2real(img)

    img = motion_blur(img)

    return img

def camera_effect(img):
    if random.random() < 0.3:
        h, w = img.shape[:2]
        for i in range(h):
            shift = int(np.sin(i / 8) * 2)
            img[i] = np.roll(img[i], shift, axis=0)

    if random.random() < 0.35:
        noise = np.random.poisson(np.clip(img/10,0,255)).astype(np.uint8)
        img = cv2.add(img, noise)

    return img

# ================= OCCLUSION =================
def occlusion(alpha):
    if random.random() < 0.5:
        h,w = alpha.shape
        for _ in range(2):
            x1,x2 = sorted([random.randint(0,w),random.randint(0,w)])
            y1,y2 = sorted([random.randint(0,h),random.randint(0,h)])
            alpha[y1:y2,x1:x2] *= random.uniform(0,0.5)
    return alpha

# ================= IOU =================
def iou(a,b):
    xA,yA,xB,yB = max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])
    inter = max(0,xB-xA)*max(0,yB-yA)
    ua = (a[2]-a[0])*(a[3]-a[1])
    ub = (b[2]-b[0])*(b[3]-b[1])
    return inter/(ua+ub-inter+1e-6)

# ================= PLACE =================
def place(canvas, fg, boxes, center):
    H, W = canvas.shape[:2]
    h, w = fg.shape[:2]

    if h >= H or w >= W:
        return None

    for _ in range(50):
        x = int(np.random.normal(center[0], W * 0.12))
        y = int(np.random.normal(center[1], H * 0.12))

        x = max(0, min(W-w, x))
        y = max(0, min(H-h, y))

        roi = canvas[y:y+h, x:x+w]

        if roi.shape[:2] != (h, w):
            continue

        rect = (x,y,x+w,y+h)

        if all(iou(rect,b)<0.35 for b in boxes):

            alpha = fg[:,:,3].astype(np.float32)/255.0
            alpha = occlusion(alpha)[...,None]

            fg_rgb = fg[:,:,:3].astype(np.float32)
            bg = roi.astype(np.float32)

            if fg_rgb.shape != bg.shape:
                continue

            blended = alpha*fg_rgb + (1-alpha)*bg
            canvas[y:y+h,x:x+w] = blended.astype(np.uint8)

            boxes.append(rect)
            return rect

    return None

# ================= BACKGROUND =================
def load_bg(size):
    files = list(BACKGROUND_DIR.glob("*.*"))
    if not files:
        return np.ones((size,size,3),np.uint8)*255

    bg = cv2.imread(str(random.choice(files)))
    h,w = bg.shape[:2]
    scale = max(size/w,size/h)
    bg = cv2.resize(bg,None,fx=scale,fy=scale)

    y = random.randint(0,bg.shape[0]-size)
    x = random.randint(0,bg.shape[1]-size)

    return bg[y:y+size,x:x+size]

# ================= MAIN =================
def main():
    OUT_ROOT.mkdir(parents=True,exist_ok=True)

    class_map = load_class_map()
    pool = load_processed_pool(class_map)

    valid, weights = build_class_weights(pool)

    for split,target in SPLITS.items():

        img_dir = OUT_ROOT/split/"images"
        lbl_dir = OUT_ROOT/split/"labels"
        img_dir.mkdir(parents=True,exist_ok=True)
        lbl_dir.mkdir(parents=True,exist_ok=True)

        idx = 0

        while idx < target:

            size = random.choice([IMG_SIZE,int(IMG_SIZE*0.75),int(IMG_SIZE*1.25)])
            canvas = load_bg(size)

            boxes, labels = [], []

            scene = random.choices(list(SCENE_TYPES.keys()), weights=list(SCENE_PROBS.values()))[0]
            num = random.randint(*SCENE_TYPES[scene])

            # ===== FIX CRITICAL: normalize probability EACH loop =====
            p = np.array(weights, dtype=np.float64)
            p = p / p.sum()

            selected = np.random.choice(
                valid,
                size=min(num, len(valid)),
                replace=False,
                p=p
            )

            cx, cy = random.randint(int(size*0.3),int(size*0.7)), random.randint(int(size*0.3),int(size*0.7))

            for cid in selected:

                img_path, cname = random.choice(pool[cid])
                fg = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if fg is None:
                    continue

                fg = production_pipeline(fg)
                fg = camera_effect(fg)

                scale = random.uniform(0.25, 1.1)

                h, w = fg.shape[:2]
                fg = cv2.resize(fg, (max(25,int(w*scale)), max(25,int(h*scale))))

                box = place(canvas, fg, boxes, (cx,cy))

                if box:
                    x1,y1,x2,y2 = box
                    labels.append(f"{cid} {(x1+x2)/2/size:.6f} {(y1+y2)/2/size:.6f} {(x2-x1)/size:.6f} {(y2-y1)/size:.6f}")

            if labels:
                name = f"{idx:06d}_{random.randint(0,9999)}"
                cv2.imwrite(str(img_dir/f"{name}.jpg"), canvas)
                open(lbl_dir/f"{name}.txt","w").write("\n".join(labels))
                idx += 1

            if idx % 200 == 0:
                gc.collect()
                print(split, idx)

        print("DONE", split)

if __name__ == "__main__":
    main()