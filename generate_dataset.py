import os
import json
import cv2

def convert_yolo_to_coco(root_path, split, class_names):
    image_dir = os.path.join(root_path, split, "images")
    label_dir = os.path.join(root_path, split, "labels")
    output_json = os.path.join(root_path, split, f"{split}_annotations.json")

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(class_names)]
    }

    ann_id = 1
    for img_id, img_name in enumerate(os.listdir(image_dir)):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')): continue

        # Đọc kích thước ảnh thực tế
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]

        coco["images"].append({
            "id": img_id, "file_name": img_name, "width": w, "height": h
        })

        # Đọc file label (.txt)
        label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    cls, x_c, y_c, bw, bh = map(float, line.split())
                    # YOLO (normalize) -> COCO (pixel)
                    xmin = (x_c - bw/2) * w
                    ymin = (y_c - bh/2) * h
                    abs_w, abs_h = bw * w, bh * h
                    
                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": int(cls),
                        "bbox": [xmin, ymin, abs_w, abs_h],
                        "area": abs_w * abs_h,
                        "iscrowd": 0
                    })
                    ann_id += 1
        if img_id % 100 == 0: print(f"Đang xử lý {split}: {img_id} ảnh...")

    with open(output_json, 'w') as f:
        json.dump(coco, f)
    print(f"--- Đã tạo xong: {output_json} ---")

# Danh sách class từ data11.yaml của bạn
NAMES = [
    "adzuki_bean", "allspice", "amaranth", "anchovy", "apple", "apricot", "artichoke", 
    "artichoke_heart", "arugula", "asparagus", "avocado", "avocado_oil", "bacon", 
    "baked_bean", "bamboo_shoot", "banana", "barley", "beef", "beet", "bergamot", 
    "bison", "black_bean", "black_cherry", "black_sapote", "blackberry", "blueberry", 
    "bok_choy", "boysenberry", "bread", "breadfruit", "broccoli", "broccoli_stem", 
    "brussels_sprout", "buckwheat", "buffalo", "bulgur", "butter", "cabbage", 
    "cannellini_bean", "canola_oil", "caper", "cardamom", "caribou", "carrot", 
    "cauliflower", "celeriac", "celery", "chard_stalk", "cheese", "cherimoya", 
    "cherry", "chia", "chicken", "chicken_breast", "chicken_thigh", "chickpea", 
    "chili_pepper", "chili_sauce", "chive", "cinnamon", "clam", "clove", "coconut", 
    "coconut_oil", "cod", "coffee", "collard_green", "coriander", "corn", "corn_grit", 
    "cornmeal", "couscous", "crab", "cracked_wheat", "cranberry", "cucumber", "cumin", 
    "daikon", "deer", "dragon_fruit", "duck", "dumpling_wrapper", "durian", "egg", 
    "einkorn", "elderberry", "elk", "emmer", "farfalle", "farro", "fava_bean", 
    "fennel", "fettuccine", "fish_sauce", "flaky_salt", "flaxseed_oil", "flour", 
    "freekeh", "fruit_cocktail", "fusilli", "garlic", "garlic_bulb", "ginger", 
    "glass_noodle", "goat", "goji_berry", "goose", "grapefruit", "grapeseed_oil", 
    "green_bean", "green_onion", "grouse", "guava", "guinea_fowl", "haddock", "ham", 
    "horseradish", "huckleberry", "jackfruit", "kale", "kamut", "kidney_bean", 
    "kiwi", "kohlrabi", "kumquat", "lamb", "lasagna", "leek", "lemon", "lemongrass", 
    "lentils", "lettuce", "lima_bean", "lime", "linguine", "lobster", "lychee", 
    "macaroni", "mackerel", "mandarin", "mango", "milk", "millet", "mulberry", 
    "mung_bean", "mushroom", "mustard_green", "mutton", "navy_bean", "nectarine", 
    "nutmeg", "oat", "octopus", "olive_oil", "olives", "onion", "orange", "ostrich", 
    "oyster", "papaya", "paprika", "parsnip", "partridge", "passion_fruit", "pasta", 
    "pasta_sauce", "pawpaw", "peach", "peanut_oil", "pear", "peas", "penne", "pepper", 
    "pesto", "pheasant", "pickle", "pineapple", "pinto_bean", "plum", "pluot", 
    "polenta", "pomelo", "pork", "potato", "pumpkin", "quail", "quinoa", "rabbit", 
    "radicchio", "radish", "ramp", "raspberry", "refried_bean", "rhubarb", "rice", 
    "rigatoni", "romaine", "rotini", "rutabaga", "salmon", "salsa", "salt", "santol", 
    "sapote", "sardine", "sausage", "savoy_cabbage", "scallop", "semolina", 
    "sesame_oil", "shallot", "shrimp", "sorghum", "sour_cherry", "soursop", "soybean", 
    "spaghetti", "spelt", "spinach", "squab", "squirrel", "strawberry", "sugar", 
    "sun_dried_tomato", "sunflower_oil", "sweet_potato", "swiss_chard", "tangerine", 
    "teff", "tomato", "trout", "tuna", "turkey", "turmeric", "turnip", "vegetable_oil", 
    "venison", "watercress", "wheat_bran", "white_peach", "wild_boar", "yuzu"
]

# Thực thi
DATASET_ROOT = "dataset_v11"
convert_yolo_to_coco(DATASET_ROOT, "train", NAMES)
convert_yolo_to_coco(DATASET_ROOT, "val", NAMES)