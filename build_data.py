import csv
import os
import yaml
from datasets import load_dataset

# =========================
# CONFIG
# =========================
HF_DATASETS = [
    "Scuccorese/food-ingredients-dataset"
]

EXTERNAL_PATH = "external_dataset"
OUTPUT_YAML = "data.yaml"
MAPPING_YAML = "class_mapping.yaml"

DATASET_ROOT = "dataset"
TRAIN_PATH = "images/train"
VAL_PATH = "images/val"
TEST_PATH = "images/test"

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
        "spelt", "farro", "emmer", "einkorn", "corn_grit", "cracked_wheat", 
        "freekeh", "polenta", "wheat_bran", "barley", "oat", "millet", 
        "sorghum", "teff", "spelt", "quinoa", "einkorn", "emmer", "farro", "kamut"
    
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

def build_mapping(raw_all):
    normalized_cache = {}
    final_classes = set()

    for raw in raw_all:
        raw_clean = raw.lower().strip().replace("-", "_").replace(" ", "_")

        norm = normalize_name(raw_clean)

        if norm is None:
            print(f"Removed: {raw}")
            continue

        # 🔥 luôn lưu mapping RAW → FINAL (sau merge + normalize)
        normalized_cache[raw_clean] = norm

        # 🔥 chỉ add class cuối cùng
        final_classes.add(norm)

    # ===== DEBUG QUAN TRỌNG =====
    print("\n🔍 Sample mappings:")
    for k in list(normalized_cache.keys())[:20]:
        print(f"{k} → {normalized_cache[k]}")

    # ===== SAVE YAML =====
    with open(MAPPING_YAML, "w", encoding="utf-8") as f:
        yaml.dump(normalized_cache, f, allow_unicode=True, sort_keys=True)

    print(f"\n✅ Mapping saved: {len(normalized_cache)} entries")
    print(f"✅ Final classes: {len(final_classes)}")

    return final_classes
   
# =========================
# COLLECT FROM HF
# =========================
def collect_hf_classes():
    class_set = set()

    for hf in HF_DATASETS:
        print(f"Scanning {hf}")
        ds = load_dataset(hf, split="train")

        label_feature = ds.features.get("label")

        for sample in ds:
            if "label" in sample and label_feature:
                raw = label_feature.names[sample["label"]]
            elif "ingredient" in sample:
                raw = sample["ingredient"]
            else:
                continue

            if raw:
                raw = raw.lower().strip().replace("-", "_").replace(" ", "_")
                class_set.add(raw)

    return class_set


# =========================
# COLLECT FROM EXTERNAL
# =========================
def collect_external_classes():
    class_set = set()

    if not os.path.exists(EXTERNAL_PATH):
        print("No external_dataset found.")
        return class_set

    for folder in os.listdir(EXTERNAL_PATH):
        full_path = os.path.join(EXTERNAL_PATH, folder)
        if os.path.isdir(full_path):
            raw = folder.lower().strip().replace("-", "_").replace(" ", "_")
            class_set.add(raw)

    return class_set


# =========================
# BUILD YAML
# =========================
def build_yaml(classes):
    classes = sorted(classes)

    yaml_dict = {
        "path": DATASET_ROOT,
        "train": TRAIN_PATH,
        "val": VAL_PATH,
        "test": TEST_PATH,
        "nc": len(classes),
        "names": {i: name for i, name in enumerate(classes)}
    }

    with open(OUTPUT_YAML, "w", encoding="utf-8") as f:
        yaml.dump(yaml_dict, f, allow_unicode=True, sort_keys=False)

    print("\n✅ DONE")
    print(f"Saved to: {OUTPUT_YAML}")
    print(f"Total classes: {len(classes)}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    hf_classes = collect_hf_classes()
    external_classes = collect_external_classes()

    all_classes = hf_classes.union(external_classes)

    print(f"\nHF classes: {len(hf_classes)}")
    print(f"External classes: {len(external_classes)}")
    print(f"Total unique classes: {len(all_classes)}")

    final_classes = build_mapping(all_classes)
    build_yaml(final_classes)