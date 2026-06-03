import yaml
import shutil
from pathlib import Path

# ================= CONFIG =================
CLEAN_DIR = Path("clean_output")
PROCESSED_DIR = Path("processed_ingredients")

DATA12_YAML = "data12.yaml"
DATA_YAML = "data.yaml"

# ================= LOAD YAML =================
def load_yaml_names(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {int(k): v for k, v in data["names"].items()}

data12_names = load_yaml_names(DATA12_YAML)
data_names = load_yaml_names(DATA_YAML)

# reverse map
data_name_to_id = {v: k for k, v in data_names.items()}

# ================= IMPORT NORMALIZE =================
# copy nguyên hàm normalize_name của bạn vào đây
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
        "spelt", "farro", "emmer", "einkorn", "corn_grit", "cracked_wheat", "freekeh", 
        "polenta", "wheat_bran", "barley", "oat", "millet", "sorghum", "teff", 
        "spelt", "quinoa", "einkorn", "emmer", "farro", "kamut"
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

# ================= COPY =================
def main():
    count = 0
    skipped = 0

    for class_folder in CLEAN_DIR.iterdir():
        if not class_folder.is_dir():
            continue

        try:
            old_id = int(class_folder.name)
        except:
            continue

        # ===== old class name =====
        if old_id not in data12_names:
            print(f"❌ Unknown id {old_id}")
            continue

        raw_name = data12_names[old_id]

        # ===== normalize =====
        new_name = normalize_name(raw_name)

        if not new_name:
            skipped += 1
            continue

        # ===== check tồn tại trong data.yaml =====
        if new_name not in data_name_to_id:
            print(f"⛔ Skip {raw_name} → {new_name} (not in data.yaml)")
            skipped += 1
            continue

        new_id = data_name_to_id[new_name]

        target_dir = PROCESSED_DIR / str(new_id)
        target_dir.mkdir(parents=True, exist_ok=True)

        # ===== copy ảnh =====
        for img_path in class_folder.glob("*.*"):
            new_path = target_dir / img_path.name

            # tránh overwrite
            if new_path.exists():
                new_path = target_dir / f"{img_path.stem}_{count}{img_path.suffix}"

            shutil.copy(img_path, new_path)
            count += 1

    print("\n✅ DONE")
    print(f"Copied: {count}")
    print(f"Skipped: {skipped}")

if __name__ == "__main__":
    main()