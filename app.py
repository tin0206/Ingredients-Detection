import streamlit as st
import yaml
import urllib.parse
import subprocess
import os
from pathlib import Path
from PIL import Image
from icrawler.builtin import BingImageCrawler
import imagehash

# ================= CONFIG =================
YAML_FILE = "data_test_v2.yaml"
RAW_ROOT = Path("raw_ingredients_test_v2")

RAW_ROOT.mkdir(parents=True, exist_ok=True)

# ================= LOAD YAML =================
@st.cache_data
def load_classes():
    with open(YAML_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return {int(k): v for k, v in data["names"].items()}

# ================= CATEGORY GROUPS =================
FRUITS = {
    "apple", "apricot", "avocado", "banana", "berry", "cherry", "durian",
    "kiwi", "lemon", "lime", "mango", "orange", "papaya", "passion_fruit",
    "peach", "pear", "pineapple", "plum", "pomelo", "strawberry", "tomato"
}

VEGETABLES = {
    "artichoke", "arugula", "asparagus", "bamboo_shoot", "beet", "bok_choy",
    "broccoli", "brussels_sprout", "cabbage", "carrot", "cauliflower",
    "celery", "chili_pepper", "coriander", "corn", "cucumber", "fennel",
    "garlic", "ginger", "green_bean", "green_onion", "horseradish",
    "kale", "kohlrabi", "lemongrass", "lettuce", "mushroom", "onion",
    "potato", "pumpkin", "radicchio", "radish", "spinach",
    "sweet_potato", "turnip"
}

MEATS = {
    "bacon", "beef", "chicken", "duck", "goat", "goose", "ham",
    "lamb", "pork", "rabbit", "sausage", "turkey"
}

SEAFOOD = {
    "clam", "crab", "fish", "lobster", "octopus", "oyster",
    "salmon", "scallop", "shrimp", "tuna"
}

PANTRY = {
    "bean", "bread", "butter", "caper", "cheese", "coffee",
    "dumpling_wrapper", "egg", "fish_sauce", "glass_noodle", "milk",
    "oil", "olives", "pasta", "pasta_sauce", "peas", "pesto",
    "pickle", "powder", "rice", "whole_spice"
}

# ================= FOLDER =================
def class_folder(class_id: int):
    folder = RAW_ROOT / str(class_id)
    folder.mkdir(parents=True, exist_ok=True)
    return folder

def list_images(folder: Path):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"]
    files = []
    for ext in exts:
        files.extend(folder.glob(ext))
    return sorted(files)

def count_images(folder: Path):
    return len(list_images(folder))

def create_all_class_folders(classes):
    for class_id in classes:
        class_folder(class_id)

# ================= QUERY =================
def build_query(class_name: str):
    return build_queries(class_name)[0]

def build_queries(class_name: str):
    name = class_name.replace("_", " ")

    special = {
        "apple": [
            "apple fruit",
            "fresh apple",
            "whole apple fruit",
            "apple close up",
            "apple on table",
            "apple grocery",
            "apple food",
            "ripe apple"
        ],
         "apricot": [
            "apricot fruit",
            "fresh apricot fruit",
            "whole apricot fruit",
            "apricot fruit close up",
            "apricot fruit white background"
        ],
        "orange": [
            "orange fruit",
            "fresh orange",
            "whole orange fruit",
            "orange close up",
            "orange grocery",
            "orange food"
        ],
        "tomato": [
            "tomato fruit",
            "fresh tomato",
            "whole tomato",
            "tomato close up",
            "tomato vegetable",
            "tomato food"
        ],
        "turkey": [
            "turkey meat",
            "raw turkey",
            "turkey breast",
            "fresh turkey meat",
            "turkey food ingredient"
        ],
        "fish": [
            "fish fillet",
            "raw fish fillet",
            "fresh fish fillet",
            "fish meat",
            "fish food ingredient"
        ],
        "salmon": [
            "salmon fillet",
            "raw salmon",
            "fresh salmon",
            "salmon food ingredient"
        ],
        "tuna": [
            "tuna fillet",
            "raw tuna",
            "fresh tuna",
            "tuna food ingredient"
        ],
        "powder": [
            "food powder",
            "flour powder",
            "white food powder",
            "powder ingredient"
        ],
        "whole_spice": [
            "whole spices",
            "mixed whole spices",
            "cardamom clove cinnamon",
            "whole spice ingredient"
        ],
        "bean": [
            "beans",
            "dry beans",
            "mixed beans",
            "bean ingredient"
        ],
        "rice": [
            "rice grain",
            "uncooked rice",
            "white rice grain",
            "rice ingredient"
        ],
        "coffee": [
            "coffee beans",
            "roasted coffee beans",
            "coffee ingredient"
        ],
        "oil": [
            "cooking oil",
            "oil ingredient",
            "vegetable oil"
        ],
        "fish_sauce": [
            "fish sauce",
            "fish sauce bowl",
            "nước mắm"
        ],
        "pasta_sauce": [
            "pasta sauce",
            "tomato pasta sauce",
            "pasta sauce bowl"
        ],
    }

    if class_name in special:
        return special[class_name]

    if class_name in FRUITS:
        return [
            f"{name} fruit",
            f"fresh {name}",
            f"whole {name}",
            f"{name} close up",
            f"{name} on table",
            f"{name} grocery",
            f"{name} food",
            f"ripe {name}"
        ]

    if class_name in VEGETABLES:
        return [
            f"{name} vegetable",
            f"fresh {name}",
            f"whole {name}",
            f"{name} close up",
            f"{name} food",
            f"{name} ingredient",
            f"{name} grocery",
            f"raw {name}"
        ]

    if class_name in MEATS:
        return [
            f"{name} meat",
            f"raw {name}",
            f"fresh {name} meat",
            f"{name} food ingredient",
            f"{name} close up"
        ]

    if class_name in SEAFOOD:
        return [
            f"{name} seafood",
            f"raw {name}",
            f"fresh {name}",
            f"{name} food ingredient",
            f"{name} close up"
        ]

    if class_name in PANTRY:
        return [
            f"{name} ingredient",
            f"{name} food",
            f"{name} close up",
            f"{name} grocery"
        ]

    return [name]

def google_images_url(query: str):
    encoded = urllib.parse.quote_plus(query)
    return f"https://www.google.com/search?tbm=isch&q={encoded}"

# ================= OPEN FOLDER =================
def open_folder(folder: Path):
    folder = folder.resolve()

    if os.name == "nt":
        subprocess.Popen(f'explorer "{folder}"')

# ================= IMAGE HASH =================
def image_hash(img_path: Path):
    try:
        img = Image.open(img_path).convert("RGB")
        return str(imagehash.phash(img))
    except Exception:
        return None

def remove_duplicate_images(folder: Path):
    seen = {}
    removed = 0

    for img_path in list_images(folder):
        h = image_hash(img_path)

        if h is None:
            img_path.unlink(missing_ok=True)
            removed += 1
            continue

        if h in seen:
            img_path.unlink(missing_ok=True)
            removed += 1
        else:
            seen[h] = img_path

    return removed

# ================= DOWNLOAD =================
def download_images_until_target(class_id: int, class_name: str, target_count: int):
    folder = class_folder(class_id)

    if count_images(folder) >= target_count:
        return 0, 0

    queries = build_queries(class_name)

    before = count_images(folder)
    total_removed = 0

    attempts = 0
    max_attempts = 10

    while count_images(folder) < target_count and attempts < max_attempts:
        need = target_count - count_images(folder)
        per_query = max(15, need // len(queries) + 15)

        for query in queries:
            if count_images(folder) >= target_count:
                break

            crawler = BingImageCrawler(
                storage={"root_dir": str(folder)}
            )

            crawler.crawl(
                keyword=query,
                max_num=per_query,
                file_idx_offset="auto"
            )

            total_removed += remove_duplicate_images(folder)

        attempts += 1

    after = count_images(folder)
    total_removed += remove_duplicate_images(folder)

    return after - before, total_removed

# ================= UI =================
st.set_page_config(
    page_title="Ingredient Image Collector",
    layout="wide"
)

st.title("Ingredient Image Collector")

classes = load_classes()
create_all_class_folders(classes)

selected_id = st.sidebar.selectbox(
    "Select class",
    list(classes.keys()),
    format_func=lambda x: f"{x}: {classes[x]}"
)

class_name = classes[selected_id]
folder = class_folder(selected_id)

images = list_images(folder)
image_count = len(images)

query_list = build_queries(class_name)
default_query = query_list[0]

st.header(f"{selected_id}: {class_name}")

st.write("Current folder:")
st.code(str(folder))

st.metric("Current images", image_count)

TARGET_PER_CLASS = st.number_input(
    "Target images per class",
    min_value=10,
    max_value=300,
    value=100,
    step=10
)

query = st.text_input("Google search query", value=default_query)
google_url = google_images_url(query)

with st.expander("Auto-download query list"):
    for q in query_list:
        st.code(q)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <a href="{google_url}" target="_blank">
            <button style="
                background-color:#4CAF50;
                color:white;
                padding:12px 24px;
                border:none;
                border-radius:8px;
                font-size:16px;
                cursor:pointer;">
                Open Google Images
            </button>
        </a>
        """,
        unsafe_allow_html=True
    )

with col2:
    if st.button("Open Current Folder"):
        open_folder(folder)

with col3:
    if st.button("Refresh"):
        st.rerun()

st.write("Google Images URL:")
st.code(google_url)

# ================= DOWNLOAD =================
st.divider()
st.subheader("Auto Download with Bing")

if st.button("Download Until Target"):
    with st.spinner(f"Downloading until {TARGET_PER_CLASS} images for {class_name}..."):
        added, removed = download_images_until_target(
            selected_id,
            class_name,
            int(TARGET_PER_CLASS)
        )

    st.success(f"Added: {added}, removed duplicates/invalid: {removed}")
    st.rerun()

if st.button("Remove Duplicates Only"):
    removed = remove_duplicate_images(folder)
    st.success(f"Removed {removed} duplicate/invalid images.")
    st.rerun()

# ================= PROGRESS =================
st.divider()
st.subheader("Class Progress")

image_count = count_images(folder)
progress = min(image_count / TARGET_PER_CLASS, 1.0)
st.progress(progress)

if image_count >= TARGET_PER_CLASS:
    st.success(f"{class_name} has enough images.")
else:
    st.warning(f"Need {TARGET_PER_CLASS - image_count} more images.")

# ================= PREVIEW =================
st.divider()
st.subheader("Preview images")

images = list_images(folder)

if not images:
    st.warning("No images yet.")
else:
    cols = st.columns(5)

    for i, img_path in enumerate(images):
        with cols[i % 5]:
            try:
                img = Image.open(img_path)
                st.image(img, width="stretch")
                st.caption(img_path.name)

                if st.button("Delete", key=f"delete_{img_path}"):
                    img_path.unlink(missing_ok=True)
                    st.rerun()

            except Exception:
                st.error("Invalid image")
                if st.button("Delete invalid", key=f"bad_{img_path}"):
                    img_path.unlink(missing_ok=True)
                    st.rerun()

# ================= OVERALL PROGRESS =================
st.divider()
st.subheader("Overall Progress")

summary = []
for cid, cname in classes.items():
    f = class_folder(cid)
    n = count_images(f)
    summary.append((cid, cname, n))

done = sum(1 for _, _, n in summary if n >= TARGET_PER_CLASS)
total = len(summary)

st.write(f"Classes completed: **{done}/{total}**")

with st.expander("View all class counts"):
    for cid, cname, n in summary:
        status = "✅" if n >= TARGET_PER_CLASS else "❌"
        st.write(f"{status} {cid}: {cname} — {n} images")