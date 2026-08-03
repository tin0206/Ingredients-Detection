import streamlit as st
import pandas as pd
import os
import re
from PIL import Image
from urllib.parse import quote_plus
from streamlit_paste_button import paste_image_button

CSV_PATH = "final_recipes.csv"
IMG_DIR = "final_recipe_imgs"
MAPPING_PATH = "recipe_image_mapping.csv"

os.makedirs(IMG_DIR, exist_ok=True)


def slugify(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


@st.cache_data
def load_data():
    return pd.read_csv(CSV_PATH)


def load_mapping():
    if os.path.exists(MAPPING_PATH):
        return pd.read_csv(MAPPING_PATH)

    return pd.DataFrame(columns=["id", "slug", "real_name", "image_path"])


def save_mapping(recipe_id, slug, real_name, image_path):
    mapping = load_mapping()

    new_row = pd.DataFrame([{
        "id": recipe_id,
        "slug": slug,
        "real_name": real_name,
        "image_path": image_path
    }])

    mapping = mapping[mapping["id"] != recipe_id]
    mapping = pd.concat([mapping, new_row], ignore_index=True)
    mapping.to_csv(MAPPING_PATH, index=False)


df = load_data()

st.title("Recipe Image Uploader")

if "recipe_idx" not in st.session_state:
    st.session_state.recipe_idx = 0

search = st.text_input("Search ingredient / recipe name")

filtered_df = df.copy()

if search:
    search_lower = search.lower()
    filtered_df = df[
        df["title"].astype(str).str.lower().str.contains(search_lower, na=False)
        | df["ingredients"].astype(str).str.lower().str.contains(search_lower, na=False)
        | df["ner"].astype(str).str.lower().str.contains(search_lower, na=False)
        | df["mapped_ingredients"].astype(str).str.lower().str.contains(search_lower, na=False)
    ]

st.write(f"Found {len(filtered_df)} recipes")

if len(filtered_df) == 0:
    st.warning("No recipe found.")
    st.stop()

if st.session_state.recipe_idx >= len(filtered_df):
    st.session_state.recipe_idx = 0

col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    if st.button("⬅ Previous"):
        st.session_state.recipe_idx = max(
            0,
            st.session_state.recipe_idx - 1
        )

with col2:
    st.write(
        f"Recipe {st.session_state.recipe_idx + 1}/{len(filtered_df)}"
    )

with col3:
    if st.button("Next ➡"):
        st.session_state.recipe_idx = min(
            len(filtered_df) - 1,
            st.session_state.recipe_idx + 1
        )

selected = st.selectbox(
    "Jump to recipe",
    range(len(filtered_df)),
    format_func=lambda i:
        f"{filtered_df.iloc[i]['id']} - {filtered_df.iloc[i]['title']}",
    index=st.session_state.recipe_idx
)

st.session_state.recipe_idx = selected

recipe = filtered_df.iloc[st.session_state.recipe_idx]

st.divider()

st.subheader(recipe["title"])

st.write("ID:", recipe["id"])
st.write("Estimated servings:", recipe.get("estimated_servings", ""))

with st.expander("Ingredients"):
    st.write(recipe["ingredients"])

with st.expander("Directions"):
    st.write(recipe["directions"])

with st.expander("NER"):
    st.write(recipe["ner"])

with st.expander("Mapped ingredients"):
    st.write(recipe["mapped_ingredients"])

with st.expander("Dietary restrictions"):
    st.write(recipe["dietary_restrictions"])

slug = slugify(recipe["title"])
st.write("Slug:", slug)

google_query = quote_plus(f"{recipe['title']}")
google_image_url = f"https://www.google.com/search?tbm=isch&q={google_query}"

st.link_button("🔍 Search image on Google", google_image_url)

existing_imgs = [
    f for f in os.listdir(IMG_DIR)
    if f.startswith(slug + ".")
]

if existing_imgs:
    existing_path = os.path.join(IMG_DIR, existing_imgs[0])
    st.subheader("Current saved image")
    st.image(existing_path, use_container_width=True)

st.subheader("Add new image")

uploaded_file = st.file_uploader(
    "Upload image file",
    type=["jpg", "jpeg", "png", "webp"],
    key=f"upload_{recipe['id']}"
)

st.write("Or paste image from clipboard:")

paste_result = paste_image_button(
    label="📋 Paste image",
    key=f"paste_{recipe['id']}"
)

image = None
ext = None

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    ext = uploaded_file.name.split(".")[-1].lower()

elif paste_result.image_data is not None:
    image = paste_result.image_data.convert("RGB")
    ext = "png"

if image is not None:
    st.subheader("Preview image")
    st.image(image, use_container_width=True)

    image_name = f"{slug}.{ext}"
    image_path = os.path.join(IMG_DIR, image_name)

    if st.button("Save image", key=f"save_{recipe['id']}"):
        image.save(image_path)

        save_mapping(
            recipe_id=recipe["id"],
            slug=slug,
            real_name=recipe["title"],
            image_path=image_path
        )

        st.success(f"Saved image to {image_path}")
        st.success(f"Updated mapping file: {MAPPING_PATH}")