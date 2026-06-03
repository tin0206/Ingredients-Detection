import streamlit as st
from pathlib import Path
from PIL import Image
import os

# ================= CONFIG =================
PROCESSED_DIR = Path("processed_ingredients")

st.set_page_config(layout="wide")

# ================= LOAD CLASSES =================
def get_classes():
    return sorted([p for p in PROCESSED_DIR.iterdir() if p.is_dir()], key=lambda x: int(x.name))

classes = get_classes()

if not classes:
    st.error("❌ processed_ingredients empty!")
    st.stop()

# ================= SESSION STATE =================
if "class_idx" not in st.session_state:
    st.session_state.class_idx = 0

if "selected" not in st.session_state:
    st.session_state.selected = set()

# ================= NAVIGATION =================
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    if st.button("⬅️ Prev"):
        st.session_state.class_idx = max(0, st.session_state.class_idx - 1)
        st.session_state.selected = set()

with col3:
    if st.button("Next ➡️"):
        st.session_state.class_idx = min(len(classes)-1, st.session_state.class_idx + 1)
        st.session_state.selected = set()

# ================= CURRENT CLASS =================
current_class = classes[st.session_state.class_idx]
st.title(f"Class: {current_class.name} ({st.session_state.class_idx+1}/{len(classes)})")

images = list(current_class.glob("*.png"))

if not images:
    st.warning("⚠ No images in this class")
    st.stop()

# ================= IMAGE GRID =================
cols = st.columns(6)

for i, img_path in enumerate(images):
    col = cols[i % 6]

    with col:
        img = Image.open(img_path)
        st.image(img, use_container_width=True)

        checked = st.checkbox(
            f"Delete {i}",
            key=str(img_path)
        )

        if checked:
            st.session_state.selected.add(img_path)
        else:
            st.session_state.selected.discard(img_path)

# ================= DELETE BUTTON =================
st.divider()

st.write(f"🗑 Selected: {len(st.session_state.selected)} images")

if st.button("🔥 DELETE SELECTED"):
    for img_path in list(st.session_state.selected):
        try:
            os.remove(img_path)
        except Exception as e:
            st.error(f"Error deleting {img_path}: {e}")

    st.session_state.selected = set()
    st.rerun()
