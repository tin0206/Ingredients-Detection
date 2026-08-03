# import streamlit as st
# from pathlib import Path
# from PIL import Image
# import subprocess
# import os

# BACKGROUND_ROOT = Path("backgrounds")

# TARGET = {
#     "wood_cutting_board": 100,
#     "bamboo_cutting_board": 80,
#     "plastic_cutting_board": 50,
#     "wood_table": 80,
#     "rustic_table": 50,
#     "marble_countertop": 70,
#     "granite_countertop": 50,
#     "kitchen_counter": 80,
#     "kitchen_scene": 80,
#     "cloth_table": 40,
# }

# EXTS = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"]


# def list_images(folder):
#     files = []
#     for ext in EXTS:
#         files.extend(folder.glob(ext))
#     return sorted(files)


# def open_folder(folder):
#     folder = folder.resolve()

#     if os.name == "nt":
#         subprocess.Popen(f'explorer "{folder}"')
#     else:
#         subprocess.Popen(["xdg-open", str(folder)])


# st.set_page_config(layout="wide")

# st.title("Background Dataset Manager")

# total = 0
# target_total = sum(TARGET.values())

# for folder_name, target in TARGET.items():

#     folder = BACKGROUND_ROOT / folder_name
#     folder.mkdir(parents=True, exist_ok=True)

#     imgs = list_images(folder)
#     n = len(imgs)

#     total += n

#     col1, col2, col3 = st.columns([4,2,2])

#     with col1:
#         st.subheader(folder_name)

#         st.progress(min(n/target,1.0))

#         st.write(f"**{n} / {target}**")

#     with col2:

#         if st.button(f"Open {folder_name}"):
#             open_folder(folder)

#     with col3:

#         if imgs:
#             st.image(Image.open(imgs[0]), width="stretch")
#         else:
#             st.warning("Empty")

#     st.divider()

# st.header("Overall")

# st.metric(
#     "Backgrounds",
#     f"{total}/{target_total}"
# )

# st.progress(min(total/target_total,1.0))

# if st.button("Refresh"):
#     st.rerun()

from pathlib import Path

BACKGROUND_ROOT = Path("backgrounds")

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

for folder in BACKGROUND_ROOT.iterdir():

    if not folder.is_dir():
        continue

    files = [f for f in folder.iterdir() if f.suffix.lower() in EXTS]
    files.sort()

    for i, file in enumerate(files, start=1):
        new_name = f"bg_{i:04d}{file.suffix.lower()}"
        new_path = folder / new_name

        if file == new_path:
            continue

        file.rename(new_path)

    print(f"✓ {folder.name}: {len(files)} images renamed")

print("\nDone!")