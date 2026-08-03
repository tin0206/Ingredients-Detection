import math
import os
import re
from urllib.parse import quote_plus

import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_paste_button import paste_image_button

# =========================================================
# CONFIG
# =========================================================

CSV_PATH = "final_recipes.csv"
IMG_DIR = "final_recipe_imgs"
MAPPING_PATH = "recipe_image_mapping.csv"
LOG_PATH = "recipe_image_download.log"

PAGE_SIZE = 100

SUPPORTED_EXISTING_EXTENSIONS = [
    "jpg",
    "jpeg",
    "png",
]

os.makedirs(IMG_DIR, exist_ok=True)

st.set_page_config(
    page_title="Recipe Image Review Tool",
    layout="wide",
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def slugify(text):
    """
    Chuyển tên recipe thành slug, giống hệt cách các script
    find_recipe_img trước đó đặt tên file ảnh.
    """
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text)

    slug = text.strip("-")

    return slug if slug else "untitled-recipe"


@st.cache_data
def load_recipes():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Không tìm thấy file CSV: {CSV_PATH}"
        )

    dataframe = pd.read_csv(CSV_PATH, usecols=["id", "title"])

    return dataframe


def load_mapping():
    columns = [
        "id",
        "slug",
        "real_name",
        "image_path",
        "image_url",
        "image_format",
    ]

    if not os.path.exists(MAPPING_PATH):
        return pd.DataFrame(columns=columns)

    try:
        mapping = pd.read_csv(MAPPING_PATH)

        for column in columns:
            if column not in mapping.columns:
                mapping[column] = ""

        return mapping[columns]

    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)


def save_mapping(recipe_id, slug, real_name, image_path, image_url, image_format):
    """
    Thêm hoặc cập nhật mapping của một recipe.
    """
    mapping = load_mapping()

    recipe_id_string = str(recipe_id)

    if not mapping.empty:
        mapping = mapping[mapping["id"].astype(str) != recipe_id_string]

    new_row = pd.DataFrame(
        [
            {
                "id": recipe_id,
                "slug": slug,
                "real_name": real_name,
                "image_path": image_path,
                "image_url": image_url,
                "image_format": image_format,
            }
        ]
    )

    mapping = pd.concat([mapping, new_row], ignore_index=True)

    mapping.to_csv(MAPPING_PATH, index=False, encoding="utf-8-sig")


def load_log_status():
    """
    Đọc recipe_image_download.log, trả về dict id -> (status, detail)
    để biết ngay recipe nào đã lỗi khi tải ảnh tự động.
    """
    status_by_id = {}

    if not os.path.exists(LOG_PATH):
        return status_by_id

    with open(LOG_PATH, "r", encoding="utf-8") as log_file:
        for line in log_file:
            line = line.strip()

            if not line:
                continue

            parts = line.split(" | ", 3)

            if len(parts) < 2:
                continue

            status = parts[0].strip()
            recipe_id = parts[1].strip()
            detail = parts[3].strip() if len(parts) >= 4 else ""

            status_by_id[recipe_id] = (status, detail)

    return status_by_id


def update_log_entry(recipe_id, title, status, detail):
    """
    Cập nhật (hoặc thêm mới) dòng log của một recipe sau khi
    người dùng paste/save ảnh thủ công.
    """
    new_line = f"{status} | {recipe_id} | {title} | {detail}"

    lines = []

    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8") as log_file:
            lines = log_file.read().splitlines()

    recipe_id_string = str(recipe_id)
    replaced = False

    for index, line in enumerate(lines):
        parts = line.split(" | ", 3)

        if len(parts) >= 2 and parts[1].strip() == recipe_id_string:
            lines[index] = new_line
            replaced = True
            break

    if not replaced:
        lines.append(new_line)

    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        log_file.write("\n".join(lines) + "\n")


def get_existing_image_path(slug):
    """
    Kiểm tra chính xác file ảnh của recipe đã tồn tại hay chưa.
    """
    for extension in SUPPORTED_EXISTING_EXTENSIONS:
        image_path = os.path.join(IMG_DIR, f"{slug}.{extension}")

        if os.path.isfile(image_path):
            return image_path

    return None


def delete_old_recipe_images(slug, keep_path=None):
    """
    Xóa các file ảnh cũ có cùng slug nhưng khác định dạng,
    để tránh còn sót lại ảnh cũ (vd: đổi từ .jpg sang .png).
    """
    for extension in SUPPORTED_EXISTING_EXTENSIONS:
        old_path = os.path.join(IMG_DIR, f"{slug}.{extension}")

        if os.path.isfile(old_path) and old_path != keep_path:
            try:
                os.remove(old_path)
            except OSError:
                pass


def determine_output_format(image):
    """
    Ảnh paste từ clipboard: PNG nếu có alpha, còn lại lưu JPG.
    """
    pil_format = str(image.format or "").upper()

    if (
        pil_format == "PNG"
        or image.mode in ("RGBA", "LA")
        or "transparency" in image.info
    ):
        return "png"

    return "jpg"


def save_recipe_image(image, slug, output_format):
    """
    Lưu ảnh đã paste dưới dạng PNG hoặc JPG, xóa ảnh cũ khác định dạng.
    """
    output_format = output_format.lower()

    if output_format == "png":
        image_path = os.path.join(IMG_DIR, f"{slug}.png")

        if image.mode not in ("RGB", "RGBA", "L", "LA"):
            image = image.convert("RGBA")

        image.save(image_path, format="PNG", optimize=True)

    else:
        image_path = os.path.join(IMG_DIR, f"{slug}.jpg")

        if image.mode in ("RGBA", "LA"):
            background = Image.new("RGB", image.size, (255, 255, 255))

            if image.mode == "RGBA":
                background.paste(image, mask=image.getchannel("A"))
            else:
                rgba_image = image.convert("RGBA")
                background.paste(rgba_image, mask=rgba_image.getchannel("A"))

            image = background
        else:
            image = image.convert("RGB")

        image.save(image_path, format="JPEG", quality=90, optimize=True)

        output_format = "jpg"

    delete_old_recipe_images(slug, keep_path=image_path)

    return image_path, output_format


def google_image_search_url(title):
    query = quote_plus(f"{title} recipe")

    return f"https://www.google.com/search?tbm=isch&q={query}"


@st.cache_data
def build_recipe_table(recipes_df, mapping_mtime, log_mtime):
    """
    Ghép recipes + mapping + log thành 1 bảng, xếp các recipe
    chưa có ảnh / bị lỗi tải lên đầu danh sách.

    mapping_mtime và log_mtime chỉ dùng để invalidate cache khi
    2 file này thay đổi (mtime đổi -> cache tính lại). Không được
    thêm dấu gạch dưới vào tên tham số vì st.cache_data bỏ qua
    hashing (và bỏ qua luôn việc invalidate cache) cho các tham
    số bắt đầu bằng "_".
    """
    mapping_df = load_mapping()
    log_status = load_log_status()

    mapping_by_id = {
        str(row["id"]): row["slug"]
        for _, row in mapping_df.iterrows()
    }

    rows = []

    for _, recipe in recipes_df.iterrows():
        recipe_id = recipe["id"]
        title = str(recipe["title"]).strip()

        slug = mapping_by_id.get(str(recipe_id)) or slugify(title)

        existing_path = get_existing_image_path(slug)
        has_image = existing_path is not None

        status, detail = log_status.get(str(recipe_id), ("", ""))

        priority = 0 if (not has_image or status == "FAILED") else 1

        rows.append(
            {
                "id": recipe_id,
                "title": title,
                "slug": slug,
                "has_image": has_image,
                "image_path": existing_path,
                "log_status": status,
                "log_detail": detail,
                "priority": priority,
            }
        )

    table = pd.DataFrame(rows)

    table = table.sort_values(
        by="priority",
        kind="stable",
    ).reset_index(drop=True)

    return table


# =========================================================
# STREAMLIT UI
# =========================================================

st.title("Recipe Image Review Tool")

st.write(
    "Duyệt qua từng recipe, search ảnh trên Google, paste ảnh từ "
    "clipboard rồi lưu lại để thay thế ảnh hiện tại. Các recipe "
    "chưa có ảnh hoặc tải lỗi (theo `recipe_image_download.log`) "
    "sẽ được xếp lên đầu danh sách."
)

try:
    recipes_df = load_recipes()
except Exception as error:
    st.error(str(error))
    st.stop()

mapping_mtime = os.path.getmtime(MAPPING_PATH) if os.path.exists(MAPPING_PATH) else 0
log_mtime = os.path.getmtime(LOG_PATH) if os.path.exists(LOG_PATH) else 0

if "v3_refresh_token" not in st.session_state:
    st.session_state.v3_refresh_token = 0

recipe_table = build_recipe_table(
    recipes_df,
    mapping_mtime + st.session_state.v3_refresh_token,
    log_mtime,
)

total_count = len(recipe_table)
missing_count = int((~recipe_table["has_image"]).sum())
failed_count = int((recipe_table["log_status"] == "FAILED").sum())

summary_col1, summary_col2, summary_col3 = st.columns(3)
summary_col1.metric("Tổng recipe", total_count)
summary_col2.metric("Chưa có ảnh", missing_count)
summary_col3.metric("Tải lỗi (log)", failed_count)

st.divider()

total_pages = max(1, math.ceil(total_count / PAGE_SIZE))

if "v3_page" not in st.session_state:
    st.session_state.v3_page = 0

st.session_state.v3_page = min(st.session_state.v3_page, total_pages - 1)
st.session_state.v3_page = max(st.session_state.v3_page, 0)


def go_previous_page():
    st.session_state.v3_page = max(0, st.session_state.v3_page - 1)
    st.session_state.v3_page_input = st.session_state.v3_page + 1


def go_next_page():
    st.session_state.v3_page = min(total_pages - 1, st.session_state.v3_page + 1)
    st.session_state.v3_page_input = st.session_state.v3_page + 1


def go_to_page():
    st.session_state.v3_page = st.session_state.v3_page_input - 1


nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])

nav_col1.button(
    "⬅ Trang trước",
    on_click=go_previous_page,
    disabled=st.session_state.v3_page <= 0,
    use_container_width=True,
)

with nav_col2:
    st.number_input(
        f"Trang (1 - {total_pages})",
        min_value=1,
        max_value=total_pages,
        value=st.session_state.v3_page + 1,
        step=1,
        key="v3_page_input",
        on_change=go_to_page,
        label_visibility="collapsed",
    )
    st.caption(
        f"Trang {st.session_state.v3_page + 1} / {total_pages}"
    )

nav_col3.button(
    "Trang sau ➡",
    on_click=go_next_page,
    disabled=st.session_state.v3_page >= total_pages - 1,
    use_container_width=True,
)

st.divider()

if "pending_images" not in st.session_state:
    st.session_state.pending_images = {}

if "paste_nonce" not in st.session_state:
    st.session_state.paste_nonce = {}

start_index = st.session_state.v3_page * PAGE_SIZE
end_index = start_index + PAGE_SIZE
page_rows = recipe_table.iloc[start_index:end_index]

for _, row in page_rows.iterrows():
    recipe_id = row["id"]
    title = row["title"]
    slug = row["slug"]

    with st.container(border=True):
        image_col, info_col = st.columns([1, 3])

        with image_col:
            if row["has_image"]:
                st.image(row["image_path"], width=180)
            else:
                st.container(height=140).caption("Không có ảnh")

        with info_col:
            st.markdown(f"**{title}**")
            st.caption(f"ID: {recipe_id} | slug: `{slug}`")

            if row["has_image"]:
                st.success("Đã có ảnh", icon="✅")
            else:
                st.error("Chưa có ảnh", icon="❌")

            if row["log_status"] == "FAILED":
                st.caption(f"Log lỗi: {row['log_detail']}")

            action_col1, action_col2 = st.columns(2)

            with action_col1:
                st.link_button(
                    "🔍 Search Google",
                    google_image_search_url(title),
                    use_container_width=True,
                )

            nonce = st.session_state.paste_nonce.get(recipe_id, 0)

            with action_col2:
                paste_result = paste_image_button(
                    "📋 Paste ảnh",
                    key=f"paste_{recipe_id}_{nonce}",
                )

            if paste_result.image_data is not None:
                st.session_state.pending_images[recipe_id] = paste_result.image_data

            pending_image = st.session_state.pending_images.get(recipe_id)

            if pending_image is not None:
                preview_col, save_col, discard_col = st.columns([2, 1, 1])

                with preview_col:
                    st.image(pending_image, width=120, caption="Ảnh vừa paste")

                with save_col:
                    if st.button("💾 Lưu", key=f"save_{recipe_id}"):
                        output_format = determine_output_format(pending_image)

                        image_path, saved_format = save_recipe_image(
                            pending_image,
                            slug,
                            output_format,
                        )

                        save_mapping(
                            recipe_id=recipe_id,
                            slug=slug,
                            real_name=title,
                            image_path=image_path,
                            image_url="",
                            image_format=saved_format,
                        )

                        update_log_entry(
                            recipe_id=recipe_id,
                            title=title,
                            status="OK",
                            detail=f"Paste ảnh thủ công | {image_path}",
                        )

                        del st.session_state.pending_images[recipe_id]
                        st.session_state.paste_nonce[recipe_id] = nonce + 1
                        st.session_state.v3_refresh_token += 1

                        st.success(f"Đã lưu ảnh cho {title}")
                        st.rerun()

                with discard_col:
                    if st.button("✖ Hủy", key=f"discard_{recipe_id}"):
                        del st.session_state.pending_images[recipe_id]
                        st.session_state.paste_nonce[recipe_id] = nonce + 1
                        st.rerun()

st.divider()

bottom_nav_col1, bottom_nav_col2 = st.columns(2)

bottom_nav_col1.button(
    "⬅ Trang trước ",
    on_click=go_previous_page,
    disabled=st.session_state.v3_page <= 0,
    use_container_width=True,
    key="bottom_prev",
)

bottom_nav_col2.button(
    "Trang sau ➡ ",
    on_click=go_next_page,
    disabled=st.session_state.v3_page >= total_pages - 1,
    use_container_width=True,
    key="bottom_next",
)
