import io
import os
import random
import re
import time

import pandas as pd
import requests
import streamlit as st
from ddgs import DDGS
from ddgs.exceptions import RatelimitException, TimeoutException
from PIL import Image, UnidentifiedImageError


# =========================================================
# CONFIG
# =========================================================

CSV_PATH = "final_recipes.csv"
IMG_DIR = "final_recipe_imgs"
MAPPING_PATH = "recipe_image_mapping.csv"
LOG_PATH = "recipe_image_download.log"

# Nghỉ giữa các recipe để tránh gửi request quá nhanh.
# Với ~13k recipe và DuckDuckGo search miễn phí (dễ bị rate limit),
# cần nghỉ vài giây kèm jitter để tránh bị chặn giữa chừng.
REQUEST_DELAY_MIN = 3
REQUEST_DELAY_MAX = 6

SEARCH_TIMEOUT = 30
IMAGE_TIMEOUT = 30

# Số lượng kết quả ảnh tối đa lấy về từ DuckDuckGo cho mỗi recipe
MAX_SEARCH_RESULTS = 10

# Khi bị DuckDuckGo rate-limit, thử lại với thời gian nghỉ tăng dần
SEARCH_RATELIMIT_RETRIES = 4
SEARCH_RATELIMIT_BACKOFF = 20

MIN_IMAGE_WIDTH = 100
MIN_IMAGE_HEIGHT = 100

SUPPORTED_EXISTING_EXTENSIONS = [
    "jpg",
    "jpeg",
    "png",
]

os.makedirs(IMG_DIR, exist_ok=True)

st.set_page_config(
    page_title="Automatic Recipe Image Downloader",
    layout="wide",
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def slugify(text):
    """
    Chuyển tên recipe thành slug.

    Ví dụ:
    Crown Roast of Pork -> crown-roast-of-pork
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

    dataframe = pd.read_csv(CSV_PATH)

    required_columns = {
        "id",
        "title",
        "ingredients",
        "directions",
        "ner",
        "estimated_servings",
        "mapped_ingredients",
        "dietary_restrictions",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "CSV đang thiếu các cột: "
            + ", ".join(sorted(missing_columns))
        )

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


def save_mapping(
    recipe_id,
    slug,
    real_name,
    image_path,
    image_url,
    image_format,
):
    """
    Thêm hoặc cập nhật mapping của một recipe.
    """
    mapping = load_mapping()

    recipe_id_string = str(recipe_id)

    if not mapping.empty:
        mapping = mapping[
            mapping["id"].astype(str) != recipe_id_string
        ]

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

    mapping = pd.concat(
        [mapping, new_row],
        ignore_index=True,
    )

    mapping.to_csv(
        MAPPING_PATH,
        index=False,
        encoding="utf-8-sig",
    )


def get_existing_image_path(slug):
    """
    Kiểm tra chính xác file ảnh của recipe đã tồn tại hay chưa.

    Ví dụ:
    chicken.jpg không bị nhầm với chicken-soup.jpg.
    """
    for extension in SUPPORTED_EXISTING_EXTENSIONS:
        image_path = os.path.join(
            IMG_DIR,
            f"{slug}.{extension}",
        )

        if os.path.isfile(image_path):
            return image_path

    return None


def search_image_urls(recipe_title):
    """
    Search ảnh trên DuckDuckGo bằng:
    <recipe title> recipe

    Trả về danh sách URL ảnh, theo đúng thứ tự kết quả trả về.
    """
    recipe_title = str(recipe_title).strip()
    query = f"{recipe_title} recipe"

    image_results = None

    for attempt in range(1, SEARCH_RATELIMIT_RETRIES + 1):
        try:
            image_results = DDGS(timeout=SEARCH_TIMEOUT).images(
                query=query,
                safesearch="moderate",
                max_results=MAX_SEARCH_RESULTS,
            )

            break

        except (RatelimitException, TimeoutException) as error:
            if attempt == SEARCH_RATELIMIT_RETRIES:
                raise RuntimeError(
                    f"Search DuckDuckGo cho query '{query}' thất bại "
                    f"sau {attempt} lần thử: {error}"
                )

            backoff = SEARCH_RATELIMIT_BACKOFF * attempt

            time.sleep(backoff)

        except Exception as error:
            raise RuntimeError(
                f"Lỗi search DuckDuckGo cho query '{query}': {error}"
            ) from error

    if not image_results:
        raise RuntimeError(
            f"Không tìm thấy ảnh cho query: {query}"
        )

    image_urls = []

    for result in image_results:
        image_url = (
            result.get("image")
            or result.get("thumbnail")
        )

        if image_url:
            image_urls.append(image_url)

    if not image_urls:
        raise RuntimeError(
            "Không có kết quả ảnh nào chứa URL."
        )

    return image_urls


def detect_image_format(image, content_type):
    """
    Xác định ảnh sẽ được lưu thành PNG hoặc JPG.

    - PNG gốc -> PNG
    - JPEG/JPG gốc -> JPG
    - Ảnh có alpha -> PNG
    - Các định dạng còn lại -> JPG
    """
    content_type = str(content_type).lower()
    pil_format = str(image.format or "").upper()

    if (
        "png" in content_type
        or pil_format == "PNG"
        or image.mode in ("RGBA", "LA")
        or "transparency" in image.info
    ):
        return "png"

    return "jpg"


def download_image(image_url):
    """
    Tải ảnh và trả về:
    - PIL Image
    - định dạng lưu: png hoặc jpg
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/150.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "image/avif,image/webp,image/apng,"
            "image/svg+xml,image/*,*/*;q=0.8"
        ),
    }

    response = requests.get(
        image_url,
        headers=headers,
        timeout=IMAGE_TIMEOUT,
        allow_redirects=True,
    )

    response.raise_for_status()

    content_type = response.headers.get(
        "Content-Type",
        "",
    ).lower()

    if content_type and "image" not in content_type:
        raise ValueError(
            "URL không trả về ảnh. "
            f"Content-Type: {content_type}"
        )

    try:
        image = Image.open(
            io.BytesIO(response.content)
        )

        image.load()

    except UnidentifiedImageError as error:
        raise ValueError(
            "Dữ liệu tải về không phải ảnh hợp lệ."
        ) from error

    if (
        image.width < MIN_IMAGE_WIDTH
        or image.height < MIN_IMAGE_HEIGHT
    ):
        raise ValueError(
            f"Ảnh quá nhỏ: {image.width}x{image.height}"
        )

    output_format = detect_image_format(
        image,
        content_type,
    )

    return image, output_format


def download_first_working_image(image_urls):
    """
    Thử tải lần lượt từng URL ảnh trong danh sách,
    dừng lại ngay khi tải thành công.

    Trả về: (image, output_format, image_url) của ảnh tải được.
    """
    last_error = None

    for image_url in image_urls:
        try:
            image, output_format = download_image(image_url)

            return image, output_format, image_url

        except Exception as error:
            last_error = error

            continue

    raise RuntimeError(
        f"Không tải được ảnh nào trong {len(image_urls)} "
        f"kết quả. Lỗi cuối: {last_error}"
    )


def delete_old_recipe_images(slug, keep_path=None):
    """
    Xóa các file ảnh cũ có cùng slug nhưng khác định dạng.

    Ví dụ:
    Nếu vừa lưu tiramisu.png thì xóa tiramisu.jpg cũ.
    """
    for extension in SUPPORTED_EXISTING_EXTENSIONS:
        old_path = os.path.join(
            IMG_DIR,
            f"{slug}.{extension}",
        )

        if (
            os.path.isfile(old_path)
            and old_path != keep_path
        ):
            try:
                os.remove(old_path)
            except OSError:
                pass


def save_recipe_image(
    image,
    slug,
    output_format,
):
    """
    Lưu ảnh dưới dạng PNG hoặc JPG.
    """
    output_format = output_format.lower()

    if output_format == "png":
        image_path = os.path.join(
            IMG_DIR,
            f"{slug}.png",
        )

        # Giữ alpha nếu ảnh có transparency
        if image.mode not in ("RGB", "RGBA", "L", "LA"):
            image = image.convert("RGBA")

        image.save(
            image_path,
            format="PNG",
            optimize=True,
        )

    else:
        image_path = os.path.join(
            IMG_DIR,
            f"{slug}.jpg",
        )

        # JPEG không hỗ trợ alpha
        if image.mode in ("RGBA", "LA"):
            background = Image.new(
                "RGB",
                image.size,
                (255, 255, 255),
            )

            if image.mode == "RGBA":
                background.paste(
                    image,
                    mask=image.getchannel("A"),
                )
            else:
                rgba_image = image.convert("RGBA")
                background.paste(
                    rgba_image,
                    mask=rgba_image.getchannel("A"),
                )

            image = background

        else:
            image = image.convert("RGB")

        image.save(
            image_path,
            format="JPEG",
            quality=90,
            optimize=True,
        )

        output_format = "jpg"

    delete_old_recipe_images(
        slug,
        keep_path=image_path,
    )

    return image_path, output_format


def add_existing_image_to_mapping(recipe):
    """
    Nếu ảnh đã tồn tại nhưng mapping chưa có,
    thêm ảnh đó vào mapping.
    """
    recipe_id = recipe["id"]
    recipe_title = str(recipe["title"]).strip()
    slug = slugify(recipe_title)

    existing_path = get_existing_image_path(slug)

    if existing_path is None:
        return

    mapping = load_mapping()

    if mapping.empty:
        mapping_has_recipe = False
    else:
        mapping_has_recipe = (
            mapping["id"].astype(str)
            == str(recipe_id)
        ).any()

    if mapping_has_recipe:
        return

    extension = os.path.splitext(existing_path)[1]
    extension = extension.lower().lstrip(".")

    if extension == "jpeg":
        extension = "jpg"

    save_mapping(
        recipe_id=recipe_id,
        slug=slug,
        real_name=recipe_title,
        image_path=existing_path,
        image_url="",
        image_format=extension,
    )


# =========================================================
# AUTO DOWNLOAD
# =========================================================

def download_all_recipe_images(
    recipes,
    status_placeholder,
    current_recipe_placeholder,
    progress_bar,
    log_placeholder,
    summary_placeholders,
    initial_existing_count,
):
    total_recipes = len(recipes)

    downloaded_count = 0
    skipped_count = 0
    failed_count = 0

    existing_count = initial_existing_count

    logs = []

    for position, (_, recipe) in enumerate(
        recipes.iterrows(),
        start=1,
    ):
        recipe_id = recipe["id"]
        recipe_title = str(recipe["title"]).strip()
        slug = slugify(recipe_title)

        progress = position / total_recipes
        progress_bar.progress(progress)

        current_recipe_placeholder.write(
            f"Đang xử lý {position}/{total_recipes}: "
            f"**{recipe_title}**"
        )

        existing_image = get_existing_image_path(slug)

        if existing_image is not None:
            skipped_count += 1

            add_existing_image_to_mapping(recipe)

            logs.append(
                f"SKIP | {recipe_id} | "
                f"{recipe_title} | {existing_image}"
            )

        else:
            try:
                image_urls = search_image_urls(
                    recipe_title,
                )

                image, output_format, image_url = (
                    download_first_working_image(image_urls)
                )

                image_path, saved_format = save_recipe_image(
                    image=image,
                    slug=slug,
                    output_format=output_format,
                )

                save_mapping(
                    recipe_id=recipe_id,
                    slug=slug,
                    real_name=recipe_title,
                    image_path=image_path,
                    image_url=image_url,
                    image_format=saved_format,
                )

                downloaded_count += 1
                existing_count += 1

                logs.append(
                    f"OK | {recipe_id} | "
                    f"{recipe_title} | "
                    f"{saved_format.upper()} | "
                    f"{image_path}"
                )

            except Exception as error:
                failed_count += 1

                logs.append(
                    f"FAILED | {recipe_id} | "
                    f"{recipe_title} | {error}"
                )

            time.sleep(
                random.uniform(
                    REQUEST_DELAY_MIN,
                    REQUEST_DELAY_MAX,
                )
            )

        status_placeholder.info(
            f"Đã tải mới: {downloaded_count} | "
            f"Đã bỏ qua: {skipped_count} | "
            f"Lỗi: {failed_count}"
        )

        log_placeholder.code(
            "\n".join(logs[-20:]),
            language="text",
        )

        summary_placeholders["existing"].metric(
            "Đã có ảnh",
            existing_count,
        )

        summary_placeholders["remaining"].metric(
            "Chưa có ảnh",
            total_recipes - existing_count,
        )

    return {
        "downloaded": downloaded_count,
        "skipped": skipped_count,
        "failed": failed_count,
        "logs": logs,
    }


# =========================================================
# STREAMLIT UI
# =========================================================

st.title("Automatic Recipe Image Downloader")

st.write(
    "Chương trình tự động search theo tên recipe + `recipe`, "
    "thử tải lần lượt từng ảnh kết quả cho đến khi tải thành công, "
    "lưu dưới dạng PNG hoặc JPG và bỏ qua những recipe đã có ảnh."
)

try:
    recipes_df = load_recipes()

except Exception as error:
    st.error(str(error))
    st.stop()


total_count = len(recipes_df)
existing_count = 0

for _, recipe_row in recipes_df.iterrows():
    recipe_slug = slugify(recipe_row["title"])

    if get_existing_image_path(recipe_slug):
        existing_count += 1


remaining_count = total_count - existing_count

summary_col1, summary_col2, summary_col3 = st.columns(3)

summary_placeholders = {
    "total": summary_col1.empty(),
    "existing": summary_col2.empty(),
    "remaining": summary_col3.empty(),
}

summary_placeholders["total"].metric(
    "Tổng recipe",
    total_count,
)

summary_placeholders["existing"].metric(
    "Đã có ảnh",
    existing_count,
)

summary_placeholders["remaining"].metric(
    "Chưa có ảnh",
    remaining_count,
)


status_placeholder = st.empty()
current_recipe_placeholder = st.empty()
progress_bar = st.progress(0)
log_placeholder = st.empty()


if "auto_download_finished" not in st.session_state:
    st.session_state.auto_download_finished = False


if remaining_count == 0:
    progress_bar.progress(1.0)

    st.success(
        "Tất cả recipe đều đã có ảnh."
    )

    st.session_state.auto_download_finished = True


elif not st.session_state.auto_download_finished:
    results = download_all_recipe_images(
        recipes=recipes_df,
        status_placeholder=status_placeholder,
        current_recipe_placeholder=current_recipe_placeholder,
        progress_bar=progress_bar,
        log_placeholder=log_placeholder,
        summary_placeholders=summary_placeholders,
        initial_existing_count=existing_count,
    )

    st.session_state.auto_download_finished = True
    st.session_state.last_results = results

    progress_bar.progress(1.0)

    st.success(
        "Hoàn thành tải ảnh tự động."
    )

    result_col1, result_col2, result_col3 = st.columns(3)

    result_col1.metric(
        "Tải thành công",
        results["downloaded"],
    )

    result_col2.metric(
        "Đã bỏ qua",
        results["skipped"],
    )

    result_col3.metric(
        "Thất bại",
        results["failed"],
    )

    with open(
        LOG_PATH,
        "w",
        encoding="utf-8",
    ) as log_file:
        log_file.write(
            "\n".join(results["logs"])
        )

    st.info(
        f"Log đã được lưu tại: {LOG_PATH}"
    )


else:
    last_results = st.session_state.get(
        "last_results"
    )

    if last_results:
        st.success(
            "Quá trình tải ảnh đã chạy trong session này."
        )

        result_col1, result_col2, result_col3 = st.columns(3)

        result_col1.metric(
            "Tải thành công",
            last_results["downloaded"],
        )

        result_col2.metric(
            "Đã bỏ qua",
            last_results["skipped"],
        )

        result_col3.metric(
            "Thất bại",
            last_results["failed"],
        )

    else:
        st.info(
            "Quá trình tải ảnh đã hoàn tất trước đó."
        )


st.divider()
st.subheader("Ảnh đã lưu")

saved_images = [
    filename
    for filename in os.listdir(IMG_DIR)
    if filename.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
]

jpg_count = sum(
    filename.lower().endswith((".jpg", ".jpeg"))
    for filename in saved_images
)

png_count = sum(
    filename.lower().endswith(".png")
    for filename in saved_images
)

image_col1, image_col2, image_col3 = st.columns(3)

image_col1.metric(
    "Tổng ảnh",
    len(saved_images),
)

image_col2.metric(
    "Ảnh JPG",
    jpg_count,
)

image_col3.metric(
    "Ảnh PNG",
    png_count,
)

st.write(
    f"Ảnh được lưu trong thư mục `{IMG_DIR}`."
)