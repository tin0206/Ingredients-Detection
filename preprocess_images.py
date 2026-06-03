import cv2
import numpy as np
from pathlib import Path
from rembg import remove, new_session
import tqdm

# ================= CONFIG =================
INPUT_DIR = Path("raw_ingredients")   
OUTPUT_DIR = Path("clean_output")     
MODEL_NAME = "u2net" 
# ==========================================

def crop_to_content(img_bgra):
    alpha = img_bgra[:, :, 3]
    coords = cv2.findNonZero(alpha)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        return img_bgra[y:y+h, x:x+w]
    return img_bgra

def parse_indices(input_str):
    """Hàm xử lý chuỗi nhập vào để trả về danh sách index nguyên (int)"""
    indices = set()
    parts = input_str.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                indices.update(range(start, end + 1))
            except ValueError:
                print(f"⚠ Bỏ qua đoạn không hợp lệ: {part}")
        elif part:
            try:
                indices.add(int(part))
            except ValueError:
                print(f"⚠ Bỏ qua index không hợp lệ: {part}")
    return sorted(list(indices))

def main():
    print("--- Tool Tách Nền & Crop Ảnh Nguyên Liệu ---")
    print("Mẹo: Bạn có thể nhập '1,2,5' hoặc '10-20' hoặc '1,5,10-15'")
    target_input = input("Nhập danh sách index hoặc khoảng cần xử lý: ").strip()
    
    target_indices = []
    if target_input:
        # Chuyển đổi chuỗi nhập thành danh sách các chuỗi tên folder
        target_indices = [str(i) for i in parse_indices(target_input)]

    print(f"Đang tải model {MODEL_NAME}...")
    session = new_session(MODEL_NAME)

    # Lọc danh sách folder
    if target_indices:
        # Chỉ lấy những folder thực sự tồn tại trong INPUT_DIR
        subfolders = [INPUT_DIR / idx for idx in target_indices if (INPUT_DIR / idx).is_dir()]
    else:
        # Nếu không nhập gì thì lấy toàn bộ và sắp xếp theo số
        subfolders = sorted([f for f in INPUT_DIR.iterdir() if f.is_dir()], 
                            key=lambda x: int(x.name) if x.name.isdigit() else 999)

    if not subfolders:
        print("Không tìm thấy folder nào phù hợp trong raw_ingredients!")
        return

    print(f"Tổng cộng sẽ xử lý {len(subfolders)} class.")

    for folder in subfolders:
        print(f"\n📁 Đang xử lý Class: {folder.name}")
        out_class_dir = OUTPUT_DIR / folder.name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp")
        files = []
        for ext in extensions:
            files.extend(list(folder.glob(ext)))

        if not files:
            continue

        for img_path in tqdm.tqdm(files, desc=f"  Processing {folder.name}"):
            try:
                # Đọc ảnh hỗ trợ tiếng Việt
                img_array = np.fromfile(str(img_path), np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is None: continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                output_rgba = remove(img_rgb, session=session)
                
                output_bgra = cv2.cvtColor(np.array(output_rgba), cv2.COLOR_RGBA2BGRA)
                final_img = crop_to_content(output_bgra)

                # Lưu ảnh hỗ trợ tiếng Việt (ghi gián tiếp qua buffer)
                save_path = out_class_dir / (img_path.stem + ".png")
                is_success, im_buf_arr = cv2.imencode(".png", final_img)
                if is_success:
                    im_buf_arr.tofile(str(save_path))

            except Exception as e:
                print(f"\n  ⛔ Lỗi tại {img_path.name}: {e}")

    print(f"\n✅ Hoàn thành! Kết quả tại: {OUTPUT_DIR.absolute()}")

if __name__ == "__main__":
    main()