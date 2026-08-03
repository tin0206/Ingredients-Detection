from pathlib import Path
import zipfile

ROOT = Path("real_dataset")
IMG_DIR = ROOT / "images"
LBL_DIR = ROOT / "labels"
OUT = Path("real_dataset_yolo_ultralytics.zip")

CLASSES = [
"apple","apricot","artichoke","arugula","asparagus","avocado","bacon","bamboo_shoot","banana","bean",
"beef","beet","berry","bok_choy","bread","broccoli","brussels_sprout","butter","cabbage","caper",
"carrot","cauliflower","celery","cheese","cherry","chicken","chili_pepper","clam","coconut","coffee",
"coriander","corn","crab","cucumber","duck","dumpling_wrapper","durian","egg","fennel","fish",
"fish_sauce","garlic","ginger","glass_noodle","goat","goose","green_bean","green_onion","ham","horseradish",
"kale","kiwi","kohlrabi","lamb","lemon","lemongrass","lettuce","lime","lobster","mango",
"milk","mushroom","octopus","oil","olives","onion","orange","oyster","papaya","passion_fruit",
"pasta","pasta_sauce","peach","pear","peas","pesto","pickle","pineapple","plum","pomelo",
"pork","potato","powder","pumpkin","rabbit","radicchio","radish","rice","salmon","sausage",
"scallop","shrimp","spinach","strawberry","sweet_potato","tomato","tuna","turnip","whole_spice"
]

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"}
images = sorted([p for p in IMG_DIR.iterdir() if p.suffix.lower() in EXTS])

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    # data.yaml
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASSES))
    z.writestr(
        "data.yaml",
        "path: .\n"
        "train: train.txt\n"
        f"names:\n{names_block}\n"
    )

    train_lines = []

    for img in images:
        img_arc = f"images/train/{img.name}"
        z.write(img, img_arc)
        train_lines.append(img_arc)

        label = LBL_DIR / f"{img.stem}.txt"
        lbl_arc = f"labels/train/{img.stem}.txt"
        if label.exists():
            z.write(label, lbl_arc)
        else:
            z.writestr(lbl_arc, "")

    z.writestr("train.txt", "\n".join(train_lines))

print("Done:", OUT)
print("Images:", len(images))