"""
Loc du lieu final_recipes_30k.csv theo 9 buoc:
  1. Loai dong thieu title/ingredients/directions
  2. Parse & chuan hoa ingredients, mapped_ingredients (JSON hoa)
  3. Loai recipe <3 nguyen lieu hoac huong dan qua ngan
  4. Loai title mang tinh danh muc/menu/tips/collection/planning
  5. Loai recipe khong thuoc am thuc (soap, pet food, cleaning, ...)
  6. Chuan hoa title (bo tien to Easy/Best/Ultimate/... khi phu hop) - dung de so khop
  7. Xoa ban ghi trung tuyet doi theo title (chuan hoa) + ingredients
  8. Phat hien trung gan dung bang RapidFuzz (title) + Jaccard (mapped_ingredients)
  9. Cac cap/nhom nghi ngo (khong chac chan) duoc ghi ra manual_review.csv de kiem tra
     thu cong thay vi xoa luon. Neu ban xac nhan xoa, dua id vao confirmed_removals.txt
     (moi id 1 dong) roi chay lai script.

Output:
  - final_recipes_filtered.csv : du lieu sau khi loc
  - filter_log.csv             : moi id -> kept/removed + ly do
  - manual_review.csv          : cac truong hop nghi ngo can nguoi kiem tra
"""

import ast
import json
import re
from collections import defaultdict
from itertools import combinations

import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm

INPUT_CSV = "final_recipes_30k.csv"
OUTPUT_CSV = "final_recipes_filtered.csv"
LOG_CSV = "filter_log.csv"
REVIEW_CSV = "manual_review.csv"
CONFIRMED_REMOVALS_FILE = "confirmed_removals.txt"

MIN_INGREDIENTS = 3
MIN_DIRECTION_WORDS = 8

# ---- step 4: title mang tinh danh muc / menu / tips / collection ----
# hard: du chac chan de tu dong loai
CATEGORY_HARD_PATTERNS = [
    r"\d+\s+(?:easy|best|quick|top|simple)?\s*\w*\s*(?:recipes|dishes|meals|snacks|appetizers)\b",
    r"\btips\s+(?:for|on)\b",
    r"\d+\s+tips\b",
    r"\bways?\s+to\s+(?:use|cook|make|serve)\b",
    r"\bmeal\s*plan(?:ning)?\b",
    r"\bweekly\s+menu\b",
    r"\bmenu\s+plan(?:ning)?\b",
    r"\bshopping\s+list\b",
    r"\bgrocery\s+list\b",
    r"\bround.?up\b",
    r"\brecipe\s+collection\b",
]
# soft: nghi ngo, can nguoi kiem tra truoc khi xoa
CATEGORY_SOFT_PATTERNS = [
    r"\bmenu\b",
    r"\bideas\b",
    r"\bguide\b",
    r"\bplanning\b",
    r"\bcollection\b",
    r"\btips\b",
]

# ---- step 5: khong thuoc am thuc ----
NONFOOD_HARD_PATTERNS = [
    r"\bsoap\b",
    r"\bshampoo\b",
    r"\bdetergent\b",
    r"\bbird\s+seed\b",
    r"\bfertilizer\b",
    r"\bweed\s+killer\b",
    r"\bbug\s+spray\b",
    r"\binsect\s+repellent\b",
    r"\bsunscreen\b",
    r"\bcandle\b",
    r"\(not\s+edible\)",
    r"\bnot\s+edible\b",
    r"\blaundry\b",
]
# Luu y: "dog food"/"cat food"/"pet food" thuong la ten dua vui cho mon an cua NGUOI
# (vd "Dog Food" = mon thit bo sot ca chua, "Human Cat Food" = mon tuna cho nguoi an)
# nen chi dua vao soft list de kiem tra thu cong, khong auto-remove.
NONFOOD_SOFT_PATTERNS = [
    r"\bplay\s*dough\b",
    r"\bslime\b",
    r"\bpotpourri\b",
    r"\bcleaner\b",
    r"\bcleaning\b",
    r"\bdog\s+food\b",
    r"\bcat\s+food\b",
    r"\bpet\s+food\b",
]

# ---- step 6: tien to hay gap, bo khi so khop trung ----
TITLE_PREFIXES = [
    "easy", "best", "the best", "ultimate", "quick", "quick and easy",
    "classic", "simple", "perfect", "homemade", "old fashioned",
    "old-fashioned", "traditional", "authentic", "famous", "super",
    "delicious", "amazing", "yummy", "favorite", "favourite",
    "award winning", "world's best", "worlds best", "no-fail", "nofail",
    "foolproof", "grandma's", "grandmas", "mom's", "moms", "my",
]
TITLE_SUFFIXES = ["recipe", "recipes"]

FUZZY_HIGH_TITLE = 90
FUZZY_HIGH_JACCARD = 0.6
FUZZY_LOW_TITLE = 78
FUZZY_LOW_JACCARD = 0.35


def compile_any(patterns):
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)


CATEGORY_HARD_RE = compile_any(CATEGORY_HARD_PATTERNS)
CATEGORY_SOFT_RE = compile_any(CATEGORY_SOFT_PATTERNS)
NONFOOD_HARD_RE = compile_any(NONFOOD_HARD_PATTERNS)
NONFOOD_SOFT_RE = compile_any(NONFOOD_SOFT_PATTERNS)


def safe_literal_list(value):
    """Parse mot cot dang python-repr list (vd ingredients/directions/mapped_ingredients)."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return None
    return parsed if isinstance(parsed, list) else None


def normalize_title(title):
    t = title.strip().lower()
    t = re.sub(r"[^\w\s']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    changed = True
    while changed:
        changed = False
        for suf in TITLE_SUFFIXES:
            if t.endswith(" " + suf):
                t = t[: -(len(suf) + 1)].strip()
                changed = True
        for pre in TITLE_PREFIXES:
            if t.startswith(pre + " "):
                t = t[len(pre) + 1:].strip()
                changed = True
    return t


def direction_word_count(direction_list):
    return sum(len(str(step).split()) for step in direction_list)


def main():
    df = pd.read_csv(INPUT_CSV)
    df["id"] = df["id"].astype(str)
    log = {rid: {"title": title, "status": "kept", "reason": ""}
           for rid, title in zip(df["id"], df["title"])}
    review_rows = []

    def mark_removed(ids, reason):
        for rid in ids:
            if log[rid]["status"] == "kept":
                log[rid]["status"] = "removed"
                log[rid]["reason"] = reason

    total_start = len(df)

    # ---------- Step 1: loai dong thieu title/ingredients/directions ----------
    def is_blank(x):
        return not isinstance(x, str) or not x.strip()

    missing_mask = (
        df["title"].apply(is_blank)
        | df["ingredients"].apply(is_blank)
        | df["directions"].apply(is_blank)
    )
    mark_removed(df.loc[missing_mask, "id"], "step1_missing_field")
    df = df.loc[~missing_mask].copy()
    print(f"[step1] loai {missing_mask.sum()} dong thieu truong bat buoc, con lai {len(df)}")

    # ---------- Step 2: parse & chuan hoa ingredients / mapped_ingredients ----------
    df["_ingredients_list"] = df["ingredients"].apply(safe_literal_list)
    df["_directions_list"] = df["directions"].apply(safe_literal_list)
    df["_mapped_list"] = df["mapped_ingredients"].apply(safe_literal_list)

    parse_fail_mask = df["_ingredients_list"].isna() | df["_directions_list"].isna()
    mark_removed(df.loc[parse_fail_mask, "id"], "step2_parse_failed")
    df = df.loc[~parse_fail_mask].copy()
    print(f"[step2] loai {parse_fail_mask.sum()} dong parse loi, con lai {len(df)}")

    def clean_ingredient_items(items):
        seen = []
        for item in items:
            s = re.sub(r"\s+", " ", str(item).strip())
            if s and s not in seen:
                seen.append(s)
        return seen

    def clean_mapped_items(items):
        cleaned = []
        for entry in items:
            if not isinstance(entry, dict):
                continue
            name = re.sub(r"\s+", " ", str(entry.get("mapped_name", "")).strip())
            cleaned.append({
                "mapped_id": entry.get("mapped_id"),
                "mapped_name": name,
                "total_grams": round(float(entry.get("total_grams", 0) or 0), 2),
                "grams_per_serving": round(float(entry.get("grams_per_serving", 0) or 0), 2),
            })
        return cleaned

    df["_ingredients_list"] = df["_ingredients_list"].apply(clean_ingredient_items)
    df["_mapped_list"] = df["_mapped_list"].apply(
        lambda v: clean_mapped_items(v) if isinstance(v, list) else []
    )
    df["ingredients"] = df["_ingredients_list"].apply(json.dumps)
    df["mapped_ingredients"] = df["_mapped_list"].apply(json.dumps)
    df["directions"] = df["_directions_list"].apply(json.dumps)

    # ---------- Step 3: loai recipe qua it nguyen lieu / huong dan qua ngan ----------
    n_ing = df["_ingredients_list"].apply(len)
    n_dir_words = df["_directions_list"].apply(direction_word_count)
    too_short_mask = (n_ing < MIN_INGREDIENTS) | (n_dir_words < MIN_DIRECTION_WORDS)
    mark_removed(df.loc[too_short_mask, "id"], "step3_too_short")
    df = df.loc[~too_short_mask].copy()
    print(f"[step3] loai {too_short_mask.sum()} recipe qua ngan, con lai {len(df)}")

    # ---------- Step 4: title mang tinh danh muc/menu/tips/collection ----------
    hard_mask = df["title"].str.contains(CATEGORY_HARD_RE)
    mark_removed(df.loc[hard_mask, "id"], "step4_category_title_hard")
    for rid, title in zip(df.loc[hard_mask, "id"], df.loc[hard_mask, "title"]):
        review_rows.append({"id": rid, "title": title, "type": "category_title",
                             "confidence": "hard_auto_removed", "detail": ""})
    df = df.loc[~hard_mask].copy()
    print(f"[step4] loai {hard_mask.sum()} title danh muc (hard), con lai {len(df)}")

    soft_mask = df["title"].str.contains(CATEGORY_SOFT_RE)
    for rid, title in zip(df.loc[soft_mask, "id"], df.loc[soft_mask, "title"]):
        m = CATEGORY_SOFT_RE.search(title)
        review_rows.append({"id": rid, "title": title, "type": "category_title",
                             "confidence": "soft_needs_review", "detail": m.group(0) if m else ""})
    print(f"[step4] {soft_mask.sum()} title nghi ngo danh muc, ghi vao {REVIEW_CSV} (khong tu dong xoa)")

    # ---------- Step 5: khong thuoc am thuc ----------
    hard_mask = df["title"].str.contains(NONFOOD_HARD_RE)
    mark_removed(df.loc[hard_mask, "id"], "step5_nonfood_hard")
    for rid, title in zip(df.loc[hard_mask, "id"], df.loc[hard_mask, "title"]):
        review_rows.append({"id": rid, "title": title, "type": "nonfood",
                             "confidence": "hard_auto_removed", "detail": ""})
    df = df.loc[~hard_mask].copy()
    print(f"[step5] loai {hard_mask.sum()} recipe khong thuoc am thuc (hard), con lai {len(df)}")

    soft_mask = df["title"].str.contains(NONFOOD_SOFT_RE)
    for rid, title in zip(df.loc[soft_mask, "id"], df.loc[soft_mask, "title"]):
        m = NONFOOD_SOFT_RE.search(title)
        review_rows.append({"id": rid, "title": title, "type": "nonfood",
                             "confidence": "soft_needs_review", "detail": m.group(0) if m else ""})
    print(f"[step5] {soft_mask.sum()} recipe nghi ngo khong thuoc am thuc, ghi vao {REVIEW_CSV}")

    # ---------- Step 6: chuan hoa title de so khop (khong ghi de cot title goc) ----------
    df["_title_norm"] = df["title"].apply(normalize_title)

    # ---------- Step 7: xoa trung tuyet doi theo title (chuan hoa) + ingredients ----------
    df["_dup_key"] = df["_title_norm"] + "||" + df["_ingredients_list"].apply(
        lambda lst: "|".join(sorted(s.lower() for s in lst))
    )
    dup_mask = df.duplicated(subset="_dup_key", keep="first")
    dup_groups = df.groupby("_dup_key")["id"].apply(list)
    for key, ids in dup_groups.items():
        if len(ids) > 1:
            keep_id = ids[0]
            for rid in ids[1:]:
                log[rid]["reason"] = f"step7_exact_duplicate_of_{keep_id}"
    mark_removed(df.loc[dup_mask, "id"], "step7_exact_duplicate")
    df = df.loc[~dup_mask].copy()
    print(f"[step7] loai {dup_mask.sum()} ban ghi trung tuyet doi, con lai {len(df)}")

    # ---------- Step 8: phat hien trung gan dung (RapidFuzz + Jaccard) ----------
    df["_mapped_id_set"] = df["_mapped_list"].apply(
        lambda lst: frozenset(e["mapped_id"] for e in lst if e.get("mapped_id") is not None)
    )

    def block_key(title_norm):
        tokens = title_norm.split()
        if not tokens:
            return ("", "")
        return (tokens[0], tokens[-1])

    df["_block_key"] = df["_title_norm"].apply(block_key)

    blocks = defaultdict(list)
    for rid, key in zip(df["id"], df["_block_key"]):
        blocks[key].append(rid)

    row_by_id = df.set_index("id")
    high_conf_remove = set()
    near_dup_pairs = []

    for key, ids in tqdm(blocks.items(), desc="step8 fuzzy blocks"):
        if len(ids) < 2:
            continue
        for id_a, id_b in combinations(ids, 2):
            if id_a in high_conf_remove or id_b in high_conf_remove:
                continue
            title_a = row_by_id.at[id_a, "_title_norm"]
            title_b = row_by_id.at[id_b, "_title_norm"]
            title_sim = fuzz.token_sort_ratio(title_a, title_b)
            if title_sim < FUZZY_LOW_TITLE:
                continue
            set_a = row_by_id.at[id_a, "_mapped_id_set"]
            set_b = row_by_id.at[id_b, "_mapped_id_set"]
            union = set_a | set_b
            jaccard = len(set_a & set_b) / len(union) if union else 0.0
            if title_sim < FUZZY_LOW_TITLE or jaccard < FUZZY_LOW_JACCARD:
                continue

            if title_sim >= FUZZY_HIGH_TITLE and jaccard >= FUZZY_HIGH_JACCARD:
                high_conf_remove.add(id_b)
                log[id_b]["reason"] = (
                    f"step8_near_duplicate_of_{id_a}"
                    f"(title_sim={title_sim:.0f},jaccard={jaccard:.2f})"
                )
            else:
                near_dup_pairs.append({
                    "id_a": id_a, "title_a": row_by_id.at[id_a, "title"],
                    "id_b": id_b, "title_b": row_by_id.at[id_b, "title"],
                    "title_similarity": round(title_sim, 1),
                    "ingredient_jaccard": round(jaccard, 3),
                })

    for rid in high_conf_remove:
        if log[rid]["status"] == "kept":
            log[rid]["status"] = "removed"
            # ly do chi tiet (id ghep doi + diem so) da duoc gan trong vong lap tren
    df = df.loc[~df["id"].isin(high_conf_remove)].copy()
    print(f"[step8] loai {len(high_conf_remove)} ban ghi trung gan dung (high confidence)")
    print(f"[step8] {len(near_dup_pairs)} cap nghi ngo trung gan dung, ghi vao {REVIEW_CSV} de kiem tra")

    for pair in near_dup_pairs:
        review_rows.append({
            "id": pair["id_a"], "title": pair["title_a"], "type": "near_duplicate_pair",
            "confidence": "soft_needs_review",
            "detail": (f"paired_with_id={pair['id_b']} title='{pair['title_b']}' "
                       f"title_sim={pair['title_similarity']} jaccard={pair['ingredient_jaccard']}"),
        })

    # ---------- Step 9: ap dung xac nhan xoa thu cong (neu co) ----------
    confirmed_ids = set()
    try:
        with open(CONFIRMED_REMOVALS_FILE, "r", encoding="utf-8") as f:
            confirmed_ids = {line.strip() for line in f if line.strip()}
    except FileNotFoundError:
        pass

    if confirmed_ids:
        confirmed_ids &= set(df["id"])
        mark_removed(confirmed_ids, "step9_manual_confirmed_removal")
        df = df.loc[~df["id"].isin(confirmed_ids)].copy()
        print(f"[step9] loai {len(confirmed_ids)} ban ghi theo xac nhan thu cong tu {CONFIRMED_REMOVALS_FILE}")
    else:
        print(f"[step9] chua co {CONFIRMED_REMOVALS_FILE} (hoac rong) - chua xoa gi theo xac nhan thu cong")

    # ---------- Xuat ket qua ----------
    output_cols = [c for c in pd.read_csv(INPUT_CSV, nrows=0).columns]
    df[output_cols].to_csv(OUTPUT_CSV, index=False)

    log_df = pd.DataFrame([
        {"id": rid, "title": info["title"], "status": info["status"], "reason": info["reason"]}
        for rid, info in log.items()
    ])
    log_df.to_csv(LOG_CSV, index=False)

    review_df = pd.DataFrame(review_rows)
    review_df.to_csv(REVIEW_CSV, index=False)

    kept = (log_df["status"] == "kept").sum()
    removed = (log_df["status"] == "removed").sum()
    print()
    print(f"Tong dong ban dau: {total_start}")
    print(f"Giu lai: {kept}  |  Loai bo: {removed}")
    print(f"Da ghi: {OUTPUT_CSV}, {LOG_CSV}, {REVIEW_CSV}")


if __name__ == "__main__":
    main()
