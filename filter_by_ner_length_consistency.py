"""
Tiep tuc kiem tra tinh nhat quan giua 'ingredients' va 'ner' tren recipes_filtered_1.0_v3.csv,
lan nay them dieu kien ve DO DAI danh sach:

  1. len(ner) phai bang len(ingredients) - so luong nguyen lieu rut gon phai khop voi so
     luong dong ingredients day du (khac nhau nghia la NER bo sot hoac gop nhieu dong)
  2. Moi ner-term van phai khop noi dung voi it nhat 1 dong ingredients (giong
     filter_by_ner_consistency.py: substring sau chuan hoa, fallback fuzzy RapidFuzz >=90)

Recipe chi duoc giu lai neu THOA CA HAI dieu kien.

Output:
  - recipes_filtered_1.0_v4.csv
  - recipes_filtered_1.0_v4_log.csv (id, title, len_ingredients, len_ner, match_pct,
    unmatched_terms, status, reason)
"""

import ast
import json
import re

import pandas as pd
from rapidfuzz import fuzz

INPUT_CSV = "recipes_filtered_1.0_v3.csv"
OUTPUT_CSV = "recipes_filtered_1.0_v4.csv"
LOG_CSV = "recipes_filtered_1.0_v4_log.csv"

FUZZY_CUTOFF = 90


def parse_list_field(raw):
    try:
        items = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        try:
            items = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return []
    return items if isinstance(items, list) else []


def normalize(s):
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def singularize(word):
    if word.endswith("ies") and len(word) > 3:
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 3:
        return word[:-2]
    if word.endswith("s") and len(word) > 2 and not word.endswith("ss"):
        return word[:-1]
    return word


def term_matches_any(term_norm, ingredient_lines_norm, combined_text):
    if term_norm in combined_text:
        return True
    words = term_norm.split()
    singular = " ".join(singularize(w) for w in words)
    if singular in combined_text:
        return True
    if any(singular in line or term_norm in line for line in ingredient_lines_norm):
        return True
    best = max(
        (max(fuzz.ratio(term_norm, line), fuzz.partial_ratio(term_norm, line))
         for line in ingredient_lines_norm),
        default=0,
    )
    return best >= FUZZY_CUTOFF


def recipe_stats(ingredients, ners):
    ner_terms = [str(n).strip() for n in ners if str(n).strip()]
    len_ing, len_ner = len(ingredients), len(ner_terms)

    if len_ner == 0:
        return len_ing, len_ner, 0.0, []

    lines_norm = [normalize(i) for i in ingredients]
    combined = " | ".join(lines_norm)

    unmatched = []
    for term in ner_terms:
        term_norm = normalize(term)
        if not term_norm:
            continue
        if not term_matches_any(term_norm, lines_norm, combined):
            unmatched.append(term)

    match_pct = (len_ner - len(unmatched)) / len_ner
    return len_ing, len_ner, match_pct, unmatched


def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Tong so recipe: {len(df)}")

    df["_ingredients"] = df["ingredients"].apply(parse_list_field)
    df["_ner"] = df["ner"].apply(parse_list_field)

    results = df.apply(lambda r: recipe_stats(r["_ingredients"], r["_ner"]), axis=1)
    df["_len_ing"] = results.apply(lambda t: t[0])
    df["_len_ner"] = results.apply(lambda t: t[1])
    df["_match_pct"] = results.apply(lambda t: t[2])
    df["_unmatched"] = results.apply(lambda t: t[3])

    len_ok = df["_len_ing"] == df["_len_ner"]
    content_ok = df["_match_pct"] >= 1.0
    keep_mask = len_ok & content_ok

    def reason(row):
        if row["_len_ing"] != row["_len_ner"]:
            return f"length_mismatch: ingredients={row['_len_ing']} vs ner={row['_len_ner']}"
        if row["_match_pct"] < 1.0:
            return f"content_mismatch: {'; '.join(row['_unmatched'])}"
        return ""

    log_df = df[["id", "title", "_len_ing", "_len_ner", "_match_pct"]].copy()
    log_df.columns = ["id", "title", "len_ingredients", "len_ner", "match_pct"]
    log_df["status"] = keep_mask.map({True: "kept", False: "removed"})
    log_df["reason"] = df.apply(reason, axis=1)
    log_df.loc[log_df["status"] == "kept", "reason"] = ""
    log_df.to_csv(LOG_CSV, index=False)

    output_cols = [c for c in pd.read_csv(INPUT_CSV, nrows=0).columns]
    df.loc[keep_mask, output_cols].to_csv(OUTPUT_CSV, index=False)

    print(f"\nLoai do lech do dai (len ner != len ingredients): {(~len_ok).sum()}")
    print(f"Loai do noi dung khong khop het: {(len_ok & ~content_ok).sum()}")
    print(f"Giu lai: {keep_mask.sum()}  |  Loai bo: {(~keep_mask).sum()}")
    print(f"Da ghi: {OUTPUT_CSV}, {LOG_CSV}")


if __name__ == "__main__":
    main()
