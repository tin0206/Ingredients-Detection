"""
Kiem tra tinh nhat quan giua cot 'ingredients' (mo ta day du, vd "2 tablespoons extra
virgin olive oil") va cot 'ner' (ban rut gon, vd "olive oil") cua tung recipe trong
recipes_filtered_1.0_v1.csv.

Voi moi ner-term, kiem tra xem no co "bat nguon" tu mot dong ingredients nao khong:
  1. Substring truc tiep (sau khi chuan hoa: bo dau/ky tu dac biet, vi du gap loi encoding
     kieu "jalape�os" van duoc coi la lien quan toi "jalapenos")
  2. Neu khong, fallback bang RapidFuzz (ratio/partial_ratio >= 90) de bat cac sai lech nho
     con lai do mat ky tu khi encode loi.

Recipe duoc giu lai neu TOAN BO ner-term deu khop duoc voi it nhat 1 dong ingredients.

Output:
  - recipes_filtered_1.0_v2.csv
  - recipes_filtered_1.0_v2_log.csv (id, title, match_pct, unmatched_terms, status)
"""

import ast
import json
import re

import pandas as pd
from rapidfuzz import fuzz

INPUT_CSV = "recipes_filtered_1.0_v1.csv"
OUTPUT_CSV = "recipes_filtered_1.0_v2.csv"
LOG_CSV = "recipes_filtered_1.0_v2_log.csv"

FUZZY_CUTOFF = 90
KEEP_THRESHOLD = 1.0  # % ner-term toi thieu phai khop de giu lai recipe


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
    s = re.sub(r"[^a-z0-9\s]", "", s)  # bo dau/ky tu dac biet/loi encoding (vd '�')
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
    # fallback: cho phep sai lech nho do loi encoding (vd mat 1 ky tu)
    best = max(
        (max(fuzz.ratio(term_norm, line), fuzz.partial_ratio(term_norm, line))
         for line in ingredient_lines_norm),
        default=0,
    )
    return best >= FUZZY_CUTOFF


def recipe_match_stats(ingredients, ners):
    ner_terms = [str(n).strip() for n in ners if str(n).strip()]
    if not ner_terms:
        return 0, 0, 0.0, []

    lines_norm = [normalize(i) for i in ingredients]
    combined = " | ".join(lines_norm)

    matched, unmatched = 0, []
    for term in ner_terms:
        term_norm = normalize(term)
        if not term_norm:
            continue
        if term_matches_any(term_norm, lines_norm, combined):
            matched += 1
        else:
            unmatched.append(term)

    total = len(ner_terms)
    return matched, total, matched / total if total else 0.0, unmatched


def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Tong so recipe: {len(df)}")

    df["_ingredients"] = df["ingredients"].apply(parse_list_field)
    df["_ner"] = df["ner"].apply(parse_list_field)

    results = df.apply(lambda r: recipe_match_stats(r["_ingredients"], r["_ner"]), axis=1)
    df["_matched"] = results.apply(lambda t: t[0])
    df["_total"] = results.apply(lambda t: t[1])
    df["_match_pct"] = results.apply(lambda t: t[2])
    df["_unmatched"] = results.apply(lambda t: t[3])

    keep_mask = df["_match_pct"] >= KEEP_THRESHOLD

    log_df = df[["id", "title", "_matched", "_total", "_match_pct", "_unmatched"]].copy()
    log_df.columns = ["id", "title", "matched_terms", "total_terms", "match_pct", "unmatched_terms"]
    log_df["unmatched_terms"] = log_df["unmatched_terms"].apply(lambda lst: "; ".join(lst))
    log_df["status"] = keep_mask.map({True: "kept", False: "removed"})
    log_df.to_csv(LOG_CSV, index=False)

    output_cols = [c for c in pd.read_csv(INPUT_CSV, nrows=0).columns]
    df.loc[keep_mask, output_cols].to_csv(OUTPUT_CSV, index=False)

    print(f"\nNguong: >= {KEEP_THRESHOLD * 100:.0f}% ner-term phai khop voi ingredients")
    print(f"Giu lai: {keep_mask.sum()}  |  Loai bo: {(~keep_mask).sum()}")
    print(f"Da ghi: {OUTPUT_CSV}, {LOG_CSV}")


if __name__ == "__main__":
    main()
