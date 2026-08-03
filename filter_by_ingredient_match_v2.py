"""
Loc recipes_filtered_1.0_v2.csv: chi giu lai recipe co 100% nguyen lieu (cot 'ner') match
duoc voi 'Natural Name' cua ingredients.csv, nhung lan nay CHI dung 2 tang (bo tang
word-overlap so voi filter_by_ingredient_match.py):

  1. Khop tuyet doi voi mot Natural Name
  2. Fallback fuzzy match (RapidFuzz WRatio) de bat loi chinh ta nhe (vd "seasame" -> "sesame")

Output:
  - recipes_filtered_1.0_v3.csv
  - recipes_filtered_1.0_v3_log.csv
"""

import ast
import json

import pandas as pd
from rapidfuzz import process, fuzz

INGREDIENTS_CSV = "ingredients.csv"
INPUT_CSV = "recipes_filtered_1.0_v2.csv"
OUTPUT_CSV = "recipes_filtered_1.0_v3.csv"
LOG_CSV = "recipes_filtered_1.0_v3_log.csv"

MATCH_THRESHOLD = 1.0
FUZZY_CUTOFF = 85


def parse_list_field(raw):
    try:
        items = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        try:
            items = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return []
    return items if isinstance(items, list) else []


def build_natural_names(ingredients_csv):
    ing = pd.read_csv(ingredients_csv)
    return ing["Natural Name"].dropna().astype(str).str.strip().str.lower().unique().tolist()


def make_term_matcher(natural_names):
    nn_set = set(natural_names)

    def is_matched(term):
        term = term.strip().lower()
        if not term:
            return False
        if term in nn_set:
            return True
        match = process.extractOne(term, natural_names, scorer=fuzz.WRatio, score_cutoff=FUZZY_CUTOFF)
        return match is not None

    return is_matched


def main():
    print("Dang xay dung danh sach Natural Name tu ingredients.csv ...")
    natural_names = build_natural_names(INGREDIENTS_CSV)
    print(f"Natural Name: {len(natural_names)} muc duy nhat")
    is_matched = make_term_matcher(natural_names)

    df = pd.read_csv(INPUT_CSV)
    print(f"Tong so recipe: {len(df)}")

    df["_ner"] = df["ner"].apply(parse_list_field)

    all_terms = set()
    for terms in df["_ner"]:
        for t in terms:
            all_terms.add(str(t).strip().lower())
    print(f"So ner-term duy nhat: {len(all_terms)}")

    print("Dang match tung term (1 lan / term, cache lai) ...")
    term_match_cache = {t: is_matched(t) for t in all_terms}
    n_matched_terms = sum(term_match_cache.values())
    print(f"So term match duoc: {n_matched_terms}/{len(all_terms)} "
          f"({n_matched_terms / len(all_terms) * 100:.1f}%)")

    def recipe_stats(terms):
        norm = [str(t).strip().lower() for t in terms]
        total = len(norm)
        if total == 0:
            return 0, 0, 0.0, []
        unmatched = [t for t in norm if not term_match_cache.get(t, False)]
        matched = total - len(unmatched)
        return matched, total, matched / total, unmatched

    stats = df["_ner"].apply(recipe_stats)
    df["_matched"] = stats.apply(lambda t: t[0])
    df["_total"] = stats.apply(lambda t: t[1])
    df["_match_pct"] = stats.apply(lambda t: t[2])
    df["_unmatched"] = stats.apply(lambda t: t[3])

    keep_mask = df["_match_pct"] >= MATCH_THRESHOLD

    log_df = df[["id", "title", "_matched", "_total", "_match_pct", "_unmatched"]].copy()
    log_df.columns = ["id", "title", "matched_ingredients", "total_ingredients", "match_pct", "unmatched_terms"]
    log_df["unmatched_terms"] = log_df["unmatched_terms"].apply(lambda lst: "; ".join(lst))
    log_df["status"] = keep_mask.map({True: "kept", False: "removed"})
    log_df.to_csv(LOG_CSV, index=False)

    output_cols = [c for c in pd.read_csv(INPUT_CSV, nrows=0).columns]
    df.loc[keep_mask, output_cols].to_csv(OUTPUT_CSV, index=False)

    print(f"\nNguong: >= {MATCH_THRESHOLD * 100:.0f}% ner-ingredient phai match")
    print(f"Giu lai: {keep_mask.sum()}  |  Loai bo: {(~keep_mask).sum()}")
    print(f"Da ghi: {OUTPUT_CSV}, {LOG_CSV}")


if __name__ == "__main__":
    main()
